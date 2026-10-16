#!/usr/bin/env bash
set -Eeuo pipefail

gpu_id="${1:?usage: run-m31-guarded-64k-dot4.sh ROCM_INDEX default|prepack}"
variant="${2:?usage: run-m31-guarded-64k-dot4.sh ROCM_INDEX default|prepack}"
workspace="${M31_CANDIDATE_WORKSPACE:-/tmp/m31-fork-candidate}"
image=localhost/miinfer-dev:rocm-7.2.1
model=/home/machinist/models/Qwen3.8-27B-Q4_K_M.gguf
model_sha=7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
model_marker=/tmp/m31-qwen35-27b-model-sha256.txt
prompt_tokens=65536

case "$gpu_id" in
    0) target_bdf=0000:04:00.0 ;;
    2) target_bdf=0000:88:00.0 ;;
    *) echo "GPU index must be 0 or 2" >&2; exit 2 ;;
esac
case "$variant" in
    default|prepack) ;;
    *) echo "variant must be default or prepack" >&2; exit 2 ;;
esac

cap_for_bdf() {
    local wanted="$1" card_path bdf cap_file
    for card_path in /sys/class/drm/card[0-9]*; do
        [[ -e "$card_path/device" ]] || continue
        bdf="$(basename "$(readlink -f "$card_path/device")")"
        [[ "$bdf" == "$wanted" ]] || continue
        for cap_file in "$card_path"/device/hwmon/hwmon*/power1_cap; do
            [[ -r "$cap_file" ]] && { cat "$cap_file"; return; }
        done
    done
    return 1
}

cap0="$(cap_for_bdf 0000:04:00.0)"
cap2="$(cap_for_bdf 0000:88:00.0)"
(( cap0 <= 180000000 && cap2 <= 180000000 )) || {
    echo "Refusing sustained 64K run: caps are ${cap0}uW and ${cap2}uW; both must be <=180000000uW" >&2
    exit 3
}
bus="$(rocm-smi -d "$gpu_id" --showbus 2>/dev/null | awk -F ': ' '/PCI Bus:/ {print $NF}')"
[[ "$bus" == "$target_bdf" ]] || { echo "GPU $gpu_id maps to $bus, expected $target_bdf" >&2; exit 4; }
kfd_info="$(rocm-smi --showpids 2>/dev/null)"
grep -q 'No KFD PIDs currently running' <<< "$kfd_info" || { echo 'Refusing run while another KFD process is active' >&2; exit 5; }

model_identity="$(stat -c '%d:%i:%s:%Y' "$model")"
actual_model_sha=""
if [[ -r "$model_marker" ]]; then
    read -r marked_identity marked_sha < "$model_marker"
    [[ "$marked_identity" != "$model_identity" ]] || actual_model_sha="$marked_sha"
fi
if [[ -z "$actual_model_sha" ]]; then
    actual_model_sha="$(sha256sum "$model" | awk '{print $1}')"
    printf '%s %s\n' "$model_identity" "$actual_model_sha" > "$model_marker"
fi
[[ "$actual_model_sha" == "$model_sha" ]] || { echo "Model SHA mismatch: $actual_model_sha" >&2; exit 6; }

start_temp="$(rocm-smi -d "$gpu_id" --showtemp 2>/dev/null | awk '/Sensor junction/ {print $NF}')"
[[ -n "$start_temp" ]] && awk -v t="$start_temp" 'BEGIN { exit !(t <= 40) }' || {
    echo "Refusing run: junction must start <=40 C (got '${start_temp:-unknown}')" >&2
    exit 7
}

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="/tmp/m31-dot4-64k-${gpu_id}-${variant}-${timestamp}"
container="m31-dot4-64k-${gpu_id}-${variant}-${timestamp}"
mkdir -p "$run_dir"
backend_lib="$workspace/build-mi50/bin/libggml-hip.so"
printf 'variant=%s\nprompt_tokens=%s\ngpu_index=%s\nbdf=%s\nsource_commit=%s\nsource_status=%s\nsource_sha256=%s\nmodel_sha256=%s\nbinary_sha256=%s\nbackend_sha256=%s\nimage_id=%s\ncap0_uw=%s\ncap2_uw=%s\nstart_junction_c=%s\n' \
    "$variant" "$prompt_tokens" "$gpu_id" "$bus" \
    "$(git -C "$workspace" rev-parse HEAD)" \
    "$(git -C "$workspace" status --short | tr '\n' ' ')" \
    "$(sha256sum "$workspace/ggml/src/ggml-cuda/fattn-tile.cuh" | awk '{print $1}')" \
    "$actual_model_sha" \
    "$(sha256sum "$workspace/build-mi50/bin/llama-bench" | awk '{print $1}')" \
    "$(sha256sum "$backend_lib" | awk '{print $1}')" \
    "$(podman image inspect --format '{{.Id}}' "$image")" "$cap0" "$cap2" "$start_temp" \
    > "$run_dir/metadata.txt"
printf 'utc,junction_c\n' > "$run_dir/temperature.csv"
printf 'load=' >> "$run_dir/metadata.txt"; uptime >> "$run_dir/metadata.txt"
printf 'nproc=' >> "$run_dir/metadata.txt"; nproc >> "$run_dir/metadata.txt"
printf '%s\n' "$kfd_info" >> "$run_dir/metadata.txt"

extra_env=()
if [[ "$variant" == prepack ]]; then
    extra_env=(-e GGML_CUDA_FA_Q8_DOT4_PREPACK=1)
fi
podman run --rm --name "$container" --device=/dev/kfd --device=/dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable \
    -e "ROCR_VISIBLE_DEVICES=$gpu_id" -e LD_LIBRARY_PATH=/workspace/build-mi50/bin "${extra_env[@]}" \
    -v "$workspace:/workspace" -v /home/machinist/models:/models:ro,z \
    -w /workspace "$image" \
    ./build-mi50/bin/llama-bench -m /models/Qwen3.8-27B-Q4_K_M.gguf \
    -p "$prompt_tokens" -n 0 -r 1 --no-warmup -b 2048 -ub 1024 \
    -ctk q8_0 -ctv q8_0 -fa on -ngl 99 -dev ROCm0 -o jsonl --progress \
    > "$run_dir/benchmark.log" 2>&1 &
bench_pid=$!
stop_reason=""
warned=0
while kill -0 "$bench_pid" 2>/dev/null; do
    now="$(date -u +%FT%TZ)"
    temp="$(rocm-smi -d "$gpu_id" --showtemp 2>/dev/null | awk '/Sensor junction/ {print $NF}')" || temp=""
    if [[ -z "$temp" ]]; then
        stop_reason='junction telemetry unavailable; fail-closed stop'
        podman kill "$container" >/dev/null 2>&1 || true
        break
    fi
    printf '%s,%s\n' "$now" "$temp" >> "$run_dir/temperature.csv"
    if (( warned == 0 )) && awk -v t="$temp" 'BEGIN { exit !(t >= 80) }'; then
        echo "$now warning: junction reached ${temp} C" | tee -a "$run_dir/benchmark.log"
        warned=1
    fi
    if awk -v t="$temp" 'BEGIN { exit !(t >= 84) }'; then
        stop_reason="junction reached ${temp} C; stopped before 85 C"
        podman kill "$container" >/dev/null 2>&1 || true
        break
    fi
    sleep 1
done

set +e
wait "$bench_pid"
run_status=$?
set -e
printf 'stop_reason=%s\nrun_status=%s\n' "$stop_reason" "$run_status" >> "$run_dir/metadata.txt"
rocm-smi -d "$gpu_id" --showtemp >> "$run_dir/metadata.txt"
cat "$run_dir/benchmark.log"
echo "evidence=$run_dir"
[[ -z "$stop_reason" && $run_status -eq 0 ]]
