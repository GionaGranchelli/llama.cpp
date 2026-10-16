#!/usr/bin/env bash
set -Eeuo pipefail

variant="${1:?usage: run-guarded-m31-8k-prepack.sh default|prepack [prompt_tokens]}"
prompt_tokens="${2:-8192}"
case "$variant" in
    default|prepack) ;;
    *) echo "variant must be default or prepack" >&2; exit 2 ;;
esac

gpu_id=2
target_bdf=0000:88:00.0
candidate_workspace="${M31_CANDIDATE_WORKSPACE:-${M31_WORKSPACE:-/tmp/m31-fork-candidate}}"
baseline_workspace="${M31_BASELINE_WORKSPACE:-/tmp/m31-fork-full}"
if [[ "$variant" == prepack ]]; then
    workspace="$candidate_workspace"
else
    workspace="$baseline_workspace"
fi
image=localhost/miinfer-dev:rocm-7.2.1
model=/home/machinist/models/Qwen3.8-27B-Q4_K_M.gguf
model_sha=7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
model_marker=/tmp/m31-qwen35-27b-model-sha256.txt

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
if (( cap0 > 180000000 || cap2 > 180000000 )); then
    echo "Refusing prefill: live MI50 caps are ${cap0}uW and ${cap2}uW; both must be <=180000000uW" >&2
    exit 3
fi

pci_info="$(rocm-smi -d "$gpu_id" --showbus 2>/dev/null)"
bus="$(awk -F ': ' '/PCI Bus:/ {print $NF}' <<< "$pci_info")"
[[ "$bus" == "$target_bdf" ]] || { echo "ROCm index $gpu_id maps to $bus" >&2; exit 3; }
pids_info="$(rocm-smi --showpids 2>/dev/null)"
if ! grep -q 'No KFD PIDs currently running' <<< "$pids_info"; then
    echo "Refusing run: a KFD process is active" >&2
    exit 4
fi
model_identity="$(stat -c '%d:%i:%s:%Y' "$model")"
actual_model_sha=""
if [[ -r "$model_marker" ]]; then
    read -r marked_identity marked_sha < "$model_marker"
    if [[ "$marked_identity" == "$model_identity" ]]; then
        actual_model_sha="$marked_sha"
    fi
fi
if [[ -z "$actual_model_sha" ]]; then
    actual_model_sha="$(sha256sum "$model" | awk '{print $1}')"
    printf '%s %s\n' "$model_identity" "$actual_model_sha" > "$model_marker"
fi
[[ "$actual_model_sha" == "$model_sha" ]] || { echo "Model SHA mismatch: $actual_model_sha" >&2; exit 5; }
temp_info="$(rocm-smi -d "$gpu_id" --showtemp 2>/dev/null)"
start_temp="$(awk '/Sensor junction/ {print $NF}' <<< "$temp_info")"
if [[ -z "$start_temp" ]] || awk -v t="$start_temp" 'BEGIN { exit !(t > 40) }'; then
    echo "Refusing run: junction must start at or below 40 C (got '$start_temp')" >&2
    exit 6
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="/tmp/m31-prepack-prefill-${prompt_tokens}-${variant}-${timestamp}"
container_name="m31-prepack-prefill-${prompt_tokens}-${variant}-${timestamp}"
source_commit="$(git -C "$workspace" rev-parse HEAD)"
mkdir -p "$run_dir"
printf 'source_base=%s\nvariant=%s\nprompt_tokens=%s\ngpu_index=%s\nbdf=%s\ncap0_uw=%s\ncap2_uw=%s\nmodel_sha256=%s\nbinary_sha256=%s\nimage=%s\nstart_junction_c=%s\n' \
    "$source_commit" \
    "$variant" "$prompt_tokens" "$gpu_id" "$target_bdf" "$cap0" "$cap2" "$actual_model_sha" \
    "$(sha256sum "$workspace/build-mi50/bin/llama-bench" | awk '{print $1}')" "$image" "$start_temp" \
    > "$run_dir/metadata.txt"
for source_file in ggml/src/ggml-cuda/fattn-common.cuh ggml/src/ggml-cuda/fattn-tile.cuh ggml/src/ggml-cuda/fattn.cu; do
    printf '%s_sha256=%s\n' "$source_file" \
        "$(sha256sum "$workspace/$source_file" | awk '{print $1}')" >> "$run_dir/metadata.txt"
done
printf 'utc,junction_c\n' > "$run_dir/temperature.csv"

extra_env=()
if [[ "$variant" == prepack ]]; then
    extra_env=(-e GGML_CUDA_FA_Q8_DOT4_PREPACK=1)
fi
if [[ "${M31_GRAPH_DEBUG:-0}" == 1 ]]; then
    extra_env+=(-e GGML_CUDA_GRAPH_DEBUG=1)
fi
verbose_args=()
if [[ "${M31_VERBOSE:-0}" == 1 ]]; then
    verbose_args=(-v)
fi

podman run --rm --name "$container_name" --device=/dev/kfd --device=/dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable \
    -e "ROCR_VISIBLE_DEVICES=$gpu_id" "${extra_env[@]}" \
    -v "$workspace:/workspace:Z" -v /home/machinist/models:/models:ro,z -w /workspace "$image" \
    ./build-mi50/bin/llama-bench -m /models/Qwen3.8-27B-Q4_K_M.gguf \
    -p "$prompt_tokens" -n 0 -r 1 --no-warmup -b 2048 -ub 1024 \
    -ctk q8_0 -ctv q8_0 -fa on -ngl 99 -dev ROCm0 -o jsonl --progress "${verbose_args[@]}" \
    > "$run_dir/benchmark.log" 2>&1 &
bench_pid=$!
stop_reason=""
warned=0

while kill -0 "$bench_pid" 2>/dev/null; do
    now="$(date -u +%FT%TZ)"
    temp_info="$(rocm-smi -d "$gpu_id" --showtemp 2>/dev/null)" || temp_info=""
    temp="$(awk '/Sensor junction/ {print $NF}' <<< "$temp_info")"
    if [[ -z "$temp" ]]; then
        stop_reason="junction telemetry unavailable; fail-closed stop"
        podman kill "$container_name" >/dev/null 2>&1 || true
        break
    fi
    printf '%s,%s\n' "$now" "$temp" >> "$run_dir/temperature.csv"
    if (( warned == 0 )) && awk -v t="$temp" 'BEGIN { exit !(t >= 80) }'; then
        echo "$now warning: junction reached ${temp} C" | tee -a "$run_dir/benchmark.log"
        warned=1
    fi
    if awk -v t="$temp" 'BEGIN { exit !(t >= 84) }'; then
        stop_reason="junction reached ${temp} C; stopped before 85 C"
        podman kill "$container_name" >/dev/null 2>&1 || true
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
