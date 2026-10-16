#!/usr/bin/env bash
set -Eeuo pipefail

gpu_id="${1:?usage: run-guarded-m31-64k.sh ROCM_INDEX [default|j40|dot4]}"
variant="${2:-default}"
workspace="${WORKSPACE:-/tmp/m31-gqa48-rocm721-20261016}"
build_dir="${BUILD_DIR:-build-gqa48}"
bench_binary="$workspace/$build_dir/bin/llama-bench"
hip_backend="$workspace/$build_dir/bin/libggml-hip.so"
model="/home/machinist/models/Qwen3.8-27B-Q4_K_M.gguf"
model_sha="7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"

case "$gpu_id" in
    0) target_bdf="0000:04:00.0" ;;
    2) target_bdf="0000:88:00.0" ;;
    *) echo "GPU index must be 0 or 2" >&2; exit 2 ;;
esac
case "$variant" in
    default|j40|dot4) ;;
    *) echo "variant must be default, j40, or dot4" >&2; exit 2 ;;
esac

if [[ ! -x "$bench_binary" || ! -f "$hip_backend" ]]; then
    echo "Refusing run: expected llama-bench and HIP backend under $workspace/$build_dir/bin" >&2
    exit 8
fi
if [[ "$variant" == j40 ]] && ! grep -q 'GGML_CUDA_MMQ_Q4K_J40' "$workspace/ggml/src/ggml-cuda/mmq.cu"; then
    echo "Refusing J=40 run: selector is absent from the source tree" >&2
    exit 9
fi
if [[ "$variant" == dot4 ]] && ! grep -q 'GGML_CUDA_FA_Q8_DOT4_PREPACK' "$workspace/ggml/src/ggml-cuda/fattn.cu"; then
    echo "Refusing dot4 run: selector is absent from the source tree" >&2
    exit 11
fi
source_sha="$(git -C "$workspace" rev-parse HEAD)"
source_status="$(git -C "$workspace" status --short)"
if [[ -n "$source_status" ]]; then
    echo "Refusing run: source tree is dirty: $source_status" >&2
    exit 10
fi

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
    echo "Refusing sustained prefill: live MI50 caps are ${cap0}uW and ${cap2}uW; both must be <=180000000uW" >&2
    exit 3
fi

bus="$(rocm-smi -d "$gpu_id" --showbus | awk -F ': ' '/PCI Bus/ {print $NF; exit}')"
if [[ "$bus" != "$target_bdf" ]]; then
    echo "Refusing run: ROCm index $gpu_id maps to $bus, expected $target_bdf" >&2
    exit 4
fi
junction_input_for_bdf() {
    local wanted="$1" card_path bdf label input
    for card_path in /sys/class/drm/card[0-9]*; do
        [[ -e "$card_path/device" ]] || continue
        bdf="$(basename "$(readlink -f "$card_path/device")")"
        [[ "$bdf" == "$wanted" ]] || continue
        for label in "$card_path"/device/hwmon/hwmon*/temp*_label; do
            [[ -r "$label" && "$(cat "$label")" == junction ]] || continue
            input="${label%_label}_input"
            [[ -r "$input" ]] && { printf '%s\n' "$input"; return 0; }
        done
    done
    return 1
}
junction_input="$(junction_input_for_bdf "$target_bdf")" || { echo "GPU junction sysfs sensor unavailable for $target_bdf" >&2; exit 4; }
read_junction() {
    local millidegrees
    millidegrees="$(cat "$junction_input" 2>/dev/null)" || return 1
    [[ "$millidegrees" =~ ^[0-9]+$ ]] || return 1
    awk -v t="$millidegrees" 'BEGIN { printf "%.1f", t/1000 }'
}
if ! rocm-smi --showpids | grep -q 'No KFD PIDs currently running'; then
    echo "Refusing run: KFD processes are already using a GPU" >&2
    exit 5
fi
actual_model_sha="$(sha256sum "$model" | awk '{print $1}')"
if [[ "$actual_model_sha" != "$model_sha" ]]; then
    echo "Refusing run: model SHA-256 is $actual_model_sha" >&2
    exit 6
fi

start_temp="$(read_junction)" || start_temp=""
if [[ -z "$start_temp" ]] || awk -v t="$start_temp" 'BEGIN { exit !(t > 40) }'; then
    echo "Refusing run: GPU junction must start at or below 40 C (read '$start_temp')" >&2
    exit 7
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="/tmp/m31-64k-${gpu_id}-${variant}-${timestamp}"
container_name="m31-64k-${gpu_id}-${timestamp}"
mkdir -p "$run_dir"
printf 'source=%s\ngpu=%s\nbdf=%s\nvariant=%s\ncap0_uw=%s\ncap2_uw=%s\nmodel_sha256=%s\n' \
    "$source_sha" "$gpu_id" "$target_bdf" "$variant" "$cap0" "$cap2" "$actual_model_sha" > "$run_dir/metadata.txt"
printf 'binary_sha256=%s\nhip_backend_sha256=%s\nbuild_dir=%s\n' \
    "$(sha256sum "$bench_binary" | awk '{print $1}')" \
    "$(sha256sum "$hip_backend" | awk '{print $1}')" "$build_dir" >> "$run_dir/metadata.txt"
printf 'utc,junction_c\n' > "$run_dir/temperature.csv"

extra_env=()
if [[ "$variant" == j40 ]]; then
    extra_env=(-e GGML_CUDA_MMQ_Q4K_J40=1)
elif [[ "$variant" == dot4 ]]; then
    extra_env=(-e GGML_CUDA_FA_Q8_DOT4_PREPACK=1)
fi

podman run --rm --name "$container_name" --device=/dev/kfd --device=/dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable \
    -e "ROCR_VISIBLE_DEVICES=$gpu_id" -e "LD_LIBRARY_PATH=/workspace/$build_dir/bin" "${extra_env[@]}" \
    -v "$workspace:/workspace:Z" -v /home/machinist/models:/models:ro,z -w /workspace \
    localhost/miinfer-dev:rocm-7.2.1 \
    "./$build_dir/bin/llama-bench" -m /models/Qwen3.8-27B-Q4_K_M.gguf \
    -p 65536 -n 0 -r 1 --no-warmup -b 2048 -ub 1024 -ctk q8_0 -ctv q8_0 \
    -fa on -ngl 99 -dev ROCm0 -o jsonl --progress > "$run_dir/benchmark.log" 2>&1 &
bench_pid=$!
started_epoch="$(date +%s)"
stop_reason=""
warned=0

while kill -0 "$bench_pid" 2>/dev/null; do
    now="$(date -u +%FT%TZ)"
    temp="$(read_junction)" || temp=""
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
        stop_reason="junction reached ${temp} C; stopped before the 85 C hard limit"
        podman kill "$container_name" >/dev/null 2>&1 || true
        break
    fi
    if (( $(date +%s) - started_epoch >= 1800 )); then
        stop_reason="benchmark exceeded 1800-second run limit"
        podman kill "$container_name" >/dev/null 2>&1 || true
        break
    fi
    sleep 0.2
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
