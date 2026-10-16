#!/usr/bin/env bash
set -Eeuo pipefail

variant="${1:?usage: run-guarded-m31-8k-dot4.sh default|dot4 [prompt_tokens]}"
prompt_tokens="${2:-8192}"
case "$variant" in
    default|dot4) ;;
    *) echo "variant must be default or dot4" >&2; exit 2 ;;
esac

gpu_id=2
target_bdf=0000:88:00.0
workspace=/tmp/m31-fork-candidate
image=localhost/miinfer-dev:rocm-7.2.1
model=/home/machinist/models/Qwen3.8-27B-Q4_K_M.gguf
model_sha=7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
model_marker=/tmp/m31-qwen35-27b-model-sha256.txt
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
run_dir="/tmp/m31-dot4-prefill-${prompt_tokens}-${variant}-${timestamp}"
container_name="m31-dot4-prefill-${prompt_tokens}-${variant}-${timestamp}"
mkdir -p "$run_dir"
backend_lib="$workspace/build-mi50/bin/libggml-hip.so"
printf 'variant=%s\nprompt_tokens=%s\ngpu_index=%s\nbdf=%s\nsource_commit=%s\nsource_status=%s\nsource_sha256=%s\nmodel_sha256=%s\nbinary_sha256=%s\nbackend_sha256=%s\nimage=%s\nstart_junction_c=%s\n' \
    "$variant" "$prompt_tokens" "$gpu_id" "$target_bdf" \
    "$(git -C "$workspace" rev-parse HEAD)" \
    "$(git -C "$workspace" status --short | tr '\n' ' ')" \
    "$(sha256sum "$workspace/ggml/src/ggml-cuda/fattn-tile.cuh" | awk '{print $1}')" \
    "$actual_model_sha" \
    "$(sha256sum "$workspace/build-mi50/bin/llama-bench" | awk '{print $1}')" \
    "$(sha256sum "$backend_lib" | awk '{print $1}')" "$image" "$start_temp" \
    > "$run_dir/metadata.txt"
printf 'utc,junction_c\n' > "$run_dir/temperature.csv"

extra_env=()
if [[ "$variant" == dot4 ]]; then
    extra_env=(-e GGML_CUDA_FA_Q8_DOT4_PREPACK=1)
fi
verbose_args=()
if [[ "${M31_VERBOSE:-0}" == 1 ]]; then
    verbose_args=(-v)
fi

podman run --rm --name "$container_name" --device=/dev/kfd --device=/dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable \
    -e "ROCR_VISIBLE_DEVICES=$gpu_id" -e LD_LIBRARY_PATH=/out/build-mi50/bin "${extra_env[@]}" \
    -v "$workspace:/out:Z" -v /home/machinist/models:/models:ro,z -w /out "$image" \
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
