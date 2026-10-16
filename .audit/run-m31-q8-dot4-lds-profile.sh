#!/usr/bin/env bash
set -Eeuo pipefail

workspace=/tmp/m31-fork-candidate
image=localhost/miinfer-dev:rocm-7.2.1
gpu=2
expected_bdf=0000:88:00.0
model=/home/machinist/models/Qwen3.8-27B-Q4_K_M.gguf
model_sha=7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="/tmp/m31-q8-dot4-profile-${timestamp}"
mkdir -p "$run_dir"

bus="$(rocm-smi -d "$gpu" --showbus 2>/dev/null | awk -F ': ' '/PCI Bus:/ {print $NF}')"
[[ "$bus" == "$expected_bdf" ]] || { echo "GPU $gpu maps to $bus" >&2; exit 2; }
actual_model_sha="$(sha256sum "$model" | awk '{print $1}')"
[[ "$actual_model_sha" == "$model_sha" ]] || { echo "model SHA mismatch: $actual_model_sha" >&2; exit 3; }
kfd_info="$(rocm-smi --showpids 2>/dev/null)"
grep -q 'No KFD PIDs currently running' <<< "$kfd_info" || { echo 'KFD is busy' >&2; exit 4; }

{
    printf 'source_commit=%s\nsource_status=%s\nsource_sha256=%s\n' \
        "$(git -C "$workspace" rev-parse HEAD)" \
        "$(git -C "$workspace" status --short | tr '\n' ' ')" \
        "$(sha256sum "$workspace/ggml/src/ggml-cuda/fattn-tile.cuh" | awk '{print $1}')"
    printf 'binary_sha256=%s\nbackend_sha256=%s\nimage_id=%s\nmodel_sha256=%s\ngpu=%s\nbdf=%s\n' \
        "$(sha256sum "$workspace/build-mi50/bin/llama-bench" | awk '{print $1}')" \
        "$(sha256sum "$workspace/build-mi50/bin/libggml-hip.so" | awk '{print $1}')" \
        "$(podman image inspect --format '{{.Id}}' "$image")" "$actual_model_sha" "$gpu" "$bus"
    printf '%s\n' "$kfd_info"
    rocm-smi -d "$gpu" --showtemp
} > "$run_dir/metadata.txt"

run_path() {
    local variant="$1" env_args=() container temp status=0
    while true; do
        temp="$(rocm-smi -d "$gpu" --showtemp 2>/dev/null | awk '/Sensor junction/ {print $NF}')"
        [[ -n "$temp" ]] || { echo 'junction telemetry unavailable' >&2; return 5; }
        if awk -v t="$temp" 'BEGIN { exit !(t <= 40) }'; then break; fi
        echo "$variant waiting for cool start: ${temp} C" >&2
        sleep 10
    done
    [[ "$variant" != prepack ]] || env_args=(-e GGML_CUDA_FA_Q8_DOT4_PREPACK=1)
    container="m31-q8-dot4-profile-${timestamp}-${variant}"
    printf '%s start_junction_c=%s\n' "$variant" "$temp" >> "$run_dir/metadata.txt"
    podman run --rm --name "$container" --device=/dev/kfd --device=/dev/dri --group-add video \
        --security-opt seccomp=unconfined --security-opt label=disable \
        -e "ROCR_VISIBLE_DEVICES=$gpu" -e LD_LIBRARY_PATH=/workspace/build-mi50/bin "${env_args[@]}" \
        -v "$workspace:/workspace" -v "$run_dir:/out:Z" \
        -v /home/machinist/models:/models:ro,z -w /workspace "$image" \
        rocprof --stats --timestamp on -o "/out/$variant.csv" \
        ./build-mi50/bin/llama-bench -m /models/Qwen3.8-27B-Q4_K_M.gguf \
        -p 2048 -n 0 -r 1 --no-warmup -b 2048 -ub 1024 \
        -ctk q8_0 -ctv q8_0 -fa on -ngl 99 -dev ROCm0 -o jsonl --progress \
        > "$run_dir/$variant.stdout" 2>&1 &
    local pid=$!
    : > "$run_dir/$variant.temperature.csv"
    while kill -0 "$pid" 2>/dev/null; do
        temp="$(rocm-smi -d "$gpu" --showtemp 2>/dev/null | awk '/Sensor junction/ {print $NF}')"
        printf '%s,%s\n' "$(date -u +%FT%TZ)" "$temp" >> "$run_dir/$variant.temperature.csv"
        if [[ -z "$temp" ]] || awk -v t="$temp" 'BEGIN { exit !(t >= 84) }'; then
            echo "$variant thermal guard stopped run at ${temp:-unknown} C" >&2
            podman kill "$container" >/dev/null 2>&1 || true
            break
        fi
        sleep 1
    done
    wait "$pid" || status=$?
    printf '%s status=%s\n' "$variant" "$status" >> "$run_dir/metadata.txt"
    [[ "$status" == 0 ]] || return "$status"
}

run_path default
run_path prepack
rocm-smi -d "$gpu" --showtemp >> "$run_dir/metadata.txt"
echo "evidence=$run_dir"
