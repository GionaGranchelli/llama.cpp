#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${M31_CANDIDATE_WORKSPACE:-/tmp/m31-fork-candidate}"
image="localhost/miinfer-dev:rocm-7.2.1"
model=/home/machinist/models/Qwen3.8-27B-Q4_K_M.gguf
model_sha=7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
prompt=/tmp/m31-fork-context/prompt.txt
gpu_id="${M31_GPU:-0}"
case "$gpu_id" in
    0) expected_bdf=0000:04:00.0 ;;
    2) expected_bdf=0000:88:00.0 ;;
    *) echo "M31_GPU must be 0 or 2" >&2; exit 2 ;;
esac
case "${1:-both}" in
    both) variants=(default prepack) ;;
    default|prepack) variants=("$1") ;;
    *) echo "usage: $0 [both|default|prepack]" >&2; exit 2 ;;
esac
max_tokens=64
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${M31_EVIDENCE_ROOT:-/tmp}/m31-dot4-lds-cli-gpu${gpu_id}-${timestamp}"
mkdir -p "$run_dir"

bus="$(rocm-smi -d "$gpu_id" --showbus 2>/dev/null | awk -F ': ' '/PCI Bus:/ {print $NF}')"
[[ "$bus" == "$expected_bdf" ]] || { echo "GPU$gpu_id maps to '$bus', expected $expected_bdf" >&2; exit 2; }
kfd_info="$(rocm-smi --showpids 2>/dev/null)"
grep -q 'No KFD PIDs currently running' <<< "$kfd_info" || { echo 'refusing run while another KFD process is active' >&2; exit 3; }
actual_model_sha="$(sha256sum "$model" | awk '{print $1}')"
[[ "$actual_model_sha" == "$model_sha" ]] || { echo "model SHA mismatch: $actual_model_sha" >&2; exit 4; }
prompt_sha="$(sha256sum "$prompt" | awk '{print $1}')"

{
    printf 'source_commit=%s\n' "$(git -C "$workspace" rev-parse HEAD)"
    printf 'source_status=%s\n' "$(git -C "$workspace" status --short | tr '\n' ' ')"
    printf 'fattn_tile_sha256=%s\n' "$(sha256sum "$workspace/ggml/src/ggml-cuda/fattn-tile.cuh" | awk '{print $1}')"
    printf 'llama_cli_sha256=%s\n' "$(sha256sum "$workspace/build-mi50/bin/llama-cli" | awk '{print $1}')"
    printf 'libggml_hip_sha256=%s\n' "$(sha256sum "$workspace/build-mi50/bin/libggml-hip.so" | awk '{print $1}')"
    printf 'image_id=%s\nmodel_sha256=%s\nprompt_sha256=%s\ngpu_bdf=%s\n' \
        "$(podman image inspect --format '{{.Id}}' "$image")" "$actual_model_sha" "$prompt_sha" "$bus"
    printf '%s\n' "$kfd_info"
    rocm-smi --showtemp
} > "$run_dir/metadata.txt"
printf 'utc,junction_c\n' > "$run_dir/temperature.csv"

for variant in "${variants[@]}"; do
    # Run each path only from a cool start so the second path does not inherit
    # the thermal state of the first.
    while true; do
        start_temp="$(rocm-smi -d "$gpu_id" --showtemp 2>/dev/null | awk '/Sensor junction/ {print $NF}')"
        [[ -n "$start_temp" ]] || { echo 'junction telemetry unavailable' >&2; exit 6; }
        if awk -v t="$start_temp" 'BEGIN { exit !(t <= 40) }'; then break; fi
        echo "$variant waiting for GPU$gpu_id to cool (junction=${start_temp} C)" >&2
        sleep 10
    done
    extra_env=()
    [[ "$variant" != prepack ]] || extra_env=(-e GGML_CUDA_FA_Q8_DOT4_PREPACK=1)
    container="m31-dot4-lds-cli-${timestamp}-gpu${gpu_id}-${variant}"
    printf '\n%s start_junction_c=%s\n' "$variant" "$start_temp" >> "$run_dir/metadata.txt"
    podman run --rm --name "$container" --device=/dev/kfd --device=/dev/dri --group-add video \
        --security-opt seccomp=unconfined --security-opt label=disable \
        -e "ROCR_VISIBLE_DEVICES=$gpu_id" -e LD_LIBRARY_PATH=/workspace/build-mi50/bin "${extra_env[@]}" \
        -v "$workspace:/workspace" -v /home/machinist/models:/models:ro \
        -v /tmp/m31-fork-context:/context:ro -w /workspace "$image" \
        ./build-mi50/bin/llama-cli -m /models/Qwen3.8-27B-Q4_K_M.gguf -f /context/prompt.txt \
        -n "$max_tokens" -c 8192 -b 2048 -ub 512 -fa on -ctk q8_0 -ctv q8_0 -ngl 99 \
        --seed 42 --temp 0 --no-display-prompt --simple-io -v \
        > "$run_dir/${variant}.log" 2>&1 &
    pid=$!
    stop_reason=""
    while kill -0 "$pid" 2>/dev/null; do
        temp="$(rocm-smi -d "$gpu_id" --showtemp 2>/dev/null | awk '/Sensor junction/ {print $NF}')"
        printf '%s,%s\n' "$(date -u +%FT%TZ)" "$temp" >> "$run_dir/temperature.csv"
        if [[ -z "$temp" ]] || awk -v t="$temp" 'BEGIN { exit !(t >= 84) }'; then
            stop_reason="GPU$gpu_id telemetry unavailable or junction reached ${temp:-unknown} C"
            podman kill "$container" >/dev/null 2>&1 || true
            break
        fi
        sleep 1
    done
    status=0
    wait "$pid" || status=$?
    printf '%s stop_reason=%s status=%s\n' "$variant" "$stop_reason" "$status" >> "$run_dir/metadata.txt"
    [[ "$status" == 0 && -z "$stop_reason" ]] || exit 5
done

rocm-smi -d "$gpu_id" --showtemp >> "$run_dir/metadata.txt"
echo "evidence=$run_dir"
