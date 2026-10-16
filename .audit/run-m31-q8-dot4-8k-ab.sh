#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${M31_WORKSPACE:-$PWD}"
gpu="${M31_GPU:-2}"
expected_bdf="${M31_GPU_BDF:-0000:88:00.0}"
image="localhost/miinfer-dev:rocm-7.2.1"
model=/home/machinist/models/Qwen3.8-27B-Q4_K_M.gguf
model_sha=7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${M31_EVIDENCE_ROOT:-/tmp}/m31-q8-dot4-8k-ab-${timestamp}"
mkdir -p "$run_dir"

bus="$(rocm-smi -d "$gpu" --showbus 2>/dev/null | awk -F ': ' '/PCI Bus:/ {print $NF}')"
[[ "$bus" == "$expected_bdf" ]] || { echo "GPU $gpu maps to '$bus', expected $expected_bdf" >&2; exit 2; }
test -x "$workspace/build-mi50/bin/llama-bench"
source_status="$(git -C "$workspace" status --porcelain)"
[[ -z "$source_status" ]] || { echo "source worktree is dirty; use a clean, pinned build" >&2; exit 10; }
grep -q 'GGML_CUDA_FA_Q8_DOT4_PREPACK' "$workspace/ggml/src/ggml-cuda/fattn-common.cuh" || {
    echo 'source does not contain the Q8 dot4 prepack path' >&2
    exit 11
}
grep -q 'selected opt-in GFX906 Q8 dot4' "$workspace/ggml/src/ggml-cuda/fattn.cu" || {
    echo 'source does not contain the Q8 dot4 route confirmation' >&2
    exit 12
}
strings "$workspace/build-mi50/bin/libggml-hip.so" | grep -q 'GGML_CUDA_FA_Q8_DOT4_PREPACK' || {
    echo 'HIP backend binary does not contain the Q8 dot4 dispatch flag' >&2
    exit 13
}
actual_model_sha="$(sha256sum "$model" | awk '{print $1}')"
[[ "$actual_model_sha" == "$model_sha" ]] || { echo "model SHA mismatch: $actual_model_sha" >&2; exit 3; }
kfd_info="$(rocm-smi --showpids 2>/dev/null)"
grep -q 'No KFD PIDs currently running' <<< "$kfd_info" || { echo 'refusing run while a KFD process is active' >&2; exit 4; }

cap=""
for card in /sys/class/drm/card[0-9]*; do
    [[ -e "$card/device" ]] || continue
    [[ "$(basename "$(readlink -f "$card/device")")" == "$expected_bdf" ]] || continue
    for cap_file in "$card"/device/hwmon/hwmon*/power1_cap; do
        [[ -r "$cap_file" ]] && cap="$(cat "$cap_file")"
    done
done
[[ -n "$cap" ]] || { echo "unable to read power cap for $expected_bdf" >&2; exit 5; }

{
    printf 'utc=%s\nsource_commit=%s\nsource_status=%s\n' \
        "$(date -u +%FT%TZ)" "$(git -C "$workspace" rev-parse HEAD)" \
        "$(git -C "$workspace" status --short | tr '\n' ' ')"
    printf 'fattn_tile_sha256=%s\nfattn_sha256=%s\nllama_bench_sha256=%s\nlibggml_hip_sha256=%s\n' \
        "$(sha256sum "$workspace/ggml/src/ggml-cuda/fattn-tile.cuh" | awk '{print $1}')" \
        "$(sha256sum "$workspace/ggml/src/ggml-cuda/fattn.cu" | awk '{print $1}')" \
        "$(sha256sum "$workspace/build-mi50/bin/llama-bench" | awk '{print $1}')" \
        "$(sha256sum "$workspace/build-mi50/bin/libggml-hip.so" | awk '{print $1}')"
    printf 'image_id=%s\nmodel_sha256=%s\ngpu=%s\ngpu_bdf=%s\npower_cap_uw=%s\n' \
        "$(podman image inspect --format '{{.Id}}' "$image")" "$actual_model_sha" "$gpu" "$bus" "$cap"
    printf '%s\n' "$kfd_info"
    rocm-smi -d "$gpu" --showtemp --showfan
} > "$run_dir/metadata.txt"
printf 'utc,variant,junction_c\n' > "$run_dir/temperature.csv"

run_variant() {
    local variant="$1" env_args=() container temp status=0 warned80=0
    local deadline=$((SECONDS + 600))
    while true; do
        temp="$(rocm-smi -d "$gpu" --showtemp 2>/dev/null | awk '/Sensor junction/ {print $NF}')"
        [[ -n "$temp" ]] || { echo 'junction telemetry unavailable' >&2; return 6; }
        awk -v t="$temp" 'BEGIN { exit !(t <= 40) }' && break
        (( SECONDS < deadline )) || { echo "$variant did not cool to 40 C within 10 minutes" >&2; return 7; }
        sleep 10
    done
    [[ "$variant" != dot4 ]] || env_args=(-e GGML_CUDA_FA_Q8_DOT4_PREPACK=1)
    container="m31-q8-dot4-8k-${timestamp}-${variant}"
    printf '%s_start_junction_c=%s\n' "$variant" "$temp" >> "$run_dir/metadata.txt"
    podman run --rm --name "$container" --device=/dev/kfd --device=/dev/dri --group-add video \
        --security-opt seccomp=unconfined --security-opt label=disable \
        -e "ROCR_VISIBLE_DEVICES=$gpu" -e LD_LIBRARY_PATH=/workspace/build-mi50/bin "${env_args[@]}" \
        -v "$workspace:/workspace" -v /home/machinist/models:/models:ro,z -w /workspace "$image" \
        ./build-mi50/bin/llama-bench -m /models/Qwen3.8-27B-Q4_K_M.gguf \
        -p 8192 -n 0 -r 1 --no-warmup -b 2048 -ub 1024 \
        -ctk q8_0 -ctv q8_0 -fa on -ngl 99 -dev ROCm0 -o jsonl --progress \
        > "$run_dir/$variant.log" 2>&1 &
    local pid=$!
    local run_deadline=$((SECONDS + 180))
    while kill -0 "$pid" 2>/dev/null; do
        temp="$(rocm-smi -d "$gpu" --showtemp 2>/dev/null | awk '/Sensor junction/ {print $NF}')"
        printf '%s,%s,%s\n' "$(date -u +%FT%TZ)" "$variant" "$temp" >> "$run_dir/temperature.csv"
        if [[ -z "$temp" ]] || awk -v t="$temp" 'BEGIN { exit !(t >= 84) }'; then
            echo "$variant stopped at thermal guard (${temp:-no telemetry} C)" >> "$run_dir/metadata.txt"
            podman kill "$container" >/dev/null 2>&1 || true
            break
        fi
        if (( SECONDS >= run_deadline )); then
            echo "$variant stopped at 180-second timeout" >> "$run_dir/metadata.txt"
            podman kill "$container" >/dev/null 2>&1 || true
            break
        fi
        if awk -v t="$temp" 'BEGIN { exit !(t >= 80) }'; then
            if [[ "$warned80" == 0 ]]; then
                echo "$variant crossed 80 C warning at $temp C" >> "$run_dir/metadata.txt"
                warned80=1
            fi
        fi
        sleep 1
    done
    wait "$pid" || status=$?
    printf '%s_status=%s\n' "$variant" "$status" >> "$run_dir/metadata.txt"
    [[ "$status" == 0 ]] || return "$status"
    local build_commit
    build_commit="$(sed -n 's/.*"build_commit": "\([^"]*\)".*/\1/p' "$run_dir/$variant.log" | head -n1)"
    [[ "$build_commit" == "$(git -C "$workspace" rev-parse --short=7 HEAD)" ]] || {
        echo "$variant binary reports build commit '$build_commit', expected current source" >&2
        return 14
    }
    if [[ "$variant" == dot4 ]]; then
        grep -q 'GGML_CUDA_FA_Q8_DOT4_PREPACK' "$run_dir/$variant.log" || {
            echo 'dot4 run completed without a route-confirmation log' >&2
            return 8
        }
    else
        if grep -q 'GGML_CUDA_FA_Q8_DOT4_PREPACK' "$run_dir/$variant.log"; then
            echo 'default run unexpectedly selected dot4' >&2
            return 9
        fi
    fi
}

run_variant default || { status=$?; cat "$run_dir/metadata.txt" "$run_dir/temperature.csv" "$run_dir/default.log"; echo "evidence=$run_dir"; exit "$status"; }
run_variant dot4 || { status=$?; cat "$run_dir/metadata.txt" "$run_dir/temperature.csv" "$run_dir/default.log" "$run_dir/dot4.log" 2>/dev/null || true; echo "evidence=$run_dir"; exit "$status"; }
rocm-smi -d "$gpu" --showtemp --showfan >> "$run_dir/metadata.txt"
cat "$run_dir/default.log" "$run_dir/dot4.log"
echo "evidence=$run_dir"
