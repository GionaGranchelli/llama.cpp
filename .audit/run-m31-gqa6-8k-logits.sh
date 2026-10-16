#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${M31_CANDIDATE_WORKSPACE:-/tmp/m31-current-01a910937}"
build_dir="${M31_BUILD_DIR:-$workspace/build-m31}"
image="localhost/miinfer-dev:rocm-7.2.1"
gpu="${M31_GPU_INDEX:-2}"
bdf="${M31_GPU_BDF:-0000:88:00.0}"
model="/home/machinist/models/Qwen3.8-27B-Q4_K_M.gguf"
model_sha="7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
debug="$build_dir/bin/llama-debug"

[[ -x "$debug" ]] || { echo "missing $debug" >&2; exit 2; }
[[ -z "$(git -C "$workspace" status --porcelain)" ]] || { echo "source tree is dirty" >&2; exit 3; }
bus="$(rocm-smi -d "$gpu" --showbus 2>/dev/null | awk -F ': ' '/PCI Bus:/ {print $NF}')"
[[ "$bus" == "$bdf" ]] || { echo "GPU$gpu maps to '$bus', expected $bdf" >&2; exit 4; }
[[ "$(sha256sum "$model" | awk '{print $1}')" == "$model_sha" ]] || { echo "model SHA mismatch" >&2; exit 5; }
kfd_info="$(rocm-smi --showpids 2>/dev/null)"
grep -q 'No KFD PIDs currently running' <<< "$kfd_info" || { echo "$kfd_info" >&2; exit 6; }

cap_for_bdf() {
    local card path
    for card in /sys/class/drm/card[0-9]*; do
        [[ -e "$card/device" ]] || continue
        [[ "$(basename "$(readlink -f "$card/device")")" == "$1" ]] || continue
        for path in "$card"/device/hwmon/hwmon*/power1_cap; do
            [[ -r "$path" ]] && { cat "$path"; return; }
        done
    done
    return 1
}

image_id="$(podman image inspect --format '{{.Id}}' "$image")"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="/tmp/m31-gqa6-8k-logits-gpu${gpu}-$timestamp"
mkdir -p "$run_dir"
trap 'echo evidence="$run_dir"' EXIT

if [[ -n "${M31_PROMPT_FILE:-}" ]]; then
    cp "$M31_PROMPT_FILE" "$run_dir/prompt.txt"
else
{
    cat <<'EOF'
Review this GPU attention implementation. Explain how its Q8 dot4 candidate
computes attention and identify numerical or execution risks. Use the source
excerpts as context; do not assume either path is correct. Distinguish the
default Q8 attention path from the opt-in candidate and state what evidence is
needed before claiming full-model output equivalence.

EOF
    printf 'File: fattn-tile.cuh, lines 583-985\n'
    sed -n '583,985p' "$workspace/ggml/src/ggml-cuda/fattn-tile.cuh"
    printf '\nFile: fattn-tile.cuh, lines 986-1250\n'
    sed -n '986,1250p' "$workspace/ggml/src/ggml-cuda/fattn-tile.cuh"
    printf '\nFile: fattn.cu, lines 660-800\n'
    sed -n '660,800p' "$workspace/ggml/src/ggml-cuda/fattn.cu"
} > "$run_dir/prompt.txt"
fi

{
    printf 'source_commit=%s\nmodel_sha256=%s\nimage_id=%s\n' "$(git -C "$workspace" rev-parse HEAD)" "$model_sha" "$image_id"
    printf 'gpu_index=%s\ngpu_bdf=%s\n' "$gpu" "$bdf"
    printf 'cap_gpu0_uw=%s\ncap_gpu2_uw=%s\n' \
        "$(cap_for_bdf 0000:04:00.0)" "$(cap_for_bdf "$bdf")"
    printf 'prompt_bytes=%s\nprompt_sha256=%s\n' \
        "$(wc -c < "$run_dir/prompt.txt")" "$(sha256sum "$run_dir/prompt.txt" | awk '{print $1}')"
} > "$run_dir/metadata.txt"
printf 'utc,junction_c,fan_rpm,pwm\n' > "$run_dir/temperature.csv"

junction_input=""
for card in /sys/class/drm/card[0-9]*; do
    [[ -e "$card/device" ]] || continue
    [[ "$(basename "$(readlink -f "$card/device")")" == "$bdf" ]] || continue
    for label in "$card"/device/hwmon/hwmon*/temp*_label; do
        [[ -r "$label" && "$(cat "$label")" == junction ]] || continue
        junction_input="${label%_label}_input"
        break 2
    done
done
[[ -r "$junction_input" ]] || { echo "junction sensor missing for $bdf" >&2; exit 7; }
hwmon_dir="${junction_input%/*}"
fan_input="$hwmon_dir/fan1_input"
pwm_input="$hwmon_dir/pwm1"
read_temp() {
    local raw
    raw="$(cat "$junction_input")" || return 1
    [[ "$raw" =~ ^[0-9]+$ ]] || return 1
    awk -v t="$raw" 'BEGIN { printf "%.1f", t/1000 }'
}

run_variant() {
    local variant="$1" start temp elapsed reason="" status=0 container
    local output_dir="/artifacts/logits-$variant"
    local host_output_dir="$run_dir/logits-$variant"
    local extra_env=() cool_started="$(date +%s)"
    mkdir -p "$host_output_dir"
    [[ "$variant" != candidate ]] || extra_env=(-e GGML_CUDA_FA_Q8_DOT4_PREPACK=1)
    while true; do
        start="$(read_temp)" || { echo "junction telemetry unavailable" >&2; return 7; }
        awk -v t="$start" 'BEGIN { exit !(t <= 40) }' && break
        (( $(date +%s) - cool_started < 600 )) || { echo "GPU2 did not cool to 40 C (now ${start} C)" >&2; return 8; }
        sleep 5
    done
    container="m31-gqa6-logits-${timestamp}-${variant}"
    printf '%s start_junction_c=%s\n' "$variant" "$start" >> "$run_dir/metadata.txt"
    podman run --rm --name "$container" --device=/dev/kfd --device=/dev/dri --group-add video \
        --security-opt seccomp=unconfined --security-opt label=disable \
        -e "ROCR_VISIBLE_DEVICES=$gpu" -e "LD_LIBRARY_PATH=/workspace/$(basename "$build_dir")/bin" "${extra_env[@]}" \
        -v "$workspace:/workspace" -v /home/machinist/models:/models:ro -v "$run_dir:/artifacts" -w /workspace "$image" \
        "./$(basename "$build_dir")/bin/llama-debug" -m /models/Qwen3.8-27B-Q4_K_M.gguf \
        -f /artifacts/prompt.txt -c 16384 -b 16384 -ub 1024 -ctk q8_0 -ctv q8_0 \
        -fa on -ngl 99 -dev ROCm0 --save-logits --logits-output-dir "$output_dir" -v \
        > "$run_dir/$variant.log" 2>&1 &
    local pid=$!
    local started="$(date +%s)"
    while kill -0 "$pid" 2>/dev/null; do
        if ! temp="$(read_temp)"; then
            reason="junction telemetry unavailable"
            podman kill "$container" >/dev/null 2>&1 || true
            break
        fi
        elapsed=$(( $(date +%s) - started ))
        fan_rpm=na
        fan_pwm=na
        [[ ! -r "$fan_input" ]] || fan_rpm="$(cat "$fan_input")"
        [[ ! -r "$pwm_input" ]] || fan_pwm="$(cat "$pwm_input")"
        printf '%s,%s,%s,%s\n' "$(date -u +%FT%TZ)" "$temp" "$fan_rpm" "$fan_pwm" >> "$run_dir/temperature.csv"
        if awk -v t="$temp" 'BEGIN { exit !(t >= 84) }'; then
            reason="junction reached ${temp} C; stopped before 85 C"
        elif (( elapsed >= 300 )); then
            reason="run exceeded 300 seconds"
        fi
        if [[ -n "$reason" ]]; then podman kill "$container" >/dev/null 2>&1 || true; break; fi
        sleep 0.2
    done
    wait "$pid" || status=$?
    printf '%s stop_reason=%s status=%s\n' "$variant" "$reason" "$status" >> "$run_dir/metadata.txt"
    [[ "$status" == 0 && -z "$reason" ]]
}

run_variant default
run_variant candidate
grep -Fq 'selected opt-in GFX906 Q8 dot4 flash-attention tile' "$run_dir/candidate.log" || {
    echo "candidate dot4 dispatch was not confirmed" >&2
    exit 9
}

default_dir="$run_dir/logits-default/llamacpp-Qwen3.8-27B-Q4_K_M"
candidate_dir="$run_dir/logits-candidate/llamacpp-Qwen3.8-27B-Q4_K_M"
cmp "${default_dir}-tokens.bin" "${candidate_dir}-tokens.bin"
python3 - "${default_dir}.bin" "${candidate_dir}.bin" <<'PY'
import array, heapq, math, sys

def read(path):
    values = array.array("f")
    with open(path, "rb") as stream:
        values.frombytes(stream.read())
    if sys.byteorder != "little":
        values.byteswap()
    return values

a, b = map(read, sys.argv[1:])
if len(a) != len(b) or not a:
    raise SystemExit("logit lengths differ or are empty")
if not all(math.isfinite(x) for x in a) or not all(math.isfinite(x) for x in b):
    raise SystemExit("non-finite logits")
diff = [x-y for x, y in zip(a, b)]
dot = sum(x*y for x, y in zip(a, b))
cosine = dot / math.sqrt(sum(x*x for x in a) * sum(y*y for y in b))
top_a = heapq.nlargest(10, range(len(a)), key=a.__getitem__)
top_b = heapq.nlargest(10, range(len(b)), key=b.__getitem__)
print("logit_count", len(a))
print("max_abs", max(map(abs, diff)))
print("rms", math.sqrt(sum(x*x for x in diff) / len(diff)))
print("cosine", cosine)
print("top1", max(range(len(a)), key=a.__getitem__), max(range(len(b)), key=b.__getitem__))
print("top10_overlap", len(set(top_a) & set(top_b)))
PY
sha256sum "${default_dir}.bin" "${candidate_dir}.bin" "${default_dir}-tokens.bin" "${candidate_dir}-tokens.bin" >> "$run_dir/metadata.txt"
grep -E 'Token ids \(' "$run_dir/default.log" | tail -1 >> "$run_dir/metadata.txt"
grep -F 'selected opt-in GFX906 Q8 dot4 flash-attention tile' "$run_dir/candidate.log" | wc -l >> "$run_dir/metadata.txt"
cat "$run_dir/metadata.txt"
