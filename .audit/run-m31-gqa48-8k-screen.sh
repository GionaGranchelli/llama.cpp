#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${M31_CANDIDATE_WORKSPACE:-/tmp/m31-gqa48-rocm721-20261016}"
build_dir="${M31_BUILD_DIR:-$workspace/build-gqa48}"
image="localhost/miinfer-dev:rocm-7.2.1"
gpu="${M31_GPU:-2}"
ubatch="${M31_UB:-1024}"
prompt_tokens="${M31_PROMPT_TOKENS:-8192}"
run_candidate="${M31_RUN_CANDIDATE:-1}"
candidate_mode="${M31_CANDIDATE_MODE:-dot4}"
model="/home/machinist/models/Qwen3.8-27B-Q4_K_M.gguf"
model_sha="7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
case "$gpu" in
    0) expected_bdf=0000:04:00.0 ;;
    2) expected_bdf=0000:88:00.0 ;;
    *) echo "M31_GPU must be 0 or 2" >&2; exit 2 ;;
esac
case "$build_dir" in
    "$workspace"/*) build_rel="${build_dir#"$workspace"/}" ;;
    *) echo "M31_BUILD_DIR must be inside M31_CANDIDATE_WORKSPACE" >&2; exit 2 ;;
esac
[[ "$ubatch" =~ ^[0-9]+$ ]] && (( ubatch > 0 && ubatch <= 2048 )) || {
    echo "M31_UB must be an integer from 1 to 2048" >&2
    exit 2
}
[[ "$run_candidate" == 0 || "$run_candidate" == 1 ]] || {
    echo "M31_RUN_CANDIDATE must be 0 or 1" >&2
    exit 2
}
[[ "$prompt_tokens" =~ ^[0-9]+$ ]] && (( prompt_tokens > 0 && prompt_tokens <= 65536 )) || {
    echo "M31_PROMPT_TOKENS must be an integer from 1 to 65536" >&2
    exit 2
}
[[ "$candidate_mode" == dot4 || "$candidate_mode" == direct_q8 ]] || {
    echo "M31_CANDIDATE_MODE must be dot4 or direct_q8" >&2
    exit 2
}

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${M31_EVIDENCE_ROOT:-/tmp}/m31-gqa48-8k-screen-gpu${gpu}-${timestamp}"
mkdir -p "$run_dir"

bus="$(rocm-smi -d "$gpu" --showbus 2>/dev/null | awk -F ': ' '/PCI Bus:/ {print $NF}')"
[[ "$bus" == "$expected_bdf" ]] || { echo "GPU$gpu maps to '$bus', expected $expected_bdf" >&2; exit 3; }
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
junction_input="$(junction_input_for_bdf "$expected_bdf")" || { echo "GPU junction sysfs sensor unavailable for $expected_bdf" >&2; exit 3; }
kfd_info="$(rocm-smi --showpids 2>/dev/null)"
grep -q 'No KFD PIDs currently running' <<< "$kfd_info" || { echo 'refusing run while a KFD process is active' >&2; exit 4; }
actual_model_sha="$(sha256sum "$model" | awk '{print $1}')"
[[ "$actual_model_sha" == "$model_sha" ]] || { echo "model SHA mismatch: $actual_model_sha" >&2; exit 5; }

if [[ ! -x "$build_dir/bin/llama-bench" ]]; then
    podman run --rm --security-opt seccomp=unconfined --security-opt label=disable \
        -v "$workspace:/workspace" -w /workspace "$image" \
        cmake --build "$build_rel" --target llama-bench --parallel 4
fi
binary="$build_dir/bin/llama-bench"
backend="$build_dir/bin/libggml-hip.so"
[[ -x "$binary" && -f "$backend" ]] || { echo 'llama-bench or HIP backend missing after build' >&2; exit 6; }

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
image_id="$(podman image inspect --format '{{.Id}}' "$image")"
{
    printf 'source_commit=%s\nsource_status=%s\n' \
        "$(git -C "$workspace" rev-parse HEAD)" "$(git -C "$workspace" status --short | tr '\n' ' ')"
    printf 'source_sha256=%s\nbinary_sha256=%s\nbackend_sha256=%s\n' \
        "$(sha256sum "$workspace/ggml/src/ggml-cuda/fattn-tile.cuh" | awk '{print $1}')" \
        "$(sha256sum "$binary" | awk '{print $1}')" "$(sha256sum "$backend" | awk '{print $1}')"
    printf 'image_id=%s\nmodel_sha256=%s\ngpu_index=%s\ngpu_bdf=%s\ncap0_uw=%s\ncap2_uw=%s\nprompt_tokens=%s\nubatch=%s\nrun_candidate=%s\ncandidate_mode=%s\n' \
        "$image_id" "$actual_model_sha" "$gpu" "$bus" "$cap0" "$cap2" "$prompt_tokens" "$ubatch" "$run_candidate" "$candidate_mode"
    printf '%s\n' "$kfd_info"
    uptime
    nproc
    rocm-smi -d "$gpu" --showtemp
} > "$run_dir/metadata.txt"
printf 'utc,junction_c\n' > "$run_dir/temperature.csv"

read_junction() {
    local millidegrees
    millidegrees="$(cat "$junction_input" 2>/dev/null)" || return 1
    [[ "$millidegrees" =~ ^[0-9]+$ ]] || return 1
    awk -v t="$millidegrees" 'BEGIN { printf "%.1f", t/1000 }'
}

run_variant() {
    local variant="$1" extra_env=() start_temp container status=0 stop_reason="" cool_started started_epoch last_heartbeat=-60
    if [[ "$variant" == candidate ]]; then
        if [[ "$candidate_mode" == dot4 ]]; then
            extra_env=(-e GGML_CUDA_FA_Q8_DOT4_PREPACK=1)
        else
            extra_env=(-e GGML_CUDA_FA_Q8_DIRECT_TILE=1)
        fi
    fi

    cool_started="$(date +%s)"
    while true; do
        start_temp="$(read_junction)" || start_temp=""
        [[ -n "$start_temp" ]] || { echo 'junction telemetry unavailable' >&2; return 7; }
        if awk -v t="$start_temp" 'BEGIN { exit !(t <= 40) }'; then break; fi
        if (( $(date +%s) - cool_started >= 600 )); then
            echo "GPU$gpu did not cool to 40 C within 10 minutes (now ${start_temp} C)" >&2
            return 8
        fi
        echo "$variant waiting for GPU$gpu to cool (junction=${start_temp} C)" >&2
        sleep 10
    done
    started_epoch="$(date +%s)"
    container="m31-gqa48-8k-${timestamp}-gpu${gpu}-ub${ubatch}-${variant}"
    printf '\n%s start_junction_c=%s\n' "$variant" "$start_temp" >> "$run_dir/metadata.txt"
    podman run --rm --name "$container" --device=/dev/kfd --device=/dev/dri --group-add video \
        --security-opt seccomp=unconfined --security-opt label=disable \
        -e "ROCR_VISIBLE_DEVICES=$gpu" -e LD_LIBRARY_PATH="/workspace/$build_rel/bin" "${extra_env[@]}" \
        -v "$workspace:/workspace" -v /home/machinist/models:/models:ro -w /workspace "$image" \
        "./$build_rel/bin/llama-bench" -m /models/Qwen3.8-27B-Q4_K_M.gguf \
        -p "$prompt_tokens" -n 0 -r 1 --no-warmup -b 2048 -ub "$ubatch" \
        -ctk q8_0 -ctv q8_0 -fa on -ngl 99 -dev ROCm0 -o jsonl --progress \
        > "$run_dir/${variant}.log" 2>&1 &
    local pid=$!
    while kill -0 "$pid" 2>/dev/null; do
        local temp now elapsed
        now="$(date -u +%FT%TZ)"
        elapsed=$(( $(date +%s) - started_epoch ))
        temp="$(read_junction)" || temp=""
        printf '%s,%s\n' "$now" "$temp" >> "$run_dir/temperature.csv"
        if [[ -z "$temp" ]]; then
            stop_reason='junction telemetry unavailable; fail-closed stop'
        elif awk -v t="$temp" 'BEGIN { exit !(t >= 84) }'; then
            stop_reason="junction reached ${temp} C; stopped before 85 C"
        elif (( elapsed >= 300 )); then
            stop_reason='benchmark exceeded 300-second limit'
        fi
        if [[ -n "$stop_reason" ]]; then
            podman kill "$container" >/dev/null 2>&1 || true
            break
        fi
        if (( elapsed - last_heartbeat >= 30 )); then
            echo "$now $variant elapsed=${elapsed}s junction=${temp} C" >&2
            last_heartbeat="$elapsed"
        fi
        sleep 0.2
    done
    wait "$pid" || status=$?
    printf '%s stop_reason=%s status=%s\n' "$variant" "$stop_reason" "$status" >> "$run_dir/metadata.txt"
    [[ "$status" == 0 && -z "$stop_reason" ]]
}

run_status=0
if run_variant default; then
    if [[ "$run_candidate" == 1 ]]; then
        if run_variant candidate; then
            :
        else
            run_status=$?
        fi
    fi
else
    run_status=$?
fi
rocm-smi -d "$gpu" --showtemp >> "$run_dir/metadata.txt" 2>&1 || true
printf '\n--- metadata ---\n'
cat "$run_dir/metadata.txt"
for variant in default candidate; do
    [[ -f "$run_dir/$variant.log" ]] || continue
    printf '\n--- %s log ---\n' "$variant"
    cat "$run_dir/$variant.log"
done
echo "evidence=$run_dir"
exit "$run_status"
