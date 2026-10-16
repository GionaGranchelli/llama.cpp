#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${M31_CANDIDATE_WORKSPACE:-/tmp/m31-fork-candidate}"
build_dir="${M31_BUILD_DIR:-$workspace/build-mi50}"
image="localhost/miinfer-dev:rocm-7.2.1"
# Select the Q8_0 GQA6 correctness fixture (KV=1024, batch=64), not every
# FLASH_ATTN_EXT case that happens to share head width and batch size.
filter='hsk=256,hsv=256,nh=4,nr23=\[6,1\],kv=1024,nb=64.*type_K=q8_0,type_V=q8_0'
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${M31_EVIDENCE_ROOT:-/tmp}/m31-q8-dot4-lds-${timestamp}"
mkdir -p "$run_dir"

case "$build_dir" in
    "$workspace"/*) build_rel="${build_dir#"$workspace"/}" ;;
    *) echo "M31_BUILD_DIR must be inside M31_CANDIDATE_WORKSPACE" >&2; exit 2 ;;
esac
test_binary="$build_dir/bin/test-backend-ops"
[[ -x "$test_binary" ]] || {
    echo "missing test-backend-ops at $test_binary" >&2
    exit 2
}

bus_for_gpu() {
    rocm-smi -d "$1" --showbus 2>/dev/null | awk -F ': ' '/PCI Bus:/ {print $NF}'
}

temp_for_gpu() {
    rocm-smi -d "$1" --showtemp 2>/dev/null | awk '/Sensor junction/ {print $NF}'
}

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

check_card() {
    local gpu="$1" expected="$2" bus temp
    bus="$(bus_for_gpu "$gpu")"
    [[ "$bus" == "$expected" ]] || { echo "GPU $gpu maps to '$bus', expected $expected" >&2; return 1; }
    temp="$(temp_for_gpu "$gpu")"
    [[ -n "$temp" ]] && awk -v t="$temp" 'BEGIN { exit !(t <= 40) }' || {
        echo "GPU $gpu junction must be <=40 C (got '$temp')" >&2
        return 1
    }
}

check_card 0 0000:04:00.0
check_card 2 0000:88:00.0
printf 'utc,gpu,junction_c\n' > "$run_dir/temperature.csv"
kfd_info="$(rocm-smi --showpids 2>/dev/null)"
if ! grep -q 'No KFD PIDs currently running' <<< "$kfd_info"; then
    echo "refusing test while another KFD process is active" >&2
    exit 3
fi

{
    printf 'source_commit=%s\n' "$(git -C "$workspace" rev-parse HEAD)"
    printf 'source_status=%s\n' "$(git -C "$workspace" status --short | tr '\n' ' ')"
    printf 'fattn_tile_sha256=%s\n' "$(sha256sum "$workspace/ggml/src/ggml-cuda/fattn-tile.cuh" | awk '{print $1}')"
    printf 'test_binary_sha256=%s\n' "$(sha256sum "$test_binary" | awk '{print $1}')"
    printf 'image_id=%s\n' "$(podman image inspect --format '{{.Id}}' "$image")"
    printf 'filter=%s\n' "$filter"
    printf 'gpu0_bdf=%s\ngpu2_bdf=%s\n' "$(bus_for_gpu 0)" "$(bus_for_gpu 2)"
    printf 'gpu0_cap_uw=%s\ngpu2_cap_uw=%s\n' \
        "$(cap_for_bdf 0000:04:00.0)" "$(cap_for_bdf 0000:88:00.0)"
    printf '%s\n' "$kfd_info"
    rocm-smi --showtemp
} > "$run_dir/metadata.txt"

declare -A pid_by_gpu container_by_gpu status_by_gpu
for gpu in 0 2; do
    container="m31-q8-dot4-lds-${timestamp}-gpu${gpu}"
    container_by_gpu[$gpu]="$container"
    podman run --rm --name "$container" --device=/dev/kfd --device=/dev/dri --group-add video \
        --security-opt seccomp=unconfined --security-opt label=disable \
        -e "ROCR_VISIBLE_DEVICES=$gpu" -e GGML_CUDA_FA_Q8_DOT4_PREPACK=1 \
        -v "$workspace:/workspace" -w /workspace "$image" \
        "./$build_rel/bin/test-backend-ops" -b ROCm0 -o FLASH_ATTN_EXT -p "$filter" \
        > "$run_dir/gpu${gpu}.log" 2>&1 &
    pid_by_gpu[$gpu]=$!
done

stop_reason=""
while kill -0 "${pid_by_gpu[0]}" 2>/dev/null || kill -0 "${pid_by_gpu[2]}" 2>/dev/null; do
    for gpu in 0 2; do
        pid="${pid_by_gpu[$gpu]}"
        kill -0 "$pid" 2>/dev/null || continue
        temp="$(temp_for_gpu "$gpu")"
        printf '%s,%s,%s\n' "$(date -u +%FT%TZ)" "$gpu" "$temp" >> "$run_dir/temperature.csv"
        if [[ -z "$temp" ]] || awk -v t="$temp" 'BEGIN { exit !(t >= 84) }'; then
            stop_reason="GPU $gpu telemetry unavailable or junction reached ${temp:-unknown} C"
            for stop_gpu in 0 2; do
                podman kill "${container_by_gpu[$stop_gpu]}" >/dev/null 2>&1 || true
            done
        fi
    done
    [[ -z "$stop_reason" ]] || break
    sleep 1
done

for gpu in 0 2; do
    status=0
    wait "${pid_by_gpu[$gpu]}" || status=$?
    status_by_gpu[$gpu]="$status"
done

{
    printf 'stop_reason=%s\n' "$stop_reason"
    printf 'gpu0_status=%s\ngpu2_status=%s\n' "${status_by_gpu[0]}" "${status_by_gpu[2]}"
    rocm-smi --showtemp
} >> "$run_dir/metadata.txt"
cat "$run_dir/gpu0.log" "$run_dir/gpu2.log"
echo "evidence=$run_dir"
[[ -z "$stop_reason" && "${status_by_gpu[0]}" == 0 && "${status_by_gpu[2]}" == 0 ]]
