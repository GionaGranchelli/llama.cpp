#!/usr/bin/env bash
set -Eeuo pipefail

repo="$(git rev-parse --show-toplevel)"
gpu_index=2
target_bdf=0000:88:00.0
gpu_uuid=GPU-2152414173497dd3
image=localhost/miinfer-dev:rocm-7.2.1
model=/home/machinist/models/Qwen3.8-27B-Q4_K_M.gguf
model_sha=7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
program="${M31_PROGRAM:-llama-server}"
executable="$repo/build-mi50/bin/$program"
backend="$repo/build-mi50/bin/libggml-hip.so"
name="${M31_CONTAINER_NAME:-m31-fork-agent-${program}-$(date -u +%Y%m%dT%H%M%SZ)}"
run_dir="${M31_RUN_DIR:-/tmp/$name}"
variant="${M31_VARIANT:-default}"
max_seconds="${M31_MAX_SECONDS:-14400}"

if [[ "$program" != llama-server && "$program" != llama-bench ]]; then
    echo "M31_PROGRAM must be llama-server or llama-bench" >&2
    exit 2
fi
if [[ ! -x "$executable" || ! -f "$backend" ]]; then
    echo "Missing qualified $program build under $repo/build-mi50/bin" >&2
    exit 2
fi
if [[ "$program" == llama-bench && ! -f "$repo/build-mi50/bin/libllama-bench-impl.so" ]]; then
    echo "Missing llama-bench implementation library" >&2
    exit 2
fi
changed="$(git diff --name-only HEAD)"
source_diff_sha="$(git diff --binary HEAD -- ggml/src/ggml-cuda/mmq.cu tools/llama-bench/llama-bench.cpp | sha256sum | awk '{print $1}')"
untracked_outside="$(git ls-files --others --exclude-standard | awk '$0 !~ /^\.audit\/m31-fork-agent-0001\//')"
allowed_changed=$'ggml/src/ggml-cuda/mmq.cu\ntools/llama-bench/llama-bench.cpp'
if [[ ( -n "$changed" && ( "$changed" != "$allowed_changed" || "$source_diff_sha" != "${M31_ALLOWED_SOURCE_DIFF_SHA256:-}" ) ) || -n "$untracked_outside" ]]; then
    echo "Refusing run: source changes do not match the explicitly allowed route instrumentation" >&2
    exit 3
fi
if [[ "$variant" != default && "$variant" != j40 ]]; then
    echo "M31_VARIANT must be default or j40" >&2
    exit 2
fi

cap_for_bdf() {
    local card_path bdf cap_file
    for card_path in /sys/class/drm/card[0-9]*; do
        [[ -e "$card_path/device" ]] || continue
        bdf="$(basename "$(readlink -f "$card_path/device")")"
        [[ "$bdf" == "$1" ]] || continue
        for cap_file in "$card_path"/device/hwmon/hwmon*/power1_cap; do
            [[ -r "$cap_file" ]] && { cat "$cap_file"; return; }
        done
    done
    return 1
}

junction_input_for_bdf() {
    local card_path bdf label input
    for card_path in /sys/class/drm/card[0-9]*; do
        [[ -e "$card_path/device" ]] || continue
        bdf="$(basename "$(readlink -f "$card_path/device")")"
        [[ "$bdf" == "$1" ]] || continue
        for label in "$card_path"/device/hwmon/hwmon*/temp*_label; do
            [[ -r "$label" && "$(cat "$label")" == junction ]] || continue
            input="${label%_label}_input"
            [[ -r "$input" ]] && { printf '%s\n' "$input"; return 0; }
        done
    done
    return 1
}

power_input_for_bdf() {
    local card_path bdf input
    for card_path in /sys/class/drm/card[0-9]*; do
        [[ -e "$card_path/device" ]] || continue
        bdf="$(basename "$(readlink -f "$card_path/device")")"
        [[ "$bdf" == "$1" ]] || continue
        for input in "$card_path"/device/hwmon/hwmon*/power1_input; do
            [[ -r "$input" ]] && { printf '%s\n' "$input"; return 0; }
        done
    done
    return 1
}

read_junction() {
    local millidegrees
    millidegrees="$(cat "$junction_input" 2>/dev/null)" || return 1
    [[ "$millidegrees" =~ ^[0-9]+$ ]] || return 1
    awk -v t="$millidegrees" 'BEGIN { printf "%.1f", t/1000 }'
}

read_power() {
    local microwatts
    microwatts="$(cat "$power_input" 2>/dev/null)" || return 1
    [[ "$microwatts" =~ ^[0-9]+$ ]] || return 1
    awk -v p="$microwatts" 'BEGIN { printf "%.1f", p/1000000 }'
}

bus="$(rocm-smi -d "$gpu_index" --showbus | awk -F ': ' '/PCI Bus:/ && !found { print $NF; found=1 }')"
[[ "$bus" == "$target_bdf" ]] || { echo "ROCm index $gpu_index maps to $bus, expected $target_bdf" >&2; exit 4; }
cap0="$(cap_for_bdf 0000:04:00.0)"
cap2="$(cap_for_bdf "$target_bdf")"
[[ "$cap0" == 180000000 && "$cap2" == 180000000 ]] || {
    echo "Both MI50 power caps must be 180000000 uW; got $cap0 and $cap2" >&2
    exit 5
}
kfd="$(rocm-smi --showpids)"
[[ "$kfd" == *'No KFD PIDs currently running'* ]] || { echo "$kfd" >&2; exit 6; }
actual_model_sha="$(sha256sum "$model" | awk '{print $1}')"
[[ "$actual_model_sha" == "$model_sha" ]] || { echo "Model SHA-256 mismatch: $actual_model_sha" >&2; exit 7; }
junction_input="$(junction_input_for_bdf "$target_bdf")" || { echo "No junction sensor for $target_bdf" >&2; exit 8; }
power_input="$(power_input_for_bdf "$target_bdf")" || { echo "No power sensor for $target_bdf" >&2; exit 8; }
start_temp="$(read_junction)" || { echo "Cannot read junction temperature" >&2; exit 8; }
start_power="$(read_power)" || { echo "Cannot read power telemetry" >&2; exit 8; }
awk -v t="$start_temp" 'BEGIN { exit !(t < 80) }' || { echo "GPU is already at the 80 C warning threshold; got $start_temp C" >&2; exit 9; }

mkdir -p /tmp/m31-fork-agent-0001-slots
mkdir -p "$(dirname "$run_dir")"
mkdir "$run_dir" || { echo "Refusing to overwrite evidence directory: $run_dir" >&2; exit 10; }
printf 'source_sha=%s\nsource_diff_sha256=%s\nprogram=%s\nexecutable_sha256=%s\nruntime_libraries_sha256=%s\nhip_sha256=%s\nmodel_sha256=%s\ngpu_bdf=%s\ngpu_uuid=%s\ncap0_uw=%s\ncap2_uw=%s\nstart_junction_c=%s\nstart_power_w=%s\nimage_id=%s\nvariant=%s\n' \
    "$(git rev-parse HEAD)" "$source_diff_sha" "$program" "$(sha256sum "$executable" | awk '{print $1}')" \
    "$(sha256sum "$repo"/build-mi50/bin/*.so | sha256sum | awk '{print $1}')" \
    "$(sha256sum "$backend" | awk '{print $1}')" "$actual_model_sha" "$target_bdf" \
    "$gpu_uuid" "$cap0" "$cap2" "$start_temp" "$start_power" \
    "$(podman image inspect "$image" --format '{{.Id}}')" "$variant" > "$run_dir/metadata.txt"
printf 'utc,junction_c,power_w\n' > "$run_dir/temperature.csv"

extra_env=()
if [[ "$variant" == j40 ]]; then
    extra_env=(-e GGML_CUDA_MMQ_Q4K_J40=1)
fi
extra_env+=(-e GGML_CUDA_MMQ_Q4K_ROUTE_LOG=1)
if [[ "$program" == llama-bench ]]; then
    extra_env+=(-e M31_PROMPT_HASH=1)
fi
podman run --rm --name "$name" --network host --device=/dev/kfd --device=/dev/dri \
    --group-add keep-groups --security-opt label=disable \
    -e "ROCR_VISIBLE_DEVICES=$gpu_uuid" -e "LD_LIBRARY_PATH=/workspace/build-mi50/bin" "${extra_env[@]}" \
    -v "$repo:/workspace:ro" -v /home/machinist/models:/models:ro \
    -v /tmp/m31-fork-agent-0001-slots:/state \
    "$image" "/workspace/build-mi50/bin/$program" "$@" \
    > "$run_dir/server.log" 2>&1 &
server_pid=$!
started="$(date +%s)"
stop_reason=""
warned=0
planned_stop=0
request_stop() {
    planned_stop=1
    podman stop -t 5 "$name" >/dev/null 2>&1 || true
}
trap request_stop INT TERM

while kill -0 "$server_pid" 2>/dev/null; do
    now="$(date -u +%FT%TZ)"
    temp="$(read_junction)" || temp=""
    power="$(read_power)" || power=""
    if [[ -z "$temp" || -z "$power" ]]; then
        stop_reason="junction or power telemetry unavailable; fail-closed stop"
        podman kill "$name" >/dev/null 2>&1 || true
        break
    fi
    printf '%s,%s,%s\n' "$now" "$temp" "$power" >> "$run_dir/temperature.csv"
    if (( warned == 0 )) && awk -v t="$temp" 'BEGIN { exit !(t >= 80) }'; then
        echo "$now warning: junction reached $temp C" >> "$run_dir/server.log"
        warned=1
    fi
    if awk -v t="$temp" 'BEGIN { exit !(t >= 84) }'; then
        stop_reason="junction reached $temp C; stopped before 85 C"
        podman kill "$name" >/dev/null 2>&1 || true
        break
    fi
    if (( $(date +%s) - started >= max_seconds )); then
        stop_reason="server exceeded ${max_seconds}-second run limit"
        podman stop -t 5 "$name" >/dev/null 2>&1 || true
        break
    fi
    sleep 1
done

set +e
wait "$server_pid"
run_status=$?
set -e
if (( planned_stop != 0 )); then
    stop_reason="requested shutdown"
    run_status=0
fi
printf 'stop_reason=%s\nrun_status=%s\n' "$stop_reason" "$run_status" >> "$run_dir/metadata.txt"
[[ ( -z "$stop_reason" || "$stop_reason" == "requested shutdown" ) && $run_status -eq 0 ]]
