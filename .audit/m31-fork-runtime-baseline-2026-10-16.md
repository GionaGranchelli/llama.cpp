# MI50 fork campaign: runtime and prefill checkpoint

Date: 2026-10-16 (Europe/Amsterdam)

## Pinned environment

- Fork checkout: `/home/machinist/Development/llama.cpp-miinfer`
- Upstream source commit: `b76b13379d3b7d67ab0bc424516bb6fd2522bd8e`
- Fork branch: `miinfer/gfx906-base`
- Build: Release, `GGML_HIP=ON`, `GPU_TARGETS=gfx906`, ROCm 7.2.1
- Container image ID: `50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`
- Model: `/home/machinist/models/Qwen3.8-27B-Q4_K_M.gguf`
- Model SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Target card used for the 8K control: gfx906 at PCI BDF `0000:04:00.0`, ROCm unique ID `0x21678e17348c2f7` (ROCR device 0)
- Second gfx906: PCI BDF `0000:88:00.0`, unique ID `0x2152414173497dd3` (ROCR device 2)
- `0000:85:00.0` is an RX 9060 XT (gfx1200), not an MI50.
- Host kernel: `7.2.10-200.fc44.x86_64`; SELinux is Enforcing.
- The user set both MI50 power caps to 180 W earlier in the campaign. The caps were not re-read during these runs.
- After the aborted runs, idle `rocm-smi --showfan` reported fan level 37 (15%) and 0 RPM on both MI50s. Fan telemetry during the hot runs was not captured, so this may indicate an airflow/control issue but does not yet establish that the fans stayed at idle.

## Runtime failure and fix

With the default rootless Podman SELinux label, `hipGetDeviceCount`, `hipSetDevice`, and `hipMalloc` succeeded, but the first device operation aborted with:

```text
Memory critical error by agent node-0 ... Reason: Memory in use.
```

The kernel journal showed SIGSEGV at libc's deliberate `abort()` instruction, called from `libhsa-runtime64`; this was a ROCm fatal error, not an arbitrary application segfault. A traced 4 KiB HIP probe localized it to the first HIP operation. Setting `HSA_ENABLE_SDMA=0` did not change the failure.

The same `hipMemset` probe passed when the container added `--security-opt label=disable`. The prebuilt MIInfer gfx906 vector-add/H2D/D2H smoke also passed with that option. That is a direct A/B showing the default SELinux container label blocked a ROCm memory mapping on this host. Keep `label=disable` on ROCm Podman runs; mounts remain explicitly scoped and the model mount is read-only.

Working container pattern:

```sh
podman run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable \
  -e ROCR_VISIBLE_DEVICES=0 \
  -v /home/machinist/Development/llama.cpp-miinfer:/workspace:Z \
  -v /home/machinist/models:/models:ro,z \
  -w /workspace localhost/miinfer-dev:rocm-7.2.1 <command>
```

## 8K full-model control

Command:

```sh
./build-mi50/bin/llama-bench \
  -m /models/Qwen3.8-27B-Q4_K_M.gguf \
  -p 8192 -n 0 -r 1 --no-warmup \
  -b 2048 -ub 512 -ctk q8_0 -ctv q8_0 -fa on \
  -ngl 99 -dev ROCm0 -o jsonl --progress
```

Result: completed on BDF `0000:04:00.0`, using 99 GPU layers, Q8 K/V cache, Flash Attention, batch 2048 and ubatch 512. The JSON result reported `avg_ns=35462865469`, `avg_ts=231.002202`, one sample, zero generation tokens. This proves the pinned model and backend can execute; it is one diagnostic sample without warmup and must not be reported as a qualified performance baseline.

## 64K thermal attempts

The same command with `-p 65536` was attempted on each MI50, with `-r 1 --no-warmup` and the same backend settings. Neither produced a completed result.

- BDF `0000:04:00.0`: junction reached 97°C after about 100 seconds. The benchmark did not exit on SIGTERM, so it was SIGKILLed; GPU activity then returned to 0% and temperature fell to 68°C.
- BDF `0000:88:00.0`: junction reached 84°C after 78 seconds (80°C at 57 seconds). The benchmark was SIGKILLed before the 85°C hard stop; activity returned to 0% and temperature fell to 55°C.

These are aborted attempts, not 64K timing results. A guarded full-context baseline remains unverified. Both cards were otherwise idle; `ROCR_VISIBLE_DEVICES` isolated each run to one gfx906. The user-reported 180 W caps and observed cooling did not sustain these 64K attempts under the 80°C warning / 85°C stop policy.

## Next action

Keep the 8K baseline as the known-good control. Resolve sustainable cooling or power limits before another 64K run, then capture at least one completed 64K baseline. Do not start an attention port comparison until that long-context control is available.
