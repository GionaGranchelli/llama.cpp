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

The same diagnostic command on BDF `0000:88:00.0` completed in `36.004 s` at `227.528` prompt tokens/s, also one sample without warmup. During this short run the card drew 253 W and reached 74 C junction; the reported fan level was 22% with 0 RPM. This is not a controlled card-performance comparison, but confirms the second MI50 also runs the pinned backend with `label=disable`.

## Power-cap check

After both 64K attempts had been stopped, sysfs reported the live caps equal to their defaults/maxima: GPU0 (`0000:04:00.0`) `225000000` µW and GPU2 (`0000:88:00.0`) `300000000` µW. During the second-card 8K run, `rocm-smi` reported 253 W. Therefore the earlier user-reported 180 W settings are not reflected in the live cap values. The account has no passwordless sudo, so a privileged reapply and verification is pending. Do not launch another sustained prefill until both `power1_cap` values are verified.

## 64K thermal attempts

The same command with `-p 65536` was attempted on each MI50, with `-r 1 --no-warmup` and the same backend settings. Neither produced a completed result.

- BDF `0000:04:00.0`: junction reached 97°C after about 100 seconds. The benchmark did not exit on SIGTERM, so it was SIGKILLed; GPU activity then returned to 0% and temperature fell to 68°C.
- BDF `0000:88:00.0`: junction reached 84°C after 78 seconds (80°C at 57 seconds). The benchmark was SIGKILLed before the 85°C hard stop; activity returned to 0% and temperature fell to 55°C.

These are aborted attempts, not 64K timing results. A guarded full-context baseline remains unverified. Both cards were otherwise idle; `ROCR_VISIBLE_DEVICES` isolated each run to one gfx906. The user-reported 180 W caps and observed cooling did not sustain these 64K attempts under the 80°C warning / 85°C stop policy.

## Attention candidate transfer gate

The MIInfer Q8 dot4 candidate is not safe to copy as-is. Its earlier 64K full-model run had max-logit error `0.9121723`, cosine `0.9935138`, and a different top token versus production Q8. The later four-plane query-residual screen still failed at 8K: max-logit error `1.09448`, cosine `0.9989098`, despite a one-sample speedup of 1.309–1.374x. That screen used ROCm 7.14, so use it to reject the math by correctness, not as a performance comparison with this ROCm 7.2.1 fork control. Evidence: `/home/gionag/Development/MIInfer-stage1-rewrite-gpu2/experiments/EXP-0409-m31-q8-dot4-query-attention.md` and `EXP-0428-q8-dot4-multi-plane-screen.md`.

llama.cpp already has gfx906 `sdot4` support in `ggml/src/ggml-cuda/common.cuh`, a Q8 vector attention path in `fattn-common.cuh`, and GQA tile grouping in `fattn-tile.cuh`. The transfer opportunity is the exact-arithmetic dataflow that avoids converting quantized K/V to F16 for multi-query tiles, if measurement shows that conversion is a bottleneck. Do not port the failed Q8-query approximation unchanged. First establish the full-context control with verified thermal settings, then compare an opt-in exact-Q variant through `FLASH_ATTN_EXT`.

## Q8 vector-prefill dispatch screen — rejected

As a minimal `FLASH_ATTN_EXT` integration probe, temporarily selected the existing Q8 vector kernel on GFX906 for 256-wide Q8_0 K/V attention when query batch exceeded two. The code used the existing workspace path and left normal dispatch opt-in. Added a Q8/Q8 64-query test case.

With `GGML_CUDA_FA_Q8_VEC_PREFILL=1`, the HIP backend test exercised that exact shape and passed all 31 selected 64-query Flash Attention cases against CPU. The route log was observed in a verbose full-model 1K run. This proved dispatch and operator-level numerical tolerance, but not whole-model output equivalence.

Matched single-sample, no-warmup 8K screens on GPU2 / BDF `0000:88:00.0`:

| Route | Wall time | Prompt throughput | Sampled junction |
|---|---:|---:|---:|
| Default | 36.0219 s | 227.417 tok/s | 77 C |
| Existing Q8 vector route | 43.6538 s | 187.658 tok/s | 81 C |

Both runs used the same build (`9e057e6`), Qwen3.8-27B Q4_K_M checksum, Q8 K/V cache, `-b 2048 -ub 512`, Flash Attention, GPU2 and one sample without warmup. The vector route took about 21% longer. This is a screening result, not a repeated benchmark; thermal state and the 300 W cap remain uncontrolled. It is enough to reject the existing per-head vector route for this prefill shape. Its likely structural disadvantage is that it does not share KV work across the model's GQA ratio of six.

The temporary selector and its test-only case were removed after the negative screen. The default llama.cpp route remains intact. Keep the evidence and pursue a Q8 K/V tile path that shares fragments across GQA heads while using full-precision Q in QK, avoiding the failed MIInfer Q8-query approximation.

## Next action

Keep the pinned 8K result as the runnable control. Next inspect the existing tile kernel's Q8-to-F16 conversion and GQA grouping, then implement the smallest direct-Q8 K/V tile experiment that reuses grouped heads and retains full-precision Q. Run backend numerical checks before another full-model timing. Resolve sustainable power/cooling before attempting 64K again.
