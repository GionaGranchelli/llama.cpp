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

A second cold-start default 8K run completed in `36.0021 s` (`227.542` tok/s), with junction rising from 29 C to a sampled 77 C and cooling afterward. This supports the repeatability of the control. The earlier Q8-vector candidate began from 38 C, so its one-sample comparison remains directional; rerun it under a matched thermal start only if that discarded route is needed again.

## Next action

Keep the pinned 8K result as the runnable control. Run a matched 8K screen of the direct-Q8 GQA tile after the card returns to its cold-start temperature. If it is correct and faster, compare a deterministic greedy token against default, then pursue the 64K control only after the live power/cooling setup is sustainable.

## Direct-Q8 GQA tile candidate

The existing `FLASH_ATTN_EXT` tile path converts Q8 K/V to full-size F16 buffers before attention. Added an opt-in GFX906 route for the model's 256-wide Q8/Q8, GQA-ratio-6 prefill. It keeps the upstream grouped tile attention and full-precision Q math, but decodes Q8 K/V directly into each shared-memory tile. It avoids full-cache F16 materialization; it does not use MIInfer's inaccurate Q8-query approximation. Enable with `GGML_CUDA_FA_Q8_DIRECT_TILE=1`; default dispatch is unchanged.

The HIP build completed. With the flag enabled, `test-backend-ops` selected the direct route for the 256-wide GQA-6 Q8/Q8 64-query case and passed all 31 selected 64-query Flash Attention cases against the CPU backend. Full-model behavior and speed are pending.

A same-build default 8K control on GPU2 / BDF `0000:88:00.0` completed in `36.0105 s` (`227.489` prompt tok/s), one sample without warmup. It started at 32 C junction and reached a sampled 76 C. The model checksum and all benchmark settings match the earlier control. The direct-Q8 candidate timing is pending.

## GQA-6 direct-Q8 tile screens — rejected for 8K prefill

Expanded the opt-in tile to process all six query heads for one KV head at a time, with 4-, 8-, and 16-token query tiles. The 4-token version uses 384 threads and a 24-column tile, matching the custom GQA-6 shape more closely. All variants keep full-precision QK math and decode Q8 K/V in the tile. The HIP builds succeeded, and the target `FLASH_ATTN_EXT` suite passed 31/31 cases on ROCm0, including Q8/Q8 GQA-6 at 64 query tokens.

Same-build, same-device, one-sample 8K screens on GPU2 / BDF `0000:88:00.0`, with model checksum `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`, `-b 2048 -ub 512`, Q8 K/V, Flash Attention, 99 GPU layers, no warmup, and start junction 32 C:

| Route | Query heads per KV tile | Query tokens per tile | Wall time | Prompt throughput | Sampled junction |
|---|---:|---:|---:|---:|---:|
| Default control | 2 | 16 | 36.0105 s | 227.489 tok/s | 76 C |
| Direct Q8 | 6 | 8 | 39.3711 s | 208.072 tok/s | 74 C |
| Direct Q8 | 6 | 16 | 40.0589 s | 204.499 tok/s | 76 C |
| Direct Q8 | 6 | 4 | 39.1352 s | 209.326 tok/s | 75 C |

The six-head routes were slower by 8.7% (4-token), 9.3% (8-token), and 11.2% (16-token). The 4-token variant was the best of these but still lost to the default. The earlier two-head direct-Q8 screen was 37.2822 s, 3.5% slower than the same-build control. These are single-sample screens, not repeat-qualified results; their consistent direction is enough to stop tuning this direct-decode dataflow for the 8K prefill shape. Do not enable it by default. This does not determine performance at 64K, which remains unmeasured because sustained prefill exceeded the thermal guard in prior attempts.

The likely cost is repeated Q8 decode inside each attention tile. The normal path expands the cache once and reuses the F16 representation across query tiles. Direct decode avoids that temporary allocation but repeats conversion work. A useful next kernel must avoid both the failed approximate-Q path and per-tile Q8-to-F16 expansion, while preserving exact-enough QK math.

## Microbatch size screen — `-ub 1024` improves 8K prefill

Tested the larger-microbatch hypothesis with the default attention route (`GGML_CUDA_FA_Q8_DIRECT_TILE` unset), pinned ROCm 7.2.1 image, model checksum, 8K prompt, `-b 2048`, Q8 K/V, Flash Attention, and no warmup. The direct-Q8 experiment code was present but opt-in and inactive. The Machinist checkout was synchronized to `f397c1537`; the isolated repeat binary reports that exact commit.

| MI50 | Microbatch | Wall time | Prompt throughput | Starting junction | Highest sampled junction |
|---|---:|---:|---:|---:|---:|
| GPU0 / `0000:04:00.0` | 512 | 35.4545 s | 231.056 tok/s | 28 C | 83 C* |
| GPU0 / `0000:04:00.0` | 1024 | 33.3853 s | 245.377 tok/s | 29 C | 81 C |
| GPU2 / `0000:88:00.0` | 1024 | 33.9218 s | 241.497 tok/s | 35 C | 80 C* |

`*` These two runs overlapped on both MI50s. GPU0 at `-ub 512` reproduced its earlier 35.46 s control; GPU2 at `-ub 1024` was about 5.8% faster than the earlier same-card 36.0105 s `-ub 512` control. An isolated GPU0 `-ub 1024` run then took 33.3853 s, 5.8% less time than its `-ub 512` screen, starting within 1 C of the control. The direction is consistent across both cards, but there is only one sample per setting and card; treat this as a promising screen, not a repeat-qualified result.

The overlapping run briefly crossed the 80 C warning line (83 C on GPU0 and 80 C on GPU2) and stayed below the 85 C hard stop. The isolated GPU0 `-ub 1024` run peaked at 81 C. Do not run longer contexts or larger batches until the cards cool and telemetry is watched. Next screen `-ub 2048` at 8K only; stop at 85 C or on allocation failure. If it does not improve on 1024, use 1024 as the working prefill setting and focus on verifying longer context under the thermal guard.
