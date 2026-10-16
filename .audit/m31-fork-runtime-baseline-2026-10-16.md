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
| GPU2 / `0000:88:00.0` | 2048 | 33.5822 s | 243.939 tok/s | 32 C | 80 C |

`*` These two runs overlapped on both MI50s. GPU0 at `-ub 512` reproduced its earlier 35.46 s control; GPU2 at `-ub 1024` was about 5.8% faster than the earlier same-card 36.0105 s `-ub 512` control. An isolated GPU0 `-ub 1024` run then took 33.3853 s, 5.8% less time than its `-ub 512` screen, starting within 1 C of the control. The direction is consistent across both cards, but there is only one sample per setting and card; treat this as a promising screen, not a repeat-qualified result.

The 2048 microbatch completed without allocation failure, with a sampled 19.25 GB VRAM use on GPU2. It was only about 1% faster than the earlier GPU2 `-ub 1024` result, while the start temperatures differed; there is no evidence of a meaningful gain over 1024. The overlapping run briefly crossed the 80 C warning line (83 C on GPU0 and 80 C on GPU2) and stayed below the 85 C hard stop. The isolated GPU0 `-ub 1024` run peaked at 81 C, and GPU2 `-ub 2048` sampled 80 C. Use 1024 as the working setting because it gives the clear ~5.8% gain over 512 with less activation memory than 2048. Longer-context verification remains blocked on thermal capacity: live sysfs caps are currently 225 W on GPU0 and 300 W on GPU2, so the earlier reported 180 W settings are not active. Do not launch a longer prefill until the caps are re-applied and verified.

## 64K context-capacity screen — allocation succeeds, full prompt pending

`llama-bench` normally sets `n_ctx` to prompt + generation length. Therefore the earlier 8K screens used an 8K context even though the GGUF supports longer contexts. Re-ran an 8K prompt with `-fitc 65536 -fitt 0`; the JSON output records `fit_min_ctx=65536`, `n_gpu_layers=99`, `-ub 1024`, and build commit `f397c15`. It completed in `33.9574 s` (`241.244` prompt tok/s), with about `20.16 GB` VRAM used during the run and sampled junction reaching `80 C` from about `32 C`.

This verifies that the pinned fork can allocate a 64K context on GPU2 while keeping all model layers there. It does not verify processing 64K prompt tokens. The full-length prefill remains pending because both power caps are currently at their stock maxima and the 8K run reached the thermal warning line.

For a future `llama-bench -p 65536` run, the benchmark creates 65,536 random token IDs and processes them as a single prompt measurement in batches of up to `-b` tokens, internally limited further by `-ub`. This is a genuine full-length prefill, but it is not a natural coding-agent prompt. Separate product validation should use a tokenizer-counted repository prompt; the random-token workload remains useful for a reproducible kernel-performance A/B.

## Existing sequence checkpoint support

The fork already exposes sequence state size/data/save/load APIs in `include/llama.h`. `llama_context::state_seq_save_file()` writes the token history and memory state, and the server's slot save/restore path calls these APIs. The pinned model is identified as Qwen3.5 27B in the benchmark output; `llama-model.cpp` routes Qwen3.5 through hybrid memory, and both hybrid and hybrid-ISWA implementations serialize the attention cache and recurrent state. This is strong source evidence that MIInfer's checkpoint machinery should not be transplanted yet. A real Machinist ROCm save/restart/restore run is still needed before claiming the agent workflow is covered.

That runtime check now passes at a small scale. On GPU2 / BDF `0000:88:00.0`, ROCm 7.2.1, the pinned Qwen model, Q8 K/V, Flash Attention and a 4K slot, `/completion` processed a 22-token prompt with `cache_prompt=true`. `/slots/0?action=save` wrote a 314,555,776-byte state file in 142.436 ms. I stopped the server, started a new process with the same model/settings, and restored the slot; the API reported all 22 tokens restored from the full file in 184.817 ms. An extended follow-up reported `cache_n=22` and `prompt_n=6`, confirming that the restored attention/recurrent state was reused and only the six-token suffix was evaluated. The temporary 300 MB state file and server were removed after the check. This proves short in-process-server-restart recovery for this model and backend; it does not yet prove 64K checkpoint size, latency, or an agent workflow.

## GFX906 MFMA attempt — invalidated before performance use

The direct-Q8 MFMA prototype did not run on MI50. `amd_mfma_available()` accepts CDNA1 and newer; GFX906 is Vega20 and is below that range. The HIP source defines `AMD_MFMA_AVAILABLE` only for architectures marked `CDNA`, and its `hip.h` defines `CDNA` for gfx908, gfx90a, gfx942, and gfx950, not gfx906. Therefore `GGML_CUDA_FA_Q8_DIRECT_MFMA=1` never selected the prototype on either Machinist MI50.

The earlier `test-backend-ops` run and both 8K timings with that flag exercised the default route. In particular, the 35.967 s candidate and 35.951 s default readings on GPU2 cannot be used to judge the prototype. The observed lack of a route log during the verbose model run was the clue; inspecting the architecture guards established the cause. I removed the unreachable prototype.

The real MI50 direct-Q8 candidate remains the GFX906 GQA tile route. I reran the exact 256-wide, GQA-6, Q8/Q8, 4K KV, 64-query case on GPU0 with `GGML_CUDA_FA_Q8_DIRECT_TILE=1`. The runtime logged `selected direct-Q8 GFX906 GQA tile attention`, and the HIP result passed its CPU-reference check (`1/1`). Earlier full-model screens of this route were slower than default at 8K (best six-head tile: 39.135 s vs 36.011 s). The next implementation should exploit GFX906's `sdot4`/DP4A path while sharing K/V work across the six query heads; another MFMA dispatch change cannot run on this GPU.

## 8K kernel profile — matrix multiplication remains the largest measured cost

On 2026-10-15 23:57 UTC, profiled one default-route 8K run on Machinist GPU2, BDF `0000:88:00.0`, unique ID `0x2152414173497dd3`. The source and binary were at `f83346b` / build 21, ROCm 7.2.1, model SHA-256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`, Q8 K/V, Flash Attention, `-b 2048 -ub 1024`, 99 GPU layers, one sample without warmup. `GGML_CUDA_FA_Q8_DIRECT_TILE` was unset. The input completed in 33.968 s (241.171 tok/s) under profiling. GPU2 started at 32 C junction, reached 76 C, then cooled to 47 C after completion.

Summing HIP dispatch durations from rocprofv3's `1_results.db` gives 33.403 s. The largest kernels were:

| Kernel | Dispatch time |
| --- | ---: |
| `mul_mat_q<GGML_TYPE_Q4_K, 64, ...>` | 21.097 s |
| `mul_mat_q<GGML_TYPE_Q6_K, 64, ...>` | 5.049 s |
| `mul_mat_q<GGML_TYPE_Q5_K, 64, ...>` | 1.967 s |
| Default `flash_attn_tile<256,256,16,2,...>` | 1.574 s |
| `gated_delta_net_cuda<128,...>` | 1.550 s |

The three listed quantized matrix kernels account for 28.112 s, about 84% of profiled dispatch time. Default attention is about 4.7%. From the matching 4K trace, attention grew about 3.9x when prompt length doubled, while Q4_K matrix time grew about 2.0x. Attention is growing faster but is still a small share at 8K. Do not infer 64K performance from this short-context profile; measure again at a thermally sustainable longer prefix before choosing a long-context attention rewrite.

This profile changes the next source target: understand the `GGML_TYPE_Q4_K` `mul_mat_q` path and its GFX906 dispatch before changing it. The current direct-Q8 attention candidate is not a performance win at 4K and remains opt-in. The 64K full-model run is still blocked by thermal headroom and must not be retried until the power/cooling state is verified.

## Q4_K MMQ J=128 screen — rejected

Added an opt-in GFX906 Q4_K `J=128` MMQ configuration, keeping the default at `J=64`. On GPU2, matched 4K rocprofv3 traces used the same binary, model, cache, batch settings, and card. With the candidate flag, the trace confirmed `mul_mat_q<GGML_TYPE_Q4_K,128,...>` executed. Q4_K dispatch time rose from 10.544 s at J=64 to 20.058 s at J=128; total kernel dispatch time rose from 16.322 s to 25.822 s. The profiled model run took 26.331 s. This one-sample result rejects J=128 for this workload. The extra work per CTA appears to reduce useful parallelism enough to outweigh the larger tile; confirm that with register/occupancy counters only if a future result depends on it.

Do not keep the J=128 configuration. The next small tile screen is J=32, which is already represented in the GCN table and tests the reverse hypothesis: more CTAs may help GFX906 saturate, at the cost of repeated weight loads. The `test-backend-ops` run under the J=128 flag passed 120/120 filtered Q4_K matrix cases, but its generated cases do not reach 128 columns, so that test did not exercise J=128; the full-model profile proves dispatch, not numerical equivalence to the reference.

## Q8 sdot4 attention tile — rejected

The opt-in GFX906 Q8 sdot4 GQA tile selected and built successfully, but the 4K profile showed 1,479 ms across 64 attention calls versus 405 ms across 64 default-tile calls. Those runs used separate MI50 cards, so treat the ratio as directional; the candidate also failed to beat the 8K full-model control in its one-sample screen. Remove the sdot4 tile code and keep the default Flash Attention path. The existing upstream Q8 cache expansion remains the control; this result does not rule out a different exact-Q attention schedule at 64K.

## Q4_K MMQ J=32 — promising 8K full-model screen

The opt-in selector is limited to `GGML_TYPE_Q4_K` on GFX906 and is disabled unless `GGML_CUDA_MMQ_Q4K_J32` is set. It selects the already-supported GCN `J=32` kernel for larger matrix batches; the production default remains J=64. On the matched GPU2 4K profile, J=32 reduced Q4_K dispatch time from 10.544 s to 9.332 s (-11.5%) and total kernel dispatch time from 16.322 s to 15.107 s (-7.4%). The J=32 kernel symbol was observed in rocprofv3 output.

The Q4_K backend operator suite passed 120/120 cases with the J=32 flag enabled; its `n=64` cases execute J=32. Three J=32 8K model samples completed at 31.463 s, 31.485 s, and 31.512 s; two completed default samples took 33.869 s and 33.831 s. All used the same build, GPU2, model checksum, 8K prompt, Q8 K/V, Flash Attention, batch 2048, ubatch 1024, 99 GPU layers, and no warmup. The two thermally safe paired candidate samples had starts 34/36 C and peaks 83/82 C; their paired defaults started 35/37 C and peaked at 80/78 C. Those two samples give median wall time 31.474 s versus 33.850 s, about 7.0% lower. The third J=32 sample completed at 31.512 s but reached the 85 C hard stop; exclude it from the thermal-safe comparison.

The code-object metadata gives a plausible mechanism. For the executed Q4_K main kernel, J=64 reports 128 VGPRs and a 44-byte private segment per work item; J=32 reports 124 VGPRs and no private segment. LLVM defines this per-work-item private segment as including argument, spill, and private memory ([AMDGPU code-object metadata](https://llvm.org/docs/AMDGPUUsage.html#amdhsa-code-object-v2-kernel-code-properties-metadata-map)). This is evidence of reduced private-memory demand, but does not establish that register spilling alone caused the gain; the ROCm profile did not report the spill count separately.

This is a promising 8K result, not a 64K qualification or grounds to change the default yet. The 180 W power cap was not active; GPU2 approached or reached the thermal stop at 300 W. Recheck the J=32 result after verified power/cooling settings, then run a guarded long-prefix test to see whether the gain holds at the actual target context.

## Q4_K MMQ J=40 — stronger 8K screen

The existing GCN Q4_K `J=40` configuration has 128 VGPRs and zero private-segment bytes per work item in the rocprof metadata. It was screened with `GGML_CUDA_MMQ_Q4K_J40=1`, keeping the default at J=64 and retaining the separate J=32 selector. The Q4_K operator suite passed with J=40 enabled. A matched 4K GPU2 profile selected `mul_mat_q<GGML_TYPE_Q4_K,40,...>` and measured 8.781 s for that kernel and 14.555 s total kernel dispatch, versus 10.544 s and 16.322 s at J=64. J=40 also beat J=32's 9.332 s Q4_K and 15.107 s total dispatch in the matched profile.

Two same-build, same-card, unprofiled 8K J=40 runs took 30.399 s and 30.404 s at 269.48 and 269.44 tok/s. The same-build default control took 33.849 s at 242.02 tok/s. The median J=40 wall time is 10.2% lower than the control. J=40 started at 34 C and 37 C; the highest sampled junction temperatures were 78 C and 80 C. The control started at 35 C and reached 77 C. These are small-sample screens, not long-context qualification. Keep J=40 opt-in until a guarded 64K prefill proves the gain remains and the thermal margin is acceptable.

## J=40 confirmation on both MI50s at 4K

Rebuilt the remote working tree in the pinned ROCm 7.2.1 container because its copied build directory had an invalid `/workspace` CMake path. The build reconfigured successfully and rebuilt `llama-bench` plus `test-backend-ops`. With `GGML_CUDA_MMQ_Q4K_J40=1`, the Q4_K `MUL_MAT` backend suite passed 120/120 cases on GPU2 and again on GPU0. A first filter used the wrong parameter name and selected 0 tests; the corrected `type_a=q4_K` filter selected and passed the 120 cases on each card.

Then screened one 4K Q4_K J=40 model run on each MI50 simultaneously, followed by one unchanged-default 4K run on each card simultaneously. All runs used the pinned model SHA, `-b 2048 -ub 1024`, Q8 K/V, Flash Attention and 99 GPU layers. The J=40 selector was enabled for candidate runs and explicitly unset for the controls. The remote source tree reports `cf6a020-dirty`; its J=40 source edit matches the pushed fork change at `0dfcf0166`.

| Card / BDF | J=40 samples | Default control | J=40 mean vs control | Highest sampled edge |
| --- | ---: | ---: | ---: | ---: |
| GPU0 / `0000:04:00.0` | 15.0615 s, 15.0580 s | 16.6408 s | 9.5% lower wall time | 56 C |
| GPU2 / `0000:88:00.0` | 15.0039 s, 15.0044 s | 16.8199 s | 10.8% lower wall time | 49 C |

This repeats the J=40 direction on both physical cards, with four candidate samples and two controls total. It is still a short-context screen: each card has only one default sample, and candidate and control runs did not start at identical temperatures. The figures in the table are edge-sensor readings from `rocm-smi`'s concise view; junction temperature was not polled during these runs, so they do not establish thermal margin against the 80 C warning threshold. `rocm-smi --showtemp` reports the junction sensor and must be used for the guarded long-context run. The result supports keeping J=40 for that check, but does not qualify 64K performance or justify changing the default.

A second simultaneous 4K J=40 screen changed only `-ub` from 1024 to 2048. The 2048-token microbatch took 14.9926 s on GPU0 and 14.9865 s on GPU2, compared with the corresponding 1024-token candidate means of 15.0597 s and 15.0041 s. These one-sample-per-card deltas are only 0.4% and 0.1%, too small to establish a gain. Peak sampled edge temperatures were 45 C and 44 C; junction was not polled during those runs. Keep 1024 as the working setting: the larger microbatch showed no useful improvement in this screen and uses more activation memory.

An 8K J=40 run on GPU2 with `-ub 2048` took 30.3458 s (269.955 tok/s), compared with the two `-ub 1024` J=40 samples at 30.399 s and 30.404 s. The difference is 0.18%, not a useful measured gain. The highest sampled edge temperature was 53 C. Junction was not sampled during the run; the later idle `rocm-smi --showtemp` reading was 43 C and is not a peak. Do not use edge temperature alone for the 80/85 C junction guard.

One simultaneous 4K screen compared `-ub2048` on GPU0 (`0000:04:00.0`) with `-ub4096` on GPU2 (`0000:88:00.0`). Both runs used `-b4096`, J=40, the pinned Qwen model, Q8 K/V, Flash Attention, 99 GPU layers, and no warmup. GPU0 took 14.9939 s at 273.177 prompt tok/s. GPU2 took 14.9115 s at 274.688 prompt tok/s. This single pair used different cards and differs by 0.55%, so it does not show a `-ub4096` gain. Junction peaked at 76 C on GPU0 and 75 C on GPU2. The caps remained at 225 W and 300 W. Do not use this screen to infer 64K performance or thermal margin.

## Decode smoke with the J=40 selector

On GPU2, ran one `-p 512 -n 64` llama-bench sample with the default dispatch and one with `GGML_CUDA_MMQ_Q4K_J40=1`. Generation throughput was 25.4135 tok/s with default dispatch and 25.4588 tok/s with J=40 enabled. This is effectively unchanged at the precision of a single sample. The J=40 selector requires at least 40 activation columns, while token-by-token decode uses a one-token batch, so this is a smoke check that the opt-in flag does not alter decode dispatch, not evidence of a decode speedup or a correctness comparison.

## Q4_K MMQ J=48 — rejected

Added and built an opt-in GFX906 J=48 selector, plus an `n=96` Q4_K MMQ correctness case. That exact case passed 1/1 against CPU on both MI50s with J=48 enabled. The prior 120-case Q4_K suites did not exercise J=48 and are not counted as its validation.

Ran a two-card crossover at 4K with `-p 4096 -n 0 -r 1 --no-warmup -b 2048 -ub 1024 -ctk q8_0 -ctv q8_0 -fa on -ngl 99`. On GPU0 / BDF `0000:04:00.0`, J=48 took `15.024882777 s` and J=40 took `15.043039420 s`. On GPU2 / BDF `0000:88:00.0`, J=40 took `14.975473665 s` and J=48 took `15.029557941 s`. Across cards, J=48 averaged `15.027220359 s` and J=40 averaged `15.009256543 s`; J=48 was 0.12% slower. This provides no useful gain, so the J=48 selector and its special test case were removed. The experiment code and full results remain in Git history.

Junction was 34 C before both crossover runs and 49-50 C after; continuous peak telemetry was not saved. Live caps read 225 W / 300 W. Keep J=40 as the better supported opt-in and do not infer 64K behavior from this short-context screen.
