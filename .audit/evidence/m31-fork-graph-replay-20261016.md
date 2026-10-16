# M31 fork campaign: opt-in attention under HIP graph capture

Date: 2026-10-16 (Europe/Amsterdam)

## Result

The candidate full-model run selected the opt-in GFX906 Q8 dot4 path. With
`GGML_HIP_GRAPHS` enabled and `GGML_CUDA_GRAPH_DEBUG=1`, llama.cpp logged
`CUDA graph warmup complete`; subsequent candidate route logs occurred while
the backend was capturing that graph. The run completed without a HIP graph
error, and the backend code instantiates and launches the captured graph.

This is a graph-capture and launch smoke, not evidence of repeated backend
graph-instance reuse: the log contains no `CUDA Graph id ... reused` message.
The final `graphs reused = 0` counter is llama.cpp's higher-level ggml graph
reuse count, not the HIP graph launch count.

## Run

- Commit in the built source: `09bca255faf238de2398344acf7322b02380099b`
- Embedded build: `0.7.0-dev`, build `11708`, commit `09bca255f`
- Host: Machinist GPU2, BDF `0000:88:00.0`
- Context screen: 2,048 prompt tokens, three runs, batch 2,048,
  microbatch 512, Q8_0 K/V, Flash Attention, 99 GPU layers
- Candidate: `GGML_CUDA_FA_Q8_DOT4_PREPACK=1`
- Graph diagnostics: `GGML_CUDA_GRAPH_DEBUG=1`, verbose logging enabled
- Temperature: 31 C before, 46 C after; 84 C stop guard remained active
- Timings: 9.45 s, 9.12 s, 9.09 s (mean 9.22 s, about 222 tok/s)

These 2K timings are diagnostic only and are not comparable to the 8K
baseline. The run had three samples but no matched baseline.

Raw output, BDF, KFD-idle check and temperature trace are in
`m31-fork-graph-replay-20261016/benchmark.log` and the adjacent files.

## Remaining gate

Run a matched full-model default-versus-candidate 8K A/B, check generated
output or logits, and then run a guarded 64K prefill. The guarded runner still
requires both MI50 caps at or below 180 W. The current caps are 225 W and
300 W; the SSH account cannot change them without an interactive sudo
password.
