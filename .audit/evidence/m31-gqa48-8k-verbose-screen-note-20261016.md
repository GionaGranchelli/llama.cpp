# GQA48 8K screen instrumentation note

The container label fix allowed `llama-bench` to build, and both 8K paths
completed on GPU2. Default reported 33.920 s (241.509 prompt tok/s), and the
candidate reported 36.957 s (221.662 prompt tok/s). However, the runner passed
`-v`; the candidate emitted 800 per-operation kernel-selection messages from the opt-in path.
That differs from the earlier quiet benchmark setup and can add host-side
logging work during prefill. Do not use this run as a valid A/B timing.

The candidate output identifies the GPU as `AMD Radeon Graphics`, BDF
`0000:88:00.0` according to the host runner, model Qwen3.8-27B Q4_K_M,
8192 prompt tokens, Q8 K/V, `-b 2048`, `-ub 1024`, and ROCm. The benchmark
reported `build_commit=unknown`; source and backend hashes are in the remote
run directory and need retrieval.

Raw logs and temperature samples remain on Machinist at
`/tmp/m31-gqa48-8k-screen-gpu2-20261016T044951Z`. The local runner now omits
`-v` and prints only the JSON result lines. Re-run the pair before drawing any
performance conclusion.
