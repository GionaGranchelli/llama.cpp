# GQA48 quiet 8K profile run

## Run status

The default and opt-in candidate 8K profiles both completed on Machinist GPU2
without reaching the 84 C stop threshold. This was a profiling run to inspect
kernel costs, not a performance timing to compare against the quiet screen.

- Remote run directory: `/tmp/m31-gqa48-8k-profile-gpu2-20261016T050033Z`
- Host source worktree commit: `176480a57a5b18be94efdc7f97b9130545775860`
- GPU index 2, BDF `0000:88:00.0`, GFX906
- Model: Qwen3.8-27B Q4_K_M, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- 8,192 prompt tokens, zero generation, `-b 2048 -ub 1024`, Q8_0 K/V,
  Flash Attention, 99 GPU layers
- Profiler: `rocprof --stats --timestamp on`
- Start junction: default 31 C, candidate 38 C; samples at 30 seconds were
  75 C and 77 C, respectively
- Current caps: GPU0 225 W, GPU2 300 W

The run directory contains per-path profiler output, benchmark logs, metadata,
and temperature CSVs. The runner attempted to print a summary with `rg`, which
is absent from the ROCm container, so no profiler tables were returned in the
terminal. The summary command is now changed to `grep`. Retrieve and inspect the
saved traces before making any claim about the limiting kernels.
