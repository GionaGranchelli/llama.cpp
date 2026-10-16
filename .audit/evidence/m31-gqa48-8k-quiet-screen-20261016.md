# GQA48 quiet 8K prefill screen

## Result

The new 48-cell GQA6 dot4 candidate completed a full-model 8K prefill, but was
slower in this one-sample quiet screen:

| Path | Time | Prompt throughput |
| --- | ---: | ---: |
| Default attention | 33.880 s | 241.792 tok/s |
| Q8 dot4 candidate | 37.034 s | 221.203 tok/s |

Candidate time was 9.31% higher and throughput was 8.52% lower. This is a
directional result from one sample per path, not a performance qualification.
The earlier verbose pair is excluded from these numbers.

## Matched setup

- Host: Machinist
- Source worktree commit: `176480a57a5b18be94efdc7f97b9130545775860`
- GPU index 2, BDF `0000:88:00.0`, GFX906, 32 GiB
- Model: Qwen3.8-27B Q4_K_M, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Prompt: 8,192 tokens; generation: 0; no warmup; one sample per path
- `-b 2048 -ub 1024`, Q8_0 K/V, Flash Attention, 99 GPU layers
- Candidate enabled `GGML_CUDA_FA_Q8_DOT4_PREPACK=1`
- Pinned image: `localhost/miinfer-dev:rocm-7.2.1`, ID
  `50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`
- 8K-run caps: GPU0 225 W, GPU2 300 W
- Both completed below the 84 C stop threshold; observed at 30 seconds:
  default 75 C, candidate 78 C. Full telemetry remains in the remote run folder.

The JSON reports `build_commit=unknown` because the container cannot follow the
host worktree's `.git` pointer. The host verified source commit above. The exact
binary hashes and full temperature range remain in
`/tmp/m31-gqa48-8k-screen-gpu2-20261016T045554Z/metadata.txt` and
`temperature.csv` on Machinist.

The candidate still lacks a demonstrated 8K speedup. The cause of the remaining
gap is not established by this single end-to-end sample. A 64K comparison is
still gated on both GPU caps being at or below 180 W.
