# GQA2 Q8 dot4 8K full-model screen

## Result

The stock 16-query by 2-head Q8 dot4 schedule passed the focused operator check
on both ROCm-visible cards, then completed one matched 8K prefill screen on
Machinist GPU2. It remains slower than the default path in this single sample:

| Path | Time | Prompt throughput |
| --- | ---: | ---: |
| Default | 33.840132720 s | 242.079 tok/s |
| Q8 dot4, GQA2 schedule | 35.974359864 s | 227.718 tok/s |

The candidate took 2.134227144 s (6.31%) longer. The previous GQA48 schedule
screen took 37.034 s, so this schedule reduced the observed gap by about 1.06 s
but did not beat default. These are one sample per path, directional only; no
performance qualification is claimed.

## Matched setup

- Host: Machinist
- Source worktree: `/tmp/m31-gqa48-rocm721-20261016`, clean at
  `cfef9b84133cc410e0a684bb4ee754b3d04534c1` before build
- GPU2 BDF `0000:88:00.0`, GFX906; benchmark reports product name `AMD Radeon
  Graphics`
- Model: Qwen3.8-27B Q4_K_M, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Prompt: 8,192 tokens; generation: zero; one sample; no warmup
- `-b 2048 -ub 1024`, Q8_0 K/V, Flash Attention, 99 GPU layers
- Pinned image: `localhost/miinfer-dev:rocm-7.2.1`
- Candidate env: `GGML_CUDA_FA_Q8_DOT4_PREPACK=1`
- Default start junction 31 C, reached 75 C at 30 s; candidate start 38 C,
  reached 78 C at 30 s. Both completed below the 84 C stop threshold.
- The 64K power guard did not pass at the last live cap check (GPU0 225 W,
  GPU2 300 W); this 8K run does not qualify 64K operation.

Exact invocation is in `.audit/run-m31-gqa48-8k-screen.sh`. Its JSON reports
`build_commit=unknown` because the container cannot follow the host worktree's
`.git` pointer. The host verified the clean source worktree commit before the
build; the remote run directory
`/tmp/m31-gqa48-8k-screen-gpu2-20261016T051600Z` contains the binary hash and
complete temperature/metadata files.

## Correctness and next step

The focused `FLASH_ATTN_EXT` Q8 GQA6 fixture passed 1/1 on each selected ROCm
device and logged the opt-in dot4 route. This checks operator output against
the backend reference; it does not check full-model logits, graph replay, or
generated tokens. No decode test ran.

The schedule change recovered part of the prior regression but remains slower
in the screen. Before investing in 64K, retrieve the saved run metadata and
attention profile for this GQA2 build. If the dot4 attention time still loses,
stop tuning the tile shape and replace the arithmetic/dataflow approach. A 64K
comparison still requires both GPU caps at or below 180 W.
