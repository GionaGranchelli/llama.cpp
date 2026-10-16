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

Exact invocation is in `.audit/run-m31-gqa48-8k-screen.sh`. The screen JSON
reports `build_commit=unknown` because the container cannot follow the host
worktree's `.git` pointer. The host verified the clean source worktree commit
before the build. The captured benchmark binary SHA-256 is
`2bac20a883561d40a97a458a2342d44ba28fa92a17ab11f0cda6dbd50ab8d54b`; its HIP
backend SHA-256 is
`fac44e460b5f2ab3ba52e5597fbe7b4f778a709ae519031ebd4a7899a888e8b6`.

## Profile attribution

The paired `rocprof --stats` run used the same source commit, binary, image,
model, GPU, prompt, and settings. It confirms the candidate attention kernel
still explains most of the timing loss:

| Path / kernel | Calls | Summed duration | Kernel-time share |
| --- | ---: | ---: | ---: |
| Default Q4_K MMQ | 2,288 | 21.138 s | 63.20% |
| Default attention `flash_attn_tile<256,256,16,2,...>` | 128 | 1.573 s | 4.70% |
| Candidate Q4_K MMQ | 2,288 | 21.603 s | 60.49% |
| Candidate attention `flash_attn_tile<256,256,16,2,false,true,true>` | 128 | 3.344 s | 9.36% |
| Candidate Q8 query prepack | 128 | 0.033 s | below 0.1% |

The dot4 attention kernel takes 2.13x default time, adding 1.771 s across the
128 launches. Q8 query prepack itself is only 33 ms. The profiler's total
llama-bench times were 33.922 s default and 36.192 s candidate, including
profiler overhead; do not treat those as standalone speed measurements. The
quiet screen independently measured a 2.134 s loss. The candidate also had
0.465 s more Q4_K MMQ time in the profile, so not all of the profiled total
difference is attention.

Profile evidence is in
`/tmp/m31-gqa48-8k-profile-gpu2-20261016T051932Z`; the source was clean at
`cfef9b84133cc410e0a684bb4ee754b3d04534c1`, kernel-source SHA-256
`db19b30d9b67e1def42975d16a8315f5d7e9f434b5f63b7d8aa81dbd9740cf8d`, image ID
`50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`, and
GPU2 BDF `0000:88:00.0`. Live caps were 225 W / 300 W. Junction reached 77 C on
default and 80 C on candidate, below the 84 C stop threshold.

## Correctness and next step

The focused `FLASH_ATTN_EXT` Q8 GQA6 fixture passed 1/1 on each selected ROCm
device and logged the opt-in dot4 route. This checks operator output against
the backend reference; it does not check full-model logits, graph replay, or
generated tokens. No decode test ran.

The GQA2 schedule recovered part of the prior regression, but Q8 dot4 remains
substantially slower than the default attention kernel, even at the same tile
shape. Stop tuning this tile. At 8K, Q4_K MMQ is the largest stage (about 63%);
test one bounded `-ub` comparison on the default path next, then decide whether
to move effort to the MMQ path or return to attention with a different
dataflow. A 64K comparison still requires both GPU caps at or below 180 W.
