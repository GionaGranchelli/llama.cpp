# 16K J=40 crossover stopped by thermal guard

## Result

The simultaneous 16,384-token default-versus-J=40 prefill screen did not
complete on either card. The 84 C guard stopped the default run on GPU0 after
37 seconds and the J=40 run on GPU2 after 41 seconds. Both containers exited
with status 137 before producing a benchmark timing. This is a thermal-stop
result, not a performance comparison.

## Setup

- Host: Machinist X99
- Source: `e6f92398f0dad240a11812bc269f634791e17e79`, clean
- Binary SHA-256:
  `2bac20a883561d40a97a458a2342d44ba28fa92a17ab11f0cda6dbd50ab8d54b`
- Model SHA-256:
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Image: `localhost/miinfer-dev:rocm-7.2.1`, ID
  `50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`
- GPU0 / `0000:04:00.0`: default dispatch, start junction 28 C, cap 225 W
- GPU2 / `0000:88:00.0`: `GGML_CUDA_MMQ_Q4K_J40=1`, start junction 30 C,
  cap 300 W
- Shared settings: Qwen3.8-27B Q4_K_M, Q8_0 K/V, Flash Attention,
  `-p 16384 -n 0 -r 1 --no-warmup -b 2048 -ub 1024 -ngl 99`
- Both GPUs were idle before launch. Each card had an independent 84 C stop.

## Evidence and next step

The temperature trace first recorded 80 C on both cards at `2026-10-16T06:03:25Z`.
It recorded GPU0 at 84 C at `06:03:31Z` and GPU2 at 84 C at `06:03:35Z`; the
run started at `06:02:54Z`. The per-card hard stop worked, but this ad-hoc
16K runner did not emit the 80 C warning. No benchmark result was emitted for
either path: both logs stop after benchmark startup. The raw logs and
per-second temperature trace are now stored in
`.audit/evidence/m31-j40-16k-crossover-20261016T060254Z/`.

The current caps do not support even this guarded 16K comparison from a cool
start. Do not retry 16K or attempt 64K at these caps. Re-run after both MI50
caps are verified at or below 180 W; then compare default and J=40 at 16K
before attempting the guarded 64K gate.
