# Per-four-value GQA6 dot4 candidate build and thermal abort

The corrected per-four-value query-scale implementation compiled successfully
in the pinned ROCm container on Machinist. The guarded comparison then started
the default path on GPU2 (`0000:88:00.0`), initially at 31 C junction. It
reached the 84 C warning threshold, so the runner stopped it before the 85 C
hard stop. The candidate did not start. This provides no numerical or timing
result for the per-four-value candidate.

## Provenance

- Source commit: `a432a8c0bd0b28cee152ba93c7ca99e5f270573d`
- ROCm image ID: `50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`
- GPU: index 2, BDF `0000:88:00.0`
- Model SHA-256:
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Live power caps: GPU0 225 W, GPU2 300 W
- Prompt: 12,738 tokens, 37,935 bytes, SHA-256
  `60c08ad04e653ca5abb082b34ff1b4f995623e5baa001ab6bbdfd4eaa3ef454a`
- Raw run directory: `/tmp/m31-gqa6-8k-logits-gpu2-20261016T090358Z`
- Stop reason: junction reached 84.0 C; process ended with status 137

The logits file was emitted in the default log, but this run is recorded as an
abort because the process was killed by the guard and no matched candidate
output exists. The runner now accepts GPU index and BDF overrides so the next
matched pair can use GPU0 if its cooler operating point sustains the workload.
