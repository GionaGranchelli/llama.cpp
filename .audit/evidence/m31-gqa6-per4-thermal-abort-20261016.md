# Per-four-value GQA6 dot4 candidate build and thermal aborts

The corrected per-four-value query-scale implementation compiled successfully
in the pinned ROCm container on Machinist. The guarded comparison started the
default path on GPU2 (`0000:88:00.0`), initially at 31 C junction, and then a
second matched comparison started on GPU0 (`0000:04:00.0`), initially at 28 C.
Both reached the 84 C warning threshold, so the runner stopped each before the
85 C hard stop. Neither candidate run started. These attempts provide no
numerical or timing result for the per-four-value candidate.

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

## Cooler-card retry

- Runner/source commit: `fe295d15806e900ba6de773108db9cefaa035bad`
- GPU: index 0, BDF `0000:04:00.0`
- Live power caps: GPU0 225 W, GPU2 225 W
- Start junction: 28 C; stop reason: junction reached 84.0 C, status 137
- Prompt SHA-256: `60c08ad04e653ca5abb082b34ff1b4f995623e5baa001ab6bbdfd4eaa3ef454a`
- Raw run directory: `/tmp/m31-gqa6-8k-logits-gpu0-20261016T090650Z`

The logits file was emitted in each default log, but both runs are recorded as
aborts because the processes were killed by the guard and neither has a
matched candidate output. The runner now accepts GPU index and BDF overrides.
Both MI50s reaching the same threshold at the current 225 W caps points to a
host cooling or power-limit issue rather than a card-specific difference; a
lower cap is needed before another long prefill attempt.
