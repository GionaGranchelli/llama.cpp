# GQA6 dot4 8K prefill screen on Machinist GPU2

The guarded default/candidate pair completed on 2026-10-16. Both benchmark
processes exited 0 with no thermal stop. Junction telemetry peaked at 81 C,
below the runner's 84 C stop threshold. This is a prefill timing screen only;
it did not generate tokens or compare logits.

| Route requested | Prompt time | Prompt rate | Start junction |
|---|---:|---:|---:|
| Default | 33.901330475 s | 241.642434 tok/s | 31 C |
| Dot4 candidate | 36.922799009 s | 221.868337 tok/s | 38 C |

The candidate sample was 8.91% slower by wall time (8.18% lower prompt rate),
but this is one sample per route and the candidate started 7 C warmer. The
candidate environment flag was passed, but the saved log has no dispatch
message, so selection of the dot4 kernel is unconfirmed. Treat this only as a
negative directional screen, not a kernel performance verdict.

## Provenance

- GPU: ROCm index 2, PCI BDF `0000:88:00.0`
- Source: `9d2493b82d68792100ccb4ed00f20db68a84de93`
- `llama-bench`: SHA-256
  `1eb91cc09d67ea57fcbffe6a59aab02acb5111a5fce087108ad5d0d33c5c3198`
- HIP backend: SHA-256
  `c0739dae80a2d6f90c4ec78db2355efd0a8127bde0045b29bc7f7b852ee96633`
- Model: Qwen3.8-27B-Q4_K_M, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Container image ID:
  `50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`
- Settings: 8,192 prompt tokens, `-b 2048 -ub 1024`, Q8_0 K/V,
  Flash Attention, 99 GPU layers, no warmup, no generation
- Live caps: GPU0 225 W, GPU2 300 W
- Default started at 31 C; candidate at 38 C. Both stop reasons were empty.
- Raw run directory:
  `/tmp/m31-gqa48-8k-screen-gpu2-20261016T083942Z`

The raw JSON fields and temperature trace were extracted from the first
remote logs. Its candidate log did not contain the route-selection message.
Full-model correctness remains open.

## Verbose route-confirmed repeat

A second guarded pair used the verbose runner on the same MI50 and workload.
The candidate log confirmed selection of `selected opt-in GFX906 Q8 dot4
flash-attention tile`.

| Route | Prompt time | Prompt rate | Start junction |
|---|---:|---:|---:|
| Default | 33.900880427 s | 241.645642 tok/s | 31 C |
| GQA6 dot4 | 36.933431136 s | 221.804467 tok/s | 39 C |

The confirmed dot4 run was 8.95% slower by wall time and 8.21% lower in prompt
rate in this single pair. Both routes exited 0 without a thermal stop; peak
junction was 79 C. The candidate started 8 C warmer, so this is not a
temperature-matched or interleaved comparison. Its result closely matches the
preceding requested-dot4 screen, but only this repeat has route confirmation.

Provenance: source `ae88eee89759a99b311e358c3a8cc7a9b1694baa`; benchmark and
HIP backend hashes match the preceding screen. Remote raw files are under
`/tmp/m31-gqa48-8k-screen-gpu2-20261016T084425Z/`. The run had no generation
tokens and did not compare logits.
