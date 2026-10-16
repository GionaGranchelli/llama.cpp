# GQA6 dot4 8K prefill screen on Machinist GPU2

The guarded default/candidate pair completed on 2026-10-16. Both benchmark
processes exited 0 with no thermal stop. Junction telemetry peaked at 81 C,
below the runner's 84 C stop threshold. This is a prefill timing screen only;
it did not generate tokens or compare logits.

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

The runner requested the opt-in dot4 candidate. The compact metadata and
temperature trace were retrieved, but the first retrieval elided the long JSON
result lines and did not show the dispatch line. Exact timings and route
selection still need extraction from the raw logs. Do not infer a speedup or
regression from this report; the screen is incomplete pending that extraction.
