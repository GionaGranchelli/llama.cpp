# Direct-Q8 environment 8K screen (dispatch unverified)

Date: 2026-10-16 (UTC)

## Result

One sequential 8,192-token full-model prefill pair completed on GPU2:

| Run | Time | Throughput |
| --- | ---: | ---: |
| Default | 33.857790223 s | 241.953 tok/s |
| `GGML_CUDA_FA_Q8_DIRECT_TILE=1` | 37.317356300 s | 219.523 tok/s |

The environment-selected run was 10.22% slower by wall time. This is only a
directional screen: the benchmark log did not include a kernel dispatch trace,
so it is not yet proven that the direct-Q8 attention kernel ran. No decode or
numerical comparison was performed.

## Provenance

- Host: Machinist
- GPU: ROCm index 2, PCI BDF `0000:88:00.0`, GFX906
- Source commit: `9d2493b82d68792100ccb4ed00f20db68a84de93`
- Model SHA-256:
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Benchmark binary SHA-256:
  `1eb91cc09d67ea57fcbffe6a59aab02acb5111a5fce087108ad5d0d33c5c3198`
- HIP backend SHA-256:
  `c0739dae80a2d6f90c4ec78db2355efd0a8127bde0045b29bc7f7b852ee96633`
- ROCm image ID:
  `50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`
- Settings: 8,192 prompt tokens, zero decode, batch 2048, microbatch 1024,
  Q8_0 K/V, Flash Attention, 99 GPU layers
- Remote raw evidence: `/tmp/m31-gqa48-8k-screen-gpu2-20261016T082614Z/`
- Start junction: default 31 C; candidate 38 C. Candidate reached 77 C in
  the runner heartbeat, below the 84 C stop threshold.

## Next decision

Verify direct-Q8 dispatch with the focused backend fixture or kernel profile.
If it did dispatch, reject this route for 8K prefill performance and move to
the Q4_K MMQ prefill bottleneck. If it did not, correct the route selection
before drawing a kernel conclusion.
