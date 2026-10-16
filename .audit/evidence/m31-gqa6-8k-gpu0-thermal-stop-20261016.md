# GQA6 dot4 8K screen stopped by the thermal guard

Date: 2026-10-16 (UTC)

## Result

No benchmark comparison was produced. The default 8,192-token prefill reached
the 84 C junction stop threshold and was terminated with status 137. No
benchmark JSON was emitted, and the dot4 candidate did not start. The
temperature trace began at 27 C and reached 84 C about 35 seconds later.

## Provenance

- Host: Machinist
- GPU: index 0, PCI BDF `0000:04:00.0`, GFX906
- Source commit: `9d2493b82d68792100ccb4ed00f20db68a84de93`
- Model SHA-256:
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Benchmark binary SHA-256:
  `1eb91cc09d67ea57fcbffe6a59aab02acb5111a5fce087108ad5d0d33c5c3198`
- HIP backend SHA-256:
  `c0739dae80a2d6f90c4ec78db2355efd0a8127bde0045b29bc7f7b852ee96633`
- ROCm image ID:
  `50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`
- Settings: batch 2048, microbatch 1024, Q8_0 K/V, flash attention,
  99 GPU layers, no decode; live power caps 225 W and 300 W.
- Remote raw evidence: `/tmp/m31-gqa48-8k-screen-gpu0-20261016T082107Z/`
- Temperature CSV SHA-256:
  `610e482a9b42cd1254676046580ed2dec3bf0669a24f5e3b826cdb5fffb8966e`

## Decision

This is a thermal abort, not evidence for or against the kernel. Run the next
screen on GPU2, which has completed prior 8K screens, and preserve failure
metadata in the runner output. Do not retry this exact GPU0 run under the same
thermal conditions.
