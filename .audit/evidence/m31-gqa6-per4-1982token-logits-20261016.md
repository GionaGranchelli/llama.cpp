# GQA6 dot4 per-four-scale 1,982-token logits comparison

The default and per-four-scale candidate completed on GPU0 using the same
truncated source prompt. The prompt token-ID files matched. The candidate log
confirmed the opt-in dot4 route. Both runs exited 0 below the thermal guard.

| Metric | Default vs per-four-scale dot4 |
| --- | ---: |
| Prompt tokens | 1,982 |
| Final logits compared | 248,320 |
| Maximum absolute difference | 0.77102590 |
| RMS difference | 0.14083825 |
| Cosine similarity | 0.998540316 |
| Top-1 token | `271` / `271` |
| Top-10 overlap | 9/10 |

This validates candidate execution and gives a short-context numerical sample.
It does not compare against the previous per-32-scale candidate on this prompt,
does not pass the strict full-logit gate, and establishes neither long-context
correctness nor speed. The separate 11-token smoke showed a smaller difference,
but prompt length changes the accumulated error.

## Provenance

- Source/runner commit: `dd11baf3fc8001e4875f52cd1b163351c94f39a1`
- Model SHA-256:
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- ROCm image ID:
  `50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`
- GPU0 BDF: `0000:04:00.0`
- Power caps: GPU0 225 W, GPU2 225 W
- Prompt: 6,000 bytes, SHA-256
  `6ea02c474f8ce2106a9014351680f2a99e81c0678d64a43e07f5fe3f55aa58da`
- Token-ID SHA-256 (both):
  `d95785776724fbdb23e2aac6958a6e62dcf3a69482316ed4e4d5fc6707729491`
- Default logits SHA-256:
  `721c8aa0da2b8432a952ac92d3b1736466ef7a43db674c31ab633d76a3f2ff0f`
- Candidate logits SHA-256:
  `e818e1624b5cf9421f9bc4b7e40560bb848c9b79425261d9ebad49f75fa46ea0`
- Start junction temperatures: default 28 C, candidate 39 C
- Both runs exited 0; no generated tokens
- Raw run directory: `/tmp/m31-gqa6-8k-logits-gpu0-20261016T091226Z`
