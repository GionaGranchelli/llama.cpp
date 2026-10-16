# GQA6 dot4 per-four-scale short-prompt logits smoke

The corrected per-four-value query-scale candidate and the default path both
completed on GPU0 without a thermal stop. Their token-ID files matched, and
the candidate log confirmed 336 dot4 dispatches.

| Metric | Default vs per-four-scale dot4 |
| --- | ---: |
| Final logits compared | 248,320 |
| Maximum absolute difference | 0.41359663 |
| RMS difference | 0.08385918 |
| Cosine similarity | 0.999717343 |
| Top-1 token | `271` / `271` |
| Top-10 overlap | 9/10 |

This is a positive short-prompt accuracy signal. Compared with the earlier
per-32-scale 11-token smoke (max absolute difference `0.629282475`, RMS
`0.115477`, cosine `0.999415556`), the per-four-scale candidate is closer to
default on this prompt. That earlier run used GPU2, so this cross-card
comparison is indicative only. The result does not pass the strict long-context
gate, establish 2K/12K/64K behavior, measure speed, or verify generated tokens.

## Provenance

- Source/runner commit: `a61fa8053fed09d5a42aab2a82140b087c95e90c`
- Model SHA-256:
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- ROCm image ID:
  `50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`
- GPU0 BDF: `0000:04:00.0`
- Power caps: GPU0 225 W, GPU2 225 W
- Prompt: 11 tokens, 44 bytes, SHA-256
  `313e396e6b4cad68835d058bc6281ea20366f2b19552096d8be3640cd242f646`
- Token-ID SHA-256 (both):
  `0812a36513fa77d765fbb6a16f1ad95d2d9c099acda7dbbb10ef4ab7d89f0bdd`
- Default logits SHA-256:
  `ba6a89e90f1e7fd2aeae4333f6d5c1963d05b8c81f6b88ed0c0dd176a41d955c`
- Candidate logits SHA-256:
  `c2f4edc8a30009f25f03b317b8300a6c014ee68da54643110dc8ba2eb28a5c16`
- Start junction temperatures: default 29 C, candidate 31 C
- Both runs exited 0; no generated tokens
- Raw run directory: `/tmp/m31-gqa6-8k-logits-gpu0-20261016T091020Z`
