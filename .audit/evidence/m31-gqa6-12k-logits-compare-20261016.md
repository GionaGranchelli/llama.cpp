# GQA6 dot4 full-model logits comparison at 12,738 prompt tokens

The corrected guarded comparator completed both default and candidate runs on
Machinist GPU2 (`0000:88:00.0`). Both processed the same 12,738-token prompt;
the binary prompt-token files matched. The candidate log confirmed selection of
the GFX906 Q8 dot4 attention tile 1,297 times. Both processes exited 0 without
a guard stop. No generation tokens were requested.

| Metric | Default vs dot4 |
|---|---:|
| Final logits compared | 248,320 |
| Maximum absolute difference | 6.5643742 |
| RMS difference | 0.8243170 |
| Cosine similarity | 0.9763538 |
| Top-1 token | `271` / `271` |
| Top-10 overlap | 7/10 |

The matching top-1 is not enough to qualify this path: its full logits differ
substantially, and it fails the stricter Q8 dot4 numerical gate used in
MIInfer EXP-0409 (`max_abs <= 0.01`, cosine `>= 0.9999`). Keep dot4 opt-in and
do not claim model correctness from the short prompt or the matching top-1.

## Provenance

- Source: `79ee3a589edcc091db1fb93708e3a59e1dab3d1f`
- Model SHA-256:
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- ROCm image ID:
  `50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`
- Prompt bytes: 37,917
- Prompt SHA-256:
  `024d5e192e4941000daaf216141042a622d9e442721a0ddaed1643f46237b5e2`
- Default logits SHA-256:
  `8a4491c8934c2d3de38fe7ede478de6a9d146c9266213c8de463bdf99a8f6eed`
- Dot4 logits SHA-256:
  `87518d2b241a6b70962ebad0f283cb1390d830cb39899ae374cb4678d53ef685`
- Prompt-token SHA-256 (both runs):
  `e26359e5e21da0762550e97d900af4412d52932c3de907e344a80291ca7c5c98`
- Live power caps: GPU0 225 W, GPU2 300 W
- Raw run directory:
  `/tmp/m31-gqa6-8k-logits-gpu2-20261016T085401Z`

The comparison was not temperature-interleaved: default started at 31 C and
dot4 at 40 C. This affects timing, not the identical-input logit comparison.
The run generated no tokens, so it does not qualify generation behavior.
