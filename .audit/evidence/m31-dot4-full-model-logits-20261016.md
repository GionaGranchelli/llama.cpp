# Q8 dot4 short full-model logits comparison

## Run provenance

- Host: Machinist, GPU2, PCI BDF `0000:88:00.0`
- Source: `7f4a20998cdc8dd151b34180a03278685c96360e`
- `llama-debug` SHA-256: `e75615ace79e243fd56772d5b0ff5a293260f38d7e8667c1ca4f0c89e1e8d365`
- Model SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- GPU selection: `ROCR_VISIBLE_DEVICES=2`; Q8_0 K/V, Flash Attention, 99 GPU layers
- Prompt: `Answer briefly: what is the factorial of 6?`
- Prompt token IDs matched between variants; 11 tokens
- Both runs exited with status 0. The candidate log contains one
  `GGML_CUDA_FA_Q8_DOT4_PREPACK` dispatch; the default log contains none.
- Remote artifacts: `/tmp/m31-dot4-logits-20261016T072330Z/`
- Remote metadata records GPU caps of 225 W and 300 W, with GPU2 junction
  temperature 31 C at start, 32 C after default and 33 C after dot4.

## Comparison

| Metric | Result |
| --- | ---: |
| Maximum absolute final-logit difference | 0.629282475 |
| RMS absolute difference | 0.115477 |
| Cosine similarity | 0.999415556494 |
| Top-1 token | Same (`271`) |
| Top-10 overlap | 9/10 |

This is a positive short-prompt signal, not a correctness qualification. It is
one prompt and one sample, the final-logit max difference is nonzero, and no
campaign acceptance tolerance has been established for this comparison. It
does not verify generated-token trajectories, long-context correctness, or
HIP graph replay of the dot4-containing graph. Keep the path opt-in.

The run used a 2,048-token context, so it is not evidence for 64K operation.
The guarded 64K runner requires both power caps at or below 180 W; the recorded
caps were above that limit.
