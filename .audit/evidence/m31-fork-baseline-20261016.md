# llama.cpp fork baseline on Machinist

## Verdict

The pinned fork baseline reproduced an 8,192-token Qwen3.8-27B Q4_K_M
prefill in 33.91 seconds (241.58 prompt tokens/s) on GPU2. This is a one-run
control, suitable for anchoring the focused port comparison, not a performance
qualification. No generated tokens were requested.

## Pinned run

- Host: Machinist X99, Xeon E5-2680 v4
- GPU: GPU2, GFX906, PCI BDF `0000:88:00.0`, 32,752 MiB VRAM
- Source commit: `56729ae70a66bf075096b0f5b4ae3d0581041b7b`; source tree clean
- Runtime image: `localhost/miinfer-dev:rocm-7.2.1`, image ID
  `50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`
- Model: `/home/machinist/models/Qwen3.8-27B-Q4_K_M.gguf`
- Model SHA-256:
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Binary SHA-256:
  `53a069db463e3bb728e30411b6190384da6fe0b3669b2eadfaa37aff9e90de11`
- Settings: Q8_0 K/V, Flash Attention, 99 GPU layers, batch 2048,
  microbatch 1024, no warmup, one run, 8,192 prompt tokens, zero generated
  tokens
- GPU2 power cap at run: 300 W. This run does not qualify the 64K thermal
  gate, which requires both MI50s at or below 180 W.
- Result: 33.910064262 s, 241.580 prompt tokens/s; process exit status 0.

Exact command:

```sh
llama-bench -m /models/Qwen3.8-27B-Q4_K_M.gguf -p 8192 -n 0 -r 1 --no-warmup -b 2048 -ub 1024 -ctk q8_0 -ctv q8_0 -fa on -ngl 99 -dev ROCm0 -o jsonl --progress
```

## Repeatability check on the unchanged default route

Four later default-route screens on the same GPU, model, prompt, and
microbatch measured 33.870, 33.814, 33.866, and 33.930 seconds. Median:
33.868 seconds; range: 33.814–33.930 seconds. The timings cluster around the
pinned control's 33.910 seconds. These runs used source commit
`e6f92398f0dad240a11812bc269f634791e17e79`; the Q8 dot4 route remained disabled.

The one-sample pinned baseline and four-run default-route check are separate
evidence: only the latter supports the narrow repeatability range. Neither
includes generated-token correctness or a decode measurement.

## Evidence files

Raw pinned-run metadata and output:

- `m31-step1-baseline-2026-10-16/gpu2-exact-metadata.txt`
- `m31-step1-baseline-2026-10-16/gpu2-exact-benchmark.log`

Repeated default-path screens:

- `m31-default-8k-ub-sweep-20261016.md`
