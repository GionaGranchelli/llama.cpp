# Route-confirmed Q8 dot4 8K prefill screen

## Result

The guarded 8,192-token prefill pair completed on GPU2. The verbose candidate
log confirmed the opt-in GFX906 Q8 dot4 route was selected 800 times; the
default log confirmed no dot4 selection. Both processes exited successfully,
and the route, source, and build checks all passed.

| Route | Prompt time | Prompt rate | Start junction | Peak junction |
| --- | ---: | ---: | ---: | ---: |
| Default | 33.887504 s | 241.741 tok/s | 33 C | 80 C |
| Q8 dot4 prepack | 37.417953 s | 218.932 tok/s | 39 C | 81 C |

The candidate was 10.42% slower by wall time and 9.43% lower by prompt rate in
this single pair. It started 6 C warmer. This is a negative screening result,
not a temperature-matched repeated benchmark. The previous route-confirmed
screen on an earlier source also showed a slowdown of about 9%.

## Provenance

- Host: Machinist
- GPU: ROCm index 2, PCI BDF `0000:88:00.0`
- Source and binary commit: `274649a6a7f78597d951d4b0899def3c2d7a0b20`
- Source tree was clean
- Model: Qwen3.8-27B Q4_K_M, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Settings: 8,192 prompt tokens, Q8_0 K/V, Flash Attention, `-b 2048 -ub
  1024`, 99 GPU layers, no warmup, one sample, no generation
- Container image ID:
  `50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`
- GPU2 power cap: 300 W
- Binary hashes and temperature trace are in the raw run directory:
  `/tmp/m31-q8-dot4-8k-ab-20261016T100506Z` on Machinist
