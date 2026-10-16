# Q8 dot4 full-model prefill screen, route not confirmed

## Result

Both 8K benchmark processes completed, but the comparison is inconclusive:
the candidate log did not confirm that the opt-in dot4 path was selected. The
runner exited 8 for this reason. The runner had not enabled verbose backend
logging, so absence of the route message does not prove that the path was not
selected.

| Requested route | Prompt time | Prompt rate | Start junction | Peak junction |
| --- | ---: | ---: | ---: | ---: |
| Default | 33.929 s | 241.447 tok/s | 32 C | 79 C |
| Dot4 prepack flag | 37.411 s | 218.973 tok/s | 39 C | 80 C |

Do not treat these as a measured dot4 speed result. The candidate requested
route was 10.3% slower by wall time, but route selection was not proven and
each path ran once. Both processes exited 0, below the 84 C stop limit.

## Provenance

- Host: Machinist
- GPU: ROCm index 2, PCI BDF `0000:88:00.0`
- Source and binary commit: `274649a`
- Model: Qwen3.8-27B Q4_K_M, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Settings: 8,192 prompt tokens, Q8_0 K/V, Flash Attention, `-b 2048 -ub
  1024`, 99 GPU layers, no warmup, one sample, no generation
- GPU power cap: 300 W
- Raw remote output and temperature trace:
  `/tmp/m31-q8-dot4-8k-ab-20261016T100119Z`

The runner now enables verbose backend logging for both paths so the next
comparison can prove route selection without changing logging between them.
