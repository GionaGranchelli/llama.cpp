# Q8 dot4 GQA2 GPU0 thermal abort

## Outcome

GPU0 completed the default 8K prefill, then the opt-in dot4 prefill reached the
thermal stop and was killed. This is not a matched A/B result; no candidate
timing was produced.

| Path | Outcome | Start junction | Highest sampled junction |
| --- | --- | ---: | ---: |
| Default | 33.483 s, 244.660 tok/s, exit 0 | 27 C | 83 C |
| Q8 dot4 prepack | killed, exit 137; no result | 38 C | 85 C |

The runner sampled via `rocm-smi` once per second and requested a stop at 84 C.
The recorded temperature jumped from 83 C to 85 C between samples, so the
actual observed stop was at 85 C. Do not use this candidate attempt as a
performance result.

At the same 38 C start temperature, the prior GPU2 dot4 8K sample completed at
36.127 s / 226.758 tok/s with a sampled peak of 79 C. Together with this GPU0
thermal abort, that points to GPU2 being the safer card for the candidate at
the current caps. This is directional evidence from one run per card, not a
cooling qualification.

## Run identity

- Host: Machinist
- Source: `96ce5481591e110a02b3a745a9ae51c6f430cf12`
- Binary was built at `01a91093734977e04de98ebd48ac2fe169b47a63`; source files
  used by the binary were unchanged through the screen.
- GPU: ROCm index 0, BDF `0000:04:00.0`, GFX906
- Model SHA-256:
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Image ID: `50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`
- Benchmark binary SHA-256:
  `2fabfb1c3a355ec18d2faf6cf901cbe487648a2ba705c3378b726710442f556a`
- HIP backend SHA-256:
  `b24d4e14559b39b6ddf6e671410524a738b4bfa61ee5faca90b5b78012d0f872`
- Workload: 8,192 prompt tokens, zero generation, `-b 2048 -ub 1024`,
  Q8_0 K/V, Flash Attention, 99 GPU layers, one sample per path
- Power caps: GPU0 225 W, GPU2 300 W
- Raw artifacts: `/tmp/m31-gqa48-8k-screen-gpu0-20261016T070237Z/`

The host exposes AMD junction temperature as a readable `temp*_input` sysfs
sensor labeled `junction` for each MI50. The screen, profile, and guarded 64K
runners now read that sensor directly at a shorter interval, avoiding the
one-second CLI polling delay seen here. The direct-read change has not yet
been exercised on a subsequent inference run.
