# Q4_K MMQ occupancy-one kernel profile

Date: 2026-10-16 (UTC)

## Finding

The occupancy-one build selected the same Q4_K `J=64` MMQ kernel 1,144 times
as the control, but changed its code-object resources and increased its
profiled time:

| Build | Option | Q4_K calls | VGPRs | `scr` field | LDS | Total Q4_K kernel time |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Control | `...OCCUPANCY_1=OFF` | 1,144 | 128 | 44 | 29,184 B | 10.387 s |
| Candidate | `...OCCUPANCY_1=ON` | 1,144 | 140 | 0 | 29,184 B | 16.509 s |

In this single profiled 4K run, the candidate Q4_K MMQ work took 6.122 s
(59.0%) longer. The `scr` field fell from 44 to 0 while VGPR use rose by 12.
The change therefore did not improve this workload's measured
kernel throughput. This supports the earlier 8K regression, but is not a
decision-grade comparison: the run used one sample per build, profiling
instrumentation was enabled, and the candidate started warmer (41 C versus
28 C).

## Run conditions and limits

- Source commit: `3310e82ebaa0280ceaf8d1f37dad5afdbef02f33`
- Host: Machinist, ROCm 7.2.1 container
- GPU: ROCm 0 / PCI BDF `0000:04:00.0` / GFX906
- Model: Qwen3.8-27B Q4_K_M, pinned SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Workload: 4,096 prompt tokens, zero generated tokens, batch 2,048,
  microbatch 1,024, Q8_0 K/V, Flash Attention, 99 GPU layers
- Profiler: `rocprof --stats --timestamp on`
- Sampled junction peak: 75 C control, 76 C candidate
- Live caps at preflight: GPU0 225 W, GPU2 300 W

The one-off wrapper compared ROCm decimal temperatures with shell integer
tests, so its automated 84 C stop was ineffective. It must not be reused.
Both recorded traces peaked below 84 C, and both profile runs exited 0, but
the run is not counted as a guarded safety result. The next run must use
floating-point-safe comparisons and matched start temperatures.

## Artifacts

Raw profile CSVs, logs, and temperature samples are on Machinist at
`/tmp/m31-q4k-occ1-profile-20261016T063750Z/`.
