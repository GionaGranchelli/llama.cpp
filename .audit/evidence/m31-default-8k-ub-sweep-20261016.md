# Default-path 8K microbatch screen

## Result

Eight sequential, one-sample default-path screens compared microbatch sizes
with the same 8,192-token prompt and `-b2048`:

| Order | `-ub` | Wall time | Prompt throughput |
| ---: | ---: | ---: | ---: |
| 1 | 1024 | 33.870042358 s | 241.866 tok/s |
| 2 | 512 | 35.939666784 s | 227.938 tok/s |
| 3 | 2048 | 33.523023472 s | 244.369 tok/s |
| 4 | 1024 | 33.814318856 s | 242.264 tok/s |
| 5 | 2048 | 33.577911158 s | 243.970 tok/s |
| 6 | 1024 | 33.866487887 s | 241.891 tok/s |
| 7 | 2048 | 33.590953946 s | 243.875 tok/s |
| 8 | 1024 | 33.929889513 s | 241.439 tok/s |

Four 1024 samples have a median of 33.868265123 s, mean of 33.870184654 s, and
range of 33.814 to 33.930 s. Three 2048 samples have a median of 33.577911158 s,
mean of 33.563962859 s, and range of 33.523 to 33.591 s. The medians favor 2048
by 0.290 s (0.86%) with non-overlapping sample ranges. The single 512 sample
remains 6.11% slower than the 1024 mean. These are small sequential samples;
the 2048 result is a repeatable screening signal, not a full qualification.

## Setup

- Host: Machinist; GPU2 BDF `0000:88:00.0`, GFX906
- Source worktree synced to `e6f92398f0dad240a11812bc269f634791e17e79`
- Model: Qwen3.8-27B Q4_K_M, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Pinned image: `localhost/miinfer-dev:rocm-7.2.1`
- Default attention path only; no dot4 environment variable
- Q8_0 K/V, Flash Attention, 99 GPU layers, `-b2048`, `-p8192`, `-n0`, one
  sample per run, no warmup
- The screen runner checked model SHA, GPU BDF, KFD idle state, source/binary
  hashes, caps, and temperatures per run. All samples completed below 84 C;
  observed junction at 30 seconds was 75, 75, 77, 78, 76, 77, 77, and 77 C.
- The last live power check was GPU0 225 W / GPU2 300 W. These 8K runs do not
  qualify 64K operation.

Runs and metadata:

- `/tmp/m31-gqa48-8k-screen-gpu2-20261016T052423Z` (`-ub1024`)
- `/tmp/m31-gqa48-8k-screen-gpu2-20261016T052545Z` (`-ub512`)
- `/tmp/m31-gqa48-8k-screen-gpu2-20261016T052709Z` (`-ub2048`)
- `/tmp/m31-gqa48-8k-screen-gpu2-20261016T052832Z` (`-ub1024`)
- `/tmp/m31-gqa48-8k-screen-gpu2-20261016T053056Z` (`-ub2048`)
- `/tmp/m31-gqa48-8k-screen-gpu2-20261016T053218Z` (`-ub1024`)
- `/tmp/m31-gqa48-8k-screen-gpu2-20261016T053340Z` (`-ub2048`)
- `/tmp/m31-gqa48-8k-screen-gpu2-20261016T053501Z` (`-ub1024`)

The benchmark JSON reports `build_commit=unknown` inside the container. The
host worktree was clean at the pinned commit before each run; the runner's
metadata has the binary and backend hashes. No full-model numerical check or
decode was run.

## Next action

Do not use 512 for this workload. The 2048 screen shows a small, consistent 8K
gain, but its higher scratch-memory demand may affect the 64K capacity target.
Test the 1024/2048 pair at 64K after both GPU caps are lowered to 180 W or less;
then select the largest context that fits without changing the default path.
