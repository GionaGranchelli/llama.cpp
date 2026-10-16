# Default-path 8K microbatch screen

## Result

Four sequential, one-sample default-path screens compared microbatch sizes with
the same 8,192-token prompt and `-b2048`:

| Order | `-ub` | Wall time | Prompt throughput |
| ---: | ---: | ---: | ---: |
| 1 | 1024 | 33.870042358 s | 241.866 tok/s |
| 2 | 512 | 35.939666784 s | 227.938 tok/s |
| 3 | 2048 | 33.523023472 s | 244.369 tok/s |
| 4 | 1024 | 33.814318856 s | 242.264 tok/s |

The two 1024 samples average 33.842180607 s (range 33.814 to 33.870 s). The
single 512 sample is 6.20% slower than that average. The single 2048 sample is
0.94% faster. The repeat suggests the 512 loss is larger than this run's
observed 1024 variation; the possible 2048 gain is small and still needs
repetition. This is a screening result, not a qualified comparison.

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
  observed junction at 30 seconds was 75, 75, 77, and 78 C respectively.
- The last live power check was GPU0 225 W / GPU2 300 W. These 8K runs do not
  qualify 64K operation.

Runs and metadata:

- `/tmp/m31-gqa48-8k-screen-gpu2-20261016T052423Z` (`-ub1024`)
- `/tmp/m31-gqa48-8k-screen-gpu2-20261016T052545Z` (`-ub512`)
- `/tmp/m31-gqa48-8k-screen-gpu2-20261016T052709Z` (`-ub2048`)
- `/tmp/m31-gqa48-8k-screen-gpu2-20261016T052832Z` (`-ub1024`)

The benchmark JSON reports `build_commit=unknown` inside the container. The
host worktree was clean at the pinned commit before each run; the runner's
metadata has the binary and backend hashes. No full-model numerical check or
decode was run.

## Next action

Do not use 512 for this workload. Run a short interleaved confirmatory screen
for 1024 and 2048. If 2048 repeats a gain, carry it into the guarded 64K run
after both GPU caps are lowered to 180 W or less. Otherwise keep 1024 and move
the next change to the dominant Q4_K MMQ implementation rather than claim a
microbatch win.
