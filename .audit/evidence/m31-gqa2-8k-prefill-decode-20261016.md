# GQA2 Q8 dot4 8K screen on Machinist

## Verdict

Inconclusive. The current GQA2 dot4 route completed its 8,192-token prefill
in 36.129 s versus 33.893 s for default, a 6.60% slower one-sample result.
The candidate route was selected, but the thermal guard stopped its run before
the decode phase. This does not establish a repeatable regression or candidate
decode behavior.

## Run provenance

- Host: Machinist
- GPU: MI50, ROCm index 2, PCI BDF `0000:88:00.0`
- Source commit: `625c7e9f603f789e97d6cd79022fb30a1c7de2e3`
- Model SHA-256:
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- `llama-bench` SHA-256:
  `bfadf21def1b2838a2e5cdf4b311a976f4de4bf9e6d7c781ac66fc088d2a6e3a`
- HIP backend SHA-256:
  `7e2b26d6615f481c81bd6c3ed7b6dcd8c8ffd6cee31cff7cf1a6877cb3d2f35e`
- Runtime: `localhost/miinfer-dev:rocm-7.2.1`
- Settings: 8,192 prompt tokens, 64 generated tokens, Q8_0 K/V, Flash
  Attention, 99 GPU layers, batch 2048, microbatch 1024, no warmup, one
  sample per route
- Exact command:
  `llama-bench -m /models/Qwen3.8-27B-Q4_K_M.gguf -p 8192 -n 64 -r 1 --no-warmup -b 2048 -ub 1024 -ctk q8_0 -ctv q8_0 -fa on -ngl 99 -dev ROCm0 -o jsonl --progress -v`
- Raw logs and temperature trace:
  `/tmp/m31-gqa2-8k-pair-20261016T080152Z/`

## Results

| Route | 8K prefill | Decode, 64 tokens | Start / end junction |
|---|---:|---:|---:|
| Default | 33.892898 s, 241.703 tok/s | 2.529664 s, 25.300 tok/s | 31 C / 50 C |
| GQA2 dot4 | 36.128865 s, 226.744 tok/s | Stopped before decode | 40 C / 85 C observed |

The candidate log confirms `GGML_CUDA_FA_Q8_DOT4_PREPACK` dispatch. The run
used one sample per route; temperatures differed at start, the paths were
sequential, and no profile was collected for this schedule. The 6.60% delta
is directional only. The 84 C guard was polled once per second and observed
85 C before stopping the candidate container. Use a faster polling cadence on
the next guarded run.

## Next action

Do not promote the candidate or call the delta repeatable. Profile the current
GQA2 schedule and run an interleaved comparison on the cooler MI50 at
`0000:04:00.0`. The sustained 64K runner remains gated because the live caps
were 225 W and 300 W.
