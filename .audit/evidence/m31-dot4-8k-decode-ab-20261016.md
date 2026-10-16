# Q8 dot4 prefill and decode screen

## Verdict

The single completed prompt sample was slower with the dot4 opt-in: 36.110 s
versus 33.901 s by the default route, a 6.5% regression. The candidate's
generation phase did not complete because the 84 C junction guard stopped it.
This is a partial, one-sample screen, not a decode comparison or a performance
qualification.

## Setup

- Host: Machinist
- GPU: GPU2, GFX906, BDF `0000:88:00.0`
- Source: `e6f92398f0dad240a11812bc269f634791e17e79`
- Model SHA-256:
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Settings: 8,192 prompt tokens, 128 generated tokens, batch 2048,
  microbatch 1024, Q8_0 K/V, Flash Attention, 99 GPU layers, no warmup,
  one sample per task
- Command:
  `llama-bench -m /models/Qwen3.8-27B-Q4_K_M.gguf -p 8192 -n 128 -r 1 --no-warmup -b 2048 -ub 1024 -ctk q8_0 -ctv q8_0 -fa on -ngl 99 -dev ROCm0 -o jsonl --progress`
- Candidate environment: `GGML_CUDA_FA_Q8_DOT4_PREPACK=1`
- The container reported `build_commit=unknown`; the host worktree was pinned
  to the source commit above.

## Results

| Route | Prompt | Prompt throughput | Generation |
| --- | ---: | ---: | --- |
| Default | 33.901406425 s | 241.642 tok/s | 128 tokens in 4.999989712 s (25.600 tok/s) |
| Dot4 opt-in | 36.109658412 s | 226.865 tok/s | Stopped by thermal guard; no valid decode result |

The default and candidate prompt samples differ by 2.208 s. They are one
sequential sample each, so treat the 6.5% regression as directional. The
candidate prompt phase returned a benchmark result before the generation task
was stopped.

The candidate runner exited 137 and reported that its junction-temperature
guard fired at the 84 C threshold. A shell interpolation error omitted the
observed peak value from the diagnostic; after the stop, GPU2 junction was
56 C. GPU0 and GPU2 caps were 225 W and 300 W. No process remained after the
stop. The run did not confirm a maximum-temperature trace, and it must not be
used to qualify the 64K guard.

The opt-in variable was set for the candidate invocation, but this run's log
did not contain the kernel-selection message. Route selection is therefore
not independently confirmed for this sample.

## Evidence on Machinist

`/tmp/m31-dot4-8k-decode-ab-20261016T054027Z/default.log`

`/tmp/m31-dot4-8k-decode-ab-20261016T054027Z/candidate.log`

`/tmp/m31-dot4-8k-decode-ab-20261016T054027Z/metadata.txt`
