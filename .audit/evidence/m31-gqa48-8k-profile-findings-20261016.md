# GQA48 8K profile findings

The profiler traces are now available. The most important observed costs are
the Q4_K matrix multiplication and, on the candidate path, the new GQA6 dot4
attention kernel:

| Path / kernel | Calls | Summed duration | Share of profiled kernel time |
| --- | ---: | ---: | ---: |
| Default Q4_K MMQ (`ggml_type 12`) | 2,288 | 21.149 s | 63.22% |
| Default Q6_K MMQ (`ggml_type 14`) | 504 | 5.051 s | 15.10% |
| Default Q5_K MMQ (`ggml_type 13`) | 384 | 1.964 s | 5.87% |
| Candidate Q4_K MMQ (`ggml_type 12`) | 2,288 | 21.164 s | 58.02% |
| Candidate Q6_K MMQ (`ggml_type 14`) | 504 | 5.050 s | 13.85% |
| Candidate `flash_attn_tile<256,256,8,6,...>` | 128 | 4.553 s | 12.48% |

These are summed kernel durations from `rocprof --stats`, not end-to-end
benchmark times. The profiled llama-bench JSON also recorded 33.939 s for
default and 36.949 s for candidate, but profiler overhead makes those numbers
unsuitable for performance reporting. The separate quiet screen remains the
valid directional result: one sample each, candidate 9.31% slower.

The Q4_K MMQ duration is nearly unchanged between paths and is the largest
measured stage. The candidate's dot4 attention adds a substantial kernel cost;
the full default attention row was not included in the compact output retrieved
so far, so the exact attention delta is still unknown. Retrieve the complete
`default.stats.csv` and `candidate.stats.csv` before attributing the whole
slowdown to attention. The next remote parser should print all rows containing
`flash_attn` plus the top MMQ rows; do not repeat the profiler run.

Run: `/tmp/m31-gqa48-8k-profile-gpu2-20261016T050033Z` on Machinist.
Source commit: `176480a57a5b18be94efdc7f97b9130545775860`.
