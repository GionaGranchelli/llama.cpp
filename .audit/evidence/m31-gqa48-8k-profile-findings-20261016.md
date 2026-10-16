# GQA48 8K profile findings

The complete profiler statistics identify the candidate attention kernel as the
main source of its 8K prefill regression.

| Path / kernel | Calls | Summed duration | Share of profiled kernel time |
| --- | ---: | ---: | ---: |
| Default Q4_K MMQ (`ggml_type 12`) | 2,288 | 21.148 s | 63.22% |
| Default Q6_K MMQ (`ggml_type 14`) | 504 | 5.051 s | 15.10% |
| Default Q5_K MMQ (`ggml_type 13`) | 384 | 1.964 s | 5.87% |
| Default attention `flash_attn_tile<256,256,16,2,...>` | 128 | 1.576 s | 4.71% |
| Candidate Q4_K MMQ (`ggml_type 12`) | 2,288 | 21.164 s | 58.02% |
| Candidate Q6_K MMQ (`ggml_type 14`) | 504 | 5.050 s | 13.85% |
| Candidate attention `flash_attn_tile<256,256,8,6,...>` | 128 | 4.553 s | 12.48% |

Over the same 128 launches, dot4 attention takes 2.977 s more than default,
or 2.89x the default time. Q4_K MMQ changes by only 0.016 s. The profiled
llama-bench timings differ by 3.011 s (33.939 s default, 36.949 s candidate),
close to the attention delta. The separate quiet screen differs by 3.154 s
(33.880 s vs 37.034 s), one sample per side. It is directional rather than a
qualified timing, but both runs show the same slowdown.

The profiler's end-to-end numbers include profiler overhead and are not
standalone performance claims. Use the CSV to diagnose kernel costs, not to
claim a user-visible speedup.

## Source-level interpretation

The candidate GFX906 tile schedules 48 query/head cells with 384 threads per
CTA; default schedules 32 cells with 256 threads. Both use occupancy target 2
in the GFX906 configuration table. The candidate processes 50% more cells but
also launches CTAs with 50% more threads, while packing Q and transposing Q8 K
into LDS. This identifies the schedule and data movement as the next area to
change, but does not isolate the cost of each component.

The earlier 24-cell dot4 tile was also slower in the 2K profile. Enlarging it
to 48 cells therefore did not solve the performance problem. Do not promote
this candidate or repeat the same short-context timing. The next meaningful
GPU measurement is a matched 64K screen only if both cards pass the current
180 W power guard; otherwise continue source-level work without sustained GPU
inference.

Run: `/tmp/m31-gqa48-8k-profile-gpu2-20261016T050033Z` on Machinist.
Source commit: `176480a57a5b18be94efdc7f97b9130545775860`.
