# M31 fork campaign: step 1 baseline screen

Date: 2026-10-16 (Europe/Amsterdam)

## Result

The pinned 8K Q4_K model control completed once on Machinist GPU2, PCI BDF
`0000:88:00.0`: **33.951 s**, or **241.288 prompt tokens/s**. This is one
no-warmup diagnostic sample, not a repeated performance qualification and not
a 64K baseline.

The matching GPU0 attempt was stopped by the thermal guard before producing a
benchmark result. Junction temperature rose from 28 C to 85 C in 36 seconds at
the live 225 W cap. GPU2 began at 31 C, peaked at 78 C, and completed at its
live 300 W cap. Both MI50s were otherwise idle before the runs.

## Pinned settings

- Host: Machinist; 56 CPUs; load average about 1.5.
- ROCm container: `localhost/miinfer-dev:rocm-7.2.1`, image ID
  `50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`.
- Intended runtime source: `56729ae70a66bf075096b0f5b4ae3d0581041b7b`
  (`ggml: drop Q4_K MMQ J48 without a gain`). J40 was unset; default dispatch
  was used.
- Model: `Qwen3.8-27B-Q4_K_M.gguf`, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
- Settings: prompt 8192, generation 0, one run, no warmup, batch 2048,
  microbatch 1024, Q8_0 K/V cache, Flash Attention on, 99 GPU layers.
- Command:

```sh
llama-bench -m /models/Qwen3.8-27B-Q4_K_M.gguf \
  -p 8192 -n 0 -r 1 --no-warmup \
  -b 2048 -ub 1024 -ctk q8_0 -ctv q8_0 -fa on \
  -ngl 99 -dev ROCm0 -o jsonl --progress
```

The two cards were selected separately with `ROCR_VISIBLE_DEVICES=0` and `2`;
inside each isolated container its selected device appeared as `ROCm0`. BDF was
checked before each run.

## Evidence and source provenance

The successful GPU2 JSON result reports `avg_ns=33951171007`,
`avg_ts=241.287701`, one sample, zero generation tokens. Junction telemetry
was sampled once per second: GPU2 `31–78 C`; GPU0 `28–85 C`, with GPU0 stopped
before completion. Raw logs and telemetry are saved beside this report under
`.audit/evidence/m31-step1-baseline-2026-10-16/`.

The executable SHA-256 is
`181bc3c30d5b29fa9db4874272b792a8cbdfa91340e58c46757dbc02c9a5e2b7`. The
isolated exported build tree has no `.git` metadata, and the benchmark JSON
reports `build_commit="unknown"`, `build_number=0`. The campaign build record
attributes its runtime source to the intended `56729ae70` snapshot, but this
run therefore has an explicit binary-to-commit provenance gap. Treat the
result as a useful baseline screen, not a fully source-attested baseline.

## Blocker and next step

The live caps are 225 W on GPU0 and 300 W on GPU2; the earlier requested 180 W
caps are not active. Do not attempt a sustained 64K prefill under these
settings. GPU0 reached the thermal stop during even this 8K screen. The next
step is to apply and verify the 180 W cap on both cards, then create a clean
build whose embedded commit identifies the pinned source before running the
guarded full-context baseline. The GPU2 8K result can serve as an initial
sanity control while that is prepared.

## Exact-source reproduction

The provenance gap above is resolved by a clean checkout and rebuild from
`56729ae70a66bf075096b0f5b4ae3d0581041b7b`. The result JSON reports
`build_commit="56729ae70"` and `build_number=11703`; the source worktree was
clean. On GPU2 (`0000:88:00.0`), the same 8K command completed in `33.910 s`
(`241.580 prompt tokens/s`), starting at 31 C and peaking at 78 C. It used the
same model SHA, image ID, cache types, batch sizes, and default attention
dispatch. This reproduces the earlier diagnostic timing with source identity
now verified; it remains a one-sample 8K control and does not establish a
64K timing or repeated-run range.

The exact build binary SHA-256 is
`53a069db463e3bb728e30411b6190384da6fe0b3669b2eadfaa37aff9e90de11`. Raw
metadata, JSON output, and junction telemetry are saved under
`.audit/evidence/m31-step1-baseline-2026-10-16/` with the `gpu2-exact-` prefix.
