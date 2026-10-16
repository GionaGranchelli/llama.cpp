# Q8 dot4 prepack: multi-query correctness check

Date: 2026-10-16 (Europe/Amsterdam)

## Change under check

This Q8 dot4 path adapts the integer dot-product dataflow from
`MIInfer-stage1-rewrite-gpu2/gfx906/kernels/qwen3_primitives.hip` in the
MIInfer repository. MIInfer is distributed under the MIT License, Copyright
(c) 2026 Constant Labs. The implementation here uses llama.cpp's existing
`FLASH_ATTN_EXT` buffers, dispatch and graph path; it does not copy the MIInfer
runtime or launcher.

`fattn-tile.cuh` now stages each CTA's prepacked Q rows into shared memory once,
then reuses them through all KV tile iterations. During staging, flattened local
query index `jc = j * ncols2 + c` is remapped to the global packed row
`(sequence * query_tokens + query_token) * query_heads + head0 + c`.
This preserves the token/head layout for the `ncols1=4`, `ncols2=6`,
24-query-head GQA6 case and avoids repeated global Q loads inside the KV loop.

## Evidence

- Physical GPU: Machinist GPU0, BDF `0000:04:00.0`, gfx906.
- ROCm 7.2.1 container: `localhost/miinfer-dev:rocm-7.2.1`.
- Focused filter: `hsk=256.*kv=1024.*nb=64`.
- Default dispatch: 4/4 selected operator tests passed.
- Prepack enabled: 4/4 selected operator tests passed, with the GQA6 Q8 case
  logging selection of the opt-in dot4 path.
- Junction temperature: 28 C before and after.
- Test executable SHA-256:
  `ebeeca504fe1bf3285a9d1269d35173a64d6fa79fcb4733ea2316f84396b450c`.
- `fattn-tile.cuh` SHA-256:
  `2c9489c282e3ca3e2d329b30d9ed58b4a1cd7228720adafa0083a649efd4b279`.

Raw logs, metadata and temperature readings are in
`m31-prepack-batched-q8-20261016/`.

The prepack-enabled filter was also repeated on GPU2, BDF `0000:88:00.0`.
It passed 4/4 with the GQA6 Q8 candidate selected; junction temperature was
31 C before and after. Its raw log and telemetry are in
`m31-prepack-batched-q8-gpu2-20261016/`.

## Limits

This establishes operator-reference agreement for the batched GQA6 Q8 shape.
Full-model logits, graph replay and performance after LDS staging remain
unverified. The earlier 8K prepack timing measured the version that reread Q
from global memory during KV iteration; it does not measure this change.
