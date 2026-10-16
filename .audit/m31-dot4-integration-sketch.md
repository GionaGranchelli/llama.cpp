# M31 Q8 dot4 attention port — design workplan

## Goal

Port the MIInfer gfx906 Q8 dot4 attention candidate into llama.cpp through the
existing `FLASH_ATTN_EXT` operation if it can be made correct and faster. Add a
new ggml operation only if the existing inputs and output contract cannot
express the candidate. Keep all candidate dispatch opt-in until operator,
graph, and full-model correctness pass.

## Grounded constraints

- Baseline source is `56729ae70a66bf075096b0f5b4ae3d0581041b7b`; one
source-attested 8K control completed on GPU2 in 33.910 s.
- The Q8 dot4 implementation in MIInfer quantizes Q into signed-byte planes and
uses `v_dot4_i32_i8` for QK. Its four-plane full-model screen still failed the
logit gate (max error 1.09448, cosine 0.9989098). Do not transplant it
unchanged or assume query quantization alone explains the failure.
- llama.cpp Q8_0 stores 32 signed values plus one FP16 scale per block. A
256-wide key/value vector therefore has eight independent scales. MIInfer's
Q8 cache uses one scale per 256-wide vector; the port must preserve the eight
llama.cpp scales in the QK result.
- The existing `GGML_CUDA_FA_Q8_DIRECT_TILE` route is already an
`FLASH_ATTN_EXT` specialization for GFX906, Q8_0 K/V, D=256, GQA=6, and
multi-token prefill. It converts only current K/V tiles, not the full cache,
and was slower than default at 8K. The dot4 candidate must avoid repeating
that conversion cost.
- GPU2's live cap is 300 W and GPU0's is 225 W. No inference service is
currently active; sudo requires the user's password. Sustained 64K runs wait
until both caps are verified at 180 W.

## Distinct shapes to compare

1. Quantize each query tile into shared-memory Q8 planes inside the existing
   attention CTA. This keeps one kernel launch and avoids global scratch, but
   repeats query conversion for every KV split and increases LDS use.
2. Launch one query-quantization kernel from the existing attention backend
   wrapper, write Q8 planes and scales to pool scratch, then consume them in
   the existing flash-attention kernel. This pays one launch and scratch cost,
   but computes each Q plane once and keeps the public ggml op unchanged.

## Current implementation choice

Use shape 2: quantize each Q row once per `FLASH_ATTN_EXT` call, then stage the
packed rows into shared memory once per attention CTA. The public
`FLASH_ATTN_EXT` operation remains unchanged. This adds one prepack launch and
scratch allocation while avoiding repeated global reads inside the KV loop.

## Rubric

- Preserves the existing `FLASH_ATTN_EXT` graph and tensor contracts; no new
  ggml op without evidence it is required.
- Correctly handles all eight Q8_0 K block scales and causal/split-K bounds.
- Keeps candidate state and dispatch opt-in; default route remains unchanged.
- Provides a real correctness path from operator output through graph replay
  to full-model logits/tokens.
- Has a plausible measured speed path over current Q8_0 conversion, without
  accepting speed that comes from reduced correctness.
- Keeps attribution to MIInfer's implementation and its license intact.

## Phases

- [x] Ground: inspect host dispatch, tile kernel, Q8_0 layout, and MIInfer
      candidate failure evidence.
- [x] Sketch: compare the in-CTA pack with a separate Q-quantization launch.
- [x] Choose: use a separate Q prepack followed by per-CTA shared-memory
      staging.
- [x] Implement: add an opt-in path inside the existing attention op; the
      Machinist ROCm 7.2.1 build succeeds.
- [ ] Verify: exact operator reference, graph replay, and full-model output;
      measure only after correctness passes.
- [ ] Scrap: remove a candidate if it fails correctness or has no plausible
      path to outperform the control.

## Initial verification evidence (superseded below)

- The focused Q8_0 GQA6 case is in the correctness test factory and the ROCm
  support probe reports it as supported.
- An initial Machinist build used a stale `fattn-tile.cuh` snapshot (hash
  `8bf05606...`), so its run is discarded. After syncing the current checkout
  (hash `760d0861...`), the ROCm 7.2.1 build of `test-backend-ops` succeeds.
- The exact-source support probe reports the focused case as supported. The
  exact-source candidate run logs the opt-in selection, then exits with
  Machinist's `Memory critical error ... Reason: Memory in use`.
- The unchanged Q8 path fails the same case on both MI50s, including with
  `HSA_ENABLE_SDMA=0`. The candidate kernel has not produced a numerical
  result. The SSH account cannot read kernel logs, so the driver-side cause is
  not yet identified. See `.audit/evidence/m31-dot4-smoke-20261016.md`.

## Historical recheck: operator passes, full-model screen regresses

The earlier runtime failure summary is superseded by an exact-source rerun.
On GPU2, the focused dot4 case passed 1/1 against the CPU reference, and the
Q8/Q8 `FLASH_ATTN_EXT` filter passed 370/370. A verbose 512-token run of the
target model confirmed the opt-in dot4 route was selected.

The matched 8K full-model screen measured 33.879 s default versus 40.053 s
with dot4 enabled (one sample per path), while both remained below the 80 C
warning threshold. The current shape packs Q inside each attention CTA; that
work repeats for each KV split. The separately packed Q workspace shape in
the design sketch is now the only plausible next variant. Keep the current
candidate opt-in. If a once-per-query pack cannot remove the measured penalty,
drop the dot4 port instead of tuning the current shape further.

Raw recheck evidence is in `.audit/evidence/m31-dot4-smoke-20261016/evidence.tar.gz`.

## Current status: prepack rows staged in shared memory

The first prepack 8K result measured the version that reread Q from global
memory inside every KV tile iteration. The updated kernel stages each packed
row in shared memory once per CTA. The copy remaps a four-query, six-head group
from global token/head order into CTA-local `jc` order.

On GPU0, default and opt-in tests passed for the `hsk=256, kv=1024, nb=64`
filter, including the GQA6 Q8 case that launches four query positions per CTA.
The opt-in test also passed on GPU2. The previous 8K timing does not measure
this source. Graph replay, full-model correctness and current full-model
performance remain unverified.
Raw operator logs are in `.audit/evidence/m31-prepack-batched-q8-20261016/`.

## Follow-up screen: block-scale integer accumulation

The staged-Q8 kernel originally converted each four-byte dot4 result to FP32,
multiplied by the same Q8_0 query and key scales, and accumulated it eight times
per 32-value block. The revised loop sums the eight exact integer dot4 results
in int32 and applies the shared scales once per block. The Q8_0 per-block scale
layout and the 8-block dimension split are unchanged.

The revised source compiled for gfx906. The focused `FLASH_ATTN_EXT` operator
filter passed 4/4 cases on each MI50, including the GQA6 Q8 case. A matched
single-sample 8K screen on GPU2 measured 33.921 s default and 36.579 s with
prepack enabled. This is 7.8% slower than default, although 2.2% faster than
the earlier 37.402 s prepack screen. Both prepack samples completed below the
84 C hard stop; the latest candidate peaked at 79 C. These screens do not
establish full-model numerical agreement or long-context performance.

A deterministic CLI comparison generated 64 tokens on the default path. The
prepack path reached the 84 C stop during its run, so it did not produce a
valid complete output comparison. Raw logs and temperature traces are under
`.audit/evidence/m31-dot4-lds-cli-20261016T034858Z/`.

Decision: the block accumulator is a real but insufficient improvement. Keep
it opt-in while checking longer-context behavior under the requested 180 W
caps; do not promote it based on these single-sample 8K timings. The live caps
at this checkpoint remain GPU0 225 W and GPU2 300 W.
