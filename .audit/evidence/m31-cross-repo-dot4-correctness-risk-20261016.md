# Cross-repo Q8 dot4 correctness risk

The MIInfer source experiment is relevant prior evidence for this fork's
Q8-query dot4 port, but it does not establish that the llama.cpp port has the
same numerical behavior. The implementations, checkpoints, and dataflows are
not identical.

In MIInfer EXP-0409, a guarded 65,536-token dot4 run completed but its logits
differed from production Q8 attention: max absolute error `0.9121723`, cosine
`0.9935138`, and a different top-1 token. A one-plane query residual run also
failed the full-logit gate: max absolute error `0.934224844`, cosine
`0.992028804163`, though top-1 matched. A later frozen-input run reported a
larger mismatch (max absolute error `28.4721289`, cosine `0.687129296209`,
different top-1); its report identifies Q8 query approximation as a likely,
not proven, cause.

The MIInfer timings are not a valid performance comparison: the first 64K
candidate and production samples used different MI50 cards. The residual run
was slower than the earlier dot4 sample, but that is also a single-run
comparison. Do not treat either timing as a speedup or regression.

The llama.cpp port's existing 2K full-model check (same top-1, cosine
`0.999415556494`) is too short to clear this long-context risk. Keep dot4
opt-in. The next correctness gate should use the same frozen model, tokenized
65,536-token prompt, GPU, build inputs, and production-Q8 reference logits;
compare all output logits and generated tokens before reporting timing. Do not
port the MIInfer residual plane without a separate hypothesis and test because
its recorded 64K result did not pass.

Sources: MIInfer
[`EXP-0409-m31-q8-dot4-query-attention.md`](../../../MIInfer-stage1-rewrite-gpu2/experiments/EXP-0409-m31-q8-dot4-query-attention.md)
and fork
[`m31-dot4-full-model-logits-20261016.md`](m31-dot4-full-model-logits-20261016.md).
