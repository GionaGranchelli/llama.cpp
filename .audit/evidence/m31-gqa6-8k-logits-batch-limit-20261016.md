# GQA6 8K logits harness batch-limit failure

The first guarded logits-comparison attempt stopped in the default run before
candidate dispatch or GPU inference. `llama-debug` passed a 12,738-token prompt
to `llama_process`, whose single-batch helper was limited to 2,048 tokens and
aborted at token index 2,048:

```text
get_sub_batch: failed to add token 26993 at index 2048 (error -1, n = 12738)
```

This is a harness batch-size mismatch, not a dot4 failure. The prompt file was
created from the tracked attention source and has SHA-256
`024d5e192e4941000daaf216141042a622d9e442721a0ddaed1643f46237b5e2`. Source,
model and container identities are in the preserved remote run directory:
`/tmp/m31-gqa6-8k-logits-gpu2-20261016T085156Z/`. The default process exited
139 after `ggml_abort`; the candidate did not start and no logits were saved.

The runner now sets `-b 16384`, which covers the full prompt, while retaining
`-ub 1024` for bounded compute microbatches. The existing 84 C junction guard
remains active. Retry the same frozen prompt and compare full logits only if
both runs complete and dot4 dispatch is logged.
