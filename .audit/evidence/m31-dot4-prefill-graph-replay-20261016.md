# Q8 dot4 prefill graph replay smoke

## Run

- Host: Machinist, GPU2, BDF `0000:88:00.0`
- Source worktree: `/tmp/m31-current-01a910937`, clean at
  `7f4a20998cdc8dd151b34180a03278685c96360e`
- Build commit: `01a91093`
- `llama-bench` SHA-256:
  `2fabfb1c3a355ec18d2faf6cf901cbe487648a2ba705c3378b726710442f556a`
- HIP backend SHA-256:
  `b24d4e14559b39b6ddf6e671410524a738b4bfa61ee5faca90b5b78012d0f872`
- Model SHA-256:
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- ROCm image: `localhost/miinfer-dev:rocm-7.2.1`
- Route: `GGML_CUDA_FA_Q8_DOT4_PREPACK=1`
- Graph diagnostics: `GGML_CUDA_GRAPH_DEBUG=1`
- Workload: `-p 512 -n 0 -r 2 --no-warmup -b 2048 -ub 1024`, Q8_0 K/V,
  Flash Attention, 99 GPU layers
- Remote log: `/tmp/m31-dot4-prefill-graph-replay-20261016T073316Z.log`
- Process exited 0; GPU2 junction temperature was 30 C before and 37 C after.
- No KFD processes remained after the run.

## Evidence

The log contains 256 selections of the opt-in GFX906 dot4 Flash Attention
route and one `CUDA Graph id 21 reused` event. The run had no decode tokens and
used two repetitions of the same prompt shape, so the reused graph was a
prefill graph while the dot4 route was enabled. This closes the earlier gap
where graph reuse had only been observed during a mixed prompt/decode run.

This is a graph-routing smoke, not a correctness or performance qualification.
It does not compare graph output with eager output, test a 64K prompt, or show
that dot4 improves full-model time. Keep the path opt-in.
