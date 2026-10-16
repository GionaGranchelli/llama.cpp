# GQA2 Q8 dot4 runtime check

## Build and test

- Source: `01a91093734977e04de98ebd48ac2fe169b47a63`
- Host: Machinist, ROCm 7.2.1 image ID
  `50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`
- GPU: ROCm index 2, BDF `0000:88:00.0`, GFX906
- Model SHA-256:
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Built `test-backend-ops` and `llama-bench` from a fresh clone of the pushed
  branch. HIP emitted loop-unroll warnings while compiling Flash Attention
  templates, but the build completed.
- With `GGML_CUDA_FA_Q8_DOT4_PREPACK=1`, the focused Q8_0 GQA6
  `FLASH_ATTN_EXT` fixture (`hsk/hsv=256`, `kv=1024`, `nb=64`) printed the
  expected dot4 dispatch and `1/1 tests passed` on ROCm0.

## Incomplete timing attempt

The remote wrapper exited with status 1 after the operator output and did not
return its post-test status, benchmark output, or evidence-directory path. No
8K A/B timing is claimed from this attempt. The temporary clone and run files
are on Machinist under `/tmp/m31-current-01a910937` and
`/tmp/m31-current-8k-gqa2-*`; inspect the newest run directory before retrying.

The live preflight before the build found no active KFD processes. At that
point GPU0/GPU2 junction temperatures were 30/27 C and their power caps were
225/300 W. This was only an 8K diagnostic attempt; no 64K run was started.
