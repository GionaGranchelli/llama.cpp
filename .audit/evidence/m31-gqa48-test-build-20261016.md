# GQA6 48-cell operator test build

## Result

Built `test-backend-ops` on Machinist from the pinned ROCm 7.2.1 image for the
GFX906 candidate worktree. The build passed. This does not yet establish that
the focused operator test passes; no GPU test was run in this pass.

## Source and build

- Host: Machinist
- Source worktree: `/tmp/m31-gqa48-rocm721-20261016`
- Source commit verified on host before build: `176480a57a5b18be94efdc7f97b9130545775860`
- Build directory: `/tmp/m31-gqa48-rocm721-20261016/build-gqa48`
- Container image ID: `50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`
- Configure: `GGML_HIP=ON`, `AMDGPU_TARGETS=gfx906`, `GGML_BUILD_TESTS=OFF`,
  `LLAMA_BUILD_TESTS=ON`, `LLAMA_BUILD_EXAMPLES=OFF`
- Target: `test-backend-ops`
- Result: `[100%] Built target test-backend-ops`

The first configure attempt used `GGML_BUILD_TESTS=ON` and failed because this
checkout does not contain `ggml/tests`. Leaving that option off while enabling
`LLAMA_BUILD_TESTS` configured successfully and built the target.

## Focused fixture and runner update

The intended eval fixture is the Q8_0 GQA6 case with head sizes 256, `nr23=[6,1]`,
KV length 1024, and query batch 64 (`tests/test-backend-ops.cpp`, around line
11730 in this source revision). The runner now filters those fields and accepts
`M31_BUILD_DIR`, so it can use the `build-gqa48` tree. `bash -n` passes.

The filter's actual selected-case count still needs confirmation from the test
binary before reporting operator correctness.

## Device state at preflight

- GPU0: BDF `0000:04:00.0`, junction 28 C, cap 225 W
- GPU2: BDF `0000:88:00.0`, junction 30 C, cap 300 W
- KFD processes: none

Both cards were idle. The guarded 64K A/B remains ineligible until both caps are
at or below 180 W. This pass stopped after building the operator test target.
