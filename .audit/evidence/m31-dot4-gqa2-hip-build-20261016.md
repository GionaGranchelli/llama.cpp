# Q8 dot4 GQA2 HIP build check

## Result

The current fork source builds the complete HIP backend for GFX906 with the
local ROCm 6.2 toolchain. This confirms the pushed GQA2 schedule change
compiles. It does not verify runtime correctness or performance on the
Machinist's pinned ROCm 7.2.1 environment.

## Provenance

- Fork worktree: `/home/gionag/Development/llama.cpp-miinfer`
- Source commit: `4e261f3fb`
- Schedule change: `ca1f121192f14d8a8f4467bb0ade9f2fa0712172`
- Local toolchain: ROCm 6.2.41133, AMD clang 18
- Target: `gfx906`
- Build directory: `/tmp/m31-fork-gqa2-build`
- Target: `ggml-hip`
- Configure: `GGML_HIP=ON`, `AMDGPU_TARGETS=gfx906`, Release,
  llama tests/examples disabled
- Build result: exit 0, `[100%] Built target ggml-hip`
- Output: `/tmp/m31-fork-gqa2-build/bin/libggml-hip.so`
- Output SHA-256: `c4b4711cdeb27dae6edd821c648204631d10c14e228dc0268e4cf197596c91ee`
- Source `fattn-tile.cuh` SHA-256:
  `db19b30d9b67e1def42975d16a8315f5d7e9f434b5f63b7d8aa81dbd9740cf8d`

The local linker requires `libxml2.so.2`, which is absent on this host. The
build used the existing private compatibility symlink at
`/tmp/m31-libxml-compat/libxml2.so.2` to the installed `libxml2.so.16`.
Linker version-metadata warnings were emitted. No source or system library
files were changed.

## Next gate

Rebuild the exact source in the pinned ROCm 7.2.1 Machinist environment, then
run the matched 8K GQA2 dot4 versus default screen and a focused operator
correctness case. Do not infer a speedup from this compile check.
