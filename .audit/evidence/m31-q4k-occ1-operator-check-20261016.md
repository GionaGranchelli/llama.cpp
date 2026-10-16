# GFX906 Q4_K occupancy-one MMQ operator check

Date: 2026-10-16 (UTC)

## Result

The opt-in Q4_K MMQ occupancy-one configuration built and passed the focused
`MUL_MAT` backend operator filter on both Machinist MI50 cards:

| ROCm index | PCI BDF | Result |
| --- | --- | --- |
| 0 | `0000:04:00.0` | 120/120 passed |
| 2 | `0000:88:00.0` | 120/120 passed |

The CMake cache had `GGML_CUDA_MMQ_Q4K_OCCUPANCY_1=ON`, `GGML_HIP=ON`,
`GGML_BUILD_TESTS=OFF`, and `LLAMA_BUILD_TESTS=ON`. The build targeted
`gfx906`. The `test-backend-ops` binary SHA-256 was
`b3aaa9443a82e0bdc5c15675861f900916c4eb5954a41cf57a4c0ff23965d4a4`.

Each card ran:

```sh
ROCR_VISIBLE_DEVICES=<0-or-2> \
LD_LIBRARY_PATH="$PWD/build-occ1/bin:${LD_LIBRARY_PATH:-}" \
./build-occ1/bin/test-backend-ops \
  -b ROCm0 -o MUL_MAT -p type_a=q4_K
```

Both processes exited 0. ROCm reported 28 C junction on GPU0 and 31 C on GPU2
after the checks. There were no KFD processes before launch.

## Scope

This validates the opt-in Q4_K MMQ operator path and its CPU-reference
comparisons on both cards. It does not establish full-model logits, prefill
speed, decode speed, or long-context behavior. The production default remains
unchanged; an 8K guarded performance comparison is the next useful gate.

## Build identity

- Fork commit: `3310e82ebaa0280ceaf8d1f37dad5afdbef02f33`
- Build: Release, ROCm 7.2.1 container, HIP `gfx906`
- CMake option: `GGML_CUDA_MMQ_Q4K_OCCUPANCY_1=ON`
