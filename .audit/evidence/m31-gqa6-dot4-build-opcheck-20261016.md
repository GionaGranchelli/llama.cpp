# Q8 dot4 GQA6 build and operator check

Date: 2026-10-16 (UTC)

## Result

The six-head opt-in Q8 dot4 tile builds with the pinned ROCm 7.2.1 environment
and passes the focused `FLASH_ATTN_EXT` Q8 K/V fixture on Machinist GPU0. The
runtime dispatch log confirms the opt-in route was selected. This validates
compilation and this operator fixture only; it does not establish full-model
correctness or a performance gain.

## Provenance

- Host: Machinist
- GPU: ROCm index 0, PCI BDF `0000:04:00.0`
- Source commit: `9d2493b82d68792100ccb4ed00f20db68a84de93`
- Container image ID: `50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`
- Source SHA-256 (`fattn-tile.cuh`):
  `b143f82b757112eac70b6b80f3644f6c3c968a693e8b5ce19ad53ab52b7ff3ac`
- `test-backend-ops` SHA-256:
  `8c9f6a7c11fdaafd6da9b92308851977823a3f913ca4e53ec73425f361b5b30a`
- `llama-bench` SHA-256:
  `1eb91cc09d67ea57fcbffe6a59aab02acb5111a5fce087108ad5d0d33c5c3198`
- HIP backend SHA-256:
  `c0739dae80a2d6f90c4ec78db2355efd0a8127bde0045b29bc7f7b852ee96633`
- Test: `test-backend-ops -b ROCm0 -o FLASH_ATTN_EXT` with
  `hsk=256,hsv=256,nh=4,nr23=[6,1],kv=1024,nb=64.*type_K=q8_0,type_V=q8_0`,
  `ROCR_VISIBLE_DEVICES=0`, and `GGML_CUDA_FA_Q8_DOT4_PREPACK=1`.
- Outcome: focused test passed; CPU backend skipped; 2/2 backend checks passed.
- Build emitted HIP loop-unroll warnings but completed successfully.

## Next decision

Run one guarded, matched full-model screen against the default route. Keep the
candidate opt-in unless both full-model correctness and performance support
promotion. The previous GQA2 dot4 screen was slower, but it does not predict
the result of this six-head tile.
