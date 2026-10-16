# Q8 GQA6 48-cell operator check

## Result

The narrowed `FLASH_ATTN_EXT` filter selected one Q8 GQA6 fixture in each
parallel run. Both outputs reported `1/1 tests passed`, and both logged the
opt-in GFX906 Q8 dot4 path, including the prepacked-Q GQA path. This verifies
the focused operator against the backend reference on the two selected ROCm
devices. It does not verify full-model logits, graph replay, or performance.

## Run details

- UTC run directory on Machinist: `/tmp/m31-q8-dot4-lds-20261016T044406Z`
- Source commit: `176480a57a5b18be94efdc7f97b9130545775860`
- Test binary SHA-256: `cd7a7613c1a054e9176ff03b1f6ee68ca724a902695912cb17d57aff62036dd4`
- Container image: pinned `localhost/miinfer-dev:rocm-7.2.1`
- Filter: `hsk=256,hsv=256,nh=4,nr23=\[6,1\],kv=1024,nb=64.*type_K=q8_0,type_V=q8_0`
- GPU0 preflight: BDF `0000:04:00.0`, junction 28 C, cap 225 W
- GPU2 preflight: BDF `0000:88:00.0`, junction 31 C, cap 300 W
- KFD processes before launch: none

The first process reported `AMD Instinct MI60 / MI50`; the second reported
`AMD Radeon Graphics`, although the host runner verified ROCm device index 2
mapped to BDF `0000:88:00.0`. Keep the second-card model attribution provisional
until the saved host metadata is retrieved and checked. Both devices reported
`gfx906` and 32,752 MiB VRAM in the test output.

## Interpretation and next step

The larger CTA now has a real focused correctness result on both visible
GFX906 devices. The dot4 candidate's prior 8K full-model sample was slower than
the default path, so this is only a kernel correctness gate, not evidence of a
speedup. Next, confirm the saved device metadata, then run a matched full-model
comparison if the card mapping and runtime path are confirmed. The 64K A/B
still requires both cards to be capped at or below 180 W.
