# Q8 dot4 operator check on both MI50 cards

Date: 2026-10-16 09:57 UTC  
Candidate source: `274649a6a`  
Image: `localhost/miinfer-dev:rocm-7.2.1`

## Result

The focused Q8 dot4 attention operation passed on both MI50 cards. Each GPU
reported `1/1 tests passed`, and the logs showed the opt-in Q8 dot4 route was
selected. The runner launched the two checks in parallel.

| ROCm device | PCI address | Result |
| --- | --- | --- |
| `ROCR_VISIBLE_DEVICES=0` | `0000:04:00.0` | 1/1 passed |
| `ROCR_VISIBLE_DEVICES=2` | `0000:88:00.0` | 1/1 passed |

The tested case used Q8 keys and values, GQA ratio 6, head width 256, 64 query
rows, and 1,024 KV rows. The prepack feature flag was enabled for both runs.

## Limits

This proves the focused operation works for this shape on both cards. It does
not prove full-model accuracy, 8K prefill speed, or 64K performance. The 8K
comparison did not start in this run because its script depended on `rg`, which
is absent on the host. The runner was changed to use `grep` and that fix was
pushed separately.

The remote raw output and temperature samples are at
`/tmp/m31-q8-dot4-lds-20261016T095751Z` on Machinist; they have not yet been
retrieved into this repository.
