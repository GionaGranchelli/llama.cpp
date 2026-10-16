# M31 fork campaign: pushed Q8 dot4 build and operator verification

Date: 2026-10-16 (Europe/Amsterdam)

## Result

The pushed Q8 dot4 prepack commit builds for gfx906 and passes the focused
GQA6 Q8 `FLASH_ATTN_EXT` operator filter on both MI50s. Default dispatch and
the opt-in candidate each pass 4/4 selected cases on GPU0. The opt-in
candidate passes 4/4 on GPU2. The GQA6 Q8 case logs the candidate dispatch.

| Host GPU | PCI BDF | Route | Result |
| --- | --- | --- | --- |
| Machinist GPU0 | `0000:04:00.0` | Default | 4/4 passed |
| Machinist GPU0 | `0000:04:00.0` | Q8 dot4 prepack | 4/4 passed |
| Machinist GPU2 | `0000:88:00.0` | Q8 dot4 prepack | 4/4 passed |

The GPU2 container reports a generic Radeon name, but host-side ROCm bus
inspection maps its selected ROCm index to `0000:88:00.0`, the second gfx906
MI50. Both cards were at 28-31 C after these short operator checks.

## Build identity

- Fork branch: `perf/m31-q8-dot4-prepack-stage`
- Commit: `09bca255faf238de2398344acf7322b02380099b`
- Embedded llama.cpp build: `0.7.0-dev`, build `11708`, commit `09bca255f`
- Container: `localhost/miinfer-dev:rocm-7.2.1`, image ID
  `50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`, digest
  `sha256:447907c697fb2c39ce9657401e354a70753d5672ad955d4ceb12de976b91224a`
- `llama-bench` SHA-256:
  `b90475a7fe602362a75a644cb71ac980aa3bec19d49e315cf3662db01c5fd473`
- `test-backend-ops` SHA-256:
  `ebeeca504fe1bf3285a9d1269d35173a64d6fa79fcb4733ea2316f84396b450c`
- Filter: `-b ROCm0 -o FLASH_ATTN_EXT -p 'hsk=256.*kv=1024.*nb=64'`

Raw command output, bus mapping, temperatures, hashes and build identity are
in `m31-fork-step2-ops-20261016/`.

## Limits and next gate

The 8K candidate timing, graph replay, full-model numerical agreement, decode
and 64K prefill are not verified by this operator screen. The guarded
full-model runner still requires both MI50 power caps at or below 180 W; the
live caps were 225 W on GPU0 and 300 W on GPU2 at the time of the check.
