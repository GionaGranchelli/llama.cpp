# M31 Q8 dot4 prepack operator screen

Date: 2026-10-16 (Europe/Amsterdam)

## Result

The separate-query-prepack variant builds for gfx906 and passes two focused
`FLASH_ATTN_EXT` cases with prepacking both enabled and disabled. Machinist
GPU0 was used: live ROCm index 0 maps to PCI BDF `0000:04:00.0`. Junction
temperature was 28 C before and after the short operator checks.

This proves focused operator/reference parity for these two shapes only. It
does not prove full-model logits, graph replay, decode correctness, or a
performance improvement.

| Case | Prepack disabled | Prepack enabled |
| --- | --- | --- |
| D=256, GQA=6, KV=512, nb=4, Q8_0 K/V | 1/1 passed | 1/1 passed |
| D=256, GQA=6, KV=1024, nb=4, Q8_0 K/V | 1/1 passed | 1/1 passed |

The test filter runs the ROCm backend and skips the CPU backend. The operator
test compares the ROCm result with its reference and reports `OK`.

## Build and source

- Worktree: `/tmp/llama.cpp-miinfer-dot4-prepack`
- Branch: `codex/m31-q8-dot4-prepack`
- Base: `e57aad81ae1fc5c8fa8d71a680a025be6e64b992`
- Container: `localhost/miinfer-dev:rocm-7.2.1`, image ID
  `50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`
- Build target: `test-backend-ops`; build exit 0
- Build binary SHA-256:
  `ebeeca504fe1bf3285a9d1269d35173a64d6fa79fcb4733ea2316f84396b450c`

The separate query pack uses 288 bytes per 256-wide query row: 256 signed
bytes and eight FP32 scales, preserving llama.cpp Q8_0's 32-value scale
granularity. It writes to the existing `FLASH_ATTN_EXT` scratch allocation
and launches on the operation's main stream.

The on/off tests used source hash `f29627aedd3ec139c2bbb5d46ae9a6a07d27e63c8efabb5c1f91934a3bbde65c`
for `fattn.cu`. The route label was corrected after those tests; the final
hash is `ad052316ac069a0f685e300a23cf3978c6a4a08fac2d37b81cfc4c122f89062d`.
That text-only edit passed `git diff --check` but was not rebuilt or rerun.

## Raw evidence

- Prepack-off logs and metadata:
  `m31-dot4-prepack-gpu0-20261016/m31-dot4-prepack-gpu0-20261016T024110Z/`
- Prepack-on logs and metadata:
  `m31-dot4-prepack-gpu0-20261016/m31-dot4-prepack-gpu0-on-20261016T024156Z/`

Both directories contain exact commands, exit codes, output logs, source and
binary checksums, image ID, and telemetry. The candidate logs captured before
the route-label correction identify the selected dot4 path; the later generic
direct-Q8 message was misleading and has been corrected in source.

## Next gate

Integrate this candidate into the main checkout, rebuild the full-model tools,
then run one guarded 8K model screen with correctness output compared against
the unchanged path. Do not claim a prefill win from the operator results. The
64K baseline still requires both Machinist MI50 power caps at or below 180 W;
the live caps remain 225 W and 300 W.
