# Q8 dot4 prepack screen on Machinist

## Verdict

Inconclusive performance screen; the latest prepack sample was slower than the
unchanged dispatch. The block-level integer accumulation reduced prepack time
from 37.402 s to 36.579 s, but the latest default sample was 33.921 s. These
are single samples, with different start temperatures. No performance default
change is justified.

## Setup

- Host: Machinist
- GPU: GPU2, PCI BDF `0000:88:00.0`
- Model: `Qwen3.8-27B-Q4_K_M.gguf`, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Source base: `09bca255faf238de2398344acf7322b02380099b`
- Prompt: 8,192 tokens; generation disabled
- Settings: Q8_0 K/V, Flash Attention, 99 GPU layers, `-b 2048 -ub 1024`,
  no warmup, one sample
- GPU2 power cap at measurement: 300 W

## Results

| Source state | Default | Q8 dot4 prepack | Start junction (default / prepack) | Peak junction |
| --- | ---: | ---: | ---: | ---: |
| LDS-staged K, before block accumulation | 33.904 s (241.622 tok/s) | 37.402 s (219.027 tok/s) | 31 / 37 °C | 78 / 79 °C |
| LDS-staged K, block-level int32 accumulation | 33.921 s (241.499 tok/s) | 36.579 s (223.954 tok/s) | 31 / 38 °C | 78 / 79 °C |

The latest prepack path was 7.8% slower by wall time than its same-build
default sample. It improved 2.2% over the preceding prepack sample. Each cell
is one run; these results do not establish a repeatable difference. The
load average was not recorded, and the limiting component has not been
profiled at 8K. A separate 2K profile found the dot4 attention kernel slower
than default at that shape, but it does not quantify its share at 8K. Host
contention and other effects are therefore not ruled out as causes of the
8K wall-time difference.

`rocprofv3` did not reach the workload: it could not load `libdw.so.1` in the
container, and a host-library mount made it exit with SIGSEGV. The legacy
`rocprof --stats --timestamp on` path did run successfully for the 2K
diagnostic described in `.audit/evidence/m31-q8-dot4-profile-20261016.md`.

The latest source hash is
`b67463b1660df7cfedfbfb9339afc173fa89b6a8792c6c481a625734914ac31e`; benchmark
binary hash is
`181bc3c30d5b29fa9db4874272b792a8cbdfa91340e58c46757dbc02c9a5e2b7`, and
`libggml-hip.so` hash is
`3dcc675ca438209555f5e37c4c055f249865753ac8d13a1c092f5c15e16e737d`. The
benchmark JSON reported `build_commit: unknown`; source, binary, and library
hashes are recorded directly in the raw metadata.

## Correctness and thermal evidence

- The focused Q8 `FLASH_ATTN_EXT` filter passed 4/4 cases on each MI50 with
  prepack enabled, including GQA6. See
  `.audit/evidence/m31-q8-dot4-lds-20261016T035835Z/`.
- The prepack dispatch was explicitly logged in a prior full-model run with
  the same dispatch and LDS staging, before block accumulation. The latest
  performance runner set the prepack environment flag but did not enable
  verbose route logging.
- A default 64-token CLI response completed. The prepack CLI attempt was
  terminated by the 84 °C thermal guard and did not yield a complete response;
  full-model token/logit agreement is therefore unverified. See
  `.audit/evidence/m31-dot4-lds-cli-20261016T034858Z/`.
- Current Machinist caps were GPU0 225 W and GPU2 300 W, not the requested
  180 W. The 8K samples stayed below the 84 °C stop, but long-context runs
  remain gated on verifying the requested caps.

Raw timing logs and temperature traces are in the adjacent run directories:

- `m31-dot4-prefill-8192-default-20261016T034621Z/`
- `m31-dot4-prefill-8192-dot4-20261016T034727Z/`
- `m31-dot4-prefill-8192-default-20261016T035853Z/`
- `m31-dot4-prefill-8192-dot4-20261016T035952Z/`
