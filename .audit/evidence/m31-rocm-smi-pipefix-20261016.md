# Fix guarded runner failures from early-closing pipelines

## Finding

The recent Machinist script exited with status 120 before cloning the fork
source. Under `set -o pipefail`, parsing `rocm-smi` with `awk ... { exit }`
caused the `rocm-smi` Python process to write to a closed pipe. The same risk
existed in the KFD check using `grep -q` and in the older dot4 prepack runner's
temperature polling.

## Change

Both guarded 64K runners now capture the complete `rocm-smi` output before
parsing it. This lets the producer finish and keeps `pipefail` enabled. The
KFD failure path also prints the captured diagnostic.

## Verification

- `bash -n` passed for both changed runners.
- A repository scan found no remaining `.audit` pipelines that parse
  `rocm-smi` with an early-exiting `grep -q` or `awk ... exit`.
- The latest failed Machinist attempt left an empty run directory and no build
  log; no pinned build or inference ran.

## Next gate

Retry the current fork commit on Machinist with output redirected to files,
then run the matched 8K GQA2 schedule comparison. The 64K runner still refuses
to start while either MI50 power cap exceeds 180 W.
