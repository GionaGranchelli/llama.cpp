# GQA2 dot4 GPU0 profile attempt

Date: 2026-10-16 (UTC)

## Result

Incomplete. The guarded default-versus-candidate profile runner stopped during
the default 8,192-token prefill when GPU0 reached the 84 C thermal threshold
after 36 seconds. The container exited with status 137, as the runner requested.
The candidate never started and no speed comparison was produced.

## Run setup

- Host: Machinist
- GPU: ROCm index 0, expected BDF `0000:04:00.0`
- Source/build: `/tmp/m31-fork-625c7e9-20261016T074946Z`,
  `build-m31-gqa2`
- Runner: `.audit/run-m31-gqa48-8k-profile.sh`, overridden for the current
  GQA2 build and GPU0
- Run shape: one profiled 8,192-token prefill for default and candidate,
  Q8_0 K/V, batch 2048, microbatch 1024
- Runner checks: GPU BDF, idle KFD, model checksum, build artifacts, junction
  sensor, and a stop at 84 C; temperature polling every 0.2 seconds
- Remote evidence: `/tmp/m31-gqa48-8k-profile-gpu0-20261016T080714Z/`
- GPU0 power cap recorded by the runner: 225 W; GPU2 cap: 300 W

## Follow-up

Use a shorter profiled prefill or lower-load kernel profiling on GPU0. The
unprofiled 8K screen remains the only completed 8K comparison; do not infer a
profiled 8K performance result from this aborted run.
