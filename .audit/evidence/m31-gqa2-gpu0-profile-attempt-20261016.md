# GQA2 dot4 GPU0 profile attempt

Date: 2026-10-16 (UTC)

## Result

Incomplete. The guarded default-versus-candidate profile runner exited with
status 1 before emitting a paired result. Its live output showed GPU0 junction
temperature at 27 C at the start and 79 C after 30 seconds. The captured SSH
output did not include the runner's stop reason or profiler summary, so the
exit cause and final temperature are unknown. No speed comparison can be
drawn.

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

## Follow-up

Retrieve the runner's remote `/tmp/m31-gqa48-8k-profile-gpu0-*` directory to
identify the exit cause before another profiled full-model run. If the stop was
thermal, profile a shorter prefill or use lower-load kernel-level profiling;
do not infer a performance result from this attempt.
