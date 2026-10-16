# GQA6 per-four-scale 24 KB prompt thermal abort

After the 1,982-token comparison completed below the guard, a bounded larger
prompt retry used a 24,000-byte prefix from the same source-context prompt on
GPU0. The default run started at 29 C and reached the 84 C warning threshold;
the runner stopped it before 85 C. The candidate did not start, so no token
count, logits, or comparison result was produced for this attempt.

## Provenance

- Source/runner commit: `15e75a8945b9b900165b514c86b1417cfc3aa3b0`
- Model SHA-256:
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- ROCm image ID:
  `50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`
- GPU0 BDF: `0000:04:00.0`
- Power caps: GPU0 225 W, GPU2 225 W
- Prompt: 24,000 bytes, SHA-256
  `787be0a668aa37b10ef63d7199a17a1fdb40c0321e26b566260237fd7f6f6591`
- Start junction: 29 C; stop at 84 C; process status 137
- Raw run directory: `/tmp/m31-gqa6-8k-logits-gpu0-20261016T091433Z`

This confirms the thermal limit persists for the larger logits workload at
225 W. Continue with shorter correctness diagnostics until the MI50 is capped
at or below 180 W; do not infer a thermal problem in the dot4 kernel itself,
because the default path was stopped before the candidate began.

## Follow-up idle sensor check

After the abort, the BDF-matched sysfs `junction` labels agreed with
`rocm-smi` at idle: GPU0 35 C and GPU2 31 C. Both cards reported fan PWM 37/255
(15%) and 0 RPM while idle. GPU0's cap was still 225 W, GPU2's 300 W; no KFD
PIDs or Podman containers were active. The runner now records fan RPM and PWM
alongside junction temperature during future runs, so load-time fan response is
captured rather than inferred from the idle reading.
