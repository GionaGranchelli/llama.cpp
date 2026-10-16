# MI50 fan telemetry during the guarded 1,982-token comparison

The instrumented default/candidate pair completed on GPU0 at 225 W. Junction
temperature rose from 29 C to a maximum recorded 74 C, then cooled between
variants. The BDF-matched fan PWM rose from 37/255 (15%) to 57/255 (22%), but
`fan1_input` remained 0 RPM in every sample. Both variants completed; their
logit metrics match the prior 1,982-token run.

This does not prove that the physical chassis fans were stopped: the exposed
GPU tachometer may be unsupported or may not represent the chassis airflow.
However, it shows the GPU fan control asked for more speed without a reported
RPM increase. The larger 24 KB run reached the 84 C guard. A manual fan override
or direct airflow check is the next thermal intervention before another large
prefill.

## Provenance

- Source/runner commit: `19ae026266738dd68a542732e04d57c03d99b999`
- GPU0 BDF: `0000:04:00.0`
- Power caps: GPU0 225 W, GPU2 225 W
- Prompt token IDs: 1,982, SHA-256
  `d95785776724fbdb23e2aac6958a6e62dcf3a69482316ed4e4d5fc6707729491`
- Raw run directory: `/tmp/m31-gqa6-8k-logits-gpu0-20261016T091747Z`
