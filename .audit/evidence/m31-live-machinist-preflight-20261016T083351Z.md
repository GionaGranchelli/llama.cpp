# Machinist live preflight

Checked 2026-10-16 08:33:51 UTC before resuming the guarded 64K work.

| ROCm index | PCI BDF | Junction | Power cap | Fan report |
|---:|---|---:|---:|---|
| 0 | `0000:04:00.0` | 27 C | 225 W | 15%, 0 RPM |
| 2 | `0000:88:00.0` | 31 C | 300 W | 15%, 0 RPM |

ROCm reported no KFD processes. `podman ps` and the host process search for
LM Studio and llama.cpp returned no entries. Both cards were idle at the
preflight. The guarded 64K runner requires both power caps at or below 180 W,
so it correctly remains ineligible. Do not interpret the idle fan reading as
loaded thermal behavior.

The card identities came from PCI BDFs under `/sys/class/drm`; ROCm index 1
is not an MI50 and was excluded.
