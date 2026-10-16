# M31-FORK-AGENT-0001 checkpoint

**Checkpoint:** 2026-10-16 16:54 UTC
**Campaign branch:** `product/m31-fork-agent-0001`
**Campaign SHA:** `8f923819cbea0b673d7b16754b4b54110bdf998e`

## Current result

The parser fix for partial Qwen3.5 tagged tool calls is committed and pushed. The focused Qwen3.5 chat tests and the full chat test suite passed on the CPU build (`test-chat --template Qwen3.5-4B.jinja`, then the complete `test-chat` suite). The fix and tests are in `common/chat.cpp` and `tests/test-chat.cpp`. This is a useful compatibility fix, but it does not qualify the MI50 agent workflow.

The running Machinist server is healthy and idle. It has one 131,072-token slot, uses the intended MI50 at `0000:88:00.0` through UUID `GPU-2152414173497dd3`, and has slot save/restore endpoints enabled. The other MI50 is idle. The model checksum matches the campaign's expected artifact. A current read-only inspection also confirmed Podman's `label=disable` setting, the active `run-guarded.sh` supervisor, the running executable SHA, immutable image ID, and ROCm driver `7.2.10-200.fc44.x86_64`.

Prior OMP traffic provides evidence that long prompts and live prefix reuse have occurred: one request processed 76,812 prompt tokens in 498.849 seconds, and a continuation reported 76,808 cached tokens. These are useful observations, not the required controlled 65,536-token suffix test. A later OMP retry reached about 97K prompt tokens, but timed out before returning a result. The retry transcript contains repeated Turn 2/Turn 4 attempts, not eight completed workflow turns.

## Gate status

| Gate | Status | Evidence and remaining proof |
| --- | --- | --- |
| 0 — Environment | PARTIAL | Correct BDF/UUID mapping, model SHA, executable SHA, immutable image ID, driver version, 131,072 context, one slot, no-context-shift, slot API, `label=disable`, 180 W caps, and active guard observed. The current server source is `58a7247`, predating campaign HEAD `1bd9f5e6`; its exact OCI repo digest and runtime-library versions remain uncaptured. |
| 1 — Matched 64K default/J40 | INCOMPLETE | Historical J40 result is 380.861 s. No valid route-confirmed matched default/J40 pair is established in the campaign evidence. |
| 2 — KV reuse and save/restore | INCOMPLETE | OMP continuation reported 76,808 cached tokens; aggregate server metrics show 7.54M cached tokens. Controlled known-prefix suffix accounting and save/restart/restore with post-restore reuse have not been run. |
| 3 — Eight-turn OMP coding task | INCOMPLETE | OMP made real tool calls in the parser task, but the session did not complete eight fixed turns, a verified repository change by OMP, and the predefined acceptance sequence. The latest retry timed out. |
| 4 — Product bottleneck | NOT RUN | No complete workflow to profile. |

## Measurements observed so far

- OMP long prompt: 76,812 tokens; cold prefill 498.849 s; 4,096 generated tokens at about 12.85 tok/s; maximum observed junction temperature 71°C. This 76.8K run exceeds the 420-second target, although it is not the canonical 65,536-token control. The model response ended at the output limit and did not count as an accepted task turn.
- OMP cached continuation: 76,808 cached tokens reported by the client/server record.
- Current server aggregate metrics: 469,386 prompt tokens processed, 7.54291M cached prompt tokens, 48,279 generated tokens, and 11.33 tokens/s reported average decode gauge. Aggregate metrics are diagnostic only; they do not replace per-request measurements.
- Current remote GPU sample: target MI50 at 30°C edge, 180 W cap, 65% VRAM, 0% GPU utilization. The last full telemetry sample in this campaign reported 39°C junction; it was not re-read in the current short preflight. The other MI50 was idle in the sample.

## Current blockers and next actions

The product gate remains open because there is no completed eight-turn agent session and no controlled save/restart/restore proof. The last OMP attempt used a very large prompt and hit the client deadline; repeating that same uncached request is not justified.

Next: finish Gate 0's missing binary/image/guard identity; run the minimum controlled 65K live-prefix and slot save/restore checks on the existing one-slot server; then run OMP with a bounded, fixed eight-turn workload and preserve actual per-turn cache and tool evidence. Do not call the project qualified until all mandatory gates and performance targets pass.

## Transcript evidence

`agent-session.jsonl` and `agent-session-initial.jsonl` preserve the two genuine OMP attempts. The second transcript includes the timeout and incomplete Turn 4; neither transcript is an eight-turn pass.
