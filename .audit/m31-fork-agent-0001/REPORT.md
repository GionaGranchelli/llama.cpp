# M31-FORK-AGENT-0001 checkpoint

**Checkpoint:** 2026-10-16 18:16 UTC
**Campaign branch:** `product/m31-fork-agent-0001`
**Parser-fix implementation SHA:** `8f923819cbea0b673d7b16754b4b54110bdf998e`

## Current result

The parser fix for partial Qwen3.5 tagged tool calls is committed and pushed. The focused Qwen3.5 chat tests and the full chat test suite passed on the CPU build (`test-chat --template Qwen3.5-4B.jinja`, then the complete `test-chat` suite). The fix and tests are in `common/chat.cpp` and `tests/test-chat.cpp`. This is a useful compatibility fix, but it does not qualify the MI50 agent workflow.

The running Machinist server is healthy and idle. It has one 131,072-token slot, uses the intended MI50 at `0000:88:00.0` through UUID `GPU-2152414173497dd3`, and has slot save/restore endpoints enabled. The other MI50 is idle. The model checksum matches the campaign's expected artifact. A current read-only inspection also confirmed Podman's `label=disable` setting, the active `run-guarded.sh` supervisor, executable/runtime-library/HIP hashes, immutable image ID, ROCm/HIP runtime `7.2.53211-e1a6bc5663`, and driver `7.2.10-200.fc44.x86_64`.

Prior OMP traffic provides evidence that long prompts and live prefix reuse have occurred: one request processed 76,812 prompt tokens in 498.849 seconds, and a continuation reported 76,808 cached tokens. These are useful observations, not the required controlled 65,536-token suffix test. A later OMP retry reached about 97K prompt tokens, but timed out before returning a result. The retry transcript contains repeated Turn 2/Turn 4 attempts, not eight completed workflow turns.

## Gate status

| Gate | Status | Evidence and remaining proof |
| --- | --- | --- |
| 0 — Environment | PARTIAL | BDF/UUID mapping, model SHA, executable/runtime/HIP hashes, immutable image ID, ROCm/HIP runtime, driver, 131,072 context, one slot, no-context-shift, slot API, `label=disable`, 180 W caps, and active guard are verified. Runtime source is `58a7247` with exactly two approved instrumentation files dirty and their fingerprint recorded. Actual route-selection log confirmation and a clean build of the parser-fix commit remain outstanding. |
| 1 — Matched 64K default/J40 | INCOMPLETE | Historical J40 result is 380.861 s. No valid route-confirmed matched default/J40 pair is established in the campaign evidence. |
| 2 — KV reuse and save/restore | PASS | A controlled 65,536-token prefix completed in 408.506 s. The live 512-token suffix reused all 65,536 cached tokens and evaluated in 4.463 s. The saved 2,754,219,996-byte slot survived a server restart; restore took 1.242 s, and a different 512-token suffix again reused all 65,536 tokens. Its greedy output matched a cold full-prompt reference. Restart-to-first-token took 48.087 s versus 407.561 s cold, an 8.48× advantage. |
| 3 — Eight-turn OMP coding task | INCOMPLETE | OMP made real tool calls in the parser task, but the session did not complete eight fixed turns, a verified repository change by OMP, and the predefined acceptance sequence. The latest retry timed out. |
| 4 — Product bottleneck | NOT RUN | No complete workflow to profile. |

## Latest Oh My Pi attempts

Two fresh runs were made from the exact pre-fix task commit in isolated worktree `/tmp/m31-omp-task-21771-run2`; both left it clean.

- Run 2 included the entire task plan in its initial 69.7K-token source bundle. It was stopped after 22m34s without finishing Turn 1. The agent issued read/search/bash tools, then produced an incomplete edit-tool call that failed JSON parsing; no source change was applied. Transcript: `agent-session-run2-aborted.jsonl` (SHA-256 `8bd41e8b15b7eac8b78b6848ddfe3c1175a4f48fdef239b801514e28b1bb79cc`).
- Run 3 separated source/issue context from the fixed turn prompts and disabled write tools. Its first request evaluated 75,800 uncached prompt tokens in about 490.6 s. The conversation reached 100,776 tokens. It made real `read` and `grep` calls, made no edits, and was stopped after 26m11s still in Turn 1. Transcript: `agent-session-run3-aborted.jsonl` (SHA-256 `6a11d68d28dad1872723cd9861b596095abe286e95f7cd60f2fe635c485597fd`).
- Run 3 server counters increased by 96,240 uncached prompt tokens, 622,790 cached prompt tokens, and 10,298 generated tokens over the attempt; prompt evaluation time increased 683.75 s. The reported decode average was 11.47 tok/s. Individual warm prompt-evaluation segments for a few thousand new tokens were about 20–58 s, already above the 15 s incremental target; these are request prompt timings, not a completed eight-turn TTFT distribution.
- Run 4 used the source/issue context, `--thinking off`, and read-only `read,grep,glob,lsp` tools. After 846 seconds it remained in Turn 1, with no completed turn or repository change. The transcript shows repeated malformed `glob` tool calls after an output-limit failure, followed by more source investigation. At interruption, the server slot reported 98,412 context tokens and `reasoning_format=deepseek`; this run does not prove OMP's thinking setting was passed through as intended. Transcript: `agent-session-run4-aborted.jsonl` (SHA-256 `cdf777cd9acfb6be71c5be2785bcee719841c744ebd4c8acf4966f50d88e9cc5`).
- After run 3, the same server stayed healthy and idle. Its 3,914-sample temperature trace ranged from 30–69°C junction; target power cap remained 180 W and the other MI50 remained idle. No thermal warning or guard stop occurred. Run 4 has no run-specific thermal trace; a subsequent preflight sampled the target at 28°C edge / 31°C junction and 180 W cap.

## Measurements observed so far

- OMP long prompt: 76,812 tokens; cold prefill 498.849 s; 4,096 generated tokens at about 12.85 tok/s; maximum observed junction temperature 71°C. This 76.8K run exceeds the 420-second target, although it is not the canonical 65,536-token control. The model response ended at the output limit and did not count as an accepted task turn.
- OMP cached continuation: 76,808 cached tokens reported by the client/server record.
- Controlled live-cache test: 65,536-token prefix, 408.506 s (160.43 tok/s), 11.5 s inside the 420-second target; saved 65,536 tokens in 2,754,219,996 bytes; the next 512-token request reused 65,536 and evaluated 512 in 4.463 s (4.679 s client wall time). The exact token IDs are preserved in `gate2-prompt-tokens.json`.
- Restore correctness: 65,536 tokens were restored in 1.242 s. A new 512-token suffix reused all 65,536 and evaluated in 4.764 s; its greedy output matched a cold 66,048-token reference. Container start to restored first output took 48.087 s, compared with 407.561 s cold (8.48× faster).
- Guard trace for the restore server and cold reference: 522 one-second samples, junction range 31–67°C, no 80°C warning or 84°C stop. The trace remains on Machinist at `/tmp/m31-fork-agent-restore-001/temperature.csv`.
- Server aggregate metrics sampled after run 4: 282,105 prompt tokens processed, 2.44814M cached prompt tokens, and 25,963 generated tokens. Aggregate metrics are diagnostic only; they do not replace per-request measurements.
- Current remote GPU sample: target MI50 at 30°C edge, 180 W cap, 65% VRAM, 0% GPU utilization. The last full telemetry sample in this campaign reported 39°C junction; it was not re-read in the current short preflight. The other MI50 was idle in the sample.

## Current blockers and next actions

The product gate remains open because no OMP turn has completed the fixed task sequence, and no OMP code change or acceptance test exists. Runs 2, 3, and 4 all remained in Turn 1 after 14–26 minutes. Run 4's `--thinking off` setting did not stop the repeated tool-call loop, and the server still reports `reasoning_format=deepseek`.

Next: inspect OMP's resolved request/provider settings and the server's chat-template configuration, then make one targeted change so the model receives the intended no-thinking setting and the agent stops issuing malformed/repeated tool calls. Keep the 65K+ context, task, model, and guarded runtime fixed. Retest Turn 1 with a bounded run; proceed to the remaining turns only after it returns a usable result. Do not claim the agent gate unless all eight turns, actual code/test outcomes, and context/cache evidence pass. The campaign remains unqualified.

## Transcript evidence

`agent-session.jsonl` and `agent-session-initial.jsonl` preserve the two genuine OMP attempts. The second transcript includes the timeout and incomplete Turn 4; neither transcript is an eight-turn pass.
