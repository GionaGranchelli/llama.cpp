# Session-state port assessment

## Decision

Do not port MIInfer's session allocator or snapshot code at this stage.
llama.cpp already exposes disk-backed slot save/restore and can persist recent
context checkpoints with the slot. That covers the basic cross-process
continuation capability the agent workflow needs. A live save, process restart,
restore, and continued generation check is still required before calling the
feature qualified.

## What llama.cpp already provides

- `--slot-save-path` enables `POST /slots/{id_slot}?action=save` and
  `POST /slots/{id_slot}?action=restore`; the server option is disabled unless
  configured. See `tools/server/README.md` and
  [`server-context.cpp`](../../tools/server/server-context.cpp#L5400).
- Save writes the slot's serialized token sequence and llama sequence state
  with `llama_state_seq_save_file`. It appends available per-context
  checkpoints and, when configured, draft and speculative state. Restore loads
  the sequence state, validates restored tokens and context length, then
  attempts to restore the checkpoint and draft-state appendices. See
  [`server-context.cpp`](../../tools/server/server-context.cpp#L3016) and
  [`server-context.cpp`](../../tools/server/server-context.cpp#L3070).
- The server defaults to 32 in-memory context checkpoints with a minimum
  spacing of 8,192 tokens. `--ctx-checkpoints` and
  `--checkpoint-min-step` configure them. The slot-save implementation
  serializes these checkpoint blobs into the slot file, so they can survive a
  server restart.
- These APIs persist the model slot and token history. The external agent
  still owns its message/tool transcript and must reconnect that transcript to
  the restored token state.

## How MIInfer differs

`MIInfer`'s `SessionCheckpoint` stores the token prefix and per-layer
recurrent state, recurrent history, and K/V buffers in allocated GPU buffers.
`capture_session_checkpoint()` allocates a separate state image and copies
state with device-to-device `hipMemcpyAsync`; lookup selects the longest
compatible token prefix and LRU eviction enforces a memory budget. This is an
in-process cache of GPU state, not a disk-backed COW snapshot.

Relevant implementation: `MIInfer/tools/miinfer_cli.cpp`, especially
`SessionCheckpoint`, `capture_session_checkpoint`, `copy_session_checkpoint`,
`restore_session_checkpoint`, and the `generate()` reuse path around lines
1043–1308 and 2215–2258.

## What remains unverified

- No live llama-server slot has yet been saved, the server restarted, and the
  slot restored.
- No request has yet continued from that restored slot with the expected
  token position and agent-visible result.
- Save-file size and save/restore latency at 64K have not been measured.
- The slot file preserves inference token state, not the external agent's
  structured tool-call history.

Next action after the MI50 guard is available: run a small real slot
save/restart/restore check, then measure file size and restore time at the
target context. Only port session behavior if that check exposes a concrete
gap.
