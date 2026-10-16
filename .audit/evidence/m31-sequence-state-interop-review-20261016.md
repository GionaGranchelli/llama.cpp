# Sequence-state and branching interop review

## llama.cpp capabilities found

- `llama_state_seq_get_size/data` and `set_data` expose per-sequence state
  capture and restore.
- `llama_state_seq_save_file/load_file` persist a sequence together with its
  token IDs. The file reader checks a magic value and format version before
  restoring the state.
- Extended sequence state APIs include an on-device transfer option. The
  file APIs use host file I/O, which is the relevant path for restart recovery.
- `llama_memory_seq_cp` and the hybrid memory backend expose sequence copy;
  the hybrid backend forwards it to attention and recurrent state. One indexed
  hybrid backend only permits whole-sequence copies.

## MIInfer comparison

MIInfer's `PersistentSession` stores the token prefix, token hash, model ID,
quantization, KV mode, recurrent GDN state, and attention KV bytes. It writes a
temporary file and renames it into place. `ReusableContext` separately keeps a
device-side GDN checkpoint for in-process prefix reuse.

The existing llama.cpp APIs cover durable sequence save/restore and sequence
copy at the state layer. The source review does not establish that sequence
copy uses copy-on-write storage, or that its on-disk identity checks match
MIInfer's model and token fingerprint checks. Those are behavioral questions
to verify only if the actual agent workflow requires branching or cross-run
cache reuse.

## Decision

Do not port MIInfer's session or allocator code now. Keep llama.cpp's native
sequence APIs as the candidate path. If the product workload exposes a gap,
test that exact lifecycle first and add only the missing behavior.

## Source locations

- llama.cpp: `include/llama.h`, `src/llama-context.cpp`,
  `src/llama-memory-hybrid.cpp`, `src/llama-memory-hybrid-idx.cpp`
- MIInfer: `include/miinfer/prefill_v2/persistent_session.hpp`,
  `src/prefill_v2/persistent_session.cpp`,
  `include/miinfer/prefill_v2/reusable_context.hpp`,
  `src/prefill_v2/reusable_context.cpp`
