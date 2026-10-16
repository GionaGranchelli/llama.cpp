# llama-server slot save/restart/restore smoke

## Result

The llama.cpp server saved a live slot, was stopped and restarted, restored
the slot from disk, and generated a follow-up response. The save and restore
both reported 548 tokens. This proves the basic disk-backed lifecycle works
for a short session on Machinist; it does not qualify a 64K session.

The slot file was **506,612,684 bytes** for this 548-token smoke. Save took
229.4 ms and restore took 314.7 ms. That file size is large enough to matter
for a real agent workload and needs investigation before choosing this as a
practical persistence format.

## Setup

- Host: Machinist, GPU2 GFX906, BDF `0000:88:00.0`
- Source: `e6f92398f0dad240a11812bc269f634791e17e79`
- Model SHA-256:
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Container image ID:
  `50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`
- Server binary SHA-256:
  `b1d878eee13330ad1495a5ca42c78823b921010c672114e4dc2b6cf10396cb0a`
- Context size 1,024; one slot; 99 GPU layers; slot save enabled; four context
  checkpoints with 64-token minimum spacing
- The prompt evaluated 541 tokens and the first request generated 8 tokens.
  The process was restarted before the slot was restored.
- Both server starts and requests stayed below the 84 C stop limit; the
  captured GPU2 junction trace peaked at 58 C. Caps were GPU0 225 W and GPU2
  300 W.

## Round trip

| Stage | Evidence |
| --- | --- |
| First request | Slot 0; 541 prompt tokens evaluated; 8 tokens generated |
| Save | 548 tokens; 506,612,684 bytes; 229.4 ms |
| Restart | First server container stopped and removed; second server started from the same pinned model and source |
| Restore | 548 tokens; 506,612,684 bytes; 314.693 ms |
| Follow-up | 8 tokens generated; Python request wall time 595.21 ms |

The follow-up response reported 558 tokens evaluated. The saved JSON's
`timings.cache_n` and `timings.prompt_n` fields were not extracted during this
run, so this smoke does **not** prove that the restored prefix avoided
full-prefix replay. That counter is the next item to inspect. The response was
generated, but no semantic agent-task result was checked.

## Decision

No MIInfer session allocator or COW snapshot code was ported. llama.cpp's
existing slot API survives a server restart and restores the saved token
position. The immediate open questions are whether the 507 MB file can be
reduced and whether the follow-up actually reuses the restored KV state. The
64K save/restore and latency check remains pending because the two MI50 caps
are above the required 180 W limit.

## Machinist artifacts

`/tmp/m31-slot-restore-smoke-20261016T054728/metadata.txt`

`/tmp/m31-slot-restore-smoke-20261016T054728/first.json`

`/tmp/m31-slot-restore-smoke-20261016T054728/save.json`

`/tmp/m31-slot-restore-smoke-20261016T054728/restore.json`

`/tmp/m31-slot-restore-smoke-20261016T054728/continuation.json`
