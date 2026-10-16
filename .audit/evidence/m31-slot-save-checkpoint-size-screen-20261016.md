# Slot-save size screen with context checkpoints on/off

## Result

For the same 548-token short session, the slot file was **506,612,684 bytes**
with four context checkpoints configured and **192,823,864 bytes** with
checkpoints disabled. The disabled run's file is 313,788,820 bytes smaller
(61.9%) in this one-run screen. Its file contained neither `SCKP` nor `SDFT`
appendix markers. The larger file's appendix has not yet been inspected, so
the size difference is measured but not yet attributed byte-for-byte.

Even with checkpoints disabled, 192.8 MB for 548 tokens remains material. A
second sample is needed before treating the size reduction as a repeatable
ratio.

## Matched setup

Both saves used Machinist GPU2 (`0000:88:00.0`), source
`e6f92398f0dad240a11812bc269f634791e17e79`, the same pinned ROCm 7.2.1 image,
and Qwen3.8-27B Q4_K_M with SHA-256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`. Each
request evaluated the same 541-token prompt and generated 8 tokens; slot save
reported 548 tokens.

| Setting | Checkpoint-enabled round trip | Checkpoint-disabled save |
| --- | ---: | ---: |
| Context checkpoints | 4, minimum spacing 64 tokens | 0 |
| Saved tokens | 548 | 548 |
| File bytes | 506,612,684 | 192,823,864 |
| Restore | 548 tokens restored in 314.693 ms | Not run |
| Slot save | 229.4 ms | Timing not retained in the summary output |

The no-checkpoint server stayed below the 84 C guard and exited successfully.
The earlier run's GPU junction trace also stayed below that guard. Both runs
used the existing 225 W / 300 W caps; neither qualifies the 64K power gate.

## Evidence on Machinist

- Checkpoint-enabled round trip:
  `/tmp/m31-slot-restore-smoke-20261016T054728/`
- Checkpoint-disabled save:
  `/tmp/m31-slot-no-checkpoints-20261016T055154/`

Next check: inspect the checkpoint-enabled file's `SCKP` appendix offset and
size, then repeat the no-checkpoint save only if the exact storage tradeoff
would change the 64K session design.
