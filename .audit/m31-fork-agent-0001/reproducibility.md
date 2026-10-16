# Reproducibility checkpoint

## Source and artifact identity

- Campaign repository: `GionaGranchelli/llama.cpp`
- Campaign branch: `product/m31-fork-agent-0001`
- Current source: `8f923819cbea0b673d7b16754b4b54110bdf998e`
- Current source working tree: clean at checkpoint
- Current campaign HEAD after checkpoint evidence: `1bd9f5e6aa77f91a0225236da39f58bf1fdff8f7`
- Model: `/models/Qwen3.8-27B-Q4_K_M.gguf`
- Model SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Current server build reports source `58a724757044fda4bbce1cd633b73274c4d32d56`; this predates the parser-fix commit and checkpoint. Its `llama-server` SHA-256 is `ffde57a456b4e2c7cb144a2decd3282e84c3a8f9ba22acdcb9324beb8da5bb09`.
- Current container image name is `localhost/miinfer-dev:rocm-7.2.1`; immutable image ID is `sha256:50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`. OCI repo digest was truncated by the inspection output and remains uncaptured.
- Host driver: `7.2.10-200.fc44.x86_64`.
- The live server is supervised by `.audit/m31-fork-agent-0001/run-guarded.sh`; its process was observed active at 2026-10-16 16:56:52 UTC.
- Controlled prefix source: `.audit/m31-fork-agent-0001/task-context-bundle.md`, SHA-256 `e98b20883c5ee75a7324bee2d7fea5524e2fc43645ab2daac7028f399f09ea98`.
- Exact request IDs: `.audit/m31-fork-agent-0001/gate2-prompt-tokens.json`, SHA-256 `6a79942bbbeea6b328ad7458e4f31d074b371e0ba767af9cf687b3ca449806c5`; the first 65,536 IDs were the prefix and the next 512 were the live suffix.
- Controlled live-cache request log: `.audit/m31-fork-agent-0001/gate2-live-cache.json`; cache counters and timings are recorded there.
- Saved slot: `/tmp/m31-fork-agent-0001-slots/m31-gate2-prefix-65536-20261016.bin` on Machinist; server reported 65,536 tokens and 2,754,219,996 bytes written. Keep this file intact for the restart/restore test; it is intentionally not copied into Git.

## Hardware and runtime observed

- Host: Machinist X99
- Target: MI50 at PCI BDF `0000:88:00.0`, ROCm index 2, UUID `GPU-2152414173497dd3`
- Other MI50: `0000:04:00.0`; idle during the observed sample
- Both MI50 power caps: 180 W
- Current server: one slot, 131,072 context, no context shifting, Q8 KV, flash attention on, GPU layers 99, batch 2048, microbatch 1024
- Device exposure: `ROCR_VISIBLE_DEVICES=GPU-2152414173497dd3`
- Podman security option: `label=disable`
- Server slot persistence: `--slot-save-path /state`; `/state` is mounted to host storage
- Driver version last reported earlier in this campaign: ROCm `7.2.10-200.fc44.x86_64`; recheck before the final qualification run.

The current runtime identity is not yet a complete reproducibility record: rebuild the server from the campaign HEAD, capture its runtime libraries and OCI repo digest, and verify guard telemetry during the final long run.

## Existing run records

- First OMP transcript: `/tmp/m31-fork-agent-0001-evidence/agent-session/2026-10-16T14-19-34-006Z_01a14515-1cf6-713d-bcd6-5e5b637b9845.jsonl`
- OMP retry transcript: `/tmp/m31-fork-agent-0001-evidence/agent-session-retry/2026-10-16T14-46-10-968Z_01a1452d-7b18-71c3-b084-44e6a85bcb90.jsonl`
- Guarded runner: `.audit/m31-fork-agent-0001/run-guarded.sh`

Both transcripts are copied into this evidence directory as `agent-session-initial.jsonl` and `agent-session.jsonl`. They contain failed/incomplete attempts and must not be reported as a passing eight-turn run.

- `agent-session-initial.jsonl` SHA-256: `c178df9bde75810521fa5e0de74a29a52c8411064baf7822bc8c359ed1a2ecef`
- `agent-session.jsonl` SHA-256: `6e49ce96afd7efaca4e9de1912ccf2f91238c7092799a440d1f8d70faf535534`

## Reproduction commands

The existing server was started with the campaign's guarded runner and these relevant options:

```text
--ctx-size 131072 --parallel 1 --batch-size 2048 --ubatch-size 1024
--cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on --n-gpu-layers 99
--device ROCm0 --host 127.0.0.1 --port 8081 --metrics --slots
--slot-save-path /state --no-context-shift --jinja
ROCR_VISIBLE_DEVICES=GPU-2152414173497dd3
--security-opt label=disable
```

The complete reproducible command is not recorded yet because the running executable SHA, image digest and final source build identity still need to be reconciled. Do not use these fragments alone as a qualified rerun recipe.

The controlled live test was driven once with:

```text
rtk python3 .audit/m31-fork-agent-0001/gate2_live_cache.py
```

It uses the local endpoint at `http://127.0.0.1:18081`, sends an exact 65,536-token prefix with `n_predict=0`, saves the slot, then sends the exact following 512 tokens. It refuses to overwrite prior evidence. To complete Gate 2, restart the guarded server with the same executable/configuration, restore the saved slot, and test a different 512-token suffix.
