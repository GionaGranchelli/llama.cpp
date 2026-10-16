# Reproducibility checkpoint

## Source and artifact identity

- Campaign repository: `GionaGranchelli/llama.cpp`
- Campaign branch: `product/m31-fork-agent-0001`
- Current source: `8f923819cbea0b673d7b16754b4b54110bdf998e`
- Current source working tree: clean at checkpoint
- Model: `/models/Qwen3.8-27B-Q4_K_M.gguf`
- Model SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Current server build reports source `58a724757044fda4bbce1cd633b73274c4d32d56`; this predates the parser-fix commit. The running binary SHA-256 and container image digest have not yet been captured.
- Current container image name is `localhost/miinfer-dev:rocm-7.2.1`; exact image ID/digest remains to be captured.

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

The current runtime identity is not yet a complete reproducibility record: capture the executable SHA-256, container image digest, runtime version and live guard ownership before starting the final long run.

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
