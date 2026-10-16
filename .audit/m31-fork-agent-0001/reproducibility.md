# Reproducibility checkpoint

## Source and artifact identity

- Campaign repository: `GionaGranchelli/llama.cpp`
- Campaign branch: `product/m31-fork-agent-0001`
- Parser-fix source commit: `8f923819cbea0b673d7b16754b4b54110bdf998e`
- Campaign HEAD before the current evidence update: `4fc1d6c45fd9ac8798fa1633af0761d199bac2bf`
- Runtime source: `58a724757044fda4bbce1cd633b73274c4d32d56` plus the two approved instrumentation files below; this is not a clean build of the parser-fix commit.
- Model: `/models/Qwen3.8-27B-Q4_K_M.gguf`
- Model SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Current server build reports source `58a724757044fda4bbce1cd633b73274c4d32d56`. Its two approved dirty paths are `ggml/src/ggml-cuda/mmq.cu` and `tools/llama-bench/llama-bench.cpp`, with combined diff SHA-256 `18d561645f0febfc5a458d71bdd9350131760e01fda02d4a93848949dc2bb593`. Its `llama-server` SHA-256 is `ffde57a456b4e2c7cb144a2decd3282e84c3a8f9ba22acdcb9324beb8da5bb09`.
- Runtime library SHA-256: `c32e116da4b6f96d38554335d2a6d1d3e818f0c18f6e18138b669a92f50fa1c4`; HIP library SHA-256: `819a43ecd31ad6650ff460e86fd3d32d4850bf3089d1b8971ac3f419ed98d9a2`.
- Current container image name is `localhost/miinfer-dev:rocm-7.2.1`; immutable image ID is `sha256:50ff28a28cc13a3ca7333cd0c70e322d16245967c5fdf0d30a1bf57bda80a512`. OCI repo digest was truncated by the inspection output and remains uncaptured.
- Host driver: `7.2.10-200.fc44.x86_64`; HIP runtime: `7.2.53211-e1a6bc5663`.
- The live server is supervised by `.audit/m31-fork-agent-0001/run-guarded.sh`; its process was observed active at 2026-10-16 16:56:52 UTC.
- Controlled prefix source: `.audit/m31-fork-agent-0001/task-context-bundle.md`, SHA-256 `e98b20883c5ee75a7324bee2d7fea5524e2fc43645ab2daac7028f399f09ea98`.
- Exact request IDs: `.audit/m31-fork-agent-0001/gate2-prompt-tokens.json`, SHA-256 `6a79942bbbeea6b328ad7458e4f31d074b371e0ba767af9cf687b3ca449806c5`; the first 65,536 IDs were the prefix and the next 512 were the live suffix.
- Controlled live-cache request log: `.audit/m31-fork-agent-0001/gate2-live-cache.json`; cache counters and timings are recorded there.
- Saved slot: `/tmp/m31-fork-agent-0001-slots/m31-gate2-prefix-65536-20261016.bin` on Machinist; 65,536 tokens, 2,754,219,996 bytes, SHA-256 `43c5743c9eb3686250224f1fcd38e80f4b065e26bf58306edc1cb2077257ea3a`. It remains on the host and is intentionally not copied into Git.
- Restore and cold-reference records: `gate2-restore.json` and `gate2-cold-reference.json`. Restore read all 65,536 tokens in 1.242 s; a different 512-token suffix reused the full prefix and returned the same greedy output as a cold 66,048-token request. Restart-to-first-output was 48.087 s versus 407.561 s cold. Probe scripts are `gate2_restore_probe.py` and `gate2_cold_reference.py`.

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

The runtime identity is not yet a complete reproducibility record: the OCI repo digest and actual route-selection logs are still missing, and the server has not been rebuilt from the parser-fix campaign commit. Gate 2 itself is complete for the recorded runtime; this identity gap remains part of Gate 0.

## Existing run records

- First OMP transcript: `/tmp/m31-fork-agent-0001-evidence/agent-session/2026-10-16T14-19-34-006Z_01a14515-1cf6-713d-bcd6-5e5b637b9845.jsonl`
- OMP retry transcript: `/tmp/m31-fork-agent-0001-evidence/agent-session-retry/2026-10-16T14-46-10-968Z_01a1452d-7b18-71c3-b084-44e6a85bcb90.jsonl`
- Guarded runner: `.audit/m31-fork-agent-0001/run-guarded.sh`

Both transcripts are copied into this evidence directory as `agent-session-initial.jsonl` and `agent-session.jsonl`. They contain failed/incomplete attempts and must not be reported as a passing eight-turn run.

- `agent-session-initial.jsonl` SHA-256: `c178df9bde75810521fa5e0de74a29a52c8411064baf7822bc8c359ed1a2ecef`
- `agent-session.jsonl` SHA-256: `6e49ce96afd7efaca4e9de1912ccf2f91238c7092799a440d1f8d70faf535534`

## Latest OMP attempts

- OMP binary: `/tmp/m31-omp-v18.11.1/omp`, version `18.11.1`.
- Isolated profile home: `/tmp/m31-omp-home`; provider model `/models/Qwen3.8-27B-Q4_K_M.gguf`, context window `131072`, `maxTokens: 4096`, API `openai-completions`, base URL `http://127.0.0.1:18081/v1`.
- Task source worktree: `/tmp/m31-omp-task-21771-run2`, branch `agent/m31-omp-21771-run2`, exact base SHA `58a724757044fda4bbce1cd633b73274c4d32d56`. It remained clean after both interrupted runs.
- Run 2 supplied `task-context-bundle.md` (69,665 tokenizer tokens) together with fixed Turn 1 and enabled `read,bash,edit,grep,glob,lsp`, `--thinking low`. It ran 1,354 seconds and was interrupted during Turn 1. Its transcript is `agent-session-run2-aborted.jsonl` (SHA-256 `8bd41e8b15b7eac8b78b6848ddfe3c1175a4f48fdef239b801514e28b1bb79cc`).
- Run 3 used source/issue-only context `agent-source-context.md` (SHA-256 `e89c41cf2358eb89a283a0c392a9dcbc866298ec737d35b02eec77c2b5d9d8eb`, 68,961 server-tokenizer tokens), fixed Turn 1, `read,grep,glob,lsp` only, and `--thinking low`. It ran 1,571 seconds and was interrupted during Turn 1. Its transcript is `agent-session-run3-aborted.jsonl` (SHA-256 `6a11d68d28dad1872723cd9861b596095abe286e95f7cd60f2fe635c485597fd`).
- Run 3’s actual prompt context reached 100,776 tokens; the initial uncached prompt evaluated 75,800 tokens in about 490.6 seconds. Aggregate per-attempt counter deltas are recorded in `metrics.json`. The attempt is not an eight-turn result.
- A follow-up diagnostic run may use `--thinking off`; this is a different OMP configuration and must be recorded separately. No source, model, server, power, or task-prompt change is implied.

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

It uses the local endpoint at `http://127.0.0.1:18081`, sends an exact 65,536-token prefix with `n_predict=0`, saves the slot, then sends the exact following 512 tokens. It refuses to overwrite prior evidence. `gate2_restore_probe.py` restarts from the saved slot and tests a different 512-token suffix; `gate2_cold_reference.py` runs the matched cold reference with caching disabled. Their JSON outputs preserve the observed timings and token counts.
