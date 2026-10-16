#!/usr/bin/env python3
"""Run one exact-token live-prefix reuse check against the guarded local server."""

import hashlib
import json
import pathlib
import time
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parent
SOURCE = ROOT / "task-context-bundle.md"
TOKENS_FILE = ROOT / "gate2-prompt-tokens.json"
RESULT_FILE = ROOT / "gate2-live-cache.json"
BASE_URL = "http://127.0.0.1:18081"
PREFIX_TOKENS = 65_536
SUFFIX_TOKENS = 512
SLOT = 0
SLOT_FILE = "m31-gate2-prefix-65536-20261016.bin"


def post(path, payload, timeout=1200):
    body = json.dumps(payload, separators=(",", ":")).encode()
    req = urllib.request.Request(
        BASE_URL + path,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.load(response)
    result.pop("prompt", None)
    result["client_elapsed_seconds"] = round(time.monotonic() - started, 3)
    return result


def write_result(result):
    RESULT_FILE.write_text(json.dumps(result, indent=2) + "\n")


def main():
    if TOKENS_FILE.exists() or RESULT_FILE.exists():
        raise SystemExit("Refusing to overwrite existing Gate 2 evidence")

    source_bytes = SOURCE.read_bytes()
    source_text = source_bytes.decode()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    tokenized = post(
        "/tokenize",
        {"content": source_text, "add_special": True, "parse_special": True},
        timeout=60,
    )
    tokens = tokenized["tokens"]
    required = PREFIX_TOKENS + SUFFIX_TOKENS
    if len(tokens) < required:
        raise SystemExit(f"Need {required} tokens; source produced {len(tokens)}")

    prompt_tokens = tokens[:required]
    token_bytes = json.dumps(prompt_tokens, separators=(",", ":")).encode()
    token_sha = hashlib.sha256(token_bytes).hexdigest()
    TOKENS_FILE.write_bytes(token_bytes + b"\n")
    result = {
        "input_source": str(SOURCE),
        "input_source_sha256": source_sha,
        "tokenizer_endpoint": BASE_URL + "/tokenize",
        "token_ids_sha256": token_sha,
        "available_tokens": len(tokens),
        "prefix_tokens_requested": PREFIX_TOKENS,
        "suffix_tokens_requested": SUFFIX_TOKENS,
        "slot_id": SLOT,
        "slot_file": SLOT_FILE,
        "stages": {},
    }
    write_result(result)
    print(f"tokenized source: {len(tokens)} tokens; saved exact prompt IDs", flush=True)

    inference_options = {
        "n_predict": 0,
        "cache_prompt": True,
        "id_slot": SLOT,
        "temperature": 0,
        "seed": 42,
        "stream": False,
    }

    print("submitting 65,536-token prefix", flush=True)
    prefix = post(
        "/completion",
        {"prompt": prompt_tokens[:PREFIX_TOKENS], **inference_options},
    )
    result["stages"]["prefix"] = prefix
    write_result(result)
    timings = prefix.get("timings", {})
    if timings.get("prompt_n", 0) + timings.get("cache_n", 0) != PREFIX_TOKENS:
        raise SystemExit(f"Prefix token accounting mismatch: {timings}")
    if prefix.get("truncated", False):
        raise SystemExit("Server truncated the 65,536-token prefix")
    print(f"prefix complete: timings={timings}", flush=True)

    print("saving slot after prefix", flush=True)
    saved = post(
        f"/slots/{SLOT}?action=save",
        {"filename": SLOT_FILE},
        timeout=300,
    )
    result["stages"]["save"] = saved
    write_result(result)
    if saved.get("n_saved", 0) < PREFIX_TOKENS or saved.get("n_written", 0) <= 0:
        raise SystemExit(f"Saved slot does not contain the expected prefix: {saved}")
    print(f"slot saved: {saved}", flush=True)

    print("submitting 512-token suffix", flush=True)
    suffix = post(
        "/completion",
        {"prompt": prompt_tokens, **inference_options},
    )
    result["stages"]["suffix"] = suffix
    write_result(result)
    suffix_timings = suffix.get("timings", {})
    cache_n = suffix_timings.get("cache_n", 0)
    prompt_n = suffix_timings.get("prompt_n", 0)
    result["reuse_ratio"] = round(cache_n / PREFIX_TOKENS, 6)
    result["suffix_prompt_tokens_evaluated"] = prompt_n
    write_result(result)
    print(
        f"suffix complete: cache_n={cache_n} prompt_n={prompt_n} "
        f"reuse_ratio={result['reuse_ratio']}",
        flush=True,
    )
    if cache_n < int(PREFIX_TOKENS * 0.95) or prompt_n > SUFFIX_TOKENS:
        raise SystemExit("Live-prefix reuse did not meet the 95% gate")


if __name__ == "__main__":
    main()
