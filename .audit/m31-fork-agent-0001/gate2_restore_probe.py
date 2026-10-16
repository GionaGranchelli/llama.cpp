#!/usr/bin/env python3
"""Restore a saved 65K slot and prove it serves a new 512-token suffix."""

import datetime
import hashlib
import json
import pathlib
import time
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parent
SOURCE = ROOT / "task-context-bundle.md"
TOKENS_FILE = ROOT / "gate2-prompt-tokens.json"
LIVE_RESULT = ROOT / "gate2-live-cache.json"
RESULT_FILE = ROOT / "gate2-restore.json"
BASE_URL = "http://127.0.0.1:18081"
SLOT = 0
PREFIX_TOKENS = 65_536
SUFFIX_TOKENS = 512
SLOT_FILE = "m31-gate2-prefix-65536-20261016.bin"


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def request(method, path, payload=None, timeout=1200):
    body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    headers = {} if body is None else {"Content-Type": "application/json"}
    req = urllib.request.Request(BASE_URL + path, data=body, headers=headers, method=method)
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.load(response)
    if isinstance(result, dict):
        result.pop("prompt", None)
        result["client_elapsed_seconds"] = round(time.monotonic() - started, 3)
    return result


def save_result(result):
    RESULT_FILE.write_text(json.dumps(result, indent=2) + "\n")


def main():
    if RESULT_FILE.exists():
        raise SystemExit("Refusing to overwrite existing restore evidence")

    ids = json.loads(TOKENS_FILE.read_text())
    if len(ids) != PREFIX_TOKENS + SUFFIX_TOKENS:
        raise SystemExit(f"Unexpected preserved token count: {len(ids)}")
    live = json.loads(LIVE_RESULT.read_text())
    expected_token_sha = live["token_ids_sha256"]
    source_bytes = SOURCE.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    tokenized = request(
        "POST",
        "/tokenize",
        {"content": source_bytes.decode(), "add_special": True, "parse_special": True},
        timeout=60,
    )["tokens"]
    all_ids_json = json.dumps(tokenized[: len(ids)], separators=(",", ":")).encode()
    if hashlib.sha256(all_ids_json).hexdigest() != expected_token_sha:
        raise SystemExit("Retokenized prefix/suffix does not match the preserved request IDs")
    if len(tokenized) < PREFIX_TOKENS + 2 * SUFFIX_TOKENS:
        raise SystemExit("Source bundle no longer contains a second 512-token suffix")

    props = request("GET", "/props")
    slots_before = request("GET", "/slots")
    if any(slot.get("is_processing") for slot in slots_before):
        raise SystemExit("Refusing restore while a slot is processing")
    result = {
        "source_bundle_sha256": source_sha,
        "token_ids_sha256_checked": expected_token_sha,
        "prefix_tokens": PREFIX_TOKENS,
        "restored_suffix_range": [PREFIX_TOKENS + SUFFIX_TOKENS,
                                  PREFIX_TOKENS + 2 * SUFFIX_TOKENS],
        "server_build_info": props.get("build_info"),
        "server_context_tokens": props.get("default_generation_settings", {}).get("n_ctx"),
        "server_slots": props.get("total_slots"),
        "slot_before_restore": slots_before,
        "restore_started_utc": utc_now(),
        "stages": {},
    }
    save_result(result)
    print("restoring saved 65,536-token slot", flush=True)
    restored = request(
        "POST",
        f"/slots/{SLOT}?action=restore",
        {"filename": SLOT_FILE},
        timeout=600,
    )
    result["stages"]["restore"] = restored
    result["restore_completed_utc"] = utc_now()
    save_result(result)
    if restored.get("n_restored", 0) != PREFIX_TOKENS or restored.get("n_read", 0) <= 0:
        raise SystemExit(f"Restored state does not match the saved prefix: {restored}")
    print(f"slot restored: {restored}", flush=True)

    prompt = tokenized[:PREFIX_TOKENS] + tokenized[
        PREFIX_TOKENS + SUFFIX_TOKENS : PREFIX_TOKENS + 2 * SUFFIX_TOKENS
    ]
    suffix_options = {
        "n_predict": 1,
        "cache_prompt": True,
        "id_slot": SLOT,
        "temperature": 0,
        "seed": 43,
        "stream": False,
    }
    result["first_token_request_utc"] = utc_now()
    save_result(result)
    print("submitting a new 512-token suffix after restore", flush=True)
    response = request(
        "POST",
        "/completion",
        {"prompt": prompt, **suffix_options},
    )
    result["stages"]["restored_suffix"] = response
    result["first_token_response_utc"] = utc_now()
    timings = response.get("timings", {})
    cache_n = timings.get("cache_n", 0)
    prompt_n = timings.get("prompt_n", 0)
    result["restored_prefix_reuse_ratio"] = round(cache_n / PREFIX_TOKENS, 6)
    result["suffix_prompt_tokens_evaluated"] = prompt_n
    result["first_token_returned"] = response.get("tokens_predicted", 0) >= 1
    save_result(result)
    print(
        f"restored suffix complete: cache_n={cache_n} prompt_n={prompt_n} "
        f"prompt_ms={timings.get('prompt_ms')}",
        flush=True,
    )
    if cache_n < int(PREFIX_TOKENS * 0.95) or prompt_n > SUFFIX_TOKENS:
        raise SystemExit("Post-restore prefix reuse did not meet the 95% gate")
    if not result["first_token_returned"]:
        raise SystemExit("Post-restore request did not return a generated token")


if __name__ == "__main__":
    main()
