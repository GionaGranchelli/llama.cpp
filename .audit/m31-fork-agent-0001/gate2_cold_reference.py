#!/usr/bin/env python3
"""Compare one restored-prefix next token with a cold full-prompt reference."""

import datetime
import hashlib
import json
import pathlib
import time
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parent
SOURCE = ROOT / "task-context-bundle.md"
TOKENS_FILE = ROOT / "gate2-prompt-tokens.json"
RESTORE_RESULT = ROOT / "gate2-restore.json"
RESULT_FILE = ROOT / "gate2-cold-reference.json"
BASE_URL = "http://127.0.0.1:18081"
PREFIX_TOKENS = 65_536
LIVE_SUFFIX_TOKENS = 512
REFERENCE_SUFFIX_TOKENS = 512


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


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


def main():
    if RESULT_FILE.exists():
        raise SystemExit("Refusing to overwrite existing cold-reference evidence")
    kept_ids = json.loads(TOKENS_FILE.read_text())
    source_bytes = SOURCE.read_bytes()
    tokenized = post(
        "/tokenize",
        {"content": source_bytes.decode(), "add_special": True, "parse_special": True},
        timeout=60,
    )["tokens"]
    live = json.loads((ROOT / "gate2-live-cache.json").read_text())
    token_sha = hashlib.sha256(
        json.dumps(tokenized[: len(kept_ids)], separators=(",", ":")).encode()
    ).hexdigest()
    if token_sha != live["token_ids_sha256"]:
        raise SystemExit("Cold reference input tokens differ from the saved Gate 2 input")
    prompt = tokenized[:PREFIX_TOKENS] + tokenized[
        PREFIX_TOKENS + LIVE_SUFFIX_TOKENS :
        PREFIX_TOKENS + LIVE_SUFFIX_TOKENS + REFERENCE_SUFFIX_TOKENS
    ]
    cached_content = json.loads(RESTORE_RESULT.read_text())["stages"][
        "restored_suffix"
    ]["content"]
    result = {
        "source_bundle_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "prompt_token_ids_sha256": hashlib.sha256(
            json.dumps(prompt, separators=(",", ":")).encode()
        ).hexdigest(),
        "prompt_tokens": len(prompt),
        "cache_prompt": False,
        "seed": 43,
        "temperature": 0,
        "restored_next_content": cached_content,
        "started_utc": utc_now(),
    }
    RESULT_FILE.write_text(json.dumps(result, indent=2) + "\n")
    print(f"cold full-prompt reference started: {len(prompt)} tokens", flush=True)
    response = post(
        "/completion",
        {
            "prompt": prompt,
            "n_predict": 1,
            "cache_prompt": False,
            "id_slot": 0,
            "temperature": 0,
            "seed": 43,
            "stream": False,
        },
    )
    result["response"] = response
    result["finished_utc"] = utc_now()
    timings = response.get("timings", {})
    result["cold_cache_n"] = timings.get("cache_n")
    result["cold_prompt_n"] = timings.get("prompt_n")
    result["content_match"] = response.get("content") == cached_content
    RESULT_FILE.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"cold complete: cache_n={result['cold_cache_n']} "
        f"prompt_n={result['cold_prompt_n']} "
        f"content_match={result['content_match']} "
        f"prompt_ms={timings.get('prompt_ms')}",
        flush=True,
    )
    if result["cold_cache_n"] != 0 or result["cold_prompt_n"] != len(prompt):
        raise SystemExit("Reference request did not prove cold full-prompt evaluation")
    if not result["content_match"]:
        raise SystemExit("Cold and restored next-token contents differ")


if __name__ == "__main__":
    main()
