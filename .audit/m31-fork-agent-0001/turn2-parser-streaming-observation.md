# Turn 2 — parser streaming observation

## Result

A throwaway C++ harness linked against the campaign's CPU build confirmed that the Qwen3.5 chat session can expose a callable tool object from `feed()` before `finish()`. The harness fed a `Qwen3.5-4B.jinja` tool call byte by byte; once `<function=special_function>` was parsed, `feed()` returned a tool call with `name=special_function` and arguments starting with `{`. Later feed calls emitted argument deltas. A syntactically complete tool call also remained present when the message was only fed and never finished.

For a malformed argument (`arg1` begins with `[1,`), a later feed invalidated the previously returned tool call. `common_chat_msg_diff::compute_diffs()` then threw:

```text
Invalid diff: now finding less tool calls!
```

The harness caught this exception so it could print the old and new tool-call counts. This proves partial tool exposure and a count-decrease failure mode in the parser/session path. It does **not** yet prove the exact issue acceptance case: the throwaway schema used an integer argument, not the required `array<object>` `sources` argument. Do not treat it as the focused regression test or a completed fix.

## Source locations

- `common/chat.cpp`: `common_chat_peg_parse()` maps partial parse results; `common_chat_session::feed()` invokes parsing with `is_partial=true`; `common_chat_session::finish()` invokes it with `is_partial=false`.
- `common/chat-peg-parser.cpp`: `common_chat_peg_mapper::map()` adds a tool to `result.tool_calls` as soon as its name is parsed and initializes arguments with `{` when no argument text has arrived.
- `common/chat.cpp`: `common_chat_msg_diff::compute_diffs()` throws when the new message has fewer tool calls than the previous one.
- `tools/server/server-task.cpp`: `task_result_state::update_chat_msg()` computes streaming diffs; the OMP investigation found the server calls it from the streaming path.

## Artifacts and limits

- Throwaway harness: `/tmp/trace21771.cpp`, SHA-256 `94cedb5393f114297240cb497557d8523d45938aa032290dc4d87e57a339ba61`.
- OMP session transcript: `/tmp/m31-fork-agent-0001-evidence/agent-session-retry/2026-10-16T14-46-10-968Z_01a1452d-7b18-71c3-b084-44e6a85bcb90.jsonl`, SHA-256 `515da7f2e5cebdd7044ed39b8d97137955a0ba3efb675e94d388552d2fcf7a43` at checkpoint time.
- The throwaway harness and transcript are outside Git; the exact `array<object>` reproduction, regression test, fix, and server-level error behavior remain unverified.
- The task worktree remains clean; no parser or test source was changed in Turn 2.
