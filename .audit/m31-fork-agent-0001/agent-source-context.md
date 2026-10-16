===== Issue #21771 =====
## Summary

When a tool has a parameter whose JSON schema is `array<object>` (e.g. `[{"type":"web"}]`) and the model (Qwen3 family using the `TAG_WITH_TAGGED` / `<tool_call><function=...><parameter=...>` format) emits it as the inner value of a `<parameter>` tag, the autoparser's PEG `p.json()` fails mid-stream. Two cascading problems follow:

1. **Root parse failure.** `build_tool_parser_tag_tagged` uses `p.json()` for any parameter that doesn't resolve to a string schema. On array-of-object values the PEG parser aborts, the request ends in a 500 with `Failed to parse input at pos N: <tool_call>...`.
2. **Partial state leaked to the client.** Before the failure, the partial-parse fallback at `common/chat.cpp:2158` returns whatever AST was built so far. For streaming (`is_partial=true`) this surfaces to the client as an SSE delta containing `tool_calls[0].function.arguments = "{"` (one character). The client, acting correctly per the OpenAI spec, appends this tool_call to conversation history.
3. **History poisoning.** Every subsequent request replays that history. `func_args_not_string` at `common/chat.cpp:1842-1860` eagerly parses `tool_calls[].function.arguments` as JSON and throws a 500 on the bare `{`, so the conversation is now stuck: every follow-up crashes with `Failed to parse tool call arguments as JSON: ... unexpected end of input`, even though the client did nothing wrong.

Net effect: a single tool-call attempt with a complex parameter permanently wedges the conversation until the client manually strips the assistant message from history.

## Name and Version

```
version: 8759 (2b2cd57de)
built with GNU 11.4.0 for Linux x86_64
```

HEAD of master as of 2026-04-11, i.e. PR #21216 is included (verified commit `26229755c` is in the binary).

## Operating systems

Linux

## GGML backends

CUDA

## Hardware

4x RTX 3090

## Models

`Qwen3.5-122B-A10B-Q4_K_M` (GGUF). The bug is not specific to this model — it reproduces with any model whose template routes to `TAG_WITH_TAGGED` tool format.

## Steps to reproduce

1. Launch the server with jinja enabled and any tool set containing an `array<object>` parameter:
   ```
   llama-server -m Qwen3.5-122B-A10B-Q4_K_M.gguf -ngl 99 -c 163840 -fa on \
                --host 0.0.0.0 --port 8080 --jinja
   ```
2. Send a chat completion with a tool definition like:
   ```json
   {
     "type": "function",
     "function": {
       "name": "firecrawl_search",
       "parameters": {
         "type": "object",
         "properties": {
           "query":   {"type": "string"},
           "limit":   {"type": "integer"},
           "sources": {
             "type": "array",
             "items": {
               "type": "object",
               "properties": {"type": {"type": "string", "enum": ["web","images","news"]}},
               "required": ["type"]
             }
           }
         },
         "required": ["query"]
       }
     }
   }
   ```
   and a user prompt that naturally requires calling it with `sources` set.
3. The model emits (captured via a request-dumping proxy in front of llama-server):
   ```
   <tool_call>
   <function=firecrawl_search>
   <parameter=limit>
   10
   </parameter>
   <parameter=query>
   pop culture trends USA March April 2026 movies music TV shows celebrities
   </parameter>
   <parameter=sources>
   [{"type": "web"},
   ```
4. Server responds with 500 in the SSE stream:
   ```
   Failed to parse input at pos 518: <tool_call>
   <function=firecrawl_search>
   ...
   <parameter=sources>
   [{"type": "web"},
   ```
5. Inspection of the SSE stream of that same request shows a **preceding** delta event:
   ```json
   {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"...",
     "type":"function","function":{"name":"firecrawl_search","arguments":"{"}}]}}]}
   ```
   i.e. `arguments = "{"` (one character) was already pushed to the client before the stream fails.
6. Any follow-up chat completion that replays the assistant message as conversation history now fails with:
   ```
   Failed to parse tool call arguments as JSON:
   [json.exception.parse_error.101] parse error at line 1, column 2:
   syntax error while parsing object key - unexpected end of input; expected string literal
   ```

## Root cause (pointers)

`common/chat-auto-parser-generator.cpp` around line 367 in `build_tool_parser_tag_tagged`:

```cpp
(schema_info.resolves_to_string(param_schema) ?
    p.tool_arg_string_value(p.schema(until_suffix, ..., param_schema, true)) :
    p.tool_arg_json_value(p.schema(p.json(), ..., param_schema, false)) + p.space())
```

For non-string parameters the generated rule calls `p.json()`, which appears to be unable to parse JSON whose value is `array<object>` inside a `<parameter>` tag context. Possibly related: nested-object backtracking, or lookahead for the closing `</parameter>` interfering with the JSON grammar. I have not pinned the exact failure site inside `p.json()` yet — happy to dig further if useful.

The partial-parse fallback at `common/chat.cpp:2158-2178` is what surfaces `"{"` to the client during streaming, and `func_args_not_string` at `common/chat.cpp:1842-1860` is the secondary crash point that poisons the conversation.

## Suggested fixes

1. **Parser:** Either fix `p.json()` to parse arbitrary nested JSON inside `<parameter>` tag context, or (simpler) switch the non-string branch to use `until_suffix`-based capture + a post-parse `json::parse` on the raw slice. The latter matches how many real-world clients emit structured parameter values and avoids coupling grammar with schema.
2. **Streaming safety:** `is_partial` should not emit a `tool_calls[].function.arguments` delta unless the accumulated content is at least a syntactically valid JSON *prefix* that the client can safely append to (or, simplest, flush only when the full tool_call is complete). Emitting `"{"` and then aborting is strictly worse than emitting nothing.
3. **Replay tolerance:** `func_args_not_string` should not throw a 500 when a prior `tool_calls[].function.arguments` in the input history is not valid JSON. It can either leave the string as-is (letting the template render it verbatim) or wrap it for the template. Throwing forces the client to clean up after the server's own broken output.

## Notes

Not a duplicate of #20867 (that one was about `MAX_REPETITION_THRESHOLD`, fixed by #21216, which is already in this build).




===== common/chat.cpp =====
#include "chat.h"

#include "chat-auto-parser-helpers.h"
#include "chat-auto-parser.h"
#include "chat-peg-parser.h"
#include "common.h"
#include "ggml.h"
#include "json-schema-to-grammar.h"
#include "json.h"
#include "log.h"
#include "parsers/parsers.h"
#include "sampling.h"

#include "jinja/value.h"
#include "jinja/runtime.h"
#include "jinja/caps.h"
#include "peg-parser.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <functional>
#include <iomanip>
#include <map>

#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using json = common_json;

static std::string format_time(const std::chrono::system_clock::time_point & now, const std::string & format) {
    auto               time       = std::chrono::system_clock::to_time_t(now);
    auto               local_time = *std::localtime(&time);
    std::ostringstream ss;
    ss << std::put_time(&local_time, format.c_str());
    auto res = ss.str();
    return res;
}

static json safe_args_parse(const std::string & to_parse) {
    std::string stripped = to_parse;
    if (to_parse.at(0) == '"' && to_parse.at(to_parse.length() - 1) == '"') {
        stripped = to_parse.substr(1, to_parse.length() - 1);
    }
    try {
        return json::parse(stripped);
    } catch (const common_json_error & e) {
        return stripped;
    }
}

static std::string string_diff(const std::string & last, const std::string & current) {
    if (last.empty()) {
        return current;
    }
    if (!string_starts_with(current, last)) {
        if (string_starts_with(last, current)) {
            // This happens if the last generation ended on a partial stop word (not erased),
            // and the current ended on a stop word (erased).
            return "";
        }
        throw std::runtime_error("Invalid diff: '" + last + "' not found at start of '" + current + "'");
    }
    return current.substr(last.size());
}

static bool has_content_or_tool_calls(const common_chat_msg & msg) {
    return !msg.content.empty() || !msg.tool_calls.empty();
}

std::string common_chat_msg::render_content(const std::string & delimiter) const {
    if (!content.empty() && !content_parts.empty()) {
        throw std::runtime_error("Cannot specify both content and content_parts");
    }
    if (!content.empty()) {
        return content;
    }

    std::string text;
    for (const auto & part : content_parts) {
        if (part.type == "text") {
            if (!text.empty()) {
                text += delimiter;
            }
            text += part.text;
        }
    }
    return text;
}

common_chat_role common_chat_role_from_string(const std::string & role) {
    if (role == "system")    { return COMMON_CHAT_ROLE_SYSTEM;    }
    if (role == "assistant") { return COMMON_CHAT_ROLE_ASSISTANT; }
    if (role == "user")      { return COMMON_CHAT_ROLE_USER;      }
    if (role == "tool")      { return COMMON_CHAT_ROLE_TOOL;      }
    return COMMON_CHAT_ROLE_UNKNOWN;
}

const char * common_chat_role_to_string(common_chat_role role) {
    switch (role) {
        case COMMON_CHAT_ROLE_SYSTEM:    return "system";
        case COMMON_CHAT_ROLE_ASSISTANT: return "assistant";
        case COMMON_CHAT_ROLE_USER:      return "user";
        case COMMON_CHAT_ROLE_TOOL:      return "tool";
        case COMMON_CHAT_ROLE_UNKNOWN:   return "";
    }
    return "";
}

void common_chat_msg_delimiters::tokenize(const llama_vocab * vocab) {
    for (auto & d : delimiters) {
        d.tokens = common_tokenize(vocab, d.delimiter, false, true);
    }
}

common_chat_msg_spans common_chat_msg_delimiters::split(const llama_tokens & tokens, const std::map<size_t, size_t> & skips) const {
    std::vector<std::pair<common_chat_role, size_t>> matches;

    auto skip = skips.begin();
    for (size_t i = 0; i < tokens.size();) {
        if (skip != skips.end() && i == skip->first) {
            i += skip->second;
            ++skip;
            continue;
        }
        for (const auto & d : delimiters) {
            if (i + d.tokens.size() > tokens.size()) {
                continue;
            }
            if (std::equal(d.tokens.begin(), d.tokens.end(), tokens.begin() + i)) {
                matches.emplace_back(d.role, i);
                break;
            }
        }
        i++;
    }

    matches.emplace_back(COMMON_CHAT_ROLE_UNKNOWN, tokens.size());

    common_chat_msg_spans spans;
    for (size_t i = 0; i + 1 < matches.size(); i++) {
        const auto & curr = matches[i];
        const auto & next = matches[i + 1];
        spans.add(curr.first, curr.second, next.second - curr.second);
    }

    return spans;
}

json common_chat_msg::to_json_oaicompat(bool concat_typed_text) const {
    if (!content.empty() && !content_parts.empty()) {
        throw std::runtime_error("Cannot specify both content and content_parts");
    }
    json jmsg {
        {"role", role},
    };
    if (!content.empty()) {
        jmsg["content"] = content;
    } else if (!content_parts.empty()) {
        if (concat_typed_text || contains_media()) {
            std::string text;
            bool last_was_media_marker = false;
            // join parts with newline, do not add newline before or after media markers
            for (const auto & part : content_parts) {
                bool add_new_line = true;
                if (part.type == "text") {
                    add_new_line = !last_was_media_marker && !text.empty();
                    last_was_media_marker = false;
                } else if (part.type == "media_marker") {
                    add_new_line = false;
                    last_was_media_marker = true;
                } else {
                    LOG_WRN("Ignoring content part type: %s\n", part.type.c_str());
                    continue;
                }

                if (add_new_line) {
                    text += '\n';
                }

                text += part.text;
            }
            jmsg["content"] = text;
        } else {
            auto & parts = jmsg["content"] = json::array();
            for (const auto & part : content_parts) {
                parts.push_back({
                    {"type", part.type},
                    {"text", part.text},
                });
            }
        }
    } else {
        jmsg["content"] = "";
    }
    if (!reasoning_content.empty()) {
        jmsg["reasoning_content"] = reasoning_content;
    }
    if (!tool_name.empty()) {
        jmsg["name"] = tool_name;
    }
    if (!tool_call_id.empty()) {
        jmsg["tool_call_id"] = tool_call_id;
    }
    if (!tool_calls.empty()) {
        jmsg["tool_calls"] = json::array();
        auto & jtool_calls = jmsg["tool_calls"];
        for (const auto & tool_call : tool_calls) {
            json tc {
                {"type", "function"},
                {"function", {
                    {"name", tool_call.name},
                    {"arguments", json(tool_call.arguments)},
                }},
            };
            if (!tool_call.id.empty()) {
                tc["id"] = tool_call.id;
            }
            // Some templates generate and require an id (sometimes in a very specific format, e.g. Mistral Nemo).
            // We only generate a random id for the ones that don't generate one by themselves
            // (they also won't get to see it as their template likely doesn't use it, so it's all for the client)
            // {"id", tc.id.empty() ? gen_tool_call_id() : tc.id},
            jtool_calls.push_back(tc);
        }
    }

    return jmsg;
}

std::vector<common_chat_msg_diff> common_chat_msg_diff::compute_diffs(const common_chat_msg & msg_prv,
                                                                      const common_chat_msg & msg_new) {
    std::vector<common_chat_msg_diff> diffs;
    if (msg_new.tool_calls.size() > msg_prv.tool_calls.size()) {
        diffs.reserve(msg_new.tool_calls.size() - msg_prv.tool_calls.size() + 3);
    } else {
        diffs.reserve(3);
    }

    // TODO: these can become expensive for long messages - how to optimize?
    if (msg_prv.reasoning_content != msg_new.reasoning_content) {
        auto & diff                  = diffs.emplace_back();
        diff.reasoning_content_delta = string_diff(msg_prv.reasoning_content, msg_new.reasoning_content);
    }
    if (msg_prv.content != msg_new.content) {
        auto & diff        = diffs.emplace_back();
        diff.content_delta = string_diff(msg_prv.content, msg_new.content);
    }

    if (msg_new.tool_calls.size() < msg_prv.tool_calls.size()) {
        std::string err = "Invalid diff: now finding less tool calls!\n";
        err += "  Previous (" + std::to_string(msg_prv.tool_calls.size()) + "):\n";
        for (const auto & tc : msg_prv.tool_calls) {
            err += "    - name: '" + tc.name + "', args: '" + tc.arguments + "'\n";
        }
        err += "  Current (" + std::to_string(msg_new.tool_calls.size()) + "):\n";
        for (const auto & tc : msg_new.tool_calls) {
            err += "    - name: '" + tc.name + "', args: '" + tc.arguments + "'\n";
        }
        err += "  Current msg text content:\n" + msg_new.content + "\n";
        throw std::runtime_error(err);
    }

    if (!msg_prv.tool_calls.empty()) {
        const auto   idx  = msg_prv.tool_calls.size() - 1;
        const auto & pref = msg_prv.tool_calls[idx];
        const auto & newf = msg_new.tool_calls[idx];
        // Allow tool name to change during incremental parsing:
        // - empty -> non-empty (initial discovery)
        // - prefix -> longer string (name grows as more input is parsed)
        if (pref.name != newf.name && !pref.name.empty() && !newf.name.empty()) {
            // Check if one is a prefix of the other (for incremental parsing where names grow or shrink)
            bool is_prefix = (newf.name.rfind(pref.name, 0) == 0);
            if (!is_prefix) {
                LOG_ERR("Tool call mismatch: prev='%s' new='%s'\n", pref.name.c_str(), newf.name.c_str());
                throw std::runtime_error("Invalid diff: tool call mismatch!");
            }
        }
        const auto args_diff = string_diff(pref.arguments, newf.arguments);
        if (!args_diff.empty() || pref.id != newf.id || pref.name != newf.name) {
            auto & diff          = diffs.emplace_back();
            diff.tool_call_index = idx;
            if (pref.id != newf.id || pref.name != newf.name) {
                diff.tool_call_delta.id   = newf.id;
                diff.tool_call_delta.name = newf.name;
            }
            diff.tool_call_delta.arguments = args_diff;
        }
    }
    for (size_t idx = msg_prv.tool_calls.size(); idx < msg_new.tool_calls.size(); ++idx) {
        auto & diff          = diffs.emplace_back();
        diff.tool_call_index = idx;
        diff.tool_call_delta = msg_new.tool_calls[idx];
    }

    return diffs;
}

using chat_template_caps = jinja::caps;

struct common_chat_templates {
    bool add_bos;
    bool add_eos;
    bool has_explicit_template;  // Model had builtin template or template overridden was specified.
    std::unique_ptr<common_chat_template> template_default;  // always set (defaults to chatml)
    std::unique_ptr<common_chat_template> template_tool_use;
};

common_chat_tool_choice common_chat_tool_choice_parse_oaicompat(const std::string & tool_choice) {
    if (tool_choice == "auto") {
        return COMMON_CHAT_TOOL_CHOICE_AUTO;
    }
    if (tool_choice == "none") {
        return COMMON_CHAT_TOOL_CHOICE_NONE;
    }
    if (tool_choice == "required") {
        return COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    }
    throw std::invalid_argument("Invalid tool_choice: " + tool_choice);
}

bool common_chat_templates_support_enable_thinking(const common_chat_templates * chat_templates) {
    common_chat_templates_inputs inputs;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    common_chat_msg msg;
    msg.role    = "user";
    msg.content = "test";
    inputs.messages = { msg };
    inputs.enable_thinking = true;
    inputs.add_generation_prompt = true;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;

    auto params = common_chat_templates_apply(chat_templates, inputs);
    return params.supports_thinking;
}

std::vector<common_chat_msg> common_chat_msgs_parse_oaicompat(const json & messages) {
    std::vector<common_chat_msg> msgs;

    try {
        if (!messages.is_array()) {
            throw std::invalid_argument("Expected 'messages' to be an array, got " + messages.dump());
        }

        for (const auto & message : messages) {
            if (!message.is_object()) {
                throw std::invalid_argument("Expected 'message' to be an object, got " + message.dump());
            }

            common_chat_msg msg;
            if (!message.contains("role")) {
                throw std::invalid_argument("Missing 'role' in message: " + message.dump());
            }
            msg.role = message.at("role");

            auto has_content    = message.contains("content");
            auto has_tool_calls = message.contains("tool_calls");
            if (has_content) {
                const auto & content = message.at("content");
                if (content.is_string()) {
                    msg.content = content;
                } else if (content.is_array()) {
                    for (const auto & part : content) {
                        if (!part.contains("type")) {
                            throw std::invalid_argument("Missing content part type: " + part.dump());
                        }
                        const auto & type = part.at("type");
                        if (type != "text" && type != "media_marker") {
                            throw std::invalid_argument("Unsupported content part type: " + type.dump());
                        }
                        common_chat_msg_content_part msg_part;
                        msg_part.type = type;
                        msg_part.text = part.at("text");
                        msg.content_parts.push_back(msg_part);
                    }
                } else if (!content.is_null()) {
                    throw std::invalid_argument("Invalid 'content' type: expected string or array, got " +
                                                content.dump() +
                                                " (ref: https://github.com/ggml-org/llama.cpp/issues/8367)");
                }
            }
            if (has_tool_calls) {
                for (const auto & tool_call : message.at("tool_calls")) {
                    common_chat_tool_call tc;
                    if (!tool_call.contains("type")) {
                        throw std::invalid_argument("Missing tool call type: " + tool_call.dump());
                    }
                    const auto & type = tool_call.at("type");
                    if (type != "function") {
                        throw std::invalid_argument("Unsupported tool call type: " + tool_call.dump());
                    }
                    if (!tool_call.contains("function")) {
                        throw std::invalid_argument("Missing tool call function: " + tool_call.dump());
                    }
                    const auto & fc = tool_call.at("function");
                    if (!fc.contains("name")) {
                        throw std::invalid_argument("Missing tool call name: " + tool_call.dump());
                    }
                    tc.name           = fc.at("name");
                    const auto & args = fc.at("arguments");
                    if (args.is_string()) {
                        tc.arguments = args;
                    } else {
                        tc.arguments = args.dump();
                    }
                    if (tool_call.contains("id")) {
                        tc.id = tool_call.at("id");
                    }
                    msg.tool_calls.push_back(tc);
                }
            }
            if (!has_content && !has_tool_calls) {
                throw std::invalid_argument(
                    "Expected 'content' or 'tool_calls' (ref: https://github.com/ggml-org/llama.cpp/issues/8367 & "
                    "https://github.com/ggml-org/llama.cpp/issues/12279)");
            }
            if (message.contains("reasoning_content")) {
                msg.reasoning_content = message.at("reasoning_content");
            }
            if (message.contains("name")) {
                msg.tool_name = message.at("name");
            }
            if (message.contains("tool_call_id")) {
                msg.tool_call_id = message.at("tool_call_id");
            }

            msgs.push_back(msg);
        }
    } catch (const std::exception & e) {
        // @ngxson : disable otherwise it's bloating the API response
        // printf("%s\n", std::string("; messages = ") + messages.dump(2));
        throw std::runtime_error("Failed to parse messages: " + std::string(e.what()));
    }

    return msgs;
}

struct messages_inp_normalizer {
    const jinja::caps & caps;

    messages_inp_normalizer(const jinja::caps & c) : caps(c) {}

    // handle supports_string_content / supports_typed_content
    // if string=true and array=false, convert array to string
    // if string=false and array=true, convert string to array
    // if both are true, do nothing
    json normalize(const json & messages) {
        bool only_string = caps.supports_string_content && !caps.supports_typed_content;
        bool only_typed  = !caps.supports_string_content && caps.supports_typed_content;
        if ((!only_string && !only_typed) || !messages.is_array()) {
            return messages;
        }
        json normalized = json::array();
        for (const auto & msg : messages) {
            json copy = msg;
            if (copy.contains("content")) {
                json & it = copy.at("content");
                if (only_typed && it.is_string()) {
                    it = json::array({
                        json{
                            {"type", "text"},
                            {"text", it.get<std::string>()},
                        }
                    });
                } else if (only_string && it.is_array()) {
                    it = concat_content_parts(it);
                }
            }
            normalized.push_back(std::move(copy));
        }
        return normalized;
    }

    // join parts with newline, do not add newline before or after media markers
    static std::string concat_content_parts(const json & parts) {
        std::string text;
        bool last_was_media_marker = false;
        for (const auto & part : parts) {
            std::string type = part.value("type", "");
            bool add_new_line = true;
            if (type == "text") {
                add_new_line = !last_was_media_marker && !text.empty();
                last_was_media_marker = false;
            } else if (type == "media_marker") {
                add_new_line = false;
                last_was_media_marker = true;
            } else {
                LOG_WRN("Ignoring content part type: %s\n", type.c_str());
                continue;
            }

            if (add_new_line) {
                text += '\n';
            }

            text += part.value("text", "");
        }
        return text;
    }
};

static json render_message_to_json(const std::vector<common_chat_msg> & msgs, const jinja::caps & c) {
    if (!c.supports_string_content && !c.supports_typed_content) {
        LOG_WRN("%s: Neither string content nor typed content is supported by the template. This is unexpected and may lead to issues.\n", __func__);
    }

    json messages = json::array();
    for (const auto & msg : msgs) {
        messages.push_back(msg.to_json_oaicompat(/* concat_typed_text= */ false));
    }
    return messages_inp_normalizer(c).normalize(messages);
}

// DEPRECATED: only used in tests
json common_chat_msgs_to_json_oaicompat(const std::vector<common_chat_msg> & msgs, bool concat_typed_text) {
    jinja::caps c;
    c.supports_string_content = true;
    c.supports_typed_content = !concat_typed_text;
    return render_message_to_json(msgs, c);
}

json common_chat_tools_to_json_oaicompat(const std::vector<common_chat_tool> & tools) {
    if (tools.empty()) {
        return json();
    }

    auto result = json::array();
    for (const auto & tool : tools) {
        result.push_back({
            { "type",     "function" },
            { "function", {
                { "name", tool.name },
                { "description", tool.description },
                { "parameters", json::parse(tool.parameters) },
            }},
        });
    }
    return result;
}

json common_chat_tool_parameters(const json & function) {
    if (function.contains("parameters")) {
        const auto & params = function.at("parameters");
        if (!params.is_null() && !(params.is_object() && params.empty())) {
            return params;
        }
    }
    return json{{"type", "object"}, {"properties", json::object()}};
}

std::vector<common_chat_tool> common_chat_tools_parse_oaicompat(const json & tools) {
    std::vector<common_chat_tool> result;

    try {
        if (!tools.is_null()) {
            if (!tools.is_array()) {
                throw std::invalid_argument("Expected 'tools' to be an array, got " + tools.dump());
            }
            for (const auto & tool : tools) {
                if (!tool.contains("type")) {
                    throw std::invalid_argument("Missing tool type: " + tool.dump());
                }
                const auto & type = tool.at("type");
                if (!type.is_string() || type != "function") {
                    throw std::invalid_argument("Unsupported tool type: " + tool.dump());
                }
                if (!tool.contains("function")) {
                    throw std::invalid_argument("Missing tool function: " + tool.dump());
                }

                const auto & function = tool.at("function");
                result.push_back({
                    /* .name = */ function.at("name"),
                    /* .description = */ function.value("description", ""),
                    /* .parameters = */ function.value("parameters", json::object()).dump(),
                });
            }
        }
    } catch (const std::exception & e) {
        throw std::runtime_error("Failed to parse tools: " + std::string(e.what()) + "; tools = " + tools.dump(2));
    }

    return result;
}

common_chat_continuation common_chat_continuation_parse(const common_json & value) {
    if (value.is_null()) {
        return COMMON_CHAT_CONTINUATION_NONE;
    }
    if (value.is_boolean()) {
        return value.get<bool>() ? COMMON_CHAT_CONTINUATION_AUTO : COMMON_CHAT_CONTINUATION_NONE;
    }
    if (value.is_string()) {
        auto value_str = value.get<std::string>();
        if (value_str == "reasoning_content") {
            return COMMON_CHAT_CONTINUATION_REASONING;
        }
        if (value_str == "content") {
            return COMMON_CHAT_CONTINUATION_CONTENT;
        }
    }
    throw std::invalid_argument("Invalid continue_final_message: expected a boolean, \"content\" or \"reasoning_content\"");
}

bool common_chat_verify_template(const std::string & tmpl, bool use_jinja) {
    if (use_jinja) {
        try {
            common_chat_msg msg;
            msg.role    = "user";
            msg.content = "test";

            auto tmpls = common_chat_templates_init(/* model= */ nullptr, tmpl);

            common_chat_templates_inputs inputs;
            inputs.messages = { msg };

            common_chat_templates_apply(tmpls.get(), inputs);
            return true;
        } catch (const std::exception & e) {
            LOG_ERR("%s: failed to apply template: %s\n", __func__, e.what());
            return false;
        }
    }
    llama_chat_message chat[] = {
        { "user", "test" }
    };
    const int res = llama_chat_apply_template(tmpl.c_str(), chat, 1, true, nullptr, 0);
    return res >= 0;
}

std::string common_chat_format_single(const struct common_chat_templates * tmpls,
                                      const std::vector<common_chat_msg> & past_msg,
                                      const common_chat_msg &              new_msg,
                                      bool                                 add_ass,
                                      bool                                 use_jinja) {
    common_chat_templates_inputs inputs;
    inputs.use_jinja = use_jinja;
    inputs.add_bos   = tmpls->add_bos;
    inputs.add_eos   = tmpls->add_eos;

    std::string fmt_past_msg;
    if (!past_msg.empty()) {
        inputs.messages              = past_msg;
        inputs.add_generation_prompt = false;
        fmt_past_msg                 = common_chat_templates_apply(tmpls, inputs).prompt.str();
    }
    std::ostringstream ss;
    // if the past_msg ends with a newline, we must preserve it in the formatted version
    if (add_ass && !fmt_past_msg.empty() && fmt_past_msg.back() == '\n') {
        ss << "\n";
    };
    // format chat with new_msg
    inputs.messages.push_back(new_msg);
    inputs.add_generation_prompt = add_ass;
    auto fmt_new_msg             = common_chat_templates_apply(tmpls, inputs).prompt.str();
    // get the diff part
    ss << fmt_new_msg.substr(fmt_past_msg.size(), fmt_new_msg.size() - fmt_past_msg.size());
    return ss.str();
}

std::string common_chat_format_example(const struct common_chat_templates *       tmpls,
                                       bool                                       use_jinja,
                                       const std::map<std::string, std::string> & chat_template_kwargs) {
    common_chat_templates_inputs inputs;
    inputs.use_jinja            = use_jinja;
    inputs.add_bos              = tmpls->add_bos;
    inputs.add_eos              = tmpls->add_eos;
    inputs.chat_template_kwargs = chat_template_kwargs;
    auto add_simple_msg         = [&](auto role, auto content) {
        common_chat_msg msg;
        msg.role    = role;
        msg.content = content;
        inputs.messages.push_back(msg);
    };
    add_simple_msg("system", "You are a helpful assistant");
    add_simple_msg("user", "Hello");
    add_simple_msg("assistant", "Hi there");
    add_simple_msg("user", "How are you?");
    return common_chat_templates_apply(tmpls, inputs).prompt.str();
}

#define CHATML_TEMPLATE_SRC                                                               \
    "{%- for message in messages -%}\n"                                                   \
    "  {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' -}}\n" \
    "{%- endfor -%}\n"                                                                    \
    "{%- if add_generation_prompt -%}\n"                                                  \
    "  {{- '<|im_start|>assistant\n' -}}\n"                                               \
    "{%- endif -%}"

void common_chat_templates_free(struct common_chat_templates * tmpls) {
    delete tmpls;
}

bool common_chat_templates_was_explicit(const struct common_chat_templates * tmpls) {
    return tmpls->has_explicit_template;
}

common_chat_prompt_preset common_chat_get_asr_prompt(const common_chat_templates * chat_templates) {
    common_chat_prompt_preset asr_preset;
    asr_preset.system = "";
    asr_preset.user   = "Transcribe audio to text";

    if (chat_templates && chat_templates->template_default && is_lfm2_template(chat_templates->template_default->source())) {
        asr_preset.system = "Perform ASR.";
        asr_preset.user   = "";
    }

    return asr_preset;
}

std::string common_chat_templates_source(const struct common_chat_templates * tmpls, const std::string & variant) {
    if (!variant.empty()) {
        if (variant == "tool_use") {
            if (tmpls->template_tool_use) {
                return tmpls->template_tool_use->source();
            }
            return "";
        }
        LOG_DBG("%s: unknown template variant: %s\n", __func__, variant.c_str());
    }
    return tmpls->template_default->source();
}

common_chat_templates_ptr common_chat_templates_init(const struct llama_model * model,
                                                     const std::string &        chat_template_override,
                                                     const std::string &        bos_token_override,
                                                     const std::string &        eos_token_override) {
    std::string default_template_src;
    std::string template_tool_use_src;

    bool has_explicit_template = !chat_template_override.empty();
    if (chat_template_override.empty()) {
        GGML_ASSERT(model != nullptr);
        const auto * str = llama_model_chat_template(model, /* name */ nullptr);
        if (str) {
            default_template_src  = str;
            has_explicit_template = true;
        }
        str = llama_model_chat_template(model, /* name */ "tool_use");
        if (str) {
            template_tool_use_src = str;
            has_explicit_template = true;
        }
    } else {
        default_template_src = chat_template_override;
    }
    if (default_template_src.empty() || default_template_src == "chatml") {
        if (!template_tool_use_src.empty()) {
            default_template_src = template_tool_use_src;
        } else {
            default_template_src = CHATML_TEMPLATE_SRC;
        }
    }

    // TODO @ngxson : this is a temporary hack to prevent chat template from throwing an error
    // Ref: https://github.com/ggml-org/llama.cpp/pull/15230#issuecomment-3173959633
    if (default_template_src.find("<|channel|>") != std::string::npos
        // search for the error message and patch it
        && default_template_src.find("in message.content or") != std::string::npos) {
        string_replace_all(default_template_src,
                           "{%- if \"<|channel|>analysis<|message|>\" in message.content or "
                           "\"<|channel|>final<|message|>\" in message.content %}",
                           "{%- if false %}");
    }

    // TODO @aldehir : this is a temporary fix, pending Minja changes
    // Ref: https://github.com/ggml-org/llama.cpp/pull/17713#issuecomment-3631342664
    if (default_template_src.find("[TOOL_CALLS]") != std::string::npos
        // search for the error message and patch it
        && default_template_src.find("if (message['content'] is none or") != std::string::npos) {
        string_replace_all(default_template_src,
                           "{%- if (message['content'] is none or message['content'] == '' or "
                           "message['content']|length == 0) and (message['tool_calls'] is not defined or "
                           "message['tool_calls'] is none or message['tool_calls']|length == 0) %}",
                           "{%- if false %}");
    }

    std::string token_bos = bos_token_override;
    std::string token_eos = eos_token_override;
    bool        add_bos   = false;
    bool        add_eos   = false;
    if (model) {
        const auto * vocab     = llama_model_get_vocab(model);
        const auto   get_token = [&](llama_token token, const char * name, const char * jinja_variable_name) {
            if (token == LLAMA_TOKEN_NULL) {
                if (default_template_src.find(jinja_variable_name) != std::string::npos ||
                    template_tool_use_src.find(jinja_variable_name) != std::string::npos) {
                    LOG_WRN(
                        "common_chat_templates_init: warning: vocab does not have a %s token, jinja template won't "
                          "work as intended.\n",
                        name);
                }
                return std::string();
            }
            return common_token_to_piece(vocab, token, true);
        };
        token_bos = get_token(llama_vocab_bos(vocab), "BOS", "bos_token");
        token_eos = get_token(llama_vocab_eos(vocab), "EOS", "eos_token");
        add_bos   = llama_vocab_get_add_bos(vocab);
        add_eos   = llama_vocab_get_add_eos(vocab);
    }
    common_chat_templates_ptr tmpls(new common_chat_templates());
    tmpls->has_explicit_template = has_explicit_template;
    tmpls->add_bos               = add_bos;
    tmpls->add_eos               = add_eos;
    try {
        tmpls->template_default = std::make_unique<common_chat_template>(default_template_src, token_bos, token_eos);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        LOG_ERR("%s: failed to initialize chat template\n", __func__);
        LOG_ERR("%s: please consider disabling jinja via --no-jinja, or using another chat template\n", __func__);
        throw e;
    }
    if (!template_tool_use_src.empty()) {
        try {
            tmpls->template_tool_use = std::make_unique<common_chat_template>(template_tool_use_src, token_bos, token_eos);
        } catch (const std::exception & e) {
            LOG_ERR("%s: failed to parse tool use chat template (ignoring it): %s\n", __func__, e.what());
        }
    }
    return tmpls;
}

const char * common_chat_format_name(common_chat_format format) {
    switch (format) {
        case COMMON_CHAT_FORMAT_CONTENT_ONLY:
            return "Content-only";
        case COMMON_CHAT_FORMAT_PEG_SIMPLE:
            return "peg-simple";
        case COMMON_CHAT_FORMAT_PEG_NATIVE:
            return "peg-native";
        case COMMON_CHAT_FORMAT_PEG_GEMMA4:
            return "peg-gemma4";
        case COMMON_CHAT_FORMAT_PEG_MINIMAX_M3:
            return "peg-minimax-m3";
        default:
            throw std::runtime_error("Unknown chat format");
    }
}

const char * common_reasoning_format_name(common_reasoning_format format) {
    switch (format) {
        case COMMON_REASONING_FORMAT_NONE:
            return "none";
        case COMMON_REASONING_FORMAT_AUTO:
            return "auto";
        case COMMON_REASONING_FORMAT_DEEPSEEK:
            return "deepseek";
        case COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY:
            return "deepseek-legacy";
        default:
            throw std::runtime_error("Unknown reasoning format");
    }
}

common_reasoning_format common_reasoning_format_from_name(const std::string & format) {
    if (format == "none") {
        return COMMON_REASONING_FORMAT_NONE;
    }
    if (format == "auto") {
        return COMMON_REASONING_FORMAT_AUTO;
    }
    if (format == "deepseek") {
        return COMMON_REASONING_FORMAT_DEEPSEEK;
    }
    if (format == "deepseek-legacy") {
        return COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY;
    }
    throw std::runtime_error("Unknown reasoning format: " + format);
}

common_chat_prompt common_chat_template_direct_apply_impl(
    const common_chat_template & tmpl,
    const autoparser::generation_params & inputs,
    const std::optional<json> & messages_override,
    const std::optional<json> & tools_override,
    const std::optional<json> & additional_context) {
    jinja::context ctx(tmpl.source());

    // messages_override is already built for this template, do not touch its content parts
    json inp = json{
        {"messages", messages_override.has_value()
            ? *messages_override
            : messages_inp_normalizer(tmpl.original_caps()).normalize(inputs.messages)},
        {"bos_token", tmpl.bos_token()},
        {"eos_token", tmpl.eos_token()},
        {"enable_thinking", inputs.enable_thinking},
    };
    if (tools_override.has_value() || !inputs.tools.empty()) {
        inp["tools"] = tools_override.has_value() ? *tools_override : inputs.tools;
    }
    if (inputs.extra_context.is_object()) {
        // TODO: do we need to merge, or replacing is fine?
        for (const auto & [k, v] : inputs.extra_context.items()) {
            inp[k] = v;
        }
    }
    if (additional_context.has_value()) {
        // TODO: merge properly instead of overwriting (matching old behavior)
        for (const auto & [k, v] : additional_context->items()) {
            inp[k] = v;
        }
    }
    if (inputs.add_generation_prompt) {
        inp["add_generation_prompt"] = true;
    }
    if (inp.contains("preserve_reasoning") && inp["preserve_reasoning"].is_boolean()) {
        bool enabled = inp["preserve_reasoning"].get<bool>();
        jinja::caps_apply_preserve_reasoning(ctx, enabled);
    }
    if (inp.contains("reasoning_effort") && inp["reasoning_effort"].is_string() && !inp["reasoning_effort"].empty()) {
        std::string reasoning_effort = inp["reasoning_effort"].get<std::string>();
        jinja::caps_apply_reasoning_effort(ctx, reasoning_effort);
    }

    jinja::global_from_json(ctx, inp, inputs.mark_input);

    // bos/eos come from the model, not from the user
    ctx.set_val("bos_token", jinja::mk_val<jinja::value_string>(tmpl.bos_token()));
    ctx.set_val("eos_token", jinja::mk_val<jinja::value_string>(tmpl.eos_token()));

    // render
    jinja::runtime runtime(ctx);
    const jinja::value results = runtime.execute(tmpl.prog);
    auto parts = jinja::runtime::gather_string_parts(results);

    common_chat_prompt result;
    for (auto & part : parts->val_str.parts) {
        result.parts.push_back({std::move(part.val), part.is_input});
    }

    // TODO: improve this later
    if (inputs.add_bos && !result.parts.empty()) {
        auto & first = result.parts.front();
        if (!first.is_input && string_starts_with(first.text, tmpl.bos_token())) {
            first.text.erase(0, tmpl.bos_token().size());
        }
    }
    if (inputs.add_eos && !result.parts.empty()) {
        auto & last = result.parts.back();
        if (!last.is_input && string_ends_with(last.text, tmpl.eos_token())) {
            last.text.resize(last.text.size() - tmpl.eos_token().size());
        }
    }
    return result;
}

common_chat_prompt common_chat_template_direct_apply(
    const common_chat_template & tmpl,
    const autoparser::generation_params & inputs) {
    return common_chat_template_direct_apply_impl(tmpl, inputs, std::nullopt, std::nullopt, std::nullopt);
}

std::string common_chat_template_generation_prompt_impl(
    const common_chat_template & tmpl,
    const autoparser::generation_params & inputs,
    const std::optional<json> & messages_override,
    const std::optional<json> & tools_override,
    const std::optional<json> & additional_context) {

    autoparser::generation_params params = inputs;
    params.add_generation_prompt = false;
    params.continue_final_message = COMMON_CHAT_CONTINUATION_NONE;
    std::string no_gen_prompt    = common_chat_template_direct_apply_impl(tmpl, params, messages_override, tools_override, additional_context).str();
    params.add_generation_prompt = true;
    std::string gen_prompt       = common_chat_template_direct_apply_impl(tmpl, params, messages_override, tools_override, additional_context).str();

    size_t prefix_len = 0;
    size_t min_size = std::min(no_gen_prompt.size(), gen_prompt.size());
    while (prefix_len < min_size && no_gen_prompt[prefix_len] == gen_prompt[prefix_len]) {
        prefix_len++;
    }
    return gen_prompt.substr(prefix_len);
}

std::string common_chat_template_generation_prompt(
    const common_chat_template & tmpl,
    const autoparser::generation_params & inputs) {
    return common_chat_template_generation_prompt_impl(tmpl, inputs, std::nullopt, std::nullopt, std::nullopt);
}

namespace workaround {

static void map_developer_role_to_system(json & messages) {
    for (auto & message : messages) {
        if (message.contains("role")) {
            if (message["role"] == "developer") {
                message["role"] = "system";
            }
        }
    }
}


// if first message is system and template does not support it, merge it with next message
static void system_message_not_supported(json & messages) {
    if (!messages.empty() && messages.front().at("role") == "system") {
        if (messages.size() > 1) {
            LOG_DBG("Merging system prompt into next message\n");
            auto & first_msg = messages.front();
            auto & second_msg = messages[1];
            second_msg["content"] = first_msg.at("content").get<std::string>()
                + "\n" + second_msg.at("content").get<std::string>();
            messages.erase(0);
        } else {
            LOG_WRN("Removing system prompt due to template not supporting system role\n");
            messages.erase(0);
        }
    }
}

static void requires_non_null_content(json & messages) {
    GGML_ASSERT(messages.is_array());
    for (auto & message : messages) {
        if (message.contains("tool_calls") && !message.contains("content")) {
            message["content"] = "";
        }
    }
}

static void func_args_not_string(json & messages) {
    GGML_ASSERT(messages.is_array());
    for (auto & message : messages) {
        if (message.contains("tool_calls")) {
            for (auto & tool_call : message["tool_calls"]) {
                if (tool_call.contains("function") && tool_call["function"].contains("arguments")) {
                    auto & args = tool_call["function"]["arguments"];
                    if (args.is_string()) {
                        try {
                            args = json::parse(args.get<std::string>());
                        } catch (const std::exception & e) {
                            throw std::runtime_error("Failed to parse tool call arguments as JSON: " + std::string(e.what()));
                        }
                    }
                }
            }
        }
    }
}

// Trim leading/trailing whitespace from message contents before rendering. This
// has to run on the messages (not on the rendered JSON) because templates with
// string-only content caps concatenate typed content parts into a single string
// during rendering, after which the per-part whitespace can no longer be reached.
// Both the plain string content and the text of typed content parts are trimmed.
static void trim_all_content(std::vector<common_chat_msg> & messages) {
    for (auto & message : messages) {
        message.content           = trim_whitespace(message.content);
        message.reasoning_content = trim_whitespace(message.reasoning_content);
        for (auto & part : message.content_parts) {
            if (part.type == "text") {
                part.text = trim_whitespace(part.text);
            }
        }
    }
}

}

static json common_chat_extra_context() {
    json ctx = json::object();
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    std::string datetime_str = format_time(now, "%b %d %Y");
    std::string date_str = format_time(now, "%d %b %Y");
    ctx["datetime"] = datetime_str;
    ctx["date_string"] = date_str;
    return ctx;
}

static common_chat_params common_chat_params_init_lfm2_tokens(const common_chat_template & tmpl, const autoparser::generation_params & inputs) {
    return common_chat_params_init_lfm2(tmpl, inputs, /* tool_list_tokens = */ true);
}

static common_chat_params common_chat_params_init_lfm2_5(const common_chat_template & tmpl, const autoparser::generation_params & inputs) {
    return common_chat_params_init_lfm2(tmpl, inputs, /* tool_list_tokens = */ false);
}

// Older gemma4 templates need their tool responses rewritten before rendering
static common_chat_params common_chat_params_init_gemma4_legacy(const common_chat_template & tmpl, const autoparser::generation_params & inputs) {
    auto adjusted = inputs;
    workaround::convert_tool_responses_gemma4(adjusted.messages);
    return common_chat_params_init_gemma4(tmpl, adjusted);
}

// Pick the dedicated handler for a template from its source, or null for the autoparser.
// Order matters: the first match wins, and later checks assume the earlier ones did not match.
static common_chat_params_init_fn common_chat_template_detect_params_init(const std::string & src) {
    // Ministral/Mistral Large 3 - uses special reasoning structure fixes, can't use autoparser
    // Note: Mistral Small 3.2 uses [CALL_ID] which Ministral doesn't have, so we can distinguish them
    if (src.find("[SYSTEM_PROMPT]") != std::string::npos && src.find("[TOOL_CALLS]") != std::string::npos &&
        src.find("[ARGS]") != std::string::npos && src.find("[CALL_ID]") == std::string::npos) {
        LOG_DBG("Using specialized template: Ministral/Magistral Large 3\n");
        return common_chat_params_init_ministral_3;
    }

    // LLM-jp-4.1 - GPT-OSS dialect (spaces after special tokens, <|end|>-separated parallel calls)
    if (src.find("chat_format=llm-jp-harmony-v1") != std::string::npos) {
        LOG_DBG("Using specialized template: LLM-jp Harmony v1\n");
        return common_chat_params_init_llm_jp_harmony;
    }

    // GPT-OSS - has unique channel-based structure that needs dedicated handler
    if (src.find("<|channel|>") != std::string::npos) {
        LOG_DBG("Using specialized template: GPT-OSS\n");
        return common_chat_params_init_gpt_oss;
    }

    // Muse Glimmer format using " to=<recipient>" recipients and <|eom|>/<|eot|> message terminators.
    if (src.find("<atem:function_calls>") != std::string::npos && src.find("<|eom|>") != std::string::npos) {
        LOG_DBG("Using specialized template: Muse Glimmer\n");
        return common_chat_params_init_muse_glimmer;
    }

    // Functionary v3.2 - uses recipient-based format with >>>recipient\n{content}
    // Detection: template has ">>>all" for content and ">>>" prefix for tool calls
    if (src.find(">>>all") != std::string::npos && src.find(">>>${recipient}") != std::string::npos) {
        LOG_DBG("Using specialized template: Functionary v3.2\n");
        return common_chat_params_init_functionary_v3_2;
    }

    // Kimi K2 Thinking - uses unique tool call ID format: functions.<name>:<index>
    // Detection: template has "<|tool_calls_section_begin|>" and "functions." prefix in tool call IDs
    if (src.find("<|tool_calls_section_begin|>") != std::string::npos &&
        src.find("<|tool_call_begin|>") != std::string::npos) {
        LOG_DBG("Using specialized template: Kimi K2 Thinking\n");
        return common_chat_params_init_kimi_k2;
    }

    // Kimi K3 - the <|open|>/<|close|>/<|end_of_msg|> markers are unique to it
    if (src.find("<|open|>") != std::string::npos && src.find("<|close|>") != std::string::npos &&
        src.find("<|end_of_msg|>") != std::string::npos) {
        LOG_DBG("Using specialized template: Kimi K3\n");
        return common_chat_params_init_kimi_k3;
    }

    // K2 Horizon - <|ifm|im_start|> turns, <ifm|think*> reasoning picked by reasoning_effort and
    // <ifm|tool_calls> sections; the three think tag pairs defeat the autoparser's reasoning detection
    if (src.find("<|ifm|im_start|>") != std::string::npos &&
        src.find("<ifm|tool_calls>") != std::string::npos) {
        LOG_DBG("Using specialized template: K2 Horizon\n");
        return common_chat_params_init_k2_horizon;
    }

    // Ling 3.0 / Bailing V3 - <role>X</role> sections with <arg_key>/<arg_value> tagged
    // tool calls. <role> sections are unique to this family among the tagged-arg templates.
    if (src.find("<role>ASSISTANT</role>") != std::string::npos &&
        src.find("<arg_key>") != std::string::npos) {
        LOG_DBG("Using specialized template: Ling 3.0 (Bailing V3)\n");
        return common_chat_params_init_ling3;
    }

    // Cohere2 MoE / North Code - marker-wrapped format with <|START_TEXT|> content and
    // <|START_ACTION|> JSON tool calls. <|START_TEXT|> is unique to this template (the older
    // Command-R templates use <|START_RESPONSE|>).
    if (src.find("<|START_TEXT|>") != std::string::npos &&
        src.find("<|START_ACTION|>") != std::string::npos) {
        LOG_DBG("Using specialized template: Cohere2 MoE\n");
        return common_chat_params_init_cohere2moe;
    }

    if (is_lfm2_template(src)) {
        LOG_DBG("Using specialized template: LFM2\n");
        return common_chat_params_init_lfm2_tokens;
    }

    // LFM2.5 format detection: template uses plain "List of tools: [...]" with no special tokens
    if (src.find("List of tools: [") != std::string::npos &&
        src.find("<|tool_list_start|>") == std::string::npos) {
        LOG_DBG("Using specialized template: LFM2.5\n");
        return common_chat_params_init_lfm2_5;
    }

    // GigaChatV3 format detection
    if (src.find("<|role_sep|>") != std::string::npos &&
        src.find("<|message_sep|>") != std::string::npos &&
        src.find("<|function_call|>") == std::string::npos) {
        LOG_DBG("Using specialized template: GigaChatV3\n");
        return common_chat_params_init_gigachat_v3;
    }

    // MiniMax-M3: the namespace token "]<]minimax[>[" collides with the autoparser's
    // markup delimiters, so detect the template and use a dedicated parser.
    if (src.find("]<]minimax[>[") != std::string::npos &&
        src.find("<tool_call>") != std::string::npos &&
        src.find("<invoke name=") != std::string::npos) {
        LOG_DBG("Using specialized template: MiniMax-M3\n");
        return common_chat_params_init_minimax_m3;
    }

    // DeepSeek V3.2/V4 format detection: template defines dsml_token and uses it for tool calls.
    // The template source contains the token as a variable assignment, not as a literal in markup.
    // V3.2 names the tool call block "function_calls", V4 names it "tool_calls".
    if (src.find("dsml_token") != std::string::npos &&
        src.find("DSML") != std::string::npos &&
        (src.find("function_calls") != std::string::npos ||
         src.find("tool_calls") != std::string::npos)) {
        LOG_DBG("Using specialized template: DeepSeek V3.2/V4\n");
        return common_chat_params_init_deepseek_v3_2;
    }

    // Gemma4 format detection
    if (src.find("'<|tool_call>call:'") != std::string::npos) {
        LOG_DBG("Using specialized template: Gemma4\n");
        if (src.find("{#- OpenAI Chat Completions:") == std::string::npos) {
            LOG_WRN("%s: detected an outdated gemma4 chat template, applying compatibility workarounds. "
                    "Consider updating to the official template.\n", __func__);
            return common_chat_params_init_gemma4_legacy;
        }
        return common_chat_params_init_gemma4;
    }

    // MiniCPM5 - XML tool calls with <function name="..."><param name="...">...</param></function>
    if (src.find("Tool usage guidelines:") != std::string::npos &&
        src.find("<function name=\"") != std::string::npos &&
        src.find("<param name=\"") != std::string::npos) {
        LOG_DBG("Using specialized template: MiniCPM5\n");
        return common_chat_params_init_minicpm5;
    }

    // TranslateGemma - user content must follow a custom schema with language codes
    if (src.find("[source_lang_code]") != std::string::npos &&
        src.find("[target_lang_code]") != std::string::npos) {
        LOG_DBG("Using specialized template: TranslateGemma\n");
        return common_chat_params_init_translate_gemma;
    }

    // Qwen3-Coder XML tool calls, also used by Nemotron Nano 3, Qwen3.5 and StepFun-3.5-Flash
    if (src.find("<tool_call>") != std::string::npos &&
        src.find("<function=") != std::string::npos &&
        src.find("<parameter=") != std::string::npos &&
        // Exclude models that don't use \n between tags
        src.find("'<tool_call><function=' ~ tool_call.name ~ '>'") == std::string::npos) {
        LOG_DBG("Using specialized template: Qwen3-Coder\n");
        return common_chat_params_init_qwen3_coder;
    }

    return nullptr;
}

common_chat_template::common_chat_template(const std::string & src, const std::string & bos_token, const std::string & eos_token) {
    jinja::lexer lexer;
    auto lexer_res = lexer.tokenize(src);
    this->prog = jinja::parse_from_tokens(lexer_res);

    this->src = lexer_res.source;
    this->bos_tok = bos_token;
    this->eos_tok = eos_token;

    this->caps = jinja::caps_get(prog);
    // LOG_INF("%s: caps:\n%s\n", __func__, this->caps.to_string().c_str());

    this->params_init = common_chat_template_detect_params_init(this->src);
    if (this->params_init) {
        return;
    }

    // The analysis depends only on the template, so run it once here instead of on every apply.
    // A failure is kept for apply to report, so a bad template still loads like it did before.
    try {
        analysis = std::make_unique<autoparser::autoparser>();
        analysis->analyze_template(*this);
    } catch (const std::exception & e) {
        analysis.reset();
        analysis_error = e.what();
    }
}

common_chat_template::~common_chat_template() = default;
common_chat_template::common_chat_template(common_chat_template &&) = default;
common_chat_template & common_chat_template::operator=(common_chat_template &&) = default;

std::optional<common_chat_params> common_chat_try_specialized_template(
        const common_chat_template &          tmpl,
        const autoparser::generation_params & params) {
    if (!tmpl.params_init) {
        return std::nullopt;
    }
    return tmpl.params_init(tmpl, params);
}

static common_chat_params common_chat_templates_apply_jinja(const struct common_chat_templates *        tmpls,
                                                            const struct common_chat_templates_inputs & inputs) {
    autoparser::generation_params params;
    params.tools = common_chat_tools_to_json_oaicompat(inputs.tools);
    const auto & tmpl =
        params.tools.is_array() && tmpls->template_tool_use ? *tmpls->template_tool_use : *tmpls->template_default;
    const auto & src             = tmpl.source();
    const auto & caps            = tmpl.original_caps();
    std::vector<common_chat_msg>        trimmed_messages;
    const std::vector<common_chat_msg> * messages_to_render = &inputs.messages;
    if (src.find("You have access to the following functions in JSONSchema format") != std::string::npos) {
        // StepFun: trim message contents (including typed content parts) before rendering,
        // otherwise leftover whitespace drives the model into reasoning loops (issue #24181)
        trimmed_messages   = inputs.messages;
        workaround::trim_all_content(trimmed_messages);
        messages_to_render = &trimmed_messages;
    }
    params.messages              = render_message_to_json(*messages_to_render, tmpl.original_caps());
    params.tool_choice           = inputs.tool_choice;
    params.reasoning_format      = inputs.reasoning_format;
    params.enable_thinking       = inputs.enable_thinking;
    params.grammar               = inputs.grammar;
    params.now                   = inputs.now;
    params.add_generation_prompt = inputs.add_generation_prompt;
    params.add_bos               = tmpls->add_bos;
    params.add_eos               = tmpls->add_eos;

    params.continue_final_message = inputs.continue_final_message;
    if (params.continue_final_message != COMMON_CHAT_CONTINUATION_NONE) {
        params.add_generation_prompt = false;

        if (!inputs.messages.empty()) {
            // Render messages[:-1] and store continuation message separately
            params.continue_msg = inputs.messages.back();
            params.messages.erase(params.messages.size() - 1);
        }

        if (params.continue_final_message == COMMON_CHAT_CONTINUATION_AUTO && !inputs.messages.empty()) {
            // Resolve based on message content
            params.continue_final_message = COMMON_CHAT_CONTINUATION_CONTENT;
            if (!params.continue_msg.reasoning_content.empty() &&
                params.continue_msg.content.empty() &&
                params.continue_msg.content_parts.empty()) {
                params.continue_final_message = COMMON_CHAT_CONTINUATION_REASONING;
            }
        }
    }

    if (src.find("<|channel|>") == std::string::npos) {
        // map developer to system for all models except for GPT-OSS
        workaround::map_developer_role_to_system(params.messages);
    }

    if (!tmpl.original_caps().supports_system_role) {
        workaround::system_message_not_supported(params.messages);
    }

    if (tmpl.original_caps().supports_tool_calls) {
        // some templates will require the content field in tool call messages
        // to still be non-null, this puts an empty string everywhere where the
        // content field is null
        workaround::requires_non_null_content(params.messages);
    }

    if (tmpl.original_caps().supports_object_arguments) {
        workaround::func_args_not_string(params.messages);
    }

    params.extra_context = common_chat_extra_context();
    for (auto el : inputs.chat_template_kwargs) {
        params.extra_context[el.first] = json::parse(el.second);
    }

    if (!inputs.json_schema.empty()) {
        params.json_schema = json::parse(inputs.json_schema);
    }

    params.parallel_tool_calls = inputs.parallel_tool_calls;

    if (params.tools.is_array()) {
        if (params.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE && !params.grammar.empty()) {
            throw std::runtime_error("Cannot specify grammar with tools");
        }
        if (caps.supports_tool_calls && !caps.supports_tools) {
            LOG_WRN(
                "Template supports tool calls but does not natively describe tools. The fallback behaviour used may "
                "produce bad results, inspect prompt w/ --verbose & consider overriding the template.\n");
        }
    }

    if (inputs.force_pure_content) {
        LOG_WRN("Forcing pure content template, will not render reasoning or tools separately.");
        // Create the result structure
        common_chat_params data;
        auto params_copy               = params;
        params_copy.reasoning_format   = COMMON_REASONING_FORMAT_NONE;
        data.prompt                    = common_chat_template_direct_apply_impl(tmpl, params_copy);
        data.generation_prompt         = common_chat_template_generation_prompt_impl(tmpl, params);
        data.format                    = COMMON_CHAT_FORMAT_PEG_NATIVE;
        data.parser                    = build_chat_peg_parser([&data](common_chat_peg_builder &p) {
            return p.literal(data.generation_prompt) << p.content(p.rest());
        });
        return data;
    }

    if (auto result = common_chat_try_specialized_template(tmpl, params)) {
        return *result;
    }

    if (!tmpl.analysis) {
        throw std::invalid_argument("Unable to generate parser for this template. Automatic parser generation failed: " + tmpl.analysis_error);
    }

    try {
        LOG_DBG("%s: using differential autoparser\n", __func__);
        const auto & autoparser = *tmpl.analysis;
        auto auto_params = autoparser::peg_generator::generate_parser(tmpl, params, autoparser);

        common_chat_msg_delimiters delimiters;
        if (!autoparser.assistant_start.empty()) {
            delimiters.add(COMMON_CHAT_ROLE_ASSISTANT, autoparser.assistant_start);
        }
        if (!autoparser.user_start.empty()) {
            delimiters.add(COMMON_CHAT_ROLE_USER, autoparser.user_start);
        }

        auto_params.message_delimiters = std::move(delimiters);

        auto_params.supports_thinking = autoparser.reasoning.mode != autoparser::reasoning_mode::NONE;
        if (auto_params.supports_thinking) {
            auto_params.thinking_start_tag = trim_whitespace(autoparser.reasoning.start);
            auto end_tag = trim_whitespace(autoparser.reasoning.end);
            if (!end_tag.empty()) {
                auto_params.thinking_end_tags = {std::move(end_tag)};
            }
        }
        const auto & arena = auto_params.parser;
        LOG_DBG("%s: generated parser:\n%s\n\nparser generation prompt: %s\n", __func__, arena.dump(arena.root()).c_str(), auto_params.generation_prompt.c_str());
        return auto_params;
    } catch (const std::exception & e) {
        throw std::invalid_argument(std::string("Unable to generate parser for this template. Automatic parser generation failed: ") + e.what());
    }
}

// Legacy template route (adhoc C++ implementation of known templates), forward to llama_chat_apply_template.
static common_chat_params common_chat_templates_apply_legacy(const struct common_chat_templates *        tmpls,
                                                             const struct common_chat_templates_inputs & inputs) {
    size_t                          alloc_size = 0;
    std::vector<llama_chat_message> chat;
    std::vector<std::string>        contents;

    for (const auto & msg : inputs.messages) {
        auto content = msg.content;
        for (const auto & part : msg.content_parts) {
            if (part.type != "text" && part.type != "media_marker") {
                LOG_WRN("Ignoring non-text content part: %s\n", part.type.c_str());
                continue;
            }
            if (!content.empty()) {
                content += "\n";
                ;
            }
            content += part.text;
        }
        contents.emplace_back(std::move(content));
    }
    for (size_t i = 0; i < contents.size(); ++i) {
        const auto & msg     = inputs.messages[i];
        const auto & content = contents[i];
        chat.push_back({ msg.role.c_str(), content.c_str() });
        size_t msg_size = msg.role.size() + content.size();
        alloc_size += msg_size + (msg_size / 4);  // == msg_size * 1.25 but avoiding float ops
    }

    std::vector<char> buf(alloc_size);

    // run the first time to get the total output length
    const auto & src = tmpls->template_default->source();
    int32_t      res = llama_chat_apply_template(src.c_str(), chat.data(), chat.size(), inputs.add_generation_prompt,
                                                 buf.data(), buf.size());

    // error: chat template is not supported
    if (res < 0) {
        // if the custom "tmpl" is not supported, we throw an error
        // this is a bit redundant (for good), since we're not sure if user validated the custom template with llama_chat_verify_template()
        throw std::runtime_error("this custom template is not supported, try using --jinja");
    }

    // if it turns out that our buffer is too small, we resize it
    if ((size_t) res > buf.size()) {
        buf.resize(res);
        res = llama_chat_apply_template(src.c_str(), chat.data(), chat.size(), inputs.add_generation_prompt, buf.data(),
                                        buf.size());
    }

    // for safety, we check the result again
    if (res < 0 || (size_t) res > buf.size()) {
        throw std::runtime_error("failed to apply chat template, try using --jinja");
    }

    common_chat_params params;
    params.prompt = common_chat_prompt(std::string(buf.data(), res));
    if (!inputs.json_schema.empty()) {
        params.grammar = json_schema_to_grammar(json::parse(inputs.json_schema));
    } else {
        params.grammar = inputs.grammar;
    }
    return params;
}

common_chat_params common_chat_templates_apply(const struct common_chat_templates *        tmpls,
                                               const struct common_chat_templates_inputs & inputs) {
    GGML_ASSERT(tmpls != nullptr);
    return inputs.use_jinja ? common_chat_templates_apply_jinja(tmpls, inputs) :
                              common_chat_templates_apply_legacy(tmpls, inputs);
}

std::string common_chat_prompt::str() const {
    if (parts.size() == 1) {
        return parts[0].text;
    }
    size_t len = 0;
    for (const auto & part : parts) {
        len += part.text.size();
    }
    std::string out;
    out.reserve(len);
    for (const auto & part : parts) {
        out += part.text;
    }
    return out;
}

void common_chat_prompt::append(const std::string & text, bool is_input) {
    if (text.empty()) {
        return;
    }
    if (!parts.empty() && parts.back().is_input == is_input) {
        parts.back().text += text;
    } else {
        parts.push_back({text, is_input});
    }
}

void common_chat_prompt::replace_last(const std::string & from, const std::string & to) {
    for (auto it = parts.rbegin(); it != parts.rend(); ++it) {
        if (it->is_input) {
            continue;
        }
        size_t pos = it->text.rfind(from);
        if (pos != std::string::npos) {
            it->text.replace(pos, from.size(), to);
            return;
        }
    }
}

void common_chat_input::append(const std::string & piece, llama_token token) {
    if (piece.empty()) {
        return;
    }
    tokens.push_back(token);
    tokens.resize(tokens.size() + piece.size() - 1, LLAMA_TOKEN_NULL);
    text += piece;
}

void common_chat_input::append(const common_chat_input & chunk) {
    tokens.insert(tokens.end(), chunk.tokens.begin(), chunk.tokens.end());
    text += chunk.text;
}

void common_chat_input::truncate(size_t pos) {
    if (pos < text.size()) {
        text.erase(pos);
        tokens.resize(pos);
    }
}

common_chat_input common_chat_input::substr(size_t pos, size_t n) const {
    common_chat_input out;
    out.text = text.substr(pos, n);
    out.tokens.assign(tokens.begin() + pos, tokens.begin() + pos + out.size());
    return out;
}

void common_chat_input::prepend(const std::string & prefix) {
    tokens.insert(tokens.begin(), prefix.size(), LLAMA_TOKEN_NULL);
    text = prefix + text;
}

void common_chat_input::prepend(const common_chat_input & prefix) {
    tokens.insert(tokens.begin(), prefix.tokens.begin(), prefix.tokens.end());
    text = prefix.text + text;
}

common_chat_input common_chat_input_tokenize(const llama_vocab * vocab, const std::string & text) {
    common_chat_input input;
    auto tokens = common_tokenize(vocab, text, false, true);
    for (size_t i = 0; i < tokens.size(); i++) {
        std::string piece = common_token_to_piece(vocab, tokens[i], true);
        if (i == 0 && std::isspace(piece[0]) && !std::isspace(text[0])) {
            // Some tokenizers will add a space before the first special token, need to exclude
            continue;
        }
        input.append(piece, tokens[i]);
    }
    if (input.text != text) {
        // the pieces do not give back the same text, keep the text without tokens
        return common_chat_input(text);
    }
    return input;
}

common_chat_msg common_chat_parse(const common_chat_input &         input,
                                  bool                              is_partial,
                                  const common_chat_parser_params & params) {
    return common_chat_peg_parse(params.parser, input, is_partial, params);
}

common_chat_msg common_chat_peg_parse(const common_peg_arena &          src_parser,
                                      const common_chat_input &         input,
                                      bool                              is_partial,
                                      const common_chat_parser_params & params) {
    // both branches must be lvalues, a temporary here would copy the arena on every call
    static const common_peg_arena content_only =
        build_chat_peg_parser([](common_chat_peg_builder & p) { return p.content(p.rest()) + p.end(); });
    const common_peg_arena & parser = src_parser.empty() ? content_only : src_parser;

    if (src_parser.empty()) {
        LOG_DBG("No parser definition detected, assuming pure content parser.");
    }

    common_chat_input effective_input = input;
    effective_input.prepend(params.generation_prompt);

    //LOG_DBG("Parsing PEG input with format %s: %s\n", common_chat_format_name(params.format), effective_input.text.c_str());

    common_peg_parse_flags flags = COMMON_PEG_PARSE_FLAG_LENIENT;
    if (params.debug) {
        flags |= COMMON_PEG_PARSE_FLAG_DEBUG;
    }

    common_peg_parse_context ctx(std::move(effective_input.text), std::move(effective_input.tokens), flags);
    auto result = parser.parse(ctx);

    if (result.fail()) {
        // During partial parsing, return partial results if any AST nodes were captured
        // This allows streaming to work correctly for formats like FUNC_MARKDOWN_CODE_BLOCK
        if (is_partial && result.end > 0) {
            // Try to extract any partial results from what was successfully parsed
            common_chat_msg msg;
            msg.role = "assistant";
            std::unique_ptr<common_chat_peg_mapper> mapper;
            if (params.format == COMMON_CHAT_FORMAT_PEG_GEMMA4) {
                mapper = std::make_unique<common_chat_peg_gemma4_mapper>(msg);
            } else if (params.format == COMMON_CHAT_FORMAT_PEG_MINIMAX_M3) {
                mapper = std::make_unique<common_chat_peg_minimax_m3_mapper>(msg);
            } else {
                mapper = std::make_unique<common_chat_peg_mapper>(msg);
            }
            mapper->from_ast(ctx.ast, result);

            if (ctx.is_debug()) {
                fprintf(stderr, "\nAST for partial parse (fail):\n%s\n", ctx.ast.dump().c_str());
                fflush(stderr);
            }
            return msg;
        }
        LOG_WRN("%s: unparsed %s output: %s\n", __func__, common_chat_format_name(params.format), ctx.input.substr(result.end).c_str());
        LOG_DBG("%s: full %s output triggering error:\n=== BEGIN ===\n%s\n=== END ===\n", __func__, common_chat_format_name(params.format), ctx.input.c_str());
        throw std::runtime_error(std::string("The model produced output that does not match the expected ") + common_chat_format_name(params.format) + " format");
    }

    common_chat_msg msg;
    msg.role = "assistant";

    std::unique_ptr<common_chat_peg_mapper> mapper;
    if (params.format == COMMON_CHAT_FORMAT_PEG_GEMMA4) {
        mapper = std::make_unique<common_chat_peg_gemma4_mapper>(msg);
    } else if (params.format == COMMON_CHAT_FORMAT_PEG_MINIMAX_M3) {
        mapper = std::make_unique<common_chat_peg_minimax_m3_mapper>(msg);
    } else {
        mapper = std::make_unique<common_chat_peg_mapper>(msg);
    }
    mapper->from_ast(ctx.ast, result);

    if (ctx.is_debug()) {
        fprintf(stderr, "\nAST for %s parse:\n%s\n", is_partial ? "partial" : "full", ctx.ast.dump().c_str());
        fflush(stderr);
    }

    if (!is_partial) {
        LOG_DBG("Parsed message: %s\n", common_chat_msgs_to_json_oaicompat({ msg }).at(0).dump().c_str());
    }
    return msg;
}

common_chat_session::common_chat_session(const common_chat_templates *        tmpls,
                                         const llama_vocab *                  vocab,
                                         const common_chat_templates_inputs & inputs,
                                         const common_chat_session_params &   params) {
    auto applied = common_chat_templates_apply(tmpls, inputs);

    templated    = true;
    prompt_parts = std::move(applied.prompt);
    result.role  = "assistant";

    grammar_text           = std::move(applied.grammar);
    grammar_lazy           = applied.grammar_lazy;
    stops                  = std::move(applied.additional_stops);
    generation_prompt_text = applied.generation_prompt;
    thinking_start         = std::move(applied.thinking_start_tag);
    thinking_ends          = std::move(applied.thinking_end_tags);

    parser_params.format            = applied.format;
    parser_params.generation_prompt = vocab ? common_chat_input_tokenize(vocab, applied.generation_prompt)
                                            : common_chat_input(applied.generation_prompt);
    parser_params.debug             = params.debug;
    parser_params.parser            = std::move(applied.parser);

    delimiters = std::move(applied.message_delimiters);

    if (vocab) {
        common_params_sampling resolved;
        resolved.grammar_lazy = applied.grammar_lazy;
        common_sampling_add_preserved_tokens(resolved, vocab, applied.preserved_tokens);
        common_sampling_add_grammar_triggers(resolved, vocab, std::move(applied.grammar_triggers));
        preserved_tokens = std::move(resolved.preserved_tokens);
        grammar_triggers = std::move(resolved.grammar_triggers);

        delimiters.tokenize(vocab);
    } else {
        grammar_triggers = std::move(applied.grammar_triggers);
    }

    if (inputs.continue_final_message != COMMON_CHAT_CONTINUATION_NONE && !params.echo) {
        // start from the prefill so it is not emitted as part of the first delta
        result = common_chat_parse(input, true, parser_params);
    }
}

void common_chat_session::apply_sampling(common_params_sampling & sampling) const {
    if (!templated) {
        return;
    }
    if (!grammar_text.empty()) {
        sampling.grammar = {COMMON_GRAMMAR_TYPE_TOOL_CALLS, grammar_text};
    }
    sampling.grammar_lazy      = grammar_lazy;
    sampling.generation_prompt = generation_prompt_text;
    sampling.preserved_tokens.insert(preserved_tokens.begin(), preserved_tokens.end());
    sampling.grammar_triggers.insert(sampling.grammar_triggers.end(), grammar_triggers.begin(), grammar_triggers.end());
}

const common_chat_msg & common_chat_session::feed(const common_chat_input & chunk) {
    GGML_ASSERT(!finished && "feed() after finish()");
    input.append(chunk);
    auto msg = common_chat_parse(input, true, parser_params);
    if (!msg.empty()) {
        result = std::move(msg);
    }
    return result;
}

const common_chat_msg & common_chat_session::finish(const common_chat_input & chunk) {
    GGML_ASSERT(!finished && "finish() called twice");
    finished = true;
    input.append(chunk);
    auto msg = common_chat_parse(input, false, parser_params);
    if (!msg.empty()) {
        result = std::move(msg);
    }
    return result;
}

std::map<std::string, bool> common_chat_templates_get_caps(const common_chat_templates * chat_templates) {
    GGML_ASSERT(chat_templates != nullptr);
    GGML_ASSERT(chat_templates->template_default != nullptr);
    if (chat_templates->template_tool_use != nullptr) {
        // take the more expressive template when available
        return chat_templates->template_tool_use->caps.to_map();
    }
    return chat_templates->template_default->caps.to_map();
}



===== common/chat.h =====
// Chat support (incl. tool call grammar constraining & output parsing) w/ generic & custom template handlers.

#pragma once

#include "common.h"
#include "peg-parser.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/caps.h"

#include "json.h"

#include <chrono>
#include <functional>
#include <map>
#include <string>
#include <vector>

using chat_template_caps = jinja::caps;

struct common_chat_templates;

namespace autoparser {
struct generation_params;
struct autoparser;
}  // namespace autoparser

struct common_chat_params;
struct common_chat_template;

// Builds the prompt and parser for a template that has a dedicated handler (see common/parsers)
using common_chat_params_init_fn = common_chat_params (*)(const common_chat_template & tmpl,
                                                          const autoparser::generation_params & inputs);

struct common_chat_tool_call {
    std::string name;
    std::string arguments;
    std::string id;

    bool operator==(const common_chat_tool_call & other) const {
        return name == other.name && arguments == other.arguments && id == other.id;
    }
};

struct common_chat_msg_content_part {
    std::string type;
    std::string text;

    // TODO @ngxson : no known chat templates support reasoning_content in content parts yet
    //                this can be useful for models with interleaved thinking (like Kimi-K2)
    //                if you see any templates explicitly support this, please ping me
    // std::string reasoning_content;

    bool operator==(const common_chat_msg_content_part & other) const {
        return type == other.type && text == other.text;
    }
};

struct common_chat_template {
    jinja::program prog;
    std::string bos_tok;
    std::string eos_tok;
    std::string src;
    chat_template_caps caps;
    // Dedicated handler picked once from the source, null when the differential autoparser is used
    common_chat_params_init_fn params_init = nullptr;

    // Differential analysis, run once here when there is no dedicated handler. Null when there
    // is one, or when the analysis failed, in which case analysis_error says why.
    std::unique_ptr<autoparser::autoparser> analysis;
    std::string analysis_error;

    common_chat_template(const std::string & src, const std::string & bos_token, const std::string & eos_token);

    // autoparser is incomplete here, so these are defined where it is complete
    ~common_chat_template();
    common_chat_template(common_chat_template &&);
    common_chat_template & operator=(common_chat_template &&);

    const std::string & source() const { return src; }
    const std::string & bos_token() const { return bos_tok; }
    const std::string & eos_token() const { return eos_tok; }

    chat_template_caps original_caps() const {
        return caps;
    }
};

struct common_chat_msg {
    std::string                               role;
    std::string                               content;
    std::vector<common_chat_msg_content_part> content_parts;
    std::vector<common_chat_tool_call>        tool_calls;
    std::string                               reasoning_content;
    std::string                               tool_name;
    std::string                               tool_call_id;

    common_json to_json_oaicompat(bool concat_typed_text = false) const;

    std::string render_content(const std::string & delimiter = "\n\n") const;

    bool empty() const {
        return content.empty() && content_parts.empty() && tool_calls.empty() && reasoning_content.empty() &&
               tool_name.empty() && tool_call_id.empty();
    }

    bool contains_media() const {
        for (const auto & part : content_parts) {
            if (part.type == "media_marker") {
                return true;
            }
        }
        return false;
    }

    void set_tool_call_ids(std::vector<std::string> &           ids_cache,
                           const std::function<std::string()> & gen_tool_call_id) {
        for (auto i = 0u; i < tool_calls.size(); i++) {
            if (ids_cache.size() <= i) {
                auto id = tool_calls[i].id;
                if (id.empty()) {
                    id = gen_tool_call_id();
                }
                ids_cache.push_back(id);
            }
            tool_calls[i].id = ids_cache[i];
        }
    }

    bool operator==(const common_chat_msg & other) const {
        return role == other.role && content == other.content && content_parts == other.content_parts &&
               tool_calls == other.tool_calls && reasoning_content == other.reasoning_content &&
               tool_name == other.tool_name && tool_call_id == other.tool_call_id;
    }

    bool operator!=(const common_chat_msg & other) const { return !(*this == other); }
};

struct common_chat_msg_diff {
    std::string           reasoning_content_delta;
    std::string           content_delta;
    size_t                tool_call_index = std::string::npos;
    common_chat_tool_call tool_call_delta;

    static std::vector<common_chat_msg_diff> compute_diffs(const common_chat_msg & msg_prv,
                                                           const common_chat_msg & msg_new);

    bool operator==(const common_chat_msg_diff & other) const {
        return content_delta == other.content_delta && tool_call_index == other.tool_call_index &&
               tool_call_delta == other.tool_call_delta;
    }
};

enum common_chat_role {
    COMMON_CHAT_ROLE_UNKNOWN,
    COMMON_CHAT_ROLE_SYSTEM,
    COMMON_CHAT_ROLE_ASSISTANT,
    COMMON_CHAT_ROLE_USER,
    COMMON_CHAT_ROLE_TOOL
};

common_chat_role common_chat_role_from_string(const std::string & role);
const char *     common_chat_role_to_string(common_chat_role role);

struct common_chat_msg_span {
    common_chat_role role = COMMON_CHAT_ROLE_UNKNOWN;
    std::size_t pos = 0;
    std::size_t len = 0;

    bool valid() const {
        return role != COMMON_CHAT_ROLE_UNKNOWN;
    }
};

struct common_chat_msg_spans {
    std::vector<common_chat_msg_span> spans;

    void add(common_chat_role role, size_t pos, size_t len) {
        spans.push_back({ role, pos, len });
    }

    bool is_user_start(int32_t pos) const {
        for (auto it = spans.begin(); it != spans.end(); ++it) {
            if (it->role == COMMON_CHAT_ROLE_USER && pos == (int32_t) it->pos) {
                return true;
            }
        }
        return false;
    }

    int32_t last_user_message_pos() const {
        for (auto it = spans.rbegin(); it != spans.rend(); ++it) {
            if (it->role == COMMON_CHAT_ROLE_USER) {
                return (int32_t) it->pos;
            }
        }
        return -1;
    }
};

struct common_chat_msg_delimiter {
    common_chat_role role = COMMON_CHAT_ROLE_UNKNOWN;
    std::string      delimiter;
    llama_tokens     tokens = {};
};

struct common_chat_msg_delimiters {
    std::vector<common_chat_msg_delimiter> delimiters;

    common_chat_msg_delimiters() = default;
    common_chat_msg_delimiters(std::initializer_list<common_chat_msg_delimiter> delims) : delimiters(delims) {}

    void add(common_chat_role role, const std::string & delimiter) {
        delimiters.push_back({ role, delimiter });
    }

    void tokenize(const llama_vocab * vocab);

    // split tokens into message spans. skips maps a start index to a length of a region to jump over without matching
    common_chat_msg_spans split(const llama_tokens & tokens, const std::map<size_t, size_t> & skips = {}) const;
};

struct common_chat_tool {
    std::string name;
    std::string description;
    std::string parameters;
};

enum common_chat_tool_choice {
    COMMON_CHAT_TOOL_CHOICE_AUTO,
    COMMON_CHAT_TOOL_CHOICE_REQUIRED,
    COMMON_CHAT_TOOL_CHOICE_NONE,
};

enum common_chat_format {
    COMMON_CHAT_FORMAT_CONTENT_ONLY,

    // These are intended to be parsed by the PEG parser
    COMMON_CHAT_FORMAT_PEG_SIMPLE,
    COMMON_CHAT_FORMAT_PEG_NATIVE,
    COMMON_CHAT_FORMAT_PEG_GEMMA4,
    COMMON_CHAT_FORMAT_PEG_MINIMAX_M3,

    COMMON_CHAT_FORMAT_COUNT,  // Not a format, just the # formats
};


// Continuation method provided via `continue_final_message`
enum common_chat_continuation {
    COMMON_CHAT_CONTINUATION_NONE,
    COMMON_CHAT_CONTINUATION_AUTO,
    COMMON_CHAT_CONTINUATION_REASONING,
    COMMON_CHAT_CONTINUATION_CONTENT,
};

struct common_chat_templates_inputs {
    std::vector<common_chat_msg>          messages;
    std::string                           grammar;
    std::string                           json_schema;
    bool                                  add_generation_prompt  = true;
    common_chat_continuation              continue_final_message = COMMON_CHAT_CONTINUATION_NONE;
    bool                                  use_jinja              = true;
    // Parameters below only supported when use_jinja is true
    std::vector<common_chat_tool>         tools;
    common_chat_tool_choice               tool_choice         = COMMON_CHAT_TOOL_CHOICE_AUTO;
    bool                                  parallel_tool_calls = false;
    common_reasoning_format               reasoning_format    = COMMON_REASONING_FORMAT_NONE; // TODO: refactor this to "bool enable_thinking"
    bool                                  enable_thinking     = true;
    std::chrono::system_clock::time_point now                 = std::chrono::system_clock::now();
    std::map<std::string, std::string>    chat_template_kwargs;
    bool                                  add_bos = false;
    bool                                  add_eos = false;
    bool                                  force_pure_content = false;
};

struct common_chat_prompt_part {
    std::string text;
    bool        is_input = false; // from user input, special tokens inside must not be parsed
};

// rendered chat template, split into template parts and user input parts
struct common_chat_prompt {
    std::vector<common_chat_prompt_part> parts;

    common_chat_prompt() = default;

    // single part of template text
    explicit common_chat_prompt(std::string text) { parts.push_back({std::move(text), false}); }

    std::string str() const;

    void append(const std::string & text, bool is_input = false);

    // replace the last occurrence of `from`, user input parts are skipped
    void replace_last(const std::string & from, const std::string & to);
};

struct common_chat_params {
    common_chat_format                  format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    common_chat_prompt                  prompt;
    std::string                         grammar;
    bool                                grammar_lazy         = false;
    std::string                         generation_prompt;
    bool                                supports_thinking    = false;
    std::string                         thinking_start_tag;  // e.g., "<think>"
    std::vector<std::string>            thinking_end_tags;   // e.g., "</think>"
    std::vector<common_grammar_trigger> grammar_triggers;
    std::vector<std::string>            preserved_tokens;
    std::vector<std::string>            additional_stops;
    common_peg_arena                    parser;
    common_chat_msg_delimiters          message_delimiters;
};

struct common_chat_input {
    std::string              text;
    std::vector<llama_token> tokens;

    common_chat_input() = default;

    // plain text, with no tokens
    explicit common_chat_input(std::string text) : text(std::move(text)), tokens(this->text.size(), LLAMA_TOKEN_NULL) {}

    size_t size() const { return text.size(); }
    bool empty() const { return text.empty(); }

    void append(const std::string & piece, llama_token token);
    void append(const common_chat_input & chunk);

    void prepend(const std::string & prefix);
    void prepend(const common_chat_input & prefix);

    void truncate(size_t pos);

    common_chat_input substr(size_t pos, size_t n = std::string::npos) const;
};

common_chat_input common_chat_input_tokenize(const llama_vocab * vocab, const std::string & text);

// per-message parsing syntax
// should be derived from common_chat_params
struct common_chat_parser_params {
    common_chat_format format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    common_chat_input  generation_prompt;
    bool               debug  = false; // Enable debug output for PEG parser
    common_peg_arena   parser = {};
    common_chat_parser_params() = default;
    common_chat_parser_params(const common_chat_params & chat_params) {
        format  = chat_params.format;
        generation_prompt = common_chat_input(chat_params.generation_prompt);
    }
};

// Check if the template supplied via "--chat-template" is supported or not. Returns true if it's valid
bool common_chat_verify_template(const std::string & tmpl, bool use_jinja);

void common_chat_templates_free(struct common_chat_templates * tmpls);

struct common_chat_templates_deleter {
    void operator()(common_chat_templates * tmpls) { common_chat_templates_free(tmpls); }
};

typedef std::unique_ptr<struct common_chat_templates, common_chat_templates_deleter> common_chat_templates_ptr;

common_chat_templates_ptr common_chat_templates_init(const struct llama_model * model,
                                                     const std::string &        chat_template_override,
                                                     const std::string &        bos_token_override = "",
                                                     const std::string &        eos_token_override = "");

bool        common_chat_templates_was_explicit(const struct common_chat_templates * tmpls);
std::string common_chat_templates_source(const struct common_chat_templates * tmpls, const std::string & variant = "");

struct common_chat_params common_chat_templates_apply(const struct common_chat_templates *        tmpls,
                                                      const struct common_chat_templates_inputs & inputs);

// Format single message, while taking into account the position of that message in chat history
std::string common_chat_format_single(const struct common_chat_templates * tmpls,
                                      const std::vector<common_chat_msg> & past_msg,
                                      const common_chat_msg &              new_msg,
                                      bool                                 add_ass,
                                      bool                                 use_jinja);

// Returns an example of formatted chat
std::string common_chat_format_example(const struct common_chat_templates *       tmpls,
                                       bool                                       use_jinja,
                                       const std::map<std::string, std::string> & chat_template_kwargs);

const char *    common_chat_format_name(common_chat_format format);
common_chat_msg common_chat_parse(const common_chat_input & input, bool is_partial, const common_chat_parser_params & params);
common_chat_msg common_chat_peg_parse(const common_peg_arena & src_parser, const common_chat_input & input, bool is_partial, const common_chat_parser_params & params);

struct common_chat_session_params {
    bool echo  = false; // include the assistant prefill in the output when continuing a message
    bool debug = false; // enable debug output for the PEG parser
};

class common_chat_session {
  public:
    common_chat_session() { result.role = "assistant"; }

    common_chat_session(const common_chat_templates *        tmpls,
                        const llama_vocab *                  vocab,
                        const common_chat_templates_inputs & inputs,
                        const common_chat_session_params &   params = {});

    const common_chat_prompt & prompt() const { return prompt_parts; }
    common_chat_format       format()   const { return parser_params.format; }
    const common_chat_msg &  msg()      const { return result; }
    const common_peg_arena & parser()   const { return parser_params.parser; }

    const std::string &              grammar()            const { return grammar_text; }
    const std::string &              generation_prompt()  const { return generation_prompt_text; }
    const std::string &              thinking_start_tag() const { return thinking_start; }
    const std::vector<std::string> & thinking_end_tags()  const { return thinking_ends; }
    const std::vector<std::string> & additional_stops()   const { return stops; }

    const common_chat_msg_delimiters & message_delimiters() const { return delimiters; }

    void apply_sampling(common_params_sampling & sampling) const;

    bool has_template() const { return templated; }

    const common_chat_msg & feed(const common_chat_input & chunk);

    const common_chat_msg & finish(const common_chat_input & chunk = {});

  private:
    common_chat_prompt                  prompt_parts;
    std::string                         grammar_text;
    bool                                grammar_lazy = false;
    std::vector<common_grammar_trigger> grammar_triggers;
    std::set<llama_token>               preserved_tokens;
    std::vector<std::string>            stops;
    std::string                         generation_prompt_text;
    std::string                         thinking_start;
    std::vector<std::string>            thinking_ends;

    common_chat_parser_params  parser_params;
    common_chat_msg_delimiters delimiters;
    common_chat_input          input;
    common_chat_msg            result;
    bool                       templated = false;
    bool                       finished  = false;
};

// used by arg and server
const char *            common_reasoning_format_name(common_reasoning_format format);
common_reasoning_format common_reasoning_format_from_name(const std::string & format);

common_chat_tool_choice common_chat_tool_choice_parse_oaicompat(const std::string & tool_choice);

bool common_chat_templates_support_enable_thinking(const common_chat_templates * chat_templates);

// Parses a JSON array of messages in OpenAI's chat completion API format.
std::vector<common_chat_msg> common_chat_msgs_parse_oaicompat(const common_json & messages);

std::vector<common_chat_tool> common_chat_tools_parse_oaicompat(const common_json & tools);

common_chat_continuation common_chat_continuation_parse(const common_json & value);

// DEPRECATED: only used in tests
common_json common_chat_msgs_to_json_oaicompat(const std::vector<common_chat_msg> & msgs, bool concat_typed_text = false);

common_json common_chat_tools_to_json_oaicompat(const std::vector<common_chat_tool> & tools);

// The parameters schema of a function tool. A tool without parameters, or with an empty {}, takes zero arguments.
common_json common_chat_tool_parameters(const common_json & function);

// get template caps, useful for reporting to server /props endpoint
std::map<std::string, bool> common_chat_templates_get_caps(const common_chat_templates * chat_templates);

common_chat_prompt common_chat_template_direct_apply(
    const common_chat_template & tmpl,
    const autoparser::generation_params & inputs);

std::string common_chat_template_generation_prompt(
    const common_chat_template &          tmpl,
    const autoparser::generation_params & inputs);

std::optional<common_chat_params> common_chat_try_specialized_template(
        const common_chat_template &          tmpl,
        const autoparser::generation_params & params);


// specialized per-task preset
struct common_chat_prompt_preset {
    std::string system;
    std::string user;
};

common_chat_prompt_preset common_chat_get_asr_prompt(const common_chat_templates * chat_templates);



===== common/chat-auto-parser-generator.cpp =====
#include "chat-auto-parser-helpers.h"
#include "chat-auto-parser.h"
#include "chat-peg-parser.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "log.h"
#include "parsers/parsers.h"
#include "peg-parser.h"

#include <stdexcept>
#include <string>

using json = common_json;

namespace autoparser {

parser_build_context::parser_build_context(common_chat_peg_builder & p, const generation_params & inputs) :
    p(p),
    inputs(inputs),
    reasoning_parser(p.eps()) {}

common_chat_params peg_generator::generate_parser(const common_chat_template &    tmpl,
                                                  const struct generation_params & inputs) {
    // Run differential analysis to extract template structure
    struct autoparser autoparser;
    autoparser.analyze_template(tmpl);
    return generate_parser(tmpl, inputs, autoparser);
}

common_chat_params peg_generator::generate_parser(const common_chat_template &    tmpl,
                                                  const struct generation_params & inputs,
                                                  const autoparser &              autoparser) {
    // Create the result structure
    common_chat_params data;
    data.prompt            = common_chat_template_direct_apply(tmpl, inputs);
    data.generation_prompt = common_chat_template_generation_prompt(tmpl, inputs);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.preserved_tokens  = autoparser.preserved_tokens;
    data.additional_stops.insert(data.additional_stops.end(),
        autoparser.additional_stops.begin(), autoparser.additional_stops.end());

    std::string parser_generation_prompt = data.generation_prompt;

    if (inputs.continue_final_message != COMMON_CHAT_CONTINUATION_NONE && !inputs.continue_msg.empty()) {
        // Build up generation prompt manually
        const auto & msg = inputs.continue_msg;

        if (!autoparser.reasoning.start.empty()) {
            data.generation_prompt = data.generation_prompt.substr(0, data.generation_prompt.find(autoparser.reasoning.start));
            data.generation_prompt += autoparser.reasoning.start + msg.reasoning_content;
            if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
                data.generation_prompt += autoparser.reasoning.end;
            }
        }

        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += msg.render_content();
        }

        data.prompt.append(data.generation_prompt);
    }

    data.parser = autoparser.build_parser(inputs, parser_generation_prompt);

    // Build grammar if tools are present
    bool has_tools =
        autoparser.tools.format.mode != tool_format::NONE && inputs.tools.is_array() && !inputs.tools.empty();
    std::string trigger_marker = !autoparser.tools.format.section_start.empty() ? autoparser.tools.format.section_start :
                                                                                  autoparser.tools.format.per_call_start;

    bool has_response_format = !inputs.json_schema.empty() && inputs.json_schema.is_object();
    bool include_grammar = has_response_format || (has_tools &&
            ((inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO && !trigger_marker.empty()) ||
              inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED));

    if (include_grammar) {
        data.grammar_lazy = !has_response_format && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            data.parser.build_grammar(builder, data.grammar_lazy);
        });

        // Set grammar triggers based on tool section markers (fall back to per-call markers)
        if (data.grammar_lazy) {
            data.grammar_triggers = {
                { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, trigger_marker }
            };
            if (autoparser.tools.format.openai_wrapper_trigger) {
                // model emits the OpenAI function wrapper, trigger on it
                data.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "{\"type\": \"function\"," });
            }
        }
    }

    return data;
}

common_peg_arena autoparser::build_parser(const generation_params & inputs, const std::string & generation_prompt) const {
    if (!analysis_complete) {
        throw std::invalid_argument("Cannot call build_parser on autoparser without performing analysis first, call analyze_template(...)");
    }
    return build_chat_peg_parser([&](common_chat_peg_builder & p) {
        parser_build_context ctx(p, inputs);
        bool                 extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;

        ctx.extracting_reasoning = extract_reasoning && reasoning.mode != reasoning_mode::NONE;
        ctx.content              = &content;
        ctx.reasoning            = &reasoning;

        // Build reasoning parser
        ctx.reasoning_parser = reasoning.build_parser(ctx);

        auto parser = p.eps();

        bool has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
        bool has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
        bool pure_content        = reasoning.mode == reasoning_mode::NONE;

        if (has_response_format) {
            auto response_format = p.rule("response-format", p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema)));
            parser = ctx.reasoning_parser + p.space() + p.choice({
                p.literal("```json") + p.space() + response_format + p.space() + p.literal("```"),
                p.space() + response_format  + p.space()
            }) + p.end();
            pure_content = false;
        } else if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE && jinja_caps.supports_tool_calls) {
            parser = tools.build_parser(ctx);
            pure_content = false;
        } else {
            parser = content.build_parser(ctx);
        }
        const std::string reasoning_start = trim_whitespace(reasoning.start);
        return pure_content ? p.prefix(generation_prompt, reasoning_start) + parser : p.prefix(generation_prompt, reasoning_start) << parser;
    });
}

common_peg_parser analyze_reasoning::build_parser(parser_build_context & ctx) const {
    auto & p = ctx.p;

    if (!ctx.extracting_reasoning) {
        return p.eps();
    }

    if (mode == reasoning_mode::TAG_BASED || mode == reasoning_mode::TOOLS_ONLY) {
        if (!end.empty()) {
            if (!start.empty()) {
                // Standard tag-based: optional(<think>reasoning</think>)
                return p.optional(p.optspace(start) + p.reasoning(p.until(trim_whitespace(end))) + p.optspace(end));
            }
            // Delimiter-style (empty start)
            return p.optional(p.reasoning(p.until(trim_whitespace(end))) + p.optspace(end));
        }
    }

    return p.eps();
}

common_peg_parser analyze_content::build_parser(parser_build_context & ctx) const {
    auto & p = ctx.p;

    if (is_always_wrapped()) {
        if (ctx.extracting_reasoning) {
            return ctx.reasoning_parser + start + p.content(p.until(end)) + end + p.end();
        }
        return p.content(p.until(start)) + start + p.content(p.until(end)) + end + p.end();
    }
    return ctx.reasoning_parser + p.content(p.rest()) + p.end();
}

common_peg_parser analyze_content::build_optional_wrapped(parser_build_context & ctx) const {
    auto & p = ctx.p;

    if (is_always_wrapped()) {
        return p.optional(start + p.content(p.until(end)) + end);
    }
    return p.eps();
}

common_peg_parser analyze_tools::build_parser(parser_build_context & ctx) const {
    switch (format.mode) {
        case tool_format::JSON_NATIVE:
            return build_tool_parser_json_native(ctx);
        case tool_format::TAG_WITH_JSON:
            return build_tool_parser_tag_json(ctx);
        case tool_format::TAG_WITH_TAGGED:
            return build_tool_parser_tag_tagged(ctx);
        default:
            LOG_ERR("[ERROR] Template seems to support tool calls, but failed to determine tool format. Tool calling will not work properly. "
                "Check for a fixed template for your model in the models/templates directory of your llama.cpp installation or "
                "report an issue at https://github.com/ggml-org/llama.cpp/issues\n");
            return ctx.p.eps();
    }
}

common_peg_parser analyze_tools::build_tool_parser_json_native(parser_build_context & ctx) const {
    auto &       p           = ctx.p;
    const auto & inputs      = ctx.inputs;

    // Build effective field names with dot notation if function_field is set
    std::string name_field = format.name_field;
    std::string args_field = format.args_field;

    if (!format.function_field.empty() && format.function_field != "function" &&
        name_field.find('.') == std::string::npos) {
        name_field = format.function_field + "." + name_field;
        args_field = format.function_field + "." + args_field;
    }

    auto tools_parser = p.eps();
    if (format.section_start.empty() && !format.per_call_start.empty()) {
        auto single_tool_parser = p.standard_json_tools(
            format.per_call_start, format.per_call_end, inputs.tools, inputs.parallel_tool_calls,
            inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED, name_field, args_field, format.tools_array_wrapped,
            format.fun_name_is_key, format.id_field, format.gen_id_field, format.parameter_order, format.openai_wrapper_trigger);
        tools_parser = p.trigger_rule("tool-calls", p.one_or_more(single_tool_parser + p.space()));
    } else {
        tools_parser = p.standard_json_tools(
            format.section_start, format.section_end, inputs.tools, inputs.parallel_tool_calls,
            inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED, name_field, args_field, format.tools_array_wrapped,
            format.fun_name_is_key, format.id_field, format.gen_id_field, format.parameter_order, format.openai_wrapper_trigger);
    }

    // Handle content wrappers if present
    if (ctx.content && ctx.content->is_always_wrapped()) {
        auto wrapped_content = ctx.content->build_optional_wrapped(ctx);
        return ctx.reasoning_parser + wrapped_content + tools_parser + p.end();
    }

    std::string tool_start = "{";
    if (!format.section_start.empty()) {
        tool_start = format.section_start;
    } else if (!format.per_call_start.empty()) {
        tool_start = format.per_call_start;
    }

    return ctx.reasoning_parser + p.optional(p.content(p.until(tool_start))) + tools_parser + p.end();
}

common_peg_parser analyze_tools::build_func_parser(common_chat_peg_builder & p, const std::string & name,
                                                    const common_peg_parser & call_id_section, bool have_call_id,
                                                    const common_peg_parser & args,
                                                    std::optional<common_peg_parser> atomic_peek) const {
    auto              open           = p.tool_open(function.name_prefix + p.tool_name(p.literal(name)) + function.name_suffix);
    bool              matched_atomic = false;
    common_peg_parser func_parser    = p.eps();

    if (!function.args_separator.empty()) {
        open = open + p.space() + p.literal(function.args_separator);
    }

    if (!function.name_suffix.empty()) {
        func_parser    = open + call_id_section + p.space() + args;
        matched_atomic = true;
    } else if (have_call_id) {
        func_parser    = p.atomic(open + call_id_section) + p.space() + args;
        matched_atomic = true;
    } else if (atomic_peek.has_value()) {
        func_parser    = p.atomic(open + call_id_section + p.space() + *atomic_peek) + args;
        matched_atomic = true;
    } else {
        func_parser = open + call_id_section + p.space() + args;
    }

    if (!function.close.empty()) {
        func_parser = func_parser + p.space() + p.tool_close(p.literal(function.close));
    } else if (!format.per_call_end.empty()) {
        // When there's no func_close but there is a per_call_end marker, use peek() to ensure
        // we only emit tool_close when we can actually see the closing marker. This prevents
        // premature closing during partial parsing when we've seen e.g. "</" which could be
        // either "</tool_call>" (end) or "<arg_key>" prefix that failed to match.
        // Laguna (v4): the model may emit whitespace between the last </arg_value> and
        // </tool_call> even though the template renders them tight. Tolerate optional
        // leading space in the close lookahead so the tool call still closes.
        auto close_peek = arguments.tolerate_intertag_whitespace
                              ? p.peek(p.space() + p.literal(format.per_call_end))
                              : p.peek(p.literal(format.per_call_end));
        func_parser = func_parser + p.tool_close(close_peek);
    } else {
        func_parser = func_parser + p.tool_close(p.space());  // force this to process tool closing callbacks in mapper
    }
    if (!matched_atomic) {
        func_parser = p.atomic(func_parser);
    }
    return func_parser;
}

common_peg_parser analyze_tools::build_tool_parser_tag_json(parser_build_context & ctx) const {
    auto &       p           = ctx.p;
    const auto & inputs      = ctx.inputs;

    common_peg_parser tool_choice = p.choice();

    foreach_function(inputs.tools, [&](size_t tool_index, const json & tool) {
        const auto & func   = tool.at("function");
        std::string  name   = func.at("name");
        const auto   schema = common_chat_tool_parameters(func);

        // Build call_id parser based on position (if supported)
        bool have_call_id = false;
        common_peg_parser call_id_section = p.eps();
        if (call_id.pos == call_id_position::BETWEEN_FUNC_AND_ARGS && !call_id.prefix.empty() &&
            (!call_id.suffix.empty() || !arguments.start.empty())) {
            if (!call_id.suffix.empty()) {
                call_id_section = p.optional(call_id.prefix + p.tool_id(p.until(call_id.suffix))) + call_id.suffix;
            } else {
                call_id_section = p.optional(call_id.prefix + p.tool_id(p.until(arguments.start)));
            }
            have_call_id = true;
        }
        auto args_parser = p.tool_args(p.schema(p.json(), "tool-" + std::to_string(tool_index) + "-schema", schema));
        if (!arguments.start.empty()) {
            args_parser = p.literal(arguments.start) + args_parser;
        }
        if (!arguments.end.empty()) {
            args_parser = args_parser + p.literal(arguments.end);
        }

        auto atomic_peek = !arguments.start.empty() ? std::optional(p.peek(p.literal(arguments.start))) : std::nullopt;
        auto func_parser = build_func_parser(p, name, call_id_section, have_call_id, args_parser, atomic_peek);
        tool_choice |= p.rule("tool-" + std::to_string(tool_index), func_parser);
    });

    auto require_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    common_peg_parser tool_calls = p.eps();

    if (!format.per_call_start.empty()) {
        auto wrapped_call = format.per_call_start + tool_choice + format.per_call_end;
        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call", wrapped_call + p.zero_or_more(p.space() + wrapped_call));
        } else {
            tool_calls = p.trigger_rule("tool-call", wrapped_call);
        }
        if (!format.section_start.empty()) {
            tool_calls = p.trigger_rule("tool-calls",
                                        p.literal(format.section_start) + p.space() + tool_calls + p.space() +
                                            (format.section_end.empty() ? p.end() : p.literal(format.section_end)));
        }
    } else {
        std::string separator = ", ";  // Default
        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call", format.section_start + tool_choice +
                                                         p.zero_or_more(separator + tool_choice) + format.section_end);
        } else {
            tool_calls = p.trigger_rule("tool-call", format.section_start + tool_choice + format.section_end);
        }
    }

    if (!require_calls) {
        tool_calls = p.optional(tool_calls);
    }

    std::string trigger_marker       = !format.section_start.empty() ? format.section_start : format.per_call_start;
    auto        content_before_tools = trigger_marker.empty() ? p.eps() : p.until(trigger_marker);
    return ctx.reasoning_parser + p.optional(p.content(content_before_tools)) + tool_calls + p.end();
}

common_peg_parser analyze_tools::build_tool_parser_tag_tagged(parser_build_context & ctx) const {
    auto &       p           = ctx.p;
    const auto & inputs      = ctx.inputs;

    auto until_suffix = p.rule("until-suffix", p.until(arguments.value_suffix));

    common_peg_parser tool_choice = p.choice();

    foreach_function(inputs.tools, [&](size_t tool_index, const json & tool) {
        const auto & func = tool.at("function");
        std::string  name = func.at("name");

        // Build parser for each argument, separating required and optional
        std::vector<common_peg_parser> required_parsers;
        std::vector<common_peg_parser> optional_parsers;
        foreach_parameter(func, [&](size_t param_index, const common_chat_schema_property & param, const common_chat_schema_document_ptr & doc) {
            auto arg =
                p.tool_arg(p.tool_arg_open(arguments.name_prefix + p.tool_arg_name(p.literal(param.name)) +
                                           arguments.name_suffix) +
                           arguments.value_prefix +
                           (param.schema->may_be_string() ?
                                p.ac(p.tool_arg_string_value(until_suffix) +
                                    p.tool_arg_close(p.literal(arguments.value_suffix)), arguments.value_suffix) :
                                (p.tool_arg_json_value(p.schema(
                                    p.json(), "tool-" + std::to_string(tool_index) + "-arg-" + std::to_string(param_index) + "-schema", doc, *param.schema)) +
                                    p.tool_arg_close(p.literal(arguments.value_suffix)))));

            auto named_arg = p.rule("tool-" + std::to_string(tool_index) + "-arg-" + std::to_string(param_index), arg);
            if (param.required) {
                required_parsers.push_back(named_arg);
            } else {
                optional_parsers.push_back(named_arg);
            }
        });

        // Build required arg sequence in definition order
        common_peg_parser args_seq = p.eps();
        for (size_t i = 0; i < required_parsers.size(); i++) {
            if (i > 0) {
                args_seq = args_seq + p.space();
            }
            args_seq = args_seq + required_parsers[i];
        }

        // Build optional args with flexible ordering
        if (!optional_parsers.empty()) {
            common_peg_parser any_opt = p.choice();
            for (const auto & opt : optional_parsers) {
                any_opt |= opt;
            }
            args_seq = args_seq + p.repeat(p.space() + any_opt, 0, -1);
        }

        if (!arguments.start.empty()) {
            args_seq = p.literal(arguments.start) + args_seq;
        }
        if (!arguments.end.empty()) {
            args_seq = args_seq + p.literal(arguments.end);
        }

        // Build call_id parser based on position (if supported)
        common_peg_parser call_id_section = p.eps();
        bool have_call_id = false;
        if (call_id.pos == call_id_position::BETWEEN_FUNC_AND_ARGS && !call_id.prefix.empty() &&
            (!call_id.suffix.empty() || !arguments.start.empty())) {
            have_call_id = true;
            if (!call_id.suffix.empty()) {
                call_id_section = p.optional(call_id.prefix + p.tool_id(p.until(call_id.suffix)) + call_id.suffix);
            } else {
                call_id_section = p.optional(call_id.prefix + p.tool_id(p.until(arguments.start)));
            }
        }

        // Only peek for an arg tag when there are required args that must follow.
        // When all args are optional, the model may emit no arg tags at all (#20650).
        auto atomic_peek = (!arguments.name_prefix.empty() && !required_parsers.empty()) ?
            std::optional(p.peek(p.literal(arguments.name_prefix))) : std::nullopt;
        auto func_parser = build_func_parser(p, name, call_id_section, have_call_id, args_seq, atomic_peek);
        tool_choice |= p.rule("tool-" + std::to_string(tool_index), func_parser);
    });

    auto require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    common_peg_parser tool_calls = p.eps();

    if (!format.per_call_start.empty()) {
        auto wrapped_call = format.per_call_start + p.space() + tool_choice + p.space() + format.per_call_end;
        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call", wrapped_call + p.zero_or_more(p.space() + wrapped_call) + p.space());
        } else {
            tool_calls = p.trigger_rule("tool-call", wrapped_call + p.space());
        }
        if (!format.section_start.empty()) {
            tool_calls = p.trigger_rule("tool-calls",
                                        p.literal(format.section_start) + p.space() + tool_calls + p.space() +
                                            (format.section_end.empty() ? p.end() : p.literal(format.section_end) + p.space()));
        }
    } else {
        std::string separator = ", ";  // Default

        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call", format.section_start + p.space() + tool_choice +
                                                         p.zero_or_more(separator + tool_choice) + p.space() +
                                                         format.section_end);
        } else {
            tool_calls = p.trigger_rule(
                "tool-call", format.section_start + p.space() + tool_choice + p.space() + format.section_end);
        }
    }

    if (!require_tools) {
        tool_calls = p.optional(tool_calls);
    }

    std::string trigger_marker       = !format.section_start.empty() ? format.section_start : format.per_call_start;
    auto        content_before_tools = trigger_marker.empty() ? p.eps() : p.until(trigger_marker);
    return ctx.reasoning_parser + p.optional(p.content(content_before_tools)) + tool_calls + p.end();
}

}  // namespace autoparser



===== common/chat-auto-parser-helpers.cpp =====
#include "chat-auto-parser-helpers.h"

#include "chat-auto-parser.h"
#include "chat-peg-parser.h"
#include "chat.h"
#include "log.h"
#include "peg-parser.h"

#include <cctype>
#include <numeric>

std::string trim_whitespace(const std::string & str) {
    size_t start = 0;
    while (start < str.length() && std::isspace(static_cast<unsigned char>(str[start]))) {
        start++;
    }

    if (start == str.length()) {
        return "";
    }

    size_t end = str.length() - 1;
    while (end > start && std::isspace(static_cast<unsigned char>(str[end]))) {
        end--;
    }

    return str.substr(start, end - start + 1);
}

std::string trim_leading_whitespace(const std::string & str) {
    size_t start = 0;
    while (start < str.length() && std::isspace(static_cast<unsigned char>(str[start]))) {
        start++;
    }

    return str.substr(start);
}

std::string trim_trailing_whitespace(const std::string & str) {
    if (str.empty()) {
        return "";
    }

    size_t end = str.length() - 1;
    while (end > 0 && std::isspace(static_cast<unsigned char>(str[end]))) {
        end--;
    }

    // If first char is also whitespace, return empty string
    if (end == 0 && std::isspace(static_cast<unsigned char>(str[0]))) {
        return "";
    }

    return str.substr(0, end + 1);
}

std::string trim_trailing_newlines(const std::string & str) {
    size_t end = str.length();
    while (end > 0 && str[end - 1] == '\n') {
        end--;
    }

    return str.substr(0, end);
}

static size_t common_prefix_len(const std::string & left, const std::string & right) {
    size_t prefix_len = 0;
    size_t min_len    = std::min(left.length(), right.length());
    while (prefix_len < min_len && left[prefix_len] == right[prefix_len]) {
        prefix_len++;
    }
    return prefix_len;
}

static size_t common_suffix_len(const std::string & left, const std::string & right) {
    size_t suffix_len = 0;
    size_t min_len    = std::min(left.length(), right.length());
    while (suffix_len < min_len && left[left.length() - 1 - suffix_len] == right[right.length() - 1 - suffix_len]) {
        suffix_len++;
    }
    return suffix_len;
}

diff_split calculate_diff_split(const std::string & left, const std::string & right) {
    diff_split result;

    auto left_seg = segmentize_markers(left);
    auto right_seg = segmentize_markers(right);

    if (left_seg.empty()) {
        result.right = right;
        return result;
    }
    if (right_seg.empty()) {
        result.left = left;
        return result;
    }

    auto left_start = left_seg.begin();
    auto left_end = --left_seg.end();
    auto right_start = right_seg.begin();
    auto right_end = --right_seg.end();

    auto test = [&] () {
        return left_start != left_end && right_start != right_end;
    };

    bool left_fully_consumed = false;
    bool right_fully_consumed = false;

    while (test()) {
        bool advanced = false;
        if (*left_start == *right_start) {
            result.prefix.append(left_start->value);
            left_start++;
            right_start++;
            advanced = true;
        }
        if (*left_end == *right_end) {
            result.suffix = left_end->value + result.suffix;
            if (left_start != left_end) {
                left_end--;
            } else {
                left_fully_consumed = true;
            }
            if (right_start != right_end) {
                right_end--;
            } else {
                right_fully_consumed = true;
            }
            advanced = true;
        }
        if (!advanced) {
            break;
        }
    }

    if (left_start == left_end && right_start != right_end) {
        if (*left_start == *right_end) {
            result.suffix = right_end->value + result.suffix;
            right_end--;
            left_fully_consumed = true;
        } else if (*left_start == *right_start) {
            result.prefix.append(right_start->value);
            right_start++;
            left_fully_consumed = true;
        }
    } else if (right_start == right_end && left_start != left_end) {
        if (*left_end == *right_start) {
            result.suffix = left_end->value + result.suffix;
            left_end--;
            right_fully_consumed = true;
        } else if (*left_start == *right_start) {
            result.prefix.append(left_start->value);
            left_start++;
            right_fully_consumed = true;
        }
    } else if (left_start == left_end && right_start == right_end && *left_start == *right_start && left_start->type == segment_type::MARKER) {
        result.prefix.append(right_start->value);
        left_fully_consumed = true;
        right_fully_consumed = true;
    }

    auto eat_segment = [](std::string str, const segment & seg) -> std::string { return std::move(str) + seg.value; };

    bool can_have_text_suffix = left_end->type == segment_type::TEXT && right_end->type == segment_type::TEXT;
    bool can_have_text_prefix = right_start->type == segment_type::TEXT && left_start->type == segment_type::TEXT;

    std::string remainder_left = std::accumulate(left_start, left_fully_consumed ? left_end : ++left_end, std::string(), eat_segment);
    std::string remainder_right = std::accumulate(right_start, right_fully_consumed ? right_end : ++right_end, std::string(), eat_segment);

    size_t suffix_len = can_have_text_suffix ? common_suffix_len(remainder_left, remainder_right) : 0;
    // avoid overlaps between prefix and suffix
    size_t prefix_len = can_have_text_prefix ? common_prefix_len(remainder_left.substr(0, remainder_left.size() - suffix_len),
        remainder_right.substr(0, remainder_right.size() - suffix_len)) : 0;

    result.prefix.append(remainder_left.substr(0, prefix_len));
    result.suffix = remainder_left.substr(remainder_left.length() - suffix_len, suffix_len) + result.suffix;
    result.left = remainder_left.substr(prefix_len, remainder_left.length() - prefix_len - suffix_len);
    result.right = remainder_right.substr(prefix_len, remainder_right.length() - prefix_len - suffix_len);

    if (result.left == "" && result.right == "") {
        // degenerate case, no diff
        result.prefix = left;
        result.suffix = "";
        // pick prefix = all as representation
    }

    // When left has no unique content (result.left is empty), left is entirely
    // shared with right. The simultaneous prefix/suffix segment matching can
    // incorrectly consume trailing segments of left as suffix when those same
    // segments also appear at the end of right (e.g. "\n" at the end of both
    // the shared content and the generation prompt). This rotates the diff.
    // Fix: if left is a prefix of right, enforce that directly.
    if (result.left.empty() && !result.right.empty() &&
            left.size() <= right.size() &&
            right.substr(0, left.size()) == left) {
        result.prefix = left;
        result.suffix = "";
        result.right  = right.substr(left.size());
    }

    return result;
}

// Returns the prefix of `full` up until the first occurrence of the common prefix of `left` and `right`
std::string until_common_prefix(const std::string & full, const std::string & left, const std::string & right) {
    // Find the common prefix of left and right
    size_t common_prefix_len = 0;
    size_t min_len           = std::min(left.length(), right.length());
    while (common_prefix_len < min_len && left[common_prefix_len] == right[common_prefix_len]) {
        common_prefix_len++;
    }

    // If there's no common prefix, return empty string
    if (common_prefix_len == 0) {
        return "";
    }

    // Find the common prefix in the full string
    std::string common_prefix = left.substr(0, common_prefix_len);
    size_t      pos           = full.find(common_prefix);

    // If not found, return empty string
    if (pos == std::string::npos) {
        return "";
    }

    // Return everything before the common prefix
    return full.substr(0, pos);
}

// Returns the suffix of `full` after the last occurrence of the common suffix of `left` and `right`
std::string after_common_suffix(const std::string & full, const std::string & left, const std::string & right) {
    // Find the common suffix of left and right (compare from the end)
    size_t common_suffix_len = 0;
    size_t min_len           = std::min(left.length(), right.length());
    while (common_suffix_len < min_len &&
           left[left.length() - 1 - common_suffix_len] == right[right.length() - 1 - common_suffix_len]) {
        common_suffix_len++;
    }

    // If there's no common suffix, return empty string
    if (common_suffix_len == 0) {
        return "";
    }

    // Extract the common suffix
    std::string common_suffix = left.substr(left.length() - common_suffix_len);

    // Find the last occurrence of the common suffix in the full string
    size_t pos = full.rfind(common_suffix);

    // If not found, return empty string
    if (pos == std::string::npos) {
        return "";
    }

    // Return everything after the common suffix
    return full.substr(pos + common_suffix_len);
}

// TODO: segmentize will treat a JSON array inside tags as a tag: <calls>[{ "fun": { ... } }]</calls> will be three markers
// not too worried about that because it hasn't turned out as a problem anywhere, but noting here in case it will
// Might have to put some restrictions on tag contents as well (like "no { }")
std::vector<segment> segmentize_markers(const std::string & text) {
    std::vector<segment> retval;
    bool in_marker = false;
    char marker_opener = '\0';

    auto is_marker_opener = [](char c) -> bool { return c == '<' || c == '['; };
    auto is_marker_closer = [](char op, char c) -> bool { return (op == '<' && c == '>') || (op == '[' && c == ']'); };

    size_t last_border = 0;

    for (size_t cur_pos = 0; cur_pos < text.length(); cur_pos++) {
        if (!in_marker && is_marker_opener(text[cur_pos])) {
            if (last_border < cur_pos) {
                retval.push_back(segment(segment_type::TEXT, text.substr(last_border, cur_pos - last_border)));
            }
            last_border = cur_pos;
            in_marker = true;
            marker_opener = text[cur_pos];
        } else if (in_marker && is_marker_closer(marker_opener, text[cur_pos])) {
            // no need to check because last_border will always be smaller
                retval.push_back(segment(segment_type::MARKER, text.substr(last_border, cur_pos - last_border + 1)));
            last_border = cur_pos + 1;
            in_marker = false;
            marker_opener = '\0';
        }
    }
    if (last_border < text.length()) {
            retval.push_back(segment(segment_type::TEXT, text.substr(last_border)));
    }
    return retval;
}

std::vector<segment> prune_whitespace_segments(const std::vector<segment> & segments) {
    std::vector<segment> result;
    for (const auto & seg : segments) {
        if (!trim_whitespace(seg.value).empty()) {
            result.push_back(seg);
        }
    }
    return result;
}

namespace autoparser {

static const std::string ERR_TMPL = "#**ERROR**#";

std::string apply_template(const common_chat_template & tmpl, const template_params & params) {
    generation_params tmpl_params;
    tmpl_params.messages              = params.messages;
    tmpl_params.tools                 = params.tools;
    tmpl_params.add_generation_prompt = params.add_generation_prompt;
    tmpl_params.enable_thinking       = params.enable_thinking;

    if (params.extra_context) {
        tmpl_params.extra_context = *params.extra_context;
    }
    tmpl_params.extra_context["enable_thinking"] = params.enable_thinking;

    try {
        return common_chat_template_direct_apply(tmpl, tmpl_params).str();
    } catch (const std::exception & e) {
        LOG_DBG("Template application failed: %s\n", e.what());
        return ERR_TMPL;
    }
}

std::optional<compare_variants_result> compare_variants(
    const common_chat_template &                   tmpl,
    const template_params &                        params_A,
    const std::function<void(template_params &)> & params_modifier) {
    // Create variant B by copying A
    template_params params_B = params_A;

    // Apply modifier to create variant B
    if (params_modifier) {
        params_modifier(params_B);
    }

    // Apply template to both variants
    std::string output_A = apply_template(tmpl, params_A);
    std::string output_B = apply_template(tmpl, params_B);

    // Check for template application failures
    if (output_A == ERR_TMPL || output_B == ERR_TMPL) {
        return std::nullopt;
    }

    // Calculate diff and return result with both outputs
    compare_variants_result result;
    result.diff     = calculate_diff_split(output_A, output_B);
    result.output_A = output_A;
    result.output_B = output_B;

    return result;
}

}  // namespace autoparser




===== common/chat-auto-parser.h =====
#pragma once

#include "chat.h"
#include "common.h"
#include "jinja/caps.h"
#include "peg-parser.h"
#include "json.h"

#include <chrono>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using json = common_json;

class common_chat_peg_builder;

// ============================================================================
// Parameters for template application (low-level, used by diff analysis)
// ============================================================================
struct template_params {
    json                messages;
    json                tools;
    bool                add_generation_prompt = false;
    bool                enable_thinking       = true;
    std::optional<json> extra_context         = std::nullopt;
};

struct diff_split {
    std::string prefix;
    std::string suffix;
    std::string left;
    std::string right;

    bool operator==(struct diff_split & other) const {
        return prefix == other.prefix && suffix == other.suffix && left == other.left && right == other.right;
    }
};

// Result of compare_variants containing diff and original outputs
struct compare_variants_result {
    diff_split  diff;
    std::string output_A;
    std::string output_B;
};

namespace autoparser {

// ============================================================================
// High-level params for parser generation
// ============================================================================

struct generation_params {
    json                                  messages;
    json                                  tools;
    common_chat_tool_choice               tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    json                                  json_schema;
    bool                                  parallel_tool_calls = true;
    common_reasoning_format               reasoning_format    = COMMON_REASONING_FORMAT_AUTO;
    bool                                  stream              = true;
    std::string                           grammar;
    bool                                  add_generation_prompt  = false;
    common_chat_continuation              continue_final_message = COMMON_CHAT_CONTINUATION_NONE;
    common_chat_msg                       continue_msg;
    bool                                  enable_thinking        = true;
    std::chrono::system_clock::time_point now                    = std::chrono::system_clock::now();
    json                                  extra_context;
    bool                                  add_bos       = false;
    bool                                  add_eos       = false;
    bool                                  is_inference  = true;
    bool                                  add_inference = false;
    bool                                  mark_input    = true;  // whether to mark input strings in the jinja context

    bool has_continuation() const {
        return continue_final_message != COMMON_CHAT_CONTINUATION_NONE && !continue_msg.empty();
    }
};

// ============================================================================
// Analysis Result Enums
// ============================================================================

// Reasoning handling mode (derived from R1-R3 comparisons)
enum class reasoning_mode {
    NONE,           // No reasoning markers detected
    TAG_BASED,      // Tag-based: <think>...</think> (start can be empty for delimiter-style)
    TOOLS_ONLY      // Only reason on tool calls, not on normal content
};

inline std::ostream & operator<<(std::ostream & os, const reasoning_mode & mode) {
    switch (mode) {
        case reasoning_mode::NONE:
            return os << "NONE";
        case reasoning_mode::TAG_BASED:
            return os << "TAG_BASED";
        case reasoning_mode::TOOLS_ONLY:
            return os << "TOOLS_ONLY";
        default:
            return os << "UNKNOWN";
    }
}

// Content wrapping mode (derived from C1 comparison)
enum class content_mode {
    PLAIN,                   // No content markers
    ALWAYS_WRAPPED,          // Content always wrapped with markers
    WRAPPED_WITH_REASONING,  // Content wrapped only when reasoning present
};

inline std::ostream & operator<<(std::ostream & os, const content_mode & mode) {
    switch (mode) {
        case content_mode::PLAIN:
            return os << "PLAIN";
        case content_mode::ALWAYS_WRAPPED:
            return os << "ALWAYS_WRAPPED";
        case content_mode::WRAPPED_WITH_REASONING:
            return os << "WRAPPED_WITH_REASONING";
        default:
            return os << "UNKNOWN";
    }
}

// Call ID position in tool calls (for non-JSON formats)
enum class call_id_position {
    NONE,                   // No call ID support detected
    PRE_FUNC_NAME,          // Call ID before function name: [CALL_ID]id[FUNC]name{args}
    BETWEEN_FUNC_AND_ARGS,  // Call ID between function and args: [FUNC]name[CALL_ID]id{args}
    POST_ARGS,              // Call ID after arguments: [FUNC]name{args}[CALL_ID]id
};

inline std::ostream & operator<<(std::ostream & os, const call_id_position & pos) {
    switch (pos) {
        case call_id_position::NONE:
            return os << "NONE";
        case call_id_position::PRE_FUNC_NAME:
            return os << "PRE_FUNC_NAME";
        case call_id_position::BETWEEN_FUNC_AND_ARGS:
            return os << "BETWEEN_FUNC_AND_ARGS";
        case call_id_position::POST_ARGS:
            return os << "POST_ARGS";
        default:
            return os << "UNKNOWN";
    }
}

// Tool call format classification (derived from T1-T5, A1-A3 comparisons)
enum class tool_format {
    NONE,             // No tool support detected
    JSON_NATIVE,      // Pure JSON: {"name": "X", "arguments": {...}}
    TAG_WITH_JSON,    // Tag-based with JSON args: <function=X>{...}</function>
    TAG_WITH_TAGGED,  // Tag-based with tagged args: <param=key>value</param>
};

inline std::ostream & operator<<(std::ostream & os, const tool_format & format) {
    switch (format) {
        case tool_format::NONE:
            return os << "NONE";
        case tool_format::JSON_NATIVE:
            return os << "JSON_NATIVE";
        case tool_format::TAG_WITH_JSON:
            return os << "TAG_WITH_JSON";
        case tool_format::TAG_WITH_TAGGED:
            return os << "TAG_WITH_TAGGED";
        default:
            return os << "UNKNOWN";
    }
}

// ============================================================================
// Sub-structs for tool analysis
// ============================================================================

struct tool_format_analysis {
    tool_format mode = tool_format::NONE;

    std::string section_start;   // e.g., "<tool_call>", "[TOOL_CALLS]", ""
    std::string section_end;     // e.g., "</tool_call>", ""
    std::string per_call_start;  // e.g., "<|tool_call_begin|>", "" (for multi-call templates)
    std::string per_call_end;    // e.g., "<|tool_call_end|>", ""

    bool fun_name_is_key = false;       // In JSON format function name is JSON key, i.e. { "<funname>": { ... arguments ... } }
    bool tools_array_wrapped = false;   // Tool calls wrapped in JSON array [...]
    bool openai_wrapper_trigger = false;  // model emits the OpenAI function wrapper, trigger on it

    std::string              function_field = "function";
    std::string              name_field     = "name";
    std::string              args_field     = "arguments";
    std::string              id_field;
    std::string              gen_id_field;
    std::vector<std::string> parameter_order;
};

struct tool_function_analysis {
    std::string name_prefix;     // e.g., "<function=", "\"name\": \"", "functions."
    std::string name_suffix;     // e.g., ">", "\"", ":0"
    std::string args_separator;  // e.g., "<tool_sep>" (marker between function name and arguments)
    std::string close;           // e.g., "</function>", "" (for tag-based)
};

struct tool_arguments_analysis {
    std::string start;          // e.g., "<|tool_call_argument_begin|>", "<args>"
    std::string end;            // e.g., "<|tool_call_argument_end|>", "</args>"
    std::string name_prefix;   // e.g., "<param=", "<arg_key>", "\""
    std::string name_suffix;   // e.g., ">", "</arg_key>", "\":"
    std::string value_prefix;  // e.g., "", "<arg_value>", ""
    std::string value_suffix;  // e.g., "</param>", "</arg_value>", ""
    std::string separator;     // e.g., "", "\n", ","
    bool tolerate_intertag_whitespace = false; // Laguna: accept optional whitespace between arg tags
};

struct tool_id_analysis {
    call_id_position pos = call_id_position::NONE;

    std::string prefix;  // e.g., "[CALL_ID]" (marker before call ID value)
    std::string suffix;  // e.g., "" (marker after call ID value, before next section)
};

// ============================================================================
// Parser build context (shared interface for build_parser methods)
// ============================================================================

struct analyze_content;
struct analyze_reasoning;

struct parser_build_context {
    common_chat_peg_builder & p;
    const generation_params &         inputs;
    common_peg_parser                 reasoning_parser;
    bool                              extracting_reasoning = false;
    const analyze_reasoning *         reasoning            = nullptr;
    const analyze_content *           content              = nullptr;

    parser_build_context(common_chat_peg_builder & p, const generation_params & inputs);
};

// ============================================================================
// Base class for analyzers with parser building
// ============================================================================

struct analyze_base {
    virtual ~analyze_base() = default;
    virtual common_peg_parser build_parser(parser_build_context & ctx) const = 0;

  protected:
    const common_chat_template * tmpl = nullptr;

    analyze_base() = default;
    explicit analyze_base(const common_chat_template & tmpl) : tmpl(&tmpl) {}
};

// ============================================================================
// Reasoning analyzer
// ============================================================================

struct analyze_reasoning : analyze_base {
    reasoning_mode mode = reasoning_mode::NONE;

    std::string start;  // e.g., "<think>", "[THINK]", "<|START_THINKING|>", ""
    std::string end;    // e.g., "</think>", "[BEGIN FINAL RESPONSE]", "<|END_THINKING|>"

    analyze_reasoning() = default;
    analyze_reasoning(const common_chat_template & tmpl, bool supports_tools);
    analyze_reasoning(std::string start_, std::string end_) : start(std::move(start_)), end(std::move(end_)) {}

    common_peg_parser build_parser(parser_build_context & ctx) const override;

  private:
    // Look for reasoning markers in rendered content
    void compare_reasoning_presence();

    // Compare generation prompt with enable_thinking=true vs false
    void compare_thinking_enabled();

    // Check if reasoning is always possible or only in tool calls
    void compare_reasoning_scope();
};

// ============================================================================
// Content analyzer
// ============================================================================

struct analyze_content : analyze_base {
    content_mode mode = content_mode::PLAIN;

    std::string start;  // e.g., "<response>", ">>>all\n", ""
    std::string end;    // e.g., "</response>", ""

    bool requires_nonnull_content = false;

    analyze_content() = default;
    analyze_content(const common_chat_template & tmpl, const analyze_reasoning & reasoning);

    common_peg_parser build_parser(parser_build_context & ctx) const override;

    bool is_always_wrapped() const;
    common_peg_parser build_optional_wrapped(parser_build_context & ctx) const;
};

// ============================================================================
// Tool analyzer
// ============================================================================

struct analyze_tools : analyze_base {
    tool_format_analysis    format;
    tool_function_analysis  function;
    tool_arguments_analysis arguments;
    tool_id_analysis        call_id;

    analyze_tools() = default;
    analyze_tools(const common_chat_template & tmpl,
                  const jinja::caps &          caps,
                  const analyze_reasoning &    reasoning);

    common_peg_parser build_parser(parser_build_context & ctx) const override;

  private:
    // Extract tool calling 'haystack' for further analysis and delegate further analysis based on format
    void analyze_tool_calls(const analyze_reasoning & reasoning, bool supports_parallel_tool_calls);

    // Analyze format based on position of function and argument name in needle
    void analyze_tool_call_format(const std::string &       haystack,
                                  const std::string &       fun_name_needle,
                                  const std::string &       arg_name_needle,
                                  const analyze_reasoning & reasoning,
                                  bool                      supports_parallel_tool_calls);

    // Analyze specifics of JSON native format (entire tool call is a JSON object)
    void analyze_tool_call_format_json_native(const std::string & clean_haystack,
                                              const std::string & fun_name_needle,
                                              const std::string & arg_name_needle);

    // Check if parallel calls in JSON native format array wrapped or tag wrapped
    void analyze_json_native_parallel_calls();

    // Analyze specifics of non-JSON native format (tags for function name or for function name and arguments)
    void analyze_tool_call_format_non_json(const std::string & clean_haystack,
                                           const std::string & fun_name_needle);

    // Check for and extract specific per-call markers for non-native-JSON templates with parallel call support
    void check_per_call_markers();

    // Extract function name markers
    void extract_function_markers();

    // Delegates to separate functions for: separator analysis, argument name analysis, argument value analysis
    void analyze_arguments();

    // Extract argument name markers
    void extract_argument_name_markers();

    // Extract argument value markers
    void extract_argument_value_markers();

    // Extract argument separator, if specified (eg. <arg=foo>...</arg><sep><arg=bar>...</arg>)
    void extract_argument_separator();

    // Extract argument wrapper markers, if present (eg. '<args><arg=foo>...</arg><arg=bar>...</arg></args>')
    void extract_args_markers();

    // Extract call ID markers, if present
    void extract_call_id_markers();

    // Per-format tool parser builders
    common_peg_parser build_tool_parser_json_native(parser_build_context & ctx) const;
    common_peg_parser build_tool_parser_tag_json(parser_build_context & ctx) const;
    common_peg_parser build_tool_parser_tag_tagged(parser_build_context & ctx) const;

    // Shared helper: builds func_parser from open+call_id+args, handling atomic wrapping and close.
    // atomic_peek: if present, used as the peek expression in the third atomicity branch.
    common_peg_parser build_func_parser(common_chat_peg_builder & p, const std::string & name,
                                        const common_peg_parser & call_id_section, bool have_call_id,
                                        const common_peg_parser & args,
                                        std::optional<common_peg_parser> atomic_peek) const;
};

// ============================================================================
// Main autoparser class
// ============================================================================

struct autoparser {
    jinja::caps          jinja_caps;
    std::string          user_start;
    std::string          assistant_start;
    analyze_reasoning    reasoning;
    analyze_content      content;
    analyze_tools        tools;
    bool                 analysis_complete = false;

    // Preserved tokens for tokenizer (union of all non-empty markers)
    std::vector<std::string> preserved_tokens;
    std::vector<std::string> additional_stops;  // literal stop strings (e.g. Laguna </assistant>) caught however tokenized

    autoparser() = default;

    // Find the starting marker for the user message and assistant message
    std::string detect_user_start_marker(const common_chat_template & tmpl);
    std::string detect_assistant_start_marker(const common_chat_template & tmpl);

    // Run full differential analysis on a template
    void analyze_template(const common_chat_template & tmpl);

    // Build the PEG parser for this template
    common_peg_arena build_parser(const generation_params & inputs, const std::string & generation_prompt) const;

  private:
    // Collect tokens from entire analysis to preserve
    void collect_preserved_tokens();
};

// ============================================================================
// Parser generator
// ============================================================================

class peg_generator {
  public:
    static common_chat_params generate_parser(const common_chat_template &    tmpl,
                                              const struct generation_params & inputs);

    static common_chat_params generate_parser(const common_chat_template &    tmpl,
                                              const struct generation_params & inputs,
                                              const autoparser &              autoparser);
};

}  // namespace autoparser

enum segment_type { TEXT, MARKER };

inline std::ostream & operator<<(std::ostream & os, const segment_type & type) {
    switch (type) {
        case segment_type::TEXT:
            return os << "TEXT";
        case segment_type::MARKER:
            return os << "MARKER";
        default:
            return os << "UNKNOWN";
    }
}

struct segment {
    segment_type type;
    std::string  value;

    segment(segment_type type, std::string value) : type(type), value(std::move(value)) {}

    bool operator==(const segment & other) const {
        return type == other.type && value == other.value;
    }

    bool operator!=(const segment & other) const {
        return !(*this == other);
    }
};



===== common/chat-peg-parser.cpp =====
#include "chat-peg-parser.h"

#include "chat-auto-parser.h"
#include "ggml.h"
#include "peg-parser.h"

#include <cstdint>
#include <functional>

using ordered_json = common_json;

static std::string_view trim_trailing_space(std::string_view sv, int max = -1) {
    int count = 0;
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.back()))) {
        if (max != -1 && count >= max) {
            break;
        }
        sv.remove_suffix(1);
        count++;
    }
    return sv;
}

static std::string_view trim_leading_space(std::string_view sv, int max = -1) {
    int count = 0;
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.front()))) {
        if (max != -1 && count >= max) {
            break;
        }
        sv.remove_prefix(1);
        count++;
    }
    return sv;
}

static std::string_view trim(std::string_view sv) {
    return trim_trailing_space(trim_leading_space(sv, 1));
}

// Count the number of unclosed '{' braces in a JSON-like string,
// properly skipping braces inside quoted strings.
static int json_brace_depth(const std::string & s) {
    int  depth     = 0;
    bool in_string = false;
    bool escaped   = false;
    for (char c : s) {
        if (escaped) {
            escaped = false;
            continue;
        }
        if (c == '\\' && in_string) {
            escaped = true;
            continue;
        }
        if (c == '"') {
            in_string = !in_string;
            continue;
        }
        if (!in_string) {
            if (c == '{') {
                depth++;
            } else if (c == '}') {
                depth--;
            }
        }
    }
    return depth;
}

// JSON-escape a string and return the inner content (without surrounding quotes).
static std::string escape_json_string_inner(const std::string & s) {
    std::string escaped = ordered_json(s).dump();
    if (escaped.size() >= 2 && escaped.front() == '"' && escaped.back() == '"') {
        return escaped.substr(1, escaped.size() - 2);
    }
    return escaped;
}

// Convert Python-style single-quoted strings to JSON double-quoted strings
// Only converts outer string delimiters, properly handling escape sequences:
// - {'key': 'value'} -> {"key": "value"}
// - {'code': 'print(\'hello\')'} -> {"code": "print('hello')"}
// - {'msg': 'He said "hi"'} -> {"msg": "He said \"hi\""}
static std::string normalize_quotes_to_json(const std::string & input) {
    std::string result;
    result.reserve(input.size() + 16);  // May need extra space for escaping

    bool in_single_quoted = false;
    bool in_double_quoted = false;

    auto is_word_char = [](char ch) { return std::isalnum(static_cast<unsigned char>(ch)) || ch == '_'; };

    for (size_t i = 0; i < input.size(); ++i) {
        char c = input[i];

        // Handle escape sequences
        if (c == '\\' && i + 1 < input.size()) {
            char next = input[i + 1];

            if (in_single_quoted) {
                // Inside a single-quoted string being converted to double quotes
                if (next == '\'') {
                    // \' -> ' (escaped single quote becomes unescaped in double-quoted string)
                    result += '\'';
                    ++i;
                    continue;
                }
                if (next == '"') {
                    // \" stays as \" (already escaped, works in double-quoted string)
                    result += "\\\"";
                    ++i;
                    continue;
                }
                // Other escapes (\n, \\, etc.): pass through both characters
                result += c;
                result += next;
                ++i;
                continue;
            }

            if (in_double_quoted) {
                // Inside a double-quoted string - pass through escape sequences as-is
                result += c;
                result += next;
                ++i;
                continue;
            }

            // Outside any string - just pass through the backslash
            result += c;
            continue;
        }

        // Handle quote characters
        if (c == '"') {
            if (in_single_quoted) {
                // Unescaped double quote inside single-quoted string -> must escape for JSON
                result += "\\\"";
            } else {
                // Double quote as string delimiter or outside strings
                in_double_quoted = !in_double_quoted;
                result += c;
            }
        } else if (c == '\'') {
            if (in_double_quoted) {
                // Single quote inside double-quoted string -> pass through
                result += c;
            } else if (in_single_quoted) {
                // Closing single quote -> convert to double quote
                in_single_quoted = false;
                result += '"';
            } else {
                // Opening single quote -> convert to double quote
                in_single_quoted = true;
                result += '"';
            }
        } else if (!in_single_quoted && !in_double_quoted && (c == 'T' || c == 'F' || c == 'N') &&
                   (i == 0 || !is_word_char(input[i - 1]))) {
            // Python literals -> JSON; prefix match keeps streamed partials monotonic.
            static constexpr std::pair<std::string_view, std::string_view> literals[] = {
                { "True", "true" }, { "False", "false" }, { "None", "null" },
            };
            size_t n = 0;
            while (i + n < input.size() && is_word_char(input[i + n])) {
                ++n;
            }
            std::string_view token(input.data() + i, n);
            bool matched = false;
            for (const auto & [py, js] : literals) {
                if (py.substr(0, n) == token) {
                    result += js.substr(0, n);
                    i += n - 1;
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                result += c;
            }
        } else {
            result += c;
        }
    }

    return result;
}

void tag_based_peg_mapper::from_ast(const common_peg_ast_arena & arena, const common_peg_parse_result & result) {
    arena.visit(result, [this](const common_peg_ast_node & node) {
        if (!node.tag.empty()) {
            tags[node.tag] = std::string(node.text);
        }
    });
}

tagged_parse_result tagged_peg_parser::parse_and_extract(const std::string & input, common_peg_parse_flags extra_flags) const {
    common_peg_parse_context ctx(input, flags | extra_flags);
    auto parse_result = arena.parse(ctx);

    tag_based_peg_mapper mapper;
    mapper.from_ast(ctx.ast, parse_result);

    return { std::move(parse_result), std::move(mapper.tags) };
}

tagged_parse_result tagged_peg_parser::parse_anywhere_and_extract(const std::string & input) const {
    if (input.empty()) {
        return parse_and_extract(input);
    }
    for (size_t i = 0; i < input.size(); i++) {
        common_peg_parse_context ctx(input, flags);
        auto parse_result = arena.parse(ctx, i);
        if (parse_result.success() || i == input.size() - 1) {
            tag_based_peg_mapper mapper;
            mapper.from_ast(ctx.ast, parse_result);
            return { std::move(parse_result), std::move(mapper.tags) };
        }
    }
    GGML_ABORT("Should not happen");
}

tagged_peg_parser build_tagged_peg_parser(
    const std::function<common_peg_parser(common_peg_parser_builder & builder)> & fn) {
    common_peg_parser_builder builder;
    builder.set_root(fn(builder));
    return { builder.build() };
}

common_peg_parser common_chat_peg_builder::tag_with_safe_content(const std::string &       tag_name,
                                                                 const std::string &       marker,
                                                                 const common_peg_parser & p) {
    if (marker.empty()) {
        return zero_or_more(choice({ p, rule(tag_name, content(any())) }));
    }
    auto content_chunk = rule(tag_name, content(negate(literal(marker)) + any() + until(marker)));
    return zero_or_more(choice({ p, content_chunk }));
}

common_peg_parser common_chat_peg_builder::permute(const std::string &                    rule_prefix,
                                                   const std::vector<common_peg_parser> & parsers) {
    if (parsers.empty()) {
        return eps();
    }

    if (parsers.size() == 1 || parsers.size() > COMMON_CHAT_MAX_PERMUTE) {
        return sequence(parsers);
    }

    std::map<uint32_t, common_peg_parser>      rules;
    std::function<common_peg_parser(uint32_t)> remaining_of;

    remaining_of = [&](uint32_t remaining) -> common_peg_parser {
        if (remaining == 0) {
            return eps();
        }

        auto cached = rules.find(remaining);
        if (cached != rules.end()) {
            return cached->second;
        }

        auto alternatives = choice();
        for (size_t i = 0; i < parsers.size(); i++) {
            const uint32_t bit = 1u << i;
            if (remaining & bit) {
                alternatives |= parsers[i] + remaining_of(remaining & ~bit);
            }
        }

        return rules.emplace(remaining, rule(rule_prefix + "-" + std::to_string(remaining), alternatives)).first->second;
    };

    return remaining_of((1u << parsers.size()) - 1);
}

std::string & common_chat_peg_mapper::args_target() {
    return (current_tool && !current_tool->name.empty()) ? current_tool->arguments : args_buffer;
}

std::string common_chat_peg_mapper::normalize_container_value(const std::string & input) {
    return normalize_quotes_to_json(input);
}

void common_chat_peg_mapper::from_ast(const common_peg_ast_arena &    arena,
                                      const common_peg_parse_result & parse_result_arg) {
    arena.visit(parse_result_arg, [this](const common_peg_ast_node & node) { map(node); });
    // Flush any pending tool call that was started but never got a name
    // This happens during partial parsing when the tool call is incomplete
    if (pending_tool_call.has_value() && !pending_tool_call->name.empty()) {
        if (!args_buffer.empty()) {
            pending_tool_call->arguments = args_buffer;
        }
        if (closing_quote_pending && !pending_tool_call->arguments.empty()) {
            pending_tool_call->arguments += "\"";
        }
        result.tool_calls.push_back(pending_tool_call.value());
        pending_tool_call.reset();
    }

    // Discard whitespace-only reasoning content (e.g. from <think></think> prefill)
    if (!result.reasoning_content.empty()) {
        bool all_whitespace = true;
        for (char c : result.reasoning_content) {
            if (c != ' ' && c != '\n' && c != '\r' && c != '\t') {
                all_whitespace = false;
                break;
            }
        }
        if (all_whitespace) {
            result.reasoning_content.clear();
        }
    }
}

void common_chat_peg_mapper::map(const common_peg_ast_node & node) {
    // Handle reasoning/content tags
    bool is_reasoning = node.tag == common_chat_peg_builder::REASONING;
    bool is_content   = node.tag == common_chat_peg_builder::CONTENT;

    if (is_reasoning) { // GPT OSS can have more than 1 reasoning block, so concatenate here
        result.reasoning_content += node.sanitized_text();
    }

    if (is_content) {
        // Concatenate content from multiple content nodes (e.g., when reasoning markers
        // are preserved before content markers in reasoning_format=NONE mode)
        result.content += node.sanitized_text();
    }

    // Handle tool-related tags (supporting both JSON and tagged formats)
    bool is_tool_open  = node.tag == common_chat_peg_builder::TOOL_OPEN;
    bool is_tool_close = node.tag == common_chat_peg_builder::TOOL_CLOSE;
    bool is_tool_name  = node.tag == common_chat_peg_builder::TOOL_NAME;
    bool is_tool_id    = node.tag == common_chat_peg_builder::TOOL_ID;
    bool is_tool_args  = node.tag == common_chat_peg_builder::TOOL_ARGS;
    bool is_arg_open   = node.tag == common_chat_peg_builder::TOOL_ARG_OPEN;
    bool is_arg_close  = node.tag == common_chat_peg_builder::TOOL_ARG_CLOSE;
    bool is_arg_name         = node.tag == common_chat_peg_builder::TOOL_ARG_NAME;
    bool is_arg_value        = node.tag == common_chat_peg_builder::TOOL_ARG_VALUE;
    bool is_arg_string_value = node.tag == common_chat_peg_builder::TOOL_ARG_STRING_VALUE;

    if (is_tool_open) {
        pending_tool_call     = common_chat_tool_call();
        current_tool          = &pending_tool_call.value();
        arg_count             = 0;
        args_buffer.clear();
        closing_quote_pending = false;
    }

    if (is_tool_id && current_tool) {
        auto text = trim_trailing_space(node.text);
        if (text.size() >= 2 && text.front() == '"' && text.back() == '"') {
            text = text.substr(1, text.size() - 2);
        }
        current_tool->id = std::string(text);
    }

    if (is_tool_name && current_tool) {
        current_tool->name = std::string(trim_trailing_space(node.text));
        // Now that we have the name, populate the arguments from the buffer
        if (!args_buffer.empty()) {
            current_tool->arguments = args_buffer;
            args_buffer.clear();
        } else if (current_tool->arguments.empty()) {
            current_tool->arguments = "{";
        }
        // Add the tool call to results so streaming can see it
        if (pending_tool_call.has_value()) {
            result.tool_calls.push_back(pending_tool_call.value());
            pending_tool_call.reset();
            current_tool = &result.tool_calls.back();
        }
    }

    if (is_tool_args && current_tool) {
        // For JSON format: arguments come as a complete JSON object
        // For tagged format: built up from individual arg_name/arg_value nodes
        auto text = trim_trailing_space(node.text);
        if (!text.empty() && text.front() == '{') {
            args_target() = std::string(text);
        }
    }

    if (is_arg_open) {
        closing_quote_pending = false;
    }

    if (is_arg_name && current_tool) {
        std::string arg_entry;
        if (arg_count > 0) {
            arg_entry = ",";
        }
        arg_entry += ordered_json(trim(node.text)).dump() + ":";
        ++arg_count;

        auto & target = args_target();
        if (target.empty()) {
            target = "{";
        }
        target += arg_entry;
    }

    if ((is_arg_value || is_arg_string_value) && current_tool) {
        std::string value_content = std::string(node.text);

        std::string value_to_add;
        if (value_content.empty() && is_arg_string_value) {
            // Empty string value - arg_close will add the closing quote
            value_to_add          = "\"";
            closing_quote_pending = true;
        } else if (!value_content.empty() && is_arg_string_value) {
            // Schema declares this as string type - always treat as literal string value
            if (!closing_quote_pending) {
                value_to_add          = "\"";
                closing_quote_pending = true;
            }
            value_to_add += escape_json_string_inner(value_content);
        } else if (!value_content.empty()) {
            // Pythonic scalars/containers -> JSON.
            value_to_add += normalize_container_value(value_content);
        }

        args_target() += value_to_add;
    }

    if (is_arg_close && current_tool) {
        if (closing_quote_pending) {
            args_target() += "\"";
            closing_quote_pending = false;
        }
    }

    if (is_tool_close && current_tool) {
        // Flush buffer to arguments if tool name was never seen
        if (current_tool->name.empty() && !args_buffer.empty()) {
            current_tool->arguments = args_buffer;
            args_buffer.clear();
        }
        // Close any pending string quote
        if (closing_quote_pending) {
            current_tool->arguments += "\"";
            closing_quote_pending = false;
        }
        // Close any unclosed braces (accounts for nested objects)
        for (int d = json_brace_depth(current_tool->arguments); d > 0; d--) {
            current_tool->arguments += "}";
        }
        // Add tool call to results if named; otherwise discard
        if (pending_tool_call.has_value()) {
            if (!current_tool->name.empty()) {
                result.tool_calls.push_back(pending_tool_call.value());
            }
            pending_tool_call.reset();
            current_tool = nullptr;
        }
    }
}

common_peg_parser common_chat_peg_builder::standard_constructed_tools(
    const std::map<std::string, std::string> & markers,
    const ordered_json &                       tools,
    bool                                       parallel_tool_calls,
    bool                                       force_tool_calls) {
    if (!tools.is_array() || tools.empty()) {
        return eps();
    }

    // Extract markers with defaults
    auto get_marker = [&markers](const std::string & key, const std::string & default_val = "") -> std::string {
        auto it = markers.find(key);
        return it != markers.end() ? it->second : default_val;
    };

    std::string section_start    = get_marker("tool_call_start_marker", "<tool_call>");
    std::string section_end      = get_marker("tool_call_end_marker", "</tool_call>");
    std::string func_opener      = get_marker("function_opener", "<function=");
    std::string func_name_suffix = get_marker("function_name_suffix", ">");
    std::string func_closer      = get_marker("function_closer", "</function>");
    std::string param_key_prefix = get_marker("parameter_key_prefix", "<param=");
    std::string param_key_suffix = get_marker("parameter_key_suffix", ">");
    std::string param_closer     = get_marker("parameter_closer", "</param>");

    // Build tool choices for tagged format
    auto tool_choices = choice();

    for (size_t i = 0; i < tools.size(); i++) {
        const auto & tool_def = tools[i];

        if (!tool_def.contains("function")) {
            continue;
        }
        const auto &   function = tool_def.at("function");
        std::string    name     = function.at("name");
        ordered_json   params   = common_chat_tool_parameters(function);

        // Build argument parsers
        auto args = eps();
        if (params.contains("properties") && !params["properties"].empty()) {
            auto arg_choice = choice();
            for (const auto & el : params["properties"].items()) {
                const std::string & prop_name = el.key();

                auto arg_name_parser =
                    choice({ literal(prop_name), literal("\"" + prop_name + "\""), literal("'" + prop_name + "'") });

                auto arg_rule = tool_arg(tool_arg_open(literal(param_key_prefix)) + tool_arg_name(arg_name_parser) +
                                         literal(param_key_suffix) + tool_arg_value(until(param_closer)) +
                                         tool_arg_close(literal(param_closer)));
                arg_choice |= arg_rule;
            }
            args = zero_or_more(arg_choice + space());
        }

        // Build function parser: <function=name>args</function>
        auto tool_parser = tool(tool_open(literal(func_opener) + tool_name(literal(name)) + literal(func_name_suffix)) +
                                space() + tool_args(args) + space() + tool_close(literal(func_closer)));

        tool_choices |= rule("tool-" + std::to_string(i), tool_parser);
    }

    // Build the section with markers
    auto section =
        parallel_tool_calls ?
            trigger_rule("tool-call", literal(section_start) + space() + one_or_more(tool_choices + space()) +
                                          literal(section_end)) :
            trigger_rule("tool-call", literal(section_start) + space() + tool_choices + space() + literal(section_end));

    return force_tool_calls ? section : optional(section);
}

// Like python_value(), but the leaf also accepts JSON-cased true/false/null, used by LFM2/LFM2.5
common_peg_parser common_chat_peg_builder::python_or_json_value() {
    return rule("python-or-json-value", [this]() {
        auto ws    = space();
        auto value = python_or_json_value();

        auto member  = sequence({ python_string(), ws, literal(":"), ws, value });
        auto members = sequence({ member, zero_or_more(sequence({ ws, literal(","), ws, member })) });
        auto dict    = rule("python-or-json-dict", [&]() {
            return sequence({ literal("{"), ws, choice({ literal("}"), sequence({ members, ws, literal("}") }) }), ws });
        });

        auto elements = sequence({ value, zero_or_more(sequence({ literal(","), ws, value })) });
        auto array    = rule("python-or-json-array", [&]() {
            return sequence({ literal("["), ws, choice({ literal("]"), sequence({ elements, ws, literal("]") }) }), ws });
        });

        return choice({ dict, array, python_string(), python_number(),
                        python_bool(), python_null(), json_bool(), json_null() });
    });
}

// Python-style tool calls: name(arg1="value1", arg2=123)
// Used only by LFM2 for now, so we don't merge it into autoparser
common_peg_parser common_chat_peg_builder::python_style_tool_calls(
    const ordered_json & tools,
    bool                 parallel_tool_calls,
    bool                 allow_json_literals) {
    if (!tools.is_array() || tools.empty()) {
        return eps();
    }

    auto tool_choices = choice();

    for (size_t i = 0; i < tools.size(); i++) {
        const auto & tool_def = tools[i];
        if (!tool_def.contains("function")) {
            continue;
        }
        const auto &   function = tool_def.at("function");
        std::string    name     = function.at("name");
        ordered_json   params   = common_chat_tool_parameters(function);

        auto args = eps();
        if (params.contains("properties") && !params["properties"].empty()) {
            auto arg_choice = choice();
            for (const auto & el : params["properties"].items()) {
                const std::string & prop_name = el.key();
                const auto & prop_def = el.value();
                bool is_string_type = (prop_def.contains("type") && prop_def["type"] == "string");

                auto arg_name_parser = literal(prop_name);

                common_peg_parser arg_value_parser = eps();
                // Quoted literal as a value: normalize_quotes_to_json preserves escapes.
                auto string_value_parser = tool_arg_value(choice({
                    literal("\"") + string_content('"') + literal("\""),
                    literal("'") + string_content('\'') + literal("'")
                }));

                if (is_string_type) {
                    arg_value_parser = string_value_parser;
                } else {
                    arg_value_parser = tool_arg_value(allow_json_literals ? python_or_json_value() : python_value());
                }

                // Full argument: name="value" or name=value
                auto arg_rule = tool_arg(
                    tool_arg_open(tool_arg_name(arg_name_parser) + literal("=")) +
                    arg_value_parser +
                    tool_arg_close(eps())
                );
                arg_choice |= arg_rule;
            }

            args = arg_choice + zero_or_more("," + space() + arg_choice);
        }

        auto tool_parser = tool(tool_open(tool_name(literal(name)) + literal("(")) +
            space() + tool_args(args) + space() + tool_close(literal(")"))
        );

        tool_choices |= rule("tool-" + std::to_string(i), tool_parser);
    }

    if (parallel_tool_calls) {
        return "[" + space() + tool_choices + zero_or_more("," + space() + tool_choices) + space() + "]";
    }
    return "[" + space() + tool_choices + space() + "]";
}

// Helper: Parse dot notation key into prefix and field name
static std::pair<std::string, std::string> parse_key_spec(const std::string & key) {
    auto dot_pos = key.find('.');
    if (dot_pos == std::string::npos) {
        return {"", key};  // Top-level field
    }
    return {key.substr(0, dot_pos), key.substr(dot_pos + 1)};
}

// Mode 1: function_is_key — parse {"function_name": {...}}
common_peg_parser common_chat_peg_builder::build_json_tools_function_is_key(
    const ordered_json & tools,
    const std::string &  args_key,
    const std::string &  effective_args_key,
    const std::string &  call_id_key,
    const std::string &  gen_call_id_key) {

    auto tool_choices = choice();

    for (size_t i = 0; i < tools.size(); i++) {
        const auto & tool_def = tools[i];
        if (!tool_def.contains("function")) {
            continue;
        }
        const auto &   function = tool_def.at("function");
        std::string    name     = function.at("name");
        ordered_json   params   = common_chat_tool_parameters(function);

        // Build inner object fields
        std::vector<common_peg_parser> inner_fields;

        if (!call_id_key.empty()) {
            auto id_parser = atomic(
                literal("\"" + call_id_key + "\"") + space() + literal(":") + space() +
                literal("\"") + tool_id(string_content('"')) + literal("\"")
            );
            inner_fields.push_back(optional(id_parser + space() + optional(literal(",") + space())));
        }

        if (!gen_call_id_key.empty()) {
            auto gen_id_parser = atomic(
                literal("\"" + gen_call_id_key + "\"") + space() + literal(":") + space() +
                choice({
                    literal("\"") + tool_id(string_content('"')) + literal("\""),
                    tool_id(json_number())
                })
            );
            inner_fields.push_back(optional(gen_id_parser + space() + optional(literal(",") + space())));
        }

        // Arguments — either wrapped in args_key or parsed directly
        common_peg_parser args_parser = eps();
        if (args_key.empty()) {
            args_parser = tool_args(schema(json(), "tool-" + std::to_string(i) + "-schema", params));
        } else {
            args_parser = literal("\"" + effective_args_key + "\"") + space() + literal(":") + space() +
                          tool_args(schema(json(), "tool-" + std::to_string(i) + "-schema", params));
        }
        inner_fields.push_back(args_parser);

        // Build inner object parser
        common_peg_parser inner_object = eps();
        if (args_key.empty() && inner_fields.size() == 1) {
            inner_object = inner_fields[0];
        } else {
            inner_object = literal("{") + space();
            for (size_t i = 0; i < inner_fields.size(); i++) {
                inner_object = inner_object + inner_fields[i];
                if (i < inner_fields.size() - 1) {
                    inner_object = inner_object + space();
                }
            }
            inner_object = inner_object + space() + literal("}");
        }

        auto tool_parser = tool(
            tool_open(literal("{")) + space() +
            literal("\"") + tool_name(literal(name)) + literal("\"") +
            space() + literal(":") + space() +
            inner_object +
            space() + tool_close(literal("}"))
        );

        tool_choices |= rule("tool-" + std::to_string(i), tool_parser);
    }

    return tool_choices;
}

// Mode 2: Nested keys (dot notation like "function.name")
common_peg_parser common_chat_peg_builder::build_json_tools_nested_keys(
    const ordered_json & tools,
    const std::string &  effective_name_key,
    const std::string &  effective_args_key,
    const std::string &  call_id_key,
    const std::string &  gen_call_id_key) {

    auto tool_choices = choice();

    auto name_spec = parse_key_spec(effective_name_key);
    auto args_spec = parse_key_spec(effective_args_key);

    std::string nested_prefix     = !name_spec.first.empty() ? name_spec.first  : args_spec.first;
    std::string nested_name_field = !name_spec.first.empty() ? name_spec.second  : effective_name_key;
    std::string nested_args_field = !args_spec.first.empty() ? args_spec.second  : effective_args_key;

    for (size_t i = 0; i < tools.size(); i++) {
        const auto & tool_def = tools[i];
        if (!tool_def.contains("function")) {
            continue;
        }
        const auto &   function = tool_def.at("function");
        std::string    name     = function.at("name");
        ordered_json   params   = common_chat_tool_parameters(function);

        auto nested_name = literal("\"" + nested_name_field + "\"") + space() + literal(":") + space() +
                          atomic(literal("\"") + tool_name(literal(name)) + literal("\""));
        auto nested_args = literal("\"" + nested_args_field + "\"") + space() + literal(":") + space() +
                          tool_args(schema(json(), "tool-" + std::to_string(i) + "-schema", params));

        auto nested_object = literal("{") + space() +
                            nested_name + space() + literal(",") + space() +
                            nested_args +
                            space() + literal("}");

        // Format: { id?, "function": {...} }
        auto tool_parser_body = tool_open(literal("{")) + space();

        if (!call_id_key.empty()) {
            auto id_spec = parse_key_spec(call_id_key);
            if (id_spec.first.empty()) {
                auto id_parser = atomic(
                    literal("\"" + call_id_key + "\"") + space() + literal(":") + space() +
                    literal("\"") + tool_id(string_content('"')) + literal("\"")
                );
                tool_parser_body = tool_parser_body + optional(id_parser + space() + literal(",") + space());
            }
        }

        if (!gen_call_id_key.empty()) {
            auto gen_id_spec = parse_key_spec(gen_call_id_key);
            if (gen_id_spec.first.empty()) {
                auto gen_id_parser = atomic(
                    literal("\"" + gen_call_id_key + "\"") + space() + literal(":") + space() +
                    choice({
                        literal("\"") + tool_id(string_content('"')) + literal("\""),
                        tool_id(json_number())
                    })
                );
                tool_parser_body = tool_parser_body + optional(gen_id_parser + space() + literal(",") + space());
            }
        }

        auto nested_field = literal("\"" + nested_prefix + "\"") + space() + literal(":") + space() + nested_object;
        tool_parser_body = tool_parser_body + nested_field + space() + tool_close(literal("}"));

        tool_choices |= rule("tool-" + std::to_string(i), tool(tool_parser_body));
    }

    return tool_choices;
}

// Mode 3: Flat keys with optional ID fields and parameter ordering
common_peg_parser common_chat_peg_builder::build_json_tools_flat_keys(
    const ordered_json &             tools,
    const std::string &              effective_name_key,
    const std::string &              effective_args_key,
    const std::string &              call_id_key,
    const std::string &              gen_call_id_key,
    const std::vector<std::string> & parameters_order,
    bool                             accept_openai_wrapper) {

    auto tool_choices    = choice();
    auto name_key_parser = literal("\"" + effective_name_key + "\"");
    auto args_key_parser = literal("\"" + effective_args_key + "\"");

    for (size_t i = 0; i < tools.size(); i++) {
        const auto & tool_def = tools[i];
        if (!tool_def.contains("function")) {
            continue;
        }
        const auto &   function = tool_def.at("function");
        std::string    name     = function.at("name");
        ordered_json   params   = common_chat_tool_parameters(function);

        auto tool_name_ = name_key_parser + space() + literal(":") + space() +
                         atomic(literal("\"") + tool_name(literal(name)) + literal("\""));
        auto tool_args_ = args_key_parser + space() + literal(":") + space() +
                         tool_args(schema(json(), "tool-" + std::to_string(i) + "-schema", params));

        // Build ID parsers if keys are provided
        common_peg_parser id_parser = eps();
        if (!call_id_key.empty()) {
            id_parser = atomic(
                literal("\"" + call_id_key + "\"") + space() + literal(":") + space() +
                choice({
                    literal("\"") + tool_id(string_content('"')) + literal("\""),
                    tool_id(json_number())
                })
            );
        }

        common_peg_parser gen_id_parser = eps();
        if (!gen_call_id_key.empty()) {
            gen_id_parser = atomic(
                literal("\"" + gen_call_id_key + "\"") + space() + literal(":") + space() +
                choice({
                    literal("\"") + tool_id(string_content('"')) + literal("\""),
                    tool_id(json_number())
                })
            );
        }

        // Create (parser, key) pairs for all fields, then sort by parameters_order
        std::vector<std::pair<common_peg_parser, std::string>> parser_pairs;
        parser_pairs.emplace_back(tool_name_, effective_name_key);
        parser_pairs.emplace_back(tool_args_, effective_args_key);
        if (!call_id_key.empty()) {
            parser_pairs.emplace_back(optional(id_parser), call_id_key);
        }
        if (!gen_call_id_key.empty()) {
            parser_pairs.emplace_back(optional(gen_id_parser), gen_call_id_key);
        }

        std::sort(parser_pairs.begin(), parser_pairs.end(),
            [&parameters_order](const auto & a, const auto & b) {
                auto pos_a = std::find(parameters_order.begin(), parameters_order.end(), a.second);
                auto pos_b = std::find(parameters_order.begin(), parameters_order.end(), b.second);
                size_t idx_a = (pos_a == parameters_order.end()) ? parameters_order.size() : std::distance(parameters_order.begin(), pos_a);
                size_t idx_b = (pos_b == parameters_order.end()) ? parameters_order.size() : std::distance(parameters_order.begin(), pos_b);
                return idx_a < idx_b;
            });

        // accept an optional leading "type": "function" field when the model emits the OpenAI wrapper
        common_peg_parser type_field = eps();
        if (accept_openai_wrapper) {
            type_field = optional(literal("\"type\"") + space() + literal(":") + space() +
                                  literal("\"function\"") + space() + literal(",") + space());
        }
        auto ordered_body = tool_open(literal("{")) + space() + type_field;
        for (size_t i = 0; i < parser_pairs.size(); i++) {
            ordered_body = ordered_body + parser_pairs[i].first;
            if (i < parser_pairs.size() - 1) {
                ordered_body = ordered_body + space() + literal(",") + space();
            }
        }
        ordered_body = ordered_body + space() + tool_close(literal("}"));

        tool_choices |= rule("tool-" + std::to_string(i), tool(ordered_body));
    }

    return tool_choices;
}

common_peg_parser common_chat_peg_builder::prefix(const std::string & s, const std::string & delimiter) {
    if (s.empty()) {
        return eps();
    }
    if (delimiter.empty()) {
        return literal(s);
    }
    return literal(s.substr(0, s.find(delimiter)));
}

common_peg_parser common_chat_peg_builder::optspace(const std::string & tag) {
    auto parser = eps();
    size_t end_of_prefix_space = tag.size();
    size_t start_of_suffix_space = tag.size();
    for (size_t i = 0; i < tag.size(); i++) {
        if (!std::isspace(tag[i])) {
            end_of_prefix_space = i;
            break;
        }
    }
    for (size_t i = tag.size(); i > 0; i--) {
        if (!std::isspace(tag[i - 1])) {
            start_of_suffix_space = i;
            break;
        }
    }
    for (size_t i = 0; i < end_of_prefix_space; i++) {
        parser += optional(literal(std::string(1, tag[i])));
    }
    parser += literal(tag.substr(end_of_prefix_space, start_of_suffix_space - end_of_prefix_space));
    for (size_t i = start_of_suffix_space; i < tag.size(); i++) {
        parser += optional(literal(std::string(1, tag[i])));
    }
    return parser;
}

common_peg_parser common_chat_peg_builder::standard_json_tools(
                                                       const std::string &              section_start,
                                                       const std::string &              section_end,
                                                       const ordered_json &             tools,
                                                       bool                             parallel_tool_calls,
                                                       bool                             force_tool_calls,
                                                       const std::string &              name_key,
                                                       const std::string &              args_key,
                                                       bool                             array_wrapped,
                                                       bool                             function_is_key,
                                                       const std::string &              call_id_key,
                                                       const std::string &              gen_call_id_key,
                                                       const std::vector<std::string> & parameters_order,
                                                       bool                             accept_openai_wrapper) {
    if (!tools.is_array() || tools.empty()) {
        return eps();
    }

    std::string effective_name_key = name_key.empty() ? "name" : name_key;
    std::string effective_args_key = args_key.empty() ? "arguments" : args_key;

    // Dispatch to the appropriate builder based on the JSON layout mode
    common_peg_parser tool_choices = eps();
    if (function_is_key) {
        tool_choices = build_json_tools_function_is_key(tools, args_key, effective_args_key, call_id_key, gen_call_id_key);
    } else {
        auto name_spec = parse_key_spec(effective_name_key);
        auto args_spec = parse_key_spec(effective_args_key);
        if (!name_spec.first.empty() || !args_spec.first.empty()) {
            tool_choices = build_json_tools_nested_keys(tools, effective_name_key, effective_args_key, call_id_key, gen_call_id_key);
        } else {
            tool_choices = build_json_tools_flat_keys(tools, effective_name_key, effective_args_key, call_id_key, gen_call_id_key, parameters_order, accept_openai_wrapper);
        }
    }

    // Build the section with markers
    auto tool_calls = tool_choices;
    if (parallel_tool_calls) {
        tool_calls = tool_calls + zero_or_more(space() + literal(",") + space() + tool_choices);
    }

    if (array_wrapped) {
        tool_calls = literal("[") + space() + tool_calls + space() + literal("]");
    }

    auto section =
        trigger_rule("tool-call", literal(section_start) + space() + tool_calls + space() + literal(section_end));

    return force_tool_calls ? section : optional(section);
}

void common_chat_peg_gemma4_mapper::from_ast(const common_peg_ast_arena & arena, const common_peg_parse_result & result) {
    for (const auto & node : result.nodes) {
        visit(arena, node);
    }
}

static std::string gemma4_to_json(const common_peg_ast_arena & arena, common_peg_ast_id id) {
    const auto & node = arena.get(id);

    if (node.text.empty()) {
        return "";
    }

    if (node.rule == "gemma4-number" || node.rule == "gemma4-bool" || node.rule == "gemma4-null") {
        return std::string(node.text);
    }

    if (node.rule == "gemma4-string-content") {
        return escape_json_string_inner(std::string(node.text));
    }

    if (node.rule == "gemma4-string") {
        std::string result = "\"";
        if (!node.children.empty()) {
            result += gemma4_to_json(arena, node.children[0]);
            if (!node.is_partial) {
                result += "\"";
            }
        }
        return result;
    }

    if (node.rule == "gemma4-array") {
        std::string result = "[";

        bool add_comma = false;
        for (auto child_id : node.children) {
            if (add_comma) {
                result += ',';
            }
            add_comma = true;
            result += gemma4_to_json(arena, child_id);
        }

        if (!node.is_partial) {
            result += ']';
        }
        return result;
    }

    if (node.rule == "gemma4-dict-key-name") {
        return std::string(node.text);
    }

    if (node.rule == "gemma4-dict-key") {
        std::string result = "\"";
        if (!node.children.empty()) {
            result += escape_json_string_inner(gemma4_to_json(arena, node.children[0]));
        }
        if (!node.is_partial) {
            result += "\":";
        }
        return result;
    }

    if (node.rule == "gemma4-dict-kv") {
        std::string result;
        for (auto child_id : node.children) {
            result += gemma4_to_json(arena, child_id);
        }
        return result;
    }

    if (node.rule == "gemma4-dict") {
        std::string result = "{";

        bool add_comma = false;
        for (auto child_id : node.children) {
            if (add_comma) {
                result += ',';
            }
            add_comma = true;
            result += gemma4_to_json(arena, child_id);
        }

        if (!node.is_partial) {
            result += '}';
        }
        return result;
    }

    if (node.rule == "gemma4-value") {
        if (!node.children.empty()) {
            return gemma4_to_json(arena, node.children[0]);
        }
        return "";
    }

    return "";
}

void common_chat_peg_gemma4_mapper::visit(const common_peg_ast_arena & arena, common_peg_ast_id id) {
    const auto & node = arena.get(id);

    if (node.tag == "reasoning") {
        result.reasoning_content += node.sanitized_text();
        return;
    }

    if (node.tag == "content") {
        result.content += node.sanitized_text();
        return;
    }

    if (node.tag == "tool") {
        auto name_id = arena.find_by_tag(node, "tool-name");
        auto args_id = arena.find_by_tag(node, "tool-args");

        if (name_id != COMMON_PEG_INVALID_AST_ID && args_id != COMMON_PEG_INVALID_AST_ID) {
            const auto & name_node = arena.get(name_id);
            const auto & args_node = arena.get(args_id);

            if (!name_node.is_partial) {
                common_chat_tool_call call;
                call.name = std::string(name_node.text);
                if (!args_node.children.empty()) {
                    call.arguments = gemma4_to_json(arena, args_node.children[0]);
                }
                result.tool_calls.push_back(call);
            }
        }

        return;
    }

    for (auto child_id : node.children) {
        visit(arena, child_id);
    }
}

static void minimax_m3_collect(const common_peg_ast_arena &     arena,
                               const common_peg_ast_node &      node,
                               const std::string &              tag,
                               std::vector<common_peg_ast_id> & out) {
    for (auto child_id : node.children) {
        const auto & child = arena.get(child_id);
        if (child.tag == tag) {
            out.push_back(child_id);
        } else {
            minimax_m3_collect(arena, child, tag, out);
        }
    }
}

static common_peg_ast_id minimax_m3_value_of(const common_peg_ast_arena & arena, const common_peg_ast_node & node) {
    for (auto child_id : node.children) {
        const auto & tag = arena.get(child_id).tag;
        if (tag == common_chat_peg_builder::TOOL_ARG_VALUE ||
            tag == common_chat_peg_builder::TOOL_ARG_STRING_VALUE ||
            tag == common_chat_peg_minimax_m3_mapper::TOOL_ARG_OBJECT ||
            tag == common_chat_peg_minimax_m3_mapper::TOOL_ARG_ARRAY) {
            return child_id;
        }
    }
    return COMMON_PEG_INVALID_AST_ID;
}

static std::string minimax_m3_value_to_json(const common_peg_ast_arena & arena, common_peg_ast_id id, bool closed);

static std::string minimax_m3_member_to_json(const common_peg_ast_arena & arena, const common_peg_ast_node & node) {
    auto name_id = arena.find_by_tag(node, common_chat_peg_builder::TOOL_ARG_NAME);
    if (name_id == COMMON_PEG_INVALID_AST_ID) {
        return "";
    }

    return ordered_json(arena.get(name_id).text).dump() + ":" +
           minimax_m3_value_to_json(arena, minimax_m3_value_of(arena, node), !node.is_partial);
}

static std::string minimax_m3_container_to_json(const common_peg_ast_arena & arena,
                                                const common_peg_ast_node & node,
                                                bool                        is_object,
                                                bool                        closed) {
    const std::string tag = is_object ? common_chat_peg_builder::TOOL_ARG
                                      : common_chat_peg_minimax_m3_mapper::TOOL_ARG_ITEM;

    std::vector<common_peg_ast_id> entries;
    minimax_m3_collect(arena, node, tag, entries);

    std::string result = is_object ? "{" : "[";

    bool add_comma = false;
    for (auto entry_id : entries) {
        const auto & entry = arena.get(entry_id);

        std::string text;
        if (is_object) {
            text = minimax_m3_member_to_json(arena, entry);
        } else {
            text = minimax_m3_value_to_json(arena, minimax_m3_value_of(arena, entry), !entry.is_partial);
        }

        if (text.empty()) {
            continue;
        }

        if (add_comma) {
            result += ",";
        }
        add_comma = true;
        result += text;
    }

    if (closed) {
        result += is_object ? "}" : "]";
    }
    return result;
}

static std::string minimax_m3_value_to_json(const common_peg_ast_arena & arena, common_peg_ast_id id, bool closed) {
    if (id == COMMON_PEG_INVALID_AST_ID) {
        return "";
    }

    const auto & node = arena.get(id);

    if (node.tag == common_chat_peg_minimax_m3_mapper::TOOL_ARG_OBJECT) {
        return minimax_m3_container_to_json(arena, node, /* is_object = */ true, closed);
    }

    if (node.tag == common_chat_peg_minimax_m3_mapper::TOOL_ARG_ARRAY) {
        return minimax_m3_container_to_json(arena, node, /* is_object = */ false, closed);
    }

    if (node.tag == common_chat_peg_builder::TOOL_ARG_STRING_VALUE) {
        return "\"" + escape_json_string_inner(std::string(node.text)) + (closed ? "\"" : "");
    }

    // Numbers and booleans are written verbatim by the template
    return std::string(node.text);
}

void common_chat_peg_minimax_m3_mapper::from_ast(const common_peg_ast_arena &    arena,
                                                 const common_peg_parse_result & result) {
    for (const auto & node : result.nodes) {
        visit(arena, node);
    }
}

void common_chat_peg_minimax_m3_mapper::visit(const common_peg_ast_arena & arena, common_peg_ast_id id) {
    const auto & node = arena.get(id);

    if (node.tag == common_chat_peg_builder::REASONING) {
        result.reasoning_content += node.sanitized_text();
        return;
    }

    if (node.tag == common_chat_peg_builder::CONTENT) {
        result.content += node.sanitized_text();
        return;
    }

    if (node.tag == common_chat_peg_builder::TOOL) {
        auto name_id = arena.find_by_tag(node, common_chat_peg_builder::TOOL_NAME);
        if (name_id != COMMON_PEG_INVALID_AST_ID) {
            common_chat_tool_call call;
            call.name      = std::string(arena.get(name_id).text);
            call.arguments = minimax_m3_container_to_json(arena, node, /* is_object = */ true, !node.is_partial);
            result.tool_calls.push_back(call);
        }
        return;
    }

    for (auto child_id : node.children) {
        visit(arena, child_id);
    }
}



===== common/chat-peg-parser.h =====
#pragma once

#include "chat.h"
#include "peg-parser.h"

#include <map>
#include <optional>
#include <vector>

class common_chat_peg_mapper {
  public:
    common_chat_msg & result;

    common_chat_peg_mapper(common_chat_msg & msg) : result(msg) {}

    virtual ~common_chat_peg_mapper() = default;

    virtual void from_ast(const common_peg_ast_arena & arena, const common_peg_parse_result & result);
    virtual void map(const common_peg_ast_node & node);
  protected:
    virtual std::string normalize_container_value(const std::string & input);
  private:
      // Tool call handling state
      std::optional<common_chat_tool_call> pending_tool_call;  // Tool call waiting for name
      common_chat_tool_call *              current_tool          = nullptr;
      int                                  arg_count             = 0;
      bool                                 closing_quote_pending = false;
      std::string                          args_buffer;  // Buffer to delay arguments until tool name is known

      // Returns a reference to the active argument destination string.
      // Before tool_name is known, writes go to args_buffer; after, to current_tool->arguments.
      std::string & args_target();
};

class common_chat_peg_gemma4_mapper : public common_chat_peg_mapper {
  public:
    common_chat_peg_gemma4_mapper(common_chat_msg & msg) : common_chat_peg_mapper(msg) {}
    virtual void from_ast(const common_peg_ast_arena & arena, const common_peg_parse_result & result);
  private:
    void visit(const common_peg_ast_arena & arena, common_peg_ast_id id);
};

class common_chat_peg_minimax_m3_mapper : public common_chat_peg_mapper {
  public:
    static constexpr const char * TOOL_ARG_OBJECT = "tool-arg-object";
    static constexpr const char * TOOL_ARG_ARRAY  = "tool-arg-array";
    static constexpr const char * TOOL_ARG_ITEM   = "tool-arg-item";

    common_chat_peg_minimax_m3_mapper(common_chat_msg & msg) : common_chat_peg_mapper(msg) {}
    virtual void from_ast(const common_peg_ast_arena & arena, const common_peg_parse_result & result);
  private:
    void visit(const common_peg_ast_arena & arena, common_peg_ast_id id);
};

struct content_structure;
struct tool_call_structure;

constexpr size_t COMMON_CHAT_MAX_PERMUTE = 6;

class common_chat_peg_builder : public common_peg_parser_builder {
  public:
    // Tag constants (from former common_chat_peg_base_builder)
    static constexpr const char * REASONING_BLOCK = "reasoning-block";
    static constexpr const char * REASONING       = "reasoning";
    static constexpr const char * CONTENT         = "content";

    // Tag constants
    static constexpr const char * TOOL           = "tool";
    static constexpr const char * TOOL_OPEN      = "tool-open";
    static constexpr const char * TOOL_CLOSE     = "tool-close";
    static constexpr const char * TOOL_ID        = "tool-id";
    static constexpr const char * TOOL_NAME      = "tool-name";
    static constexpr const char * TOOL_ARGS      = "tool-args";
    static constexpr const char * TOOL_ARG       = "tool-arg";
    static constexpr const char * TOOL_ARG_OPEN  = "tool-arg-open";
    static constexpr const char * TOOL_ARG_CLOSE = "tool-arg-close";
    static constexpr const char * TOOL_ARG_NAME         = "tool-arg-name";
    static constexpr const char * TOOL_ARG_VALUE        = "tool-arg-value";
    static constexpr const char * TOOL_ARG_STRING_VALUE = "tool-arg-string-value";  // For schema-declared string types

    // Low-level tag methods (from former common_chat_peg_base_builder)
    common_peg_parser reasoning_block(const common_peg_parser & p) { return tag(REASONING_BLOCK, p); }

    common_peg_parser reasoning(const common_peg_parser & p) { return tag(REASONING, p); }

    common_peg_parser content(const common_peg_parser & p) { return tag(CONTENT, p); }

    common_peg_parser tag_with_safe_content(const std::string &       tag_name,
                        const std::string &       marker,
                        const common_peg_parser & p);

    // Low-level tag methods
    common_peg_parser tool(const common_peg_parser & p) { return tag(TOOL, p); }
    common_peg_parser tool_open(const common_peg_parser & p) { return atomic(tag(TOOL_OPEN, p)); }
    common_peg_parser tool_close(const common_peg_parser & p) { return atomic(tag(TOOL_CLOSE, p)); }
    common_peg_parser tool_id(const common_peg_parser & p) { return atomic(tag(TOOL_ID, p)); }
    common_peg_parser tool_name(const common_peg_parser & p) { return atomic(tag(TOOL_NAME, p)); }
    common_peg_parser tool_args(const common_peg_parser & p) { return tag(TOOL_ARGS, p); }
    common_peg_parser tool_arg(const common_peg_parser & p) { return tag(TOOL_ARG, p); }
    common_peg_parser tool_arg_open(const common_peg_parser & p) { return atomic(tag(TOOL_ARG_OPEN, p)); }
    common_peg_parser tool_arg_close(const common_peg_parser & p) { return atomic(tag(TOOL_ARG_CLOSE, p)); }
    common_peg_parser tool_arg_name(const common_peg_parser & p) { return atomic(tag(TOOL_ARG_NAME, p)); }
    common_peg_parser tool_arg_value(const common_peg_parser & p) { return tag(TOOL_ARG_VALUE, p); }

    // Use for schema-declared string types - won't be treated as potential JSON container
    common_peg_parser tool_arg_string_value(const common_peg_parser & p) { return tag(TOOL_ARG_STRING_VALUE, p); }
    common_peg_parser tool_arg_json_value(const common_peg_parser & p) { return tag(TOOL_ARG_VALUE, p); }


    // Matches every parser exactly once, in any order.
    common_peg_parser permute(const std::string & rule_prefix, const std::vector<common_peg_parser> & parsers);

    // Return a parser that parses the prefix of a string, up to a given delimiter.
    common_peg_parser prefix(const std::string & s, const std::string & delimiter = {});

    // Return a parser that parses all elements of tag, but leading and trailing spaces are optional
    common_peg_parser optspace(const std::string & tag);

    // Legacy-compatible helper for building standard JSON tool calls
    // Used by tests and manual parsers
    // name_key/args_key: JSON key names for function name and arguments
    //   Empty or "name"/"arguments" will accept both common variations
    //   Supports dot notation for nested objects (e.g., "function.name")
    // array_wrapped: if true, tool calls are wrapped in JSON array [...]
    // function_is_key: if true, function name is the JSON key (e.g., {"func_name": {...}})
    // call_id_key: JSON key for string call ID (e.g., "id")
    // gen_call_id_key: JSON key for generated integer call ID (e.g., "tool_call_id")
    // parameters_order: order in which JSON fields should be parsed
    common_peg_parser standard_json_tools(const std::string &              section_start,
                                          const std::string &              section_end,
                                          const common_json &   tools,
                                          bool                             parallel_tool_calls,
                                          bool                             force_tool_calls,
                                          const std::string &              name_key = "",
                                          const std::string &              args_key = "",
                                          bool                             array_wrapped = false,
                                          bool                             function_is_key = false,
                                          const std::string &              call_id_key = "",
                                          const std::string &              gen_call_id_key = "",
                                          const std::vector<std::string> & parameters_order = {},
                                          bool                             accept_openai_wrapper = false);

    // Legacy-compatible helper for building XML/tagged style tool calls
    // Used by tests and manual parsers
    common_peg_parser standard_constructed_tools(const std::map<std::string, std::string> & markers,
                                                 const common_json &             tools,
                                                 bool                                       parallel_tool_calls,
                                                 bool                                       force_tool_calls);

    // Helper for Python-style function call format: name(arg1="value1", arg2=123)
    // Used by LFM2 and similar templates
    common_peg_parser python_style_tool_calls(const common_json & tools,
                                              bool                           parallel_tool_calls,
                                              bool                           allow_json_literals);

  private:
    // Python values plus JSON true/false/null.
    common_peg_parser python_or_json_value();

    // Implementation helpers for standard_json_tools — one per JSON tool call layout mode
    common_peg_parser build_json_tools_function_is_key(const common_json & tools,
                                                       const std::string &            args_key,
                                                       const std::string &            effective_args_key,
                                                       const std::string &            call_id_key,
                                                       const std::string &            gen_call_id_key);

    common_peg_parser build_json_tools_nested_keys(const common_json & tools,
                                                   const std::string &            effective_name_key,
                                                   const std::string &            effective_args_key,
                                                   const std::string &            call_id_key,
                                                   const std::string &            gen_call_id_key);

    common_peg_parser build_json_tools_flat_keys(const common_json &   tools,
                                                 const std::string &              effective_name_key,
                                                 const std::string &              effective_args_key,
                                                 const std::string &              call_id_key,
                                                 const std::string &              gen_call_id_key,
                                                 const std::vector<std::string> & parameters_order,
                                                 bool                             accept_openai_wrapper);
};

inline common_peg_arena build_chat_peg_parser(
  const std::function<common_peg_parser(common_chat_peg_builder & builder)> & fn) {
  common_chat_peg_builder builder;
  builder.set_root(fn(builder));
  return builder.build();
}

class tag_based_peg_mapper {
  public:
    std::map<std::string, std::string> tags;

    void from_ast(const common_peg_ast_arena & arena, const common_peg_parse_result & result);
};

struct tagged_parse_result {
    common_peg_parse_result              result;
    std::map<std::string, std::string> tags;
};

struct tagged_peg_parser {
    common_peg_arena arena;
    common_peg_parse_flags flags = COMMON_PEG_PARSE_FLAG_NONE;

    tagged_peg_parser & withDebug() {
      flags |= COMMON_PEG_PARSE_FLAG_DEBUG;
      return *this;
    }

    tagged_peg_parser & withoutDebug() {
      flags = flags & ~COMMON_PEG_PARSE_FLAG_DEBUG;
      return *this;
    }

    tagged_parse_result parse_and_extract(const std::string & input, common_peg_parse_flags extra_flags = COMMON_PEG_PARSE_FLAG_NONE) const;
    tagged_parse_result parse_anywhere_and_extract(const std::string & input) const;
};

tagged_peg_parser build_tagged_peg_parser(
    const std::function<common_peg_parser(common_peg_parser_builder & builder)> & fn);



===== tests/CMakeLists.txt =====
llama_add_compile_flags()

# Optional args:
# - LINK: libraries to link against (defaults to llama-common, which also brings llama)
function(llama_build source)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs LINK)
    cmake_parse_arguments(LLAMA_BUILD "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_BUILD_UNPARSED_ARGUMENTS})

    if (NOT LLAMA_BUILD_LINK)
        set(LLAMA_BUILD_LINK llama-common)
    endif()

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
    target_link_libraries(${TEST_TARGET} PRIVATE ${LLAMA_BUILD_LINK})
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
endfunction()

function(llama_test target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    set(TEST_TARGET ${target})

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

function(llama_test_cmd target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND ${target}
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# Builds and runs a test source file.
# Optional args:
# - NAME: name of the executable & test target (defaults to the source file name without extension)
# - LABEL: label for the test (defaults to main)
# - ARGS: arguments to pass to the test executable
# - WORKING_DIRECTORY
# - LINK: libraries to link against (defaults to llama-common, which also brings llama)
function(llama_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS LINK)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS})

    if (NOT LLAMA_TEST_LINK)
        set(LLAMA_TEST_LINK llama-common)
    endif()

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
    target_link_libraries(${TEST_TARGET} PRIVATE ${LLAMA_TEST_LINK})

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# build test-tokenizer-0 target once and add many tests
llama_build(test-tokenizer-0.cpp)

llama_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge          ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-bert-bge.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-command-r         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-command-r.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-coder.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm      ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-llm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gemma-4           ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gemma-4.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-bpe         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-spm         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-mpt               ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-phi-3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen35            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen35.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-refact            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

if (NOT WIN32)
    llama_test_cmd(
        ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizers-repo.sh
        NAME test-tokenizers-ggml-vocabs
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        ARGS https://huggingface.co/ggml-org/vocabs ${PROJECT_SOURCE_DIR}/models/ggml-vocabs
    )
endif()

if (LLAMA_LLGUIDANCE)
    llama_build_and_test(test-grammar-llguidance.cpp ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
endif ()

# llama_context state tests: KV and recurrent state save, load, copy, rollback
set(LLAMA_CONTEXT_TEST_SRC
    test-llama-context.cpp
    llama-context/cases-kv.cpp
    llama-context/cases-rs.cpp
    llama-context/test-case.cpp
    llama-context/test-log.cpp
    llama-context/test-runner.cpp
    llama-context/test-util.cpp
)
llama_build(${LLAMA_CONTEXT_TEST_SRC})

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API (when building with shared libraries)
    llama_build_and_test(test-unicode.cpp LINK llama)
    llama_build_and_test(test-sampling.cpp LINK llama)
    llama_build_and_test(test-reasoning-budget.cpp)
    llama_build_and_test(test-grammar-parser.cpp LINK llama)
    llama_build_and_test(test-grammar-integration.cpp)
    llama_build_and_test(test-llama-grammar.cpp LINK llama)
    llama_build_and_test(test-batch-alloc.cpp LINK llama)
    llama_build_and_test(test-chat.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    target_include_directories(test-chat PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-chat PRIVATE server-context)
    llama_build_and_test(test-server-component.cpp)
    target_include_directories(test-server-component PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-server-component PRIVATE server-context)
    llama_build_and_test(test-json-schema-to-grammar.cpp)

    if (NOT GGML_BACKEND_DL)
        llama_build(test-quantize-stats.cpp)
    endif()

    llama_build(test-gbnf-validator.cpp LINK llama)

    # build test-tokenizer-1-bpe target once and add many tests
    llama_build(test-tokenizer-1-bpe.cpp)

    # TODO: disabled due to slowness
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-aquila    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-aquila.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-falcon    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-2     ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-neox  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-neox.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-llama-bpe ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf --ignore-merges)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-mpt       ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-refact    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-starcoder ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

    # build test-tokenizer-1-spm target once and add many tests
    llama_build(test-tokenizer-1-spm.cpp)

    llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-llama-spm ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
    #llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-baichuan  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-baichuan.gguf)

    # llama_build_and_test(test-double-float.cpp) # SLOW

    # TODO: after `test-llama-archs` is refactored to not link internal symbols, we can move all related tests out of
    #       this branch and run them on all platforms
    llama_build(test-llama-archs.cpp)

    set(MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-models/")
    file(MAKE_DIRECTORY "${MODEL_DIR}")

    llama_test(
        test-llama-archs
        NAME test-generate-models
        LABEL main
        ARGS -o "${MODEL_DIR}"
    )
    set_tests_properties(test-generate-models PROPERTIES
        FIXTURES_SETUP generate-models
    )

    # Test context state handling across all architectures, using the generated dummy models
    llama_test(
        test-llama-context
        LABEL main
        ARGS --models "${MODEL_DIR}"
    )
    # abort on graph reallocations that the initial reserve should have covered
    set_tests_properties(test-llama-context PROPERTIES
        FIXTURES_REQUIRED generate-models
        ENVIRONMENT "GGML_SCHED_DEBUG_REALLOC=1"
    )

    llama_build(test-fusion.cpp)
endif()

llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp)
llama_build_and_test(test-jinja.cpp)
llama_test(test-jinja NAME test-jinja-py ARGS -py LABEL python)
llama_build_and_test(test-json-schema.cpp)
llama_build_and_test(test-chat-auto-parser.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
llama_build_and_test(test-chat-template.cpp)
# debug tool for chat template differential analysis (not registered as a test, run it manually)
llama_build(test-chat-analysis.cpp)
llama_build_and_test(test-log.cpp)
llama_build_and_test(
    test-peg-parser.cpp
    peg-parser/simple-tokenize.cpp
    peg-parser/test-basic.cpp
    peg-parser/test-gbnf-generation.cpp
    peg-parser/test-json-parser.cpp
    peg-parser/test-python-dict-parser.cpp
    peg-parser/test-unicode.cpp
    peg-parser/tests.h
)

if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "s390x")
    set(MODEL_NAME "tinyllamas/stories15M-q4_0.gguf")
    set(MODEL_HASH "SHA256=66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739")
else()
    set(MODEL_NAME "tinyllamas/stories15M-be.Q4_0.gguf")
    set(MODEL_HASH "SHA256=9aec857937849d976f30397e97eb1cabb53eb9dcb1ce4611ba8247fb5f44c65d")
endif()
set(MODEL_DEST "${CMAKE_BINARY_DIR}/${MODEL_NAME}")

add_test(NAME test-download-model COMMAND ${CMAKE_COMMAND}
    -DDEST=${MODEL_DEST}
    -DNAME=${MODEL_NAME}
    -DHASH=${MODEL_HASH}
    -P ${PROJECT_SOURCE_DIR}/cmake/download-models.cmake
)
set_tests_properties(test-download-model PROPERTIES FIXTURES_SETUP test-download-model)

# the context state cases against a real model, the sweep above uses the generated dummy ones
llama_test(test-llama-context NAME test-llama-context-model LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-llama-context-model PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-thread-safety.cpp ARGS -m "${MODEL_DEST}" -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2)
set_tests_properties(test-thread-safety PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-arg-parser.cpp)
llama_build_and_test(test-model-resolution.cpp)
# the test serves its repos from an httplib server, and the library links it privately
target_link_libraries(test-model-resolution PRIVATE cpp-httplib)

if (NOT LLAMA_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
  # TODO: repair known memory leaks
  llama_build_and_test(test-opt.cpp LINK ggml)
endif()
if (GGML_METAL AND NOT GGML_BACKEND_DL)
    # calls the Metal backend's graph optimizer and fusion table directly
    llama_build_and_test(test-metal-graph-optimize.cpp LINK ggml ggml-metal)
    target_include_directories(test-metal-graph-optimize PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src ${PROJECT_SOURCE_DIR}/ggml/src/ggml-metal)
endif()

llama_build(test-backend-ops.cpp LINK ggml)

llama_build_and_test(test-tiled-mulmat.cpp LINK ggml)
llama_build_and_test(test-model-load-cancel.cpp LABEL "model")
llama_build_and_test(test-autorelease.cpp       LABEL "model")
llama_build_and_test(test-backend-sampler.cpp   LABEL "model")

if (APPLE)
    llama_build(test-rset-release.cpp)
endif()

if (NOT GGML_BACKEND_DL AND GGML_CPU)
    # these tests use the backends directly and cannot be built with dynamic loading
    llama_build_and_test(test-barrier.cpp LINK ggml)
    llama_build_and_test(test-quantize-fns.cpp LINK ggml)
    llama_build_and_test(test-quantize-perf.cpp LINK ggml)
    llama_build_and_test(test-rope.cpp LINK ggml)
    llama_build_and_test(test-col2im-1d.cpp LINK ggml)
    llama_build_and_test(test-conv-1d.cpp LINK ggml)
endif()

# libmtmd
llama_build_and_test(test-mtmd-c-api.c LINK mtmd)

llama_build_and_test(test-mtmd-impl.cpp LINK mtmd)

# GGUF model data fetcher library for tests that need real model metadata
# Only compile when cpp-httplib has SSL support (CPPHTTPLIB_OPENSSL_SUPPORT)
if (TARGET cpp-httplib)
    get_target_property(_cpp_httplib_defs cpp-httplib INTERFACE_COMPILE_DEFINITIONS)
    if (_cpp_httplib_defs MATCHES "CPPHTTPLIB_OPENSSL_SUPPORT")
        add_library(gguf-model-data STATIC gguf-model-data.cpp)
        target_link_libraries(gguf-model-data PRIVATE llama-common cpp-httplib)
        target_include_directories(gguf-model-data PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

        add_executable(test-gguf-model-data test-gguf-model-data.cpp)
        target_link_libraries(test-gguf-model-data PRIVATE gguf-model-data llama-common)
        llama_test(test-gguf-model-data LABEL "model")

        # test-quant-type-selection requires gguf-model-data for remote model metadata
        llama_build_and_test(test-quant-type-selection.cpp LABEL "model" LINK llama)
        target_link_libraries(test-quant-type-selection PRIVATE gguf-model-data)
        target_compile_definitions(test-quant-type-selection PRIVATE
            SNAPSHOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/snapshots")
    endif()
endif()

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)

if (NOT LLAMA_USE_SYSTEM_GGML)
    # Needs non-public ggml-impl.h
    llama_build_and_test(test-gguf.cpp LINK ggml)

    # Needs non-public ggml{,-backend}-impl.h
    llama_build_and_test(test-alloc.cpp LINK ggml)
endif()

llama_build(test-export-graph-ops.cpp)
if (TARGET gguf-model-data)
    target_link_libraries(test-export-graph-ops PRIVATE gguf-model-data)
    target_compile_definitions(test-export-graph-ops PRIVATE LLAMA_HF_FETCH)
endif()



===== models/templates/Qwen3.5-4B.jinja =====
{%- set image_count = namespace(value=0) %}
{%- set video_count = namespace(value=0) %}
{%- macro render_content(content, do_vision_count, is_system_content=false) %}
    {%- if content is string %}
        {{- content }}
    {%- elif content is iterable and content is not mapping %}
        {%- for item in content %}
            {%- if 'image' in item or 'image_url' in item or item.type == 'image' %}
                {%- if is_system_content %}
                    {{- raise_exception('System message cannot contain images.') }}
                {%- endif %}
                {%- if do_vision_count %}
                    {%- set image_count.value = image_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}
                    {{- 'Picture ' ~ image_count.value ~ ': ' }}
                {%- endif %}
                {{- '<|vision_start|><|image_pad|><|vision_end|>' }}
            {%- elif 'video' in item or item.type == 'video' %}
                {%- if is_system_content %}
                    {{- raise_exception('System message cannot contain videos.') }}
                {%- endif %}
                {%- if do_vision_count %}
                    {%- set video_count.value = video_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}
                    {{- 'Video ' ~ video_count.value ~ ': ' }}
                {%- endif %}
                {{- '<|vision_start|><|video_pad|><|vision_end|>' }}
            {%- elif 'text' in item %}
                {{- item.text }}
            {%- else %}
                {{- raise_exception('Unexpected item type in content.') }}
            {%- endif %}
        {%- endfor %}
    {%- elif content is none or content is undefined %}
        {{- '' }}
    {%- else %}
        {{- raise_exception('Unexpected content type.') }}
    {%- endif %}
{%- endmacro %}
{%- if not messages %}
    {{- raise_exception('No messages provided.') }}
{%- endif %}
{%- if tools and tools is iterable and tools is not mapping %}
    {{- '<|im_start|>system\n' }}
    {{- "# Tools\n\nYou have access to the following functions:\n\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>" }}
    {{- '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>' }}
    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {%- if content %}
            {{- '\n\n' + content }}
        {%- endif %}
    {%- endif %}
    {{- '<|im_end|>\n' }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {{- '<|im_start|>system\n' + content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" %}
        {%- set content = render_content(message.content, false)|trim %}
        {%- if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}
            {%- set ns.multi_step_tool = false %}
            {%- set ns.last_query_index = index %}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if ns.multi_step_tool %}
    {{- raise_exception('No user query found in messages.') }}
{%- endif %}
{%- for message in messages %}
    {%- set content = render_content(message.content, true)|trim %}
    {%- if message.role == "system" %}
        {%- if not loop.first %}
            {{- raise_exception('System message must be at the beginning.') }}
        {%- endif %}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- set reasoning_content = reasoning_content|trim %}
        {%- if loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}
            {%- for tool_call in message.tool_calls %}
                {%- if tool_call.function is defined %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {%- if loop.first %}
                    {%- if content|trim %}
                        {{- '\n\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                    {%- else %}
                        {{- '<tool_call>\n<function=' + tool_call.name + '>\n' }}
                    {%- endif %}
                {%- else %}
                    {{- '\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                {%- endif %}
                {%- if tool_call.arguments is defined %}
                    {%- for args_name, args_value in tool_call.arguments|items %}
                        {{- '<parameter=' + args_name + '>\n' }}
                        {%- set args_value = args_value | tojson | safe if args_value is mapping or (args_value is sequence and args_value is not string) else args_value | string %}
                        {{- args_value }}
                        {{- '\n</parameter>\n' }}
                    {%- endfor %}
                {%- endif %}
                {{- '</function>\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.previtem and loop.previtem.role != "tool" %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if not loop.last and loop.nextitem.role != "tool" %}
            {{- '<|im_end|>\n' }}
        {%- elif loop.last %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- else %}
        {{- raise_exception('Unexpected message role.') }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}


===== CONTRIBUTING.md =====
# Contributors

The project differentiates between 3 levels of contributors:

- Contributors: people who have contributed before (no special privileges)
- Collaborators (Triage): people with significant contributions, who may be responsible for some parts of the code, and are expected to maintain and review contributions for the code they own
- Maintainers: responsible for reviewing and merging PRs, after approval from the code owners

# AI Usage Policy

> [!IMPORTANT]
>
> AI-generated code is allowed. You are 100% responsible for every line, however it was produced.
>
> Undisclosed AI usage may result in your account being permanently banned from contributing to the project.
>
> Detailed information regarding permissible and restricted uses of AI can be found in the [AGENTS.md](AGENTS.md) file.

If AI is used to generate any portion of the code, contributors must adhere to the following requirements:

1. Explicitly disclose the manner in which AI was employed.
2. Check for an existing PR addressing the same change; if one exists, comment there to work with its author instead of opening a duplicate.
3. Perform a comprehensive manual review prior to submitting the pull request. A proper code review usually takes something like one hour per 200-400 LOC and you should be spending **at least that much time on code review alone**.
4. Be prepared to explain every line of code you submit when asked about it by a maintainer.
5. It is strictly prohibited to use AI to write your posts for you (bug reports, feature requests, pull request descriptions, Github discussions, responding to humans, ...).

For more info, please refer to the [AGENTS.md](AGENTS.md) file.

# Pull requests (for contributors & collaborators)

### Before you start

- Search for existing discussions and PRs first - duplicates will likely be closed without questions.
- Features must begin with an issue, not a PR - let interest accumulate before writing code; niche features may only land as an example/tool, or on a private fork.
- Bug-fix PRs must include a reproducible issue and a regression test that fails before your change and passes after. Fixes without a test may be closed without review.
- New CLI or public API additions carry a **higher bar** than internal changes - justify why an existing mechanism doesn't suffice.
- Meeting all of the above still doesn't guarantee a merge - see [Pull requests (for maintainers)](#pull-requests-for-maintainers).
- If you are a new contributor
    - Limit your open PRs to 1
    - Do not submit trivial fixes (e.g. typos, formatting changes)

### Preparing your PR

- llama.cpp uses the ggml tensor library for model evaluation. If you are unfamiliar with ggml, consider taking a look at the [examples in the ggml repository](https://github.com/ggml-org/ggml/tree/master/examples/). [simple](https://github.com/ggml-org/ggml/tree/master/examples/simple) shows the bare minimum for using ggml. [gpt-2](https://github.com/ggml-org/ggml/tree/master/examples/gpt-2) has minimal implementations for language model inference using GPT-2. [mnist](https://github.com/ggml-org/ggml/tree/master/examples/mnist) demonstrates how to train and evaluate a simple image classifier
- Test your changes:
  - Execute [the full CI locally on your machine](ci/README.md) before publishing
  - Verify that the perplexity and the performance are not affected negatively by your changes (use `llama-perplexity` and `llama-bench`)
  - If you modified the `ggml` source, run the `test-backend-ops` tool to check whether different backend implementations of the `ggml` operators produce consistent results (this requires access to at least two different `ggml` backends)
  - If you modified a `ggml` operator or added a new one, add the corresponding test cases to `test-backend-ops`
- Create separate PRs for each feature or fix:
  - Avoid combining unrelated changes in a single PR
  - When adding support for a new model or feature, focus on **CPU support only** in the initial PR unless you have a good reason not to. Add support for other backends like CUDA in follow-up PRs
  - In particular, adding new data types (extension of the `ggml_type` enum) carries with it a disproportionate maintenance burden. As such, to add a new quantization type you will need to meet the following *additional* criteria *at minimum*:
    - convert a small model to GGUF using the new type and upload it to HuggingFace
    - provide [perplexity](https://github.com/ggml-org/llama.cpp/tree/master/tools/perplexity) comparisons to FP16/BF16 (whichever is the native precision) as well as to types of similar size
    - provide KL divergence data calculated vs. the FP16/BF16 (whichever is the native precision) version for both the new type as well as types of similar size
    - provide [performance data](https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench) for the new type in comparison to types of similar size on pure CPU
- Consider allowing write access to your branch for faster reviews, as reviewers can push commits directly

### After submitting your PR

- Expect requests for modifications to ensure the code meets llama.cpp's standards for quality and long-term maintainability
- Maintainers will rely on your insights and approval when making a final decision to approve and merge a PR
- If your PR becomes stale, rebase it on top of latest `master` to get maintainers attention
- Consider adding yourself to [CODEOWNERS](CODEOWNERS) to indicate your availability for fixing related issues and reviewing related PRs

# Pull requests (for maintainers)

- Squash-merge PRs
- Use the following format for the squashed commit title: `<module> : <commit title> (#<issue_number>)`. For example: `utils : fix typo in utils.py (#1234)`
- Optionally pick a `<module>` from here: https://github.com/ggml-org/llama.cpp/wiki/Modules
- Let other maintainers merge their own PRs
- When merging a PR, make sure you have a good understanding of the changes
- If a PR does not warrant a new release, add `[no release]` in the squashed commit to spare CI resources
- Be mindful of maintenance: most of the work going into a feature happens after the PR is merged. If the PR author is not committed to contribute long-term, someone else needs to take responsibility (you)
- Add the ["merge ready"](https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr+is%3Aopen+draft%3Ano+sort%3Aupdated-desc+label%3A%22merge+ready%22+) label to a PR to indicate when a PR can be fast-merged without waiting for 2 independent reviews. [(more info)](https://github.com/ggml-org/llama.cpp/pull/26178)
- Wait for CI results before merging

Maintainers reserve the right to decline review or close pull requests for any reason, without any questions, particularly under any of the following conditions:
- The proposed change is already mentioned in the roadmap or an existing issue, and it has been assigned to someone.
- The pull request duplicates an existing one.
- The contributor fails to adhere to this contributing guide or the AI policy.
- The change doesn't fit the existing architecture, or is too complex to justify its benefit.

# Coding guidelines

- Avoid adding third-party dependencies, extra files, extra headers, etc.
- Always consider cross-compatibility with other operating systems and architectures
- Avoid fancy-looking modern STL constructs, use basic `for` loops, avoid templates, keep it simple
- Vertical alignment makes things more readable and easier to batch edit
- Clean-up any trailing whitespaces, use 4 spaces for indentation, brackets on the same line, `void * ptr`, `int & a`
- Use sized integer types such as `int32_t` in the public API, e.g. `size_t` may also be appropriate for allocation sizes or byte offsets
- Declare structs with `struct foo {}` instead of `typedef struct foo {} foo`
    - In C++ code omit optional `struct` and `enum` keyword whenever they are not necessary
    ```cpp
    // OK
    llama_context * ctx;
    const llama_rope_type rope_type;

    // not OK
    struct llama_context * ctx;
    const enum llama_rope_type rope_type;
    ```

    _(NOTE: this guideline is yet to be applied to the `llama.cpp` codebase. New code should follow this guideline.)_

- Try to follow the existing patterns in the code (indentation, spaces, etc.). In case of doubt use `clang-format` (from clang-tools v15+) to format the added code
- For anything not covered in the current guidelines, refer to the [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
- Tensors store data in row-major order. We refer to dimension 0 as columns, 1 as rows, 2 as matrices
- Matrix multiplication is unconventional: [`C = ggml_mul_mat(ctx, A, B)`](https://github.com/ggml-org/llama.cpp/blob/880e352277fc017df4d5794f0c21c44e1eae2b84/ggml.h#L1058-L1064) means $C^T = A B^T \Leftrightarrow C = B A^T.$

![matmul](media/matmul.png)

# Naming guidelines

- Use `snake_case` for function, variable and type names
- Naming usually optimizes for longest common prefix (see https://github.com/ggml-org/ggml/pull/302#discussion_r1243240963)

    ```cpp
    // not OK
    int small_number;
    int big_number;

    // OK
    int number_small;
    int number_big;
    ```

- Enum values are always in upper case and prefixed with the enum name

    ```cpp
    enum llama_vocab_type {
        LLAMA_VOCAB_TYPE_NONE = 0,
        LLAMA_VOCAB_TYPE_SPM  = 1,
        LLAMA_VOCAB_TYPE_BPE  = 2,
        LLAMA_VOCAB_TYPE_WPM  = 3,
        LLAMA_VOCAB_TYPE_UGM  = 4,
        LLAMA_VOCAB_TYPE_RWKV = 5,
    };
    ```

- The general naming pattern is `<class>_<method>`, with `<method>` being `<action>_<noun>`

    ```cpp
    llama_model_init();           // class: "llama_model",         method: "init"
    llama_sampler_chain_remove(); // class: "llama_sampler_chain", method: "remove"
    llama_sampler_get_seed();     // class: "llama_sampler",       method: "get_seed"
    llama_set_embeddings();       // class: "llama_context",       method: "set_embeddings"
    llama_n_threads();            // class: "llama_context",       method: "n_threads"
    llama_adapter_lora_free();    // class: "llama_adapter_lora",  method: "free"
    ```

    - The `get` `<action>` can be omitted
    - The `<noun>` can be omitted if not necessary
    - The `_context` suffix of the `<class>` is optional. Use it to disambiguate symbols when needed
    - Use `init`/`free` for constructor/destructor `<action>`

- Use the `_t` suffix when a type is supposed to be opaque to the user - it's not relevant to them if it is a struct or anything else

    ```cpp
    typedef struct llama_context * llama_context_t;

    enum llama_pooling_type llama_pooling_type(const llama_context_t ctx);
    ```

    _(NOTE: this guideline is yet to be applied to the `llama.cpp` codebase. New code should follow this guideline)_

- C/C++ filenames are all lowercase with dashes. Headers use the `.h` extension. Source files use the `.c` or `.cpp` extension
- Python filenames are all lowercase with underscores

- _(TODO: abbreviations usage)_

# Preprocessor directives

- _(TODO: add guidelines with examples and apply them to the codebase)_

    ```cpp
    #ifdef FOO
    #endif // FOO
    ```

# Code maintenance

- Existing code should have designated collaborators and/or maintainers specified in the [CODEOWNERS](CODEOWNERS) file responsible for:
  - Reviewing and merging related PRs
  - Fixing related bugs
  - Providing developer guidance/support

- When adding or modifying a large piece of code:
  - If you are a collaborator, make sure to add yourself to [CODEOWNERS](CODEOWNERS) to indicate your availability for reviewing related PRs
  - If you are a contributor, find an existing collaborator who is willing to review and maintain your code long-term
  - Provide the necessary CI workflow (and hardware) to test your changes (see [ci/README.md](https://github.com/ggml-org/llama.cpp/tree/master/ci))

- New code should follow the guidelines (coding, naming, etc.) outlined in this document. Exceptions are allowed in isolated, backend-specific parts of the code that do not interface directly with the `ggml` interfaces.
  _(NOTE: for legacy reasons, existing code is not required to follow this guideline)_

- For changes in server, please make sure to refer to the [server development documentation](./tools/server/README-dev.md)

# Documentation

- Documentation is a community effort
- When you need to look into the source code to figure out how to use an API consider adding a short summary to the header file for future reference
- When you notice incorrect or outdated documentation, please update it

# Resources

The Github issues, PRs and discussions contain a lot of information that can be useful to get familiar with the codebase. For convenience, some of the more important information is referenced from Github projects:

https://github.com/ggml-org/llama.cpp/projects



===== tests/test-chat-auto-parser.cpp TAG_WITH_TAGGED tests =====
            parsed["oldString"].get<std::string>().find("\"14\"") != std::string::npos);
        t.assert_true("newString contains embedded quotes",
            parsed["newString"].get<std::string>().find("\"15\"") != std::string::npos);
    } catch (const std::exception & e) {
        t.assert_true(std::string("JSON parse failed: ") + e.what(), false);
    }
}

// ============================================================================
// TAG_WITH_TAGGED Argument Parsing Tests
// ============================================================================

// Build tools definition for edit function
static json build_edit_tool() {
    json parameters_schema = json::object();
    parameters_schema["type"] = "object";
    parameters_schema["properties"] = json::object();
    parameters_schema["properties"]["filename"] = json::object({
        {"type", "string"},
        {"description", "Path of file to edit"}
    });
    parameters_schema["properties"]["oldString"] = json::object({
        {"type", "string"},
        {"description", "String to replace"}
    });
    parameters_schema["properties"]["newString"] = json::object({
        {"type", "string"},
        {"description", "New (replacement) value"}
    });
    parameters_schema["required"] = json::array({"filename", "oldString", "newString"});

    return json::array({
        json{
            {"type", "function"},
            {"function", json{
                {"name", "edit"},
                {"description", "Edit a file"},
                {"parameters", parameters_schema}
            }}
        }
    });
}

// ============================================================================
// Role marker detection tests for all autoparser-handled templates
//
// Verifies that detect_user_start_marker / detect_assistant_start_marker
// return the correct boundary text between turns for every template that
// falls through to the differential autoparser (i.e. is not handled by a
// dedicated specialized template in common_chat_try_specialized_template).
//
// Markers were deduced manually from the jinja sources in models/templates/.
// ============================================================================
struct role_marker_case {
    std::string template_file;
    std::string expected_user_start;
    std::string expected_assistant_start;
};

static void test_role_markers_all_templates(testing & t) {
    // Each entry is { template filename, user_start, assistant_start } as
    // produced when rendering the standard chatml-like sequences. The values
    // come from reading each jinja template and tracing what text precedes
    // a user/assistant message body once the autoparser strips any reasoning
    // markers it detected first.
    const std::vector<role_marker_case> cases = {
        // ChatML family: <|im_start|>{role} ... <|im_end|>
        { "Bielik-11B-v3.0-Instruct.jinja",                  "<|im_start|>user",       "<|im_start|>assistant"      },
        { "HuggingFaceTB-SmolLM3-3B.jinja",                  "<|im_start|>user",       "<|im_start|>assistant"      },
        { "MiMo-VL.jinja",                                   "<|im_start|>user",       "<|im_start|>assistant"      },
        { "NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use.jinja", "<|im_start|>user",   "<|im_start|>assistant"      },
        { "NousResearch-Hermes-3-Llama-3.1-8B-tool_use.jinja",   "<|im_start|>user",   "<|im_start|>assistant"      },
        { "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.jinja",       "<|im_start|>user",       "<|im_start|>assistant"      },
        { "Qwen3.5-4B.jinja",                                "<|im_start|>user",       "<|im_start|>assistant"      },
        { "Qwen3-Coder.jinja",                               "<|im_start|>user",       "<|im_start|>assistant"      },
        { "Qwen-Qwen2.5-7B-Instruct.jinja",                  "<|im_start|>user",       "<|im_start|>assistant"      },
        { "Qwen-Qwen3-0.6B.jinja",                           "<|im_start|>user",       "<|im_start|>assistant"      },
        { "Qwen-QwQ-32B.jinja",                              "<|im_start|>user",       "<|im_start|>assistant"      },
        { "StepFun3.5-Flash.jinja",                          "<|im_start|>user",       "<|im_start|>assistant"      },

        // DeepSeek family
        { "deepseek-ai-DeepSeek-R1-Distill-Llama-8B.jinja",  "<｜User｜>",                "<｜Assistant｜>"             },
        { "deepseek-ai-DeepSeek-R1-Distill-Qwen-32B.jinja",  "<｜User｜>",                "<｜Assistant｜>"             },
        { "deepseek-ai-DeepSeek-V3.1.jinja",                 "<｜User｜>",                "<｜Assistant｜>"             },
        { "llama-cpp-deepseek-r1.jinja",                     "<｜User｜>",                "<｜Assistant｜>"             },

        // Llama 3 header family
        { "meetkai-functionary-medium-v3.1.jinja",           "<|start_header_id|>user<|end_header_id|>", "<|start_header_id|>assistant<|end_header_id|>" },
        { "meta-llama-Llama-3.1-8B-Instruct.jinja",          "<|start_header_id|>user<|end_header_id|>", "<|start_header_id|>assistant<|end_header_id|>" },
        { "meta-llama-Llama-3.2-3B-Instruct.jinja",          "<|start_header_id|>user<|end_header_id|>", "<|start_header_id|>assistant<|end_header_id|>" },
        { "meta-llama-Llama-3.3-70B-Instruct.jinja",         "<|start_header_id|>user<|end_header_id|>", "<|start_header_id|>assistant<|end_header_id|>" },
        // fireworks-ai forces a trailing assistant header even without add_generation_prompt,
        // so the marker is absorbed into the common suffix and assistant_start is detected as empty.
        { "fireworks-ai-llama-3-firefunction-v2.jinja",      "<|start_header_id|>user<|end_header_id|>", "<|start_header_id|>assistant<|end_header_id|>" },

        // Phi/GLM/Apriel-style: <|user|> / <|assistant|>
        { "microsoft-Phi-3.5-mini-instruct.jinja",           "<|user|>",               "<|assistant|>"              },
        { "GLM-4.6.jinja",                                   "<|user|>",               "<|assistant|>"              },
        { "unsloth-Apriel-1.5.jinja",                        "<|user|>",               "<|assistant|>"              },
        { "GLM-4.7-Flash.jinja",                             "<|user|>",                 "<|assistant|>"                },

        // Gemma 2: <start_of_turn>{user|model}
        { "google-gemma-2-2b-it.jinja",                      "<start_of_turn>user",    "<start_of_turn>model"       },

        // IBM Granite
        { "ibm-granite-granite-3.3-2B-Instruct.jinja",       "<|start_of_role|>user<|end_of_role|>", "<|start_of_role|>assistant<|end_of_role|>" },
        { "ibm-granite-granite-4.0.jinja",                   "<|start_of_role|>user<|end_of_role|>", "<|start_of_role|>assistant<|end_of_role|>" },

        // Cohere R-series
        { "CohereForAI-c4ai-command-r7b-12-2024-tool_use.jinja",
            "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>", "<|START_RESPONSE|>" },
        { "CohereForAI-c4ai-command-r-plus-tool_use.jinja",
            "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>", "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>" },

        // Mistral: assistant content follows [/INST] immediately, no header
        { "mistralai-Mistral-Nemo-Instruct-2407.jinja",      "[INST]",                   "" },
        { "Mistral-Small-3.2-24B-Instruct-2506.jinja",       "[INST]",                   "" },

        // Apertus uses <|user_start|> / <|assistant_start|> but the user diff
        // carries the preceding <|assistant_end|> from the previous turn.
        { "Apertus-8B-Instruct.jinja",                       "<|user_start|>", "<|assistant_start|>" },

        // Apriel 1.6 wraps the assistant body with <|begin_assistant|>, but
        // <|begin_assistant|> is also the detected reasoning start, so the
        // assistant_start is trimmed back to the preceding newline.
        { "Apriel-1.6-15b-Thinker-fixed.jinja",              "<|begin_user|>", "<|begin_assistant|>" },

        // ByteDance Seed-OSS: <seed:bos>{role}
        { "ByteDance-Seed-OSS.jinja",                        "<seed:bos>user",         "<seed:bos>assistant"        },

        // GigaChat 3.1: {role}<|role_sep|>
        { "GigaChat3.1-10B-A1.8B.jinja",                     "user<|role_sep|>",       "assistant<|role_sep|>"      },

        // MiniMax M2: ]~b]{user|ai}
        { "MiniMax-M2.jinja",                                "]~b]user",               "]~b]ai"                     },

        // HunYuan V3: <｜hy_User:opensource｜> / <｜hy_Assistant:opensource｜>
        { "tencent-Hy3.jinja",                               "<｜hy_User:opensource｜>", "<｜hy_Assistant:opensource｜>" },

        // Nemotron Nano v2: <SPECIAL_11>{User|Assistant}; assistant marker
        // is followed by a prefilled <think> block that gets included.
        { "NVIDIA-Nemotron-Nano-v2.jinja",                   "<SPECIAL_11>User",       "<SPECIAL_11>Assistant" },

        // Reka Edge: "human: " / "assistant: " — but the rendered preamble
        // depends on enable_thinking, which currently confuses the user-start
        // diff and trims the marker down. Lock in the observed value.
        { "Reka-Edge.jinja",                                 "human:",                     "assistant:"       },

        // RWKV-world chat preset: "User: " / "Assistant: "
        { "llama-cpp-rwkv-world.jinja",                      "User:",               "Assistant:"              },

        // Upstage Solar 100B: <|begin|>{role}... but reasoning marker absorbs
        // the "<|begin|>assistant" prefix from assistant_start.
        { "upstage-Solar-Open-100B.jinja",                   "<|begin|>user<|content|>", "<|begin|>assistant"           },
    };

    for (const auto & c : cases) {
        t.test(c.template_file, [&](testing & t) {
            common_chat_template tmpl = load_template(t, "models/templates/" + c.template_file);
            struct autoparser ap;
            ap.analyze_template(tmpl);
            t.assert_equal("user_start",      c.expected_user_start,      ap.user_start);
            t.assert_equal("assistant_start", c.expected_assistant_start, ap.assistant_start);
        });
    }
}

static void test_bailing_v3_tool_format(testing & t) {
    const std::string template_source = R"JINJA(
{# Bailing V3 chat template #}
{%- if tools %}{{ tools | tojson }}{%- endif %}
{%- for message in messages %}
    {%- if message.role == "user" %}
        {{- '<role>HUMAN</role>' + message.content + '<|role_end|>' }}
    {%- elif message.role == "assistant" %}
        {{- '<role>ASSISTANT</role>' }}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- set tc = tool_call.function %}
                {{- '<tool_call>' + tc.name }}
                {%- for k, v in tc.arguments.items() %}
                    {{- '<arg_key>' + k + '</arg_key>' }}
                    {{- '\n<arg_value>' + v + '</arg_value>' }}
                {%- endfor %}
                {{- '\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|role_end|>' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<role>ASSISTANT</role>' }}{%- endif %}
)JINJA";

    common_chat_template tmpl(template_source, "", "");
    struct autoparser analysis;
    analysis.analyze_template(tmpl);

    t.assert_equal("arg_value_suffix", "</arg_value>", analysis.tools.arguments.value_suffix);
    t.assert_true("intertag whitespace", analysis.tools.arguments.tolerate_intertag_whitespace);

    generation_params inputs;
    inputs.tools = json::array({
        {
            { "type", "function" },
            { "function", {
                { "name", "test_function_name" },
                { "parameters", {
                    { "type", "object" },
                    { "properties", {
                        { "param1", { { "type", "string" } } },
                        { "param2", { { "type", "string" } } },
                    } },
                } },
            } },
        },
    });
    inputs.reasoning_format = COMMON_REASONING_FORMAT_NONE;
    auto parser = analysis.build_parser(inputs, "");
    const std::string output =
        "<tool_call>test_function_name\n"
        "<arg_key>param1</arg_key>\n"
        "<arg_value>value1</arg_value>"
        "<arg_key>param2</arg_key>\n"
        "<arg_value>value2</arg_value>\n"
        "</tool_call>";
    common_peg_parse_context ctx(output, COMMON_PEG_PARSE_FLAG_LENIENT);
    t.assert_true("multi-argument tool call", parser.parse(ctx).success());
}

// Test that reproduces the Seed-OSS template issue with embedded quotes
static void test_tagged_args_with_embedded_quotes(testing & t) {
    json tools = build_edit_tool();

    // Build a parser for TAG_WITH_TAGGED format like Seed-OSS/Nemotron
    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        // Build tool choice for the edit function
        auto tool_choice = p.choice();

        for (const auto & tool_def : tools) {
            if (!tool_def.contains("function")) { continue; }
            const auto & function = tool_def.at("function");
            std::string name = function.at("name");
            const auto & params = function.at("parameters");

            if (!params.contains("properties") || !params.at("properties").is_object()) { continue; }

            const auto & properties = params.at("properties");

            // Build argument parsers
            std::vector<common_peg_parser> arg_parsers;
            for (const auto & [param_name, param_schema] : properties.items()) {
                auto arg = p.tool_arg(
                    p.tool_arg_open(p.literal("<parameter=") + p.tool_arg_name(p.literal(param_name)) + p.literal(">")) +
                    p.space() +
                    p.tool_arg_string_value(p.until("</parameter>")) +
                    p.space() +
                    p.tool_arg_close(p.literal("</parameter>"))
                );
                arg_parsers.push_back(p.optional(p.rule("arg-" + param_name, arg)));
            }

            // Build arg sequence with space() between
            common_peg_parser args_seq = p.eps();
            for (size_t i = 0; i < arg_parsers.size(); i++) {
                if (i > 0) {
                    args_seq = args_seq + p.space();
                }
                args_seq = args_seq + arg_parsers[i];
            }

            auto func_parser =
                p.tool_open(p.literal("<function=") + p.tool_name(p.literal(name)) + p.literal(">")) +
                p.space() + args_seq + p.space() +
                p.tool_close(p.literal("</function>"));

            tool_choice |= p.rule("tool-" + name, p.tool(func_parser));
        }

        auto tool_section =
            p.literal("<seed:tool_call>") + p.space() +
            tool_choice +
            p.space() + p.literal("</seed:tool_call>");

        return p.content(p.until("<seed:tool_call>")) + p.optional(tool_section) + p.end();
    });

    std::string input =
        "<seed:tool_call>\n"
        "<function=edit>\n"
        "<parameter=filename>"
        "foo.cpp"
        "</parameter>\n"
        "<parameter=oldString>"
        "def foo(arg = \"14\"):\n"
        "    return arg + \"bar\"\n"
        "\n"
        "</parameter>\n"
        "<parameter=newString>"
        "def foo(arg = \"15\"):\n"
        "    pass\n"


===== tests/test-chat.cpp lines 1060-1230: session test harness =====
    simple_assist_msg("", "I'm\nthinking", "special_function", "{\"arg1\":1}");
const common_chat_msg message_assist_call_thoughts_unparsed =
    simple_assist_msg("<think>I'm\nthinking</think>\n\n", "", "special_function", "{\"arg1\": 1}");
const common_chat_msg message_assist_call_thoughts_content =
    simple_assist_msg("Hello, world!\nWhat's up?", "I'm\nthinking", "special_function", "{\"arg1\": 1}");
const common_chat_msg message_assist_call_id =
    simple_assist_msg("", "", "special_function", "{\"arg1\":1}", /* .id = */ "123456789");
const common_chat_msg message_assist_call_idx =
    simple_assist_msg("", "", "special_function", "{\"arg1\":1}", /* .id = */ "0");
const common_chat_msg message_assist_thoughts_call_idx =
    simple_assist_msg("", "I'm\nthinking", "special_function", "{\"arg1\": 1}", /* id = */ "0");
const common_chat_msg message_assist_thoughts_partial_call =
    simple_assist_msg("", "I'm\nthinking", "special_function", "", /* id = */ "0");
const common_chat_msg message_assist_call_python = simple_assist_msg("", "", "python", "{\"code\":\"print('hey')\"}");
const common_chat_msg message_assist_call_python_lines =
    simple_assist_msg("", "", "python", "{\"code\":\"# This is a program:\\nprint('hey')\"}");
const common_chat_msg message_assist_call_python_lines_unclosed =
    simple_assist_msg("", "", "python", "{\"code\":\"# This is a program:\\nprint('hey')");
const common_chat_msg message_assist_json_content =
    simple_assist_msg("{\n  \"response\": \"Hello, world!\\nWhat's up?\"\n}");
const common_chat_msg message_assist_prefill_content   = simple_assist_msg("Hello, ", "I'm thinking");
const common_chat_msg message_assist_prefill_reasoning = simple_assist_msg("", "I'm");

// Use for PEG parser implementations
struct peg_test_case {
    common_chat_templates_inputs params;
    std::string                  input;
    common_chat_msg              expect;
    bool                         is_partial            = false;
    bool                         expect_reconstruction = false;
    std::vector<std::string>     expect_rules;
};

// Global template filter for --template flag
static std::string g_template_filter;

// When true, run reconstruction test on every non-partial test and report results
static bool g_force_reconstruction_test = false;

static void test_peg_parser(common_chat_templates *                      tmpls,
                            const std::function<void(peg_test_case &)> & init,
                            bool                                         detailed_debug) {
    // UTF-8-safe truncation helper (same as in test_parser_with_streaming)
    constexpr auto utf8_truncate_safe_len = [](const std::string_view s) -> size_t {
        auto len = s.size();
        if (len == 0) {
            return 0;
        }
        auto i = len;
        for (size_t back = 0; back < 4 && i > 0; ++back) {
            --i;
            unsigned char c = s[i];
            if ((c & 0x80) == 0) {
                return len;
            }
            if ((c & 0xC0) == 0xC0) {
                size_t expected_len = 0;
                if ((c & 0xE0) == 0xC0) {
                    expected_len = 2;
                } else if ((c & 0xF0) == 0xE0) {
                    expected_len = 3;
                } else if ((c & 0xF8) == 0xF0) {
                    expected_len = 4;
                } else {
                    return i;
                }
                if (len - i >= expected_len) {
                    return len;
                }
                return i;
            }
        }
        return len - std::min(len, size_t(3));
    };

    peg_test_case tc;
    init(tc);
    if (tc.params.messages.empty()) {
        tc.params.messages = { message_user };
    }
    if (tc.expect.role.empty()) {
        tc.expect.role = "assistant";
    }

    common_chat_session_params session_params;
    session_params.debug = detailed_debug;
    common_chat_session session(tmpls, nullptr, tc.params, session_params);
    const auto & parser = session.parser();
    common_params_sampling sampling;
    session.apply_sampling(sampling);
    if (detailed_debug) {
        LOG_DBG("Using parser: \n%s\n", parser.dump(parser.root()).c_str());
        LOG_DBG("Generation prompt: '%s'\n", session.generation_prompt().c_str());
    }

    for (const auto & rule : tc.expect_rules) {
        if (!parser.has_rule(rule)) {
            LOG_ERR("Missing rule: %s\n", rule.c_str());
            common_log_flush(common_log_main());
            throw std::runtime_error("Test failed");
        }
    }

    common_chat_msg msg_accum;
    common_chat_msg msg_prev;
    msg_accum.role = msg_prev.role = "assistant";

    size_t fed = 0;
    for (size_t i = 1; i <= tc.input.size(); ++i) {
        auto            is_partial  = i < tc.input.size() || tc.is_partial;
        // Use UTF-8 safe truncation to avoid corrupting multi-byte characters
        size_t          safe_len    = utf8_truncate_safe_len(std::string_view(tc.input).substr(0, i));
        common_chat_input chunk(tc.input.substr(fed, safe_len - fed));
        fed = safe_len;
        const common_chat_msg & msg_current = is_partial ? session.feed(chunk) : session.finish(chunk);

        for (const auto & diff : common_chat_msg_diff::compute_diffs(msg_prev, msg_current)) {
            if (!diff.reasoning_content_delta.empty()) {
                msg_accum.reasoning_content += diff.reasoning_content_delta;
            }
            if (!diff.content_delta.empty()) {
                msg_accum.content += diff.content_delta;
            }
            if (diff.tool_call_index != std::string::npos) {
                // During partial parsing, a new tool call may appear with empty name initially
                // The name gets filled in as more input is parsed
                while (msg_accum.tool_calls.size() <= diff.tool_call_index) {
                    msg_accum.tool_calls.push_back({ "", "", "" });
                }
                // Always update name and id from diff (may change during incremental parsing), but only if the delta
                // actually contains them
                if (!diff.tool_call_delta.name.empty()) {
                    msg_accum.tool_calls[diff.tool_call_index].name = diff.tool_call_delta.name;
                }
                if (!diff.tool_call_delta.id.empty()) {
                    msg_accum.tool_calls[diff.tool_call_index].id = diff.tool_call_delta.id;
                }
                if (!diff.tool_call_delta.arguments.empty()) {
                    msg_accum.tool_calls[diff.tool_call_index].arguments += diff.tool_call_delta.arguments;
                }
            }
        }
        try {
            assert_msg_equals(msg_current, msg_accum, true);
        } catch (std::exception & e) {
            throw std::runtime_error((std::string("Error comparing accumulated message to current: ") + e.what()).c_str());
        }

        msg_prev = msg_current;
    }

    if (!tc.is_partial) {
        if (tc.input.empty()) {
            session.finish();
        }
        assert_msg_equals(tc.expect, session.msg(), true);
    }
    assert_msg_equals(tc.expect, msg_accum, true);

    // A response format must be enforced by an eager grammar
    if (!tc.params.json_schema.empty()) {
        if (session.grammar().empty()) {
            throw std::runtime_error("json_schema is set but no grammar was produced");
        }
        if (sampling.grammar_lazy) {
            throw std::runtime_error("json_schema is set but the grammar is lazy");
        }
    }

    // Test grammar if present in params
    if (!session.grammar().empty()) {


===== tests/test-chat.cpp lines 2135-2205: Qwen3.5 parser tests =====
        .expect(simple_assist_msg("", "", "empty_args", "{}"))
        .run();

}

static void test_template_output_peg_parsers(bool detailed_debug) {
    LOG_DBG("%s\n", __func__);

    // JSON schemas
    const char * invoice_schema = R"({
        "type": "object",
        "properties": {
            "amount": {"type": "number"},
            "date": {"type": "string"}
        }
    })";

    const char * const_schema = R"({
        "const": "42"
    })";

    {
        // Qwen3.5 (basically same as Nemotron, but keeping separate tests just in case)
        auto tst = peg_tester("models/templates/Qwen3.5-4B.jinja", detailed_debug);

        tst.test("I'm\nthinking\n</think>\n\nHello, world!\nWhat's up?")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .enable_thinking(true)
            .expect(message_assist_thoughts)
            .run();

        tst.test("I'm\nthinking\n</think>\n\nHello, world!\nWhat's up?")
            .enable_thinking(true)
            .reasoning_format(COMMON_REASONING_FORMAT_NONE)
            .expect_content("<think>\nI'm\nthinking\n</think>\n\nHello, world!\nWhat's up?")
            .run();

        tst.test(
               "<tool_call>\n"
               "<function=special_function>\n"
               "<parameter=arg1>\n1\n</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .enable_thinking(false)
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .tools({ special_function_tool })
            .expect(message_assist_call)
            .expect_rules({ "tool-0", "tool-0-arg-0" })
            .run();

        tst.test(
               "I'm\nthinking\n</think>\n\n"
               "<tool_call>\n"
               "<function=special_function>\n"
               "<parameter=arg1>\n1\n</parameter>\n"
               "</function>\n"
               "</tool_call>")
            .reasoning_format(COMMON_REASONING_FORMAT_AUTO)
            .tools({ special_function_tool })
            .expect(message_assist_call_thoughts)
            .run();

        tst.test(
               "<tool_call>\n"
               "<function=special_function>\n"
               "<parameter=arg1>\n1\n</parameter>\n"
               "</function>\n"
               "</tool_call>\n"
               "<tool_call>\n"
               "<function=special_function_with_opt>\n"
               "<parameter=arg1>\n1\n</parameter>\n"


===== tests/test-chat.cpp lines 3415-3555: Qwen3.5 tagged tool tests =====
            .run();
    }

    {
        // IBM Granite 4.0 (production template shared by h-tiny, h-small, micro)
        // Uses <tool_call> XML tags for tool calls, tools in system message
        auto tst = peg_tester("models/templates/ibm-granite-granite-4.0.jinja", detailed_debug);

        tst.test("Hello, world!\nWhat's up?").expect(message_assist).run();

        tst.test(
               "<tool_call>\n"
               "{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
               "</tool_call>")
            .tools({ special_function_tool })
            .expect(message_assist_call)
            .run();

        // Continuation tests
        tst.test("world!\nWhat's up?")
            .messages({ message_user, message_assist_prefill_content })
            .add_generation_prompt(false)
            .continue_final_message(COMMON_CHAT_CONTINUATION_CONTENT)
            .expect_content("Hello, world!\nWhat's up?")
            .run();
    }

    {
        // IBM Granite 4.1 (same format as 4.0)
        auto tst = peg_tester("models/templates/ibm-granite-granite-4.1.jinja", detailed_debug);

        tst.test("Hello, world!\nWhat's up?").expect(message_assist).run();

        tst.test(
               "<tool_call>\n"
               "{\"name\": \"special_function\", \"arguments\": {\"arg1\": 1}}\n"
               "</tool_call>")
            .tools({ special_function_tool })
            .expect(message_assist_call)
            .run();
    }

    {
        // ByteDance-Seed-OSS (reasoning and tool calling model)
        auto tst = peg_tester("models/templates/ByteDance-Seed-OSS.jinja", detailed_debug);

        tst.test("Hello, world!\nWhat's up?").expect(message_assist).run();

        tst.test("<seed:think>I'm thinking about the answer</seed:think>\nHello, world!")
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .expect(simple_assist_msg("Hello, world!", "I'm thinking about the answer"))
            .run();

        tst.test(
               "<seed:tool_call>\n"
               "<function=special_function>\n"
               "<parameter=arg1>1</parameter>\n"
               "</function>\n"
               "</seed:tool_call>")
            .tools({ special_function_tool })
            .expect(message_assist_call)
            .run();

        tst.test(
               "<seed:tool_call>\n"
               "<function=special_function>\n"
               "<parameter=arg1>1</parameter>\n"
               "</function>\n"
               "</seed:tool_call>\n"
               "<seed:tool_call>\n"
               "<function=special_function_with_opt>\n"
               "<parameter=arg1>1</parameter>\n"
               "<parameter=arg2>2</parameter>\n"
               "</function>\n"
               "</seed:tool_call>")
            .parallel_tool_calls(true)
            .tools({
                special_function_tool, special_function_tool_with_optional_param
        })
            .expect_tool_calls({
                { "special_function", R"({"arg1": 1})", {} },
                { "special_function_with_opt", R"({"arg1": 1, "arg2": 2})", {} },
            })
            .run();

        tst.test(
               "<seed:tool_call>\n"
               "<function=todo_list>\n"
               "<parameter=todos>[{\"item\": \"Check stuff\", \"selected\": false}, {\"item\": \"Prepare stuff\", \"selected\": true}]</parameter>\n"
               "</function>\n"
               "</seed:tool_call>")
            .tools({
                todo_list
        })
            .expect_tool_calls({
                { "todo_list", "{\"todos\": [{\"item\": \"Check stuff\", \"selected\": false}, {\"item\": \"Prepare stuff\", \"selected\": true}]}", {} },
            })
            .run();

        // tool call with inside quotes
        tst.test(
               "<seed:tool_call>\n"
               "<function=edit>\n"
               "<parameter=filename>"
               "foo.cpp"
               "</parameter>\n"
               "<parameter=oldString>"
               "def foo(arg = \"14\"):\n"
               "    return arg + \"bar\"\n"
               "</parameter>\n"
               "<parameter=newString>"
               "def foo(arg = \"15\"):\n"
               "    pass\n"
               "</parameter>\n"
               "</function>\n"
               "</seed:tool_call>")
            .tools({
                edit_tool
        })
            .expect_tool_calls({
                { "edit", "{\"filename\": \"foo.cpp\", "
                    "\"oldString\": \"def foo(arg = \\\"14\\\"):\\n    return arg + \\\"bar\\\"\\n\", "
                    "\"newString\": \"def foo(arg = \\\"15\\\"):\\n    pass\\n\"}", {}
                }
            })
            .run();

        // Continuation tests
        tst.test("world!\nWhat's up?")
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .enable_thinking(true)
            .messages({ message_user, message_assist_prefill_content })
            .add_generation_prompt(false)
            .continue_final_message(COMMON_CHAT_CONTINUATION_CONTENT)
            .expect_reasoning("I'm thinking")
            .expect_content("Hello, world!\nWhat's up?")
            .run();

        tst.test(" thinking</seed:think>\nHello, world!\nWhat's up?")
            .reasoning_format(COMMON_REASONING_FORMAT_DEEPSEEK)
            .enable_thinking(true)


===== tools/server/README.md slot save and restore API =====
| `llamacpp:requests_processing` | Gauge | Number of requests processing. |
| `llamacpp:requests_deferred` | Gauge | Number of requests deferred. |
| `llamacpp:n_tokens_max` | Counter | High watermark of the context size observed. |
| `llamacpp:n_decode_total` | Counter | Total Number of llama_decode() calls. |
| `llamacpp:n_busy_slots_per_decode` | Gauge | Average number of busy slots per llama_decode() call. |
| `llamacpp:spec_decode_num_draft_tokens_total` | Counter | Total draft tokens generated (0 when spec-decode is off). |
| `llamacpp:spec_decode_num_accepted_tokens_total` | Counter | Total draft tokens accepted by the target model (0 when spec-decode is off). |
| `llamacpp:spec_decode_num_drafts_total` | Counter | Total speculative decoding verification steps (0 when spec-decode is off). |
| `llamacpp:spec_decode_num_accepted_tokens_per_pos_total` | Counter | Accepted tokens per draft position (labeled `position="N"`; absent when spec-decode is off or before the first completed speculative request). |

### POST `/slots/{id_slot}?action=save`: Save the prompt cache of the specified slot to a file.

*Options:*

`filename`: Name of the file to save the slot's prompt cache. The file will be saved in the directory specified by the `--slot-save-path` server parameter.

**Response format**

```json
{
    "id_slot": 0,
    "filename": "slot_save_file.bin",
    "n_saved": 1745,
    "n_written": 14309796,
    "timings": {
        "save_ms": 49.865
    }
}
```

### POST `/slots/{id_slot}?action=restore`: Restore the prompt cache of the specified slot from a file.

*Options:*

`filename`: Name of the file to restore the slot's prompt cache from. The file should be located in the directory specified by the `--slot-save-path` server parameter.

**Response format**

```json
{
    "id_slot": 0,
    "filename": "slot_save_file.bin",
    "n_restored": 1745,
    "n_read": 14309796,
    "timings": {
        "restore_ms": 42.937
    }
}
```

