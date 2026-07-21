# serve_harness

HTTP dashboard + JSON chat API for `odin-infer`. Loads a Qwen3.5 engine and
serves a single-page browser UI plus a minimal REST API for chat.

Inspired by colibri's `coli web` (live token metrics, browser-based chat).

## Run

```sh
odin run serve_harness/serve.odin -file -- \
  /path/to/ornith.gguf --metal --max-ctx 1024 --port 9748 \
  -collection:ggml=. -collection:infer=. \
  -collection:qwen3_5=. -collection:qwen3_5_tokenizer=. \
  -collection:sampler=. -collection:tokenizer=.
```

Then open `http://127.0.0.1:9748` in a browser.

## CLI flags

| Flag           | Default | Notes                                             |
|----------------|---------|---------------------------------------------------|
| `--port N`     | 9748    | Loopback port (bound to 127.0.0.1, not exposed)  |
| `--metal`      | off     | Use Metal backend (otherwise CPU)                 |
| `--max-ctx N`  | 4096    | Caps KV cache memory                              |

## Endpoints

| Method | Path          | Returns                                                |
|--------|---------------|--------------------------------------------------------|
| GET    | `/`           | HTML dashboard (chat UI + live metrics, polls /api/*) |
| GET    | `/api/info`   | JSON: model dims, vocab, seq_len, metal                |
| GET    | `/api/state`  | JSON: current pos, last tok/s, last TTFT               |
| POST   | `/api/chat`   | JSON `{message}` → `{response, tok_per_s, ttft_ms, …}` |

## Dashboard

The HTML page (inlined via `#load`) provides:

- **Left panel**: model dimensions, vocab, layer counts, metal status;
  live position + last generation tok/s + TTFT (auto-refreshes every 2s)
- **Right panel**: chat UI with per-turn timing breakdown

Each chat response shows generated token count, tok/s, TTFT, and total wall
time. Useful for comparing kernel changes or model settings.

## Architecture

Single-threaded HTTP/1.0 server inlined in `serve.odin` (~250 lines for
HTTP + handlers). No keep-alive, no chunked encoding, no TLS — designed for
local dev use only.

The engine is loaded once at startup; chat requests maintain conversation
position in `g_state.pos`. Subsequent prompts are prefilled as continuations,
not as fresh chat turns.

## Limitations (v1)

- No streaming — full response is buffered then sent
- No conversation history UI (refresh loses chat log)
- No chat over the wire for non-Qwen3.5 models (path is hardcoded to q35)
- Server is single-threaded; concurrent requests queue
- 256-token generation cap per turn

## Future work

- SSE streaming for live token-by-token display
- Conversation history persisted to localStorage
- Multi-user / multi-conversation support
- OpenAI-compatible `/v1/chat/completions` endpoint for tool integration
