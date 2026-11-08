# MCP — Model Context Protocol (stdio)

odin-serve has a built-in MCP server (JSON-RPC 2.0 over newline-delimited
stdio) that exposes the local Qwen engine as MCP tools for agent hosts like
Claude Code / Cursor / pie. Protocol version `2024-11-05`; stdout is reserved
for protocol messages, all diagnostics go to stderr.

[中文版 / Chinese version](MCP.zh.md)

## Start

```sh
bin/odin-serve <model.gguf> --mcp --metal [--max-ctx 4096]
```

The model loads before the first response (about 10–25 s depending on size);
afterwards `initialize → tools/list → tools/call` work directly.

## Tools

| Tool | Description | Parameters |
|---|---|---|
| `odin.chat` | Generate a reply with the local model | `messages` (OpenAI format, required), `max_tokens` (default 256), `temperature` (default 0.6) |
| `odin.models` | Loaded model and engine status | none |

`odin.chat` reuses the server's OpenAI handler (stateless per request: reset →
prefill → sampling); output is MCP `text` content.

## Client setup

### Claude Code

```sh
claude mcp add odin -- \
  /path/to/odin-infer/bin/odin-serve \
  /path/to/model.gguf --mcp --metal
```

Then call the `odin.chat` / `odin.models` tools from a session.

### Cursor

`~/.cursor/mcp.json` (or project-local `.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "odin-infer": {
      "command": "/path/to/odin-infer/bin/odin-serve",
      "args": ["/path/to/model.gguf", "--mcp", "--metal"]
    }
  }
}
```

### pie

pie currently uses the HTTP path (`ODINFERSERVER_BASE_URL`, see `pie.sh` at the
repo root) — no changes needed. For pie's MCP host capability consuming the
`odin.chat` tool, follow pie's own MCP config format (not verified in this
session).

### Debug

```sh
# Official inspector (needs node)
npx @modelcontextprotocol/inspector \
  /path/to/bin/odin-serve /path/to/model.gguf --mcp --metal
```

## Test

```sh
python3 server/serve_harness/test_mcp.py   # MODEL=... overrides the model
```

Covers: initialize / ping / tools/list / tools/call ×2 / unknown-method error
-32601 / unknown tool. The built binary is cached at
`server/serve_harness/.test-serve.bin`.

## Notes & limits

- stdio transport only (the 90% case for local agents); SSE/streamable HTTP is
  not implemented.
- The engine is single-request serial: concurrent MCP calls queue in arrival
  order.
- Requires a Qwen3.5-family GGUF (the server runs the qwen3_5 engine); Qwen3.5
  MoE GGUFs also work via the engine's CPU fallback.
