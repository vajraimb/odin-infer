# MCP — Model Context Protocol (stdio) · MCP 协议（stdio 传输）

odin-serve has a built-in MCP server (JSON-RPC 2.0 over newline-delimited
stdio) that exposes the local Qwen engine as MCP tools for agent hosts like
Claude Code / Cursor / pie. Protocol version `2024-11-05`; stdout is reserved
for protocol messages, all diagnostics go to stderr.
odin-serve 内置 MCP 服务端（JSON-RPC 2.0 over newline-delimited stdio），把本地
Qwen 引擎暴露为 MCP 工具，供 Claude Code / Cursor / pie 等 agent 宿主调用。
协议版本 `2024-11-05`；stdout 保留给协议消息，所有诊断走 stderr。

## Start / 启动

```sh
bin/odin-serve <model.gguf> --mcp --metal [--max-ctx 4096]
```

The model loads before the first response (about 10–25 s depending on size);
afterwards `initialize → tools/list → tools/call` work directly.
模型在第一次响应前加载（约 10–25s，取决于模型大小）；此后握手直接可用。

## Tools / 工具

| Tool / 工具 | Description / 说明 | Parameters / 参数 |
|---|---|---|
| `odin.chat` | Generate a reply with the local model / 用本地模型生成回复 | `messages` (OpenAI format, required / 必填), `max_tokens` (default / 默认 256), `temperature` (default / 默认 0.6) |
| `odin.models` | Loaded model and engine status / 已加载模型与引擎状态 | none / 无 |

`odin.chat` reuses the server's OpenAI handler (stateless per request: reset →
prefill → sampling); output is MCP `text` content.
`odin.chat` 复用服务端的 OpenAI 处理路径（每请求无状态），输出为 MCP `text` content。

## Client setup / 客户端接入

### Claude Code

```sh
claude mcp add odin -- \
  /path/to/odin-infer/bin/odin-serve \
  /path/to/model.gguf --mcp --metal
```

Then call the `odin.chat` / `odin.models` tools from a session.
之后在对话中可直接调用 `odin.chat` / `odin.models` 工具。

### Cursor

`~/.cursor/mcp.json` (or project-local `.cursor/mcp.json`) / `~/.cursor/mcp.json`（或项目内 `.cursor/mcp.json`）：

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
session). pie 当前走 HTTP 路径（根目录 `pie.sh`），无需改动；pie 的 MCP 宿主
接入 `odin.chat` 的配置格式以 pie 文档为准（本会话未验证）。

### Debug / 调试

```sh
# Official inspector (needs node) / 官方 inspector（需要 node）
npx @modelcontextprotocol/inspector \
  /path/to/bin/odin-serve /path/to/model.gguf --mcp --metal
```

## Test / 测试

```sh
python3 server/serve_harness/test_mcp.py   # MODEL=... overrides the model / MODEL=... 可覆盖模型路径
```

Covers: initialize / ping / tools/list / tools/call ×2 / unknown-method error
-32601 / unknown tool. The built binary is cached at
`server/serve_harness/.test-serve.bin`.
覆盖：initialize / ping / tools/list / tools/call×2 / 未知方法错误码 -32601 / 未知工具。

## Notes & limits / 说明与限制

- stdio transport only (the 90% case for local agents); SSE/streamable HTTP is
  not implemented. 仅 stdio 传输（本地 agent 的 90% 场景）；SSE/streamable HTTP 未实现。
- The engine is single-request serial: concurrent MCP calls queue in arrival
  order. 引擎单请求串行：并发 MCP 调用按到达顺序排队执行。
- Requires a Qwen3.5-family GGUF (the server runs the qwen3_5 engine); for
  Qwen3 dense use `odin-cli` or the HTTP server path. 需 Qwen3.5 系 GGUF；
  Qwen3 密集模型请用 `odin-cli` 或 HTTP 服务端的对应路径。
