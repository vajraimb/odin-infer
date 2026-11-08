# MCP — Model Context Protocol（stdio）

odin-serve 内置 MCP 服务端（JSON-RPC 2.0 over newline-delimited stdio），把
本地 Qwen 引擎暴露为 MCP 工具，供 Claude Code / Cursor / pie 等 agent 宿主
调用。协议版本 `2024-11-05`；stdout 保留给协议消息，所有诊断走 stderr。

## 启动

```sh
bin/odin-serve <model.gguf> --mcp --metal [--max-ctx 4096]
```

模型在第一次响应前加载（约 10–25s，取决于模型大小）；此后
`initialize → tools/list → tools/call` 直接可用。

## 工具

| 工具 | 说明 | 参数 |
|------|------|------|
| `odin.chat` | 用本地模型生成回复 | `messages`（OpenAI 格式，必填）、`max_tokens`（默认 256）、`temperature`（默认 0.6） |
| `odin.models` | 已加载模型与引擎状态 | 无 |

`odin.chat` 复用服务端的 OpenAI 处理路径（每请求无状态：reset →
prefill → 采样），输出为 MCP `text` content。

## 客户端接入

### Claude Code

```sh
claude mcp add odin -- \
  /path/to/odin-infer/bin/odin-serve \
  /path/to/model.gguf --mcp --metal
```

之后在对话中可直接调用 `odin.chat` / `odin.models` 工具。

### Cursor

`~/.cursor/mcp.json`（或项目内 `.cursor/mcp.json`）：

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

pie 当前走 HTTP 路径（`ODINFERSERVER_BASE_URL`，见根目录 `pie.sh`），
无需改动；pie 的 MCP 宿主能力接入 `odin.chat` 工具时，配置格式以 pie
文档为准（本会话未验证 pie 的 MCP 配置入口）。

### 调试

```sh
# 官方 inspector（需要 node）
npx @modelcontextprotocol/inspector \
  /path/to/bin/odin-serve /path/to/model.gguf --mcp --metal
```

## 测试

```sh
python3 server/serve_harness/test_mcp.py   # MODEL=... 可覆盖模型路径
```

覆盖：initialize / ping / tools/list / tools/call×2 / 未知方法错误码
-32601 / 未知工具。构建产物缓存于 `server/serve_harness/.test-serve.bin`。

## 说明与限制

- 仅 stdio 传输（本地 agent 的 90% 场景）；SSE/streamable HTTP 未实现。
- 引擎是单请求串行模型：并发 MCP 调用按到达顺序排队执行。
- 需 Qwen3.5 系 GGUF（服务端走 qwen3_5 引擎）；Qwen3 密集模型请用
  `odin-cli` 或 HTTP 服务端的对应路径。
