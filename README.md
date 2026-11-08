# odin-infer

**A high-efficiency local inference engine for Qwen3.5-class models on Apple Silicon (MacBook Air first).**
**Apple Silicon（MacBook Air 优先）上 Qwen3.5 级模型的高效本地推理引擎。**

A single 7 MB binary that runs Qwen3.5-class models on your Mac — no Python,
no llama.cpp dependency, 73 ms first token, native MCP, agent-ready.
一个 7MB 的单文件二进制，把 Qwen3.5 级模型跑在你的 Mac 上——无 Python、无
llama.cpp 依赖，首 token 73ms，原生 MCP，agent 就绪。

[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![language: Odin](https://img.shields.io/badge/language-Odin-9cf.svg)](https://odin-lang.org)

---

## Why / 为什么是它

| Your need / 你的需求 | odin-infer's answer / odin-infer 的回答 |
|---|---|
| Run local models on an Air without heat issues / 在 Air 上跑本地模型，不烫不卡 | Low-thread kernels tuned for passive cooling; memory budget shown at startup / 低线程内核专为被动散热设计，内存预算启动即显（[thermal protocol / 热测试协议](docs/THERMAL.md)） |
| Serve models to Cursor / Claude Code / pie / 给 Cursor / Claude Code / pie 供模型 | `--mcp` turns it into an MCP server; `odin.chat` / `odin.models` out of the box / `--mcp` 一键变 MCP 服务端，`odin.chat` / `odin.models` 开箱即用（[docs/MCP.md](docs/MCP.md)） |
| No Python / toolchain install / 不想装 Python / 编译链 | Single-file binary from GitHub Releases, zero compile / 单文件二进制，GitHub Releases 零编译下载 |
| OpenAI / Ollama compatibility / 要 OpenAI / Ollama 兼容 | `/v1/chat/completions` + `/api/chat` + Web dashboard |
| Fast first token (interactive feel) / 首 token 要快（交互体感） | 73 ms measured on M3, faster than ollama's ~90 ms / 73ms（M3 实测，快过 ollama 的 ~90ms） |

## Quick start / 30 秒上手

Zero compile, Apple Silicon (M1–M4). 零编译，Apple Silicon（M1–M4）：

```sh
curl -L -o odin-serve https://github.com/vajraimb/odin-infer/releases/latest/download/odin-serve
curl -L -o odin-cli   https://github.com/vajraimb/odin-infer/releases/latest/download/odin-cli
chmod +x odin-serve odin-cli
xattr -cr odin-serve odin-cli   # only if macOS blocks the unsigned binaries / 仅当 macOS 拦截未签名二进制时需要
./odin-serve ./models/<your-model>.gguf --metal --port 9748   # http://127.0.0.1:9748
```

MCP (for agents / 给 agent 用):

```sh
claude mcp add odin -- /path/to/odin-serve /path/to/model.gguf --mcp --metal
```

Build from source (needs the [Odin](https://odin-lang.org) toolchain).
从源码构建（需要 [Odin](https://odin-lang.org) 工具链）：

```sh
git clone https://github.com/vajraimb/odin-infer && cd odin-infer
./build.sh                                  # build + unit tests, produces bin/odin-serve bin/odin-cli / 构建 + 单测，产出 bin/odin-serve bin/odin-cli
MODEL=./models/<your-model>.gguf ./serve.sh   # http://127.0.0.1:9748
```

> Model GGUFs are not in the repo (gitignored): download from Hugging Face into
> `models/` or any path (see models/README.md for the recommended layout).
> 模型 GGUF 不在仓库里（gitignored）：从 Hugging Face 下载后放 `models/` 或任意路径。

## Measured performance / 实测性能

M3 / 16 GB, DeepSeek-V4-Pro-Qwen3.5-9B UNCENSORED heretic (Q4_K_M), solo
residency, cooled machine. M3 / 16GB，heretic Q4_K_M，单引擎驻留、冷却后：

| Metric / 指标 | odin-infer | ollama / llama.cpp |
|---|---|---|
| Decode throughput (EN) / 解码吞吐 | 13.3 tok/s（0.89×） | 14.9 tok/s |
| Wall, 128-token reply / 128-token 回复墙钟 | 10.3–10.5 s | 7.1–9.1 s |
| First token / 首 token | **73 ms** | ~90 ms |
| Binary size / 二进制体积 | **7.1 MB** | large binary + runtime deps / 大二进制 + 运行时依赖 |

> Benchmark protocol in [POSITIONING.md](POSITIONING.md) §5; the full GEMV
> kernel comparison experiments (llama.cpp family sweep, shape tuning,
> ablations) live in [packages/qwen3_5/README.md](packages/qwen3_5/README.md).
> 口径见 POSITIONING.md 第五节；GEMV 内核对照实验在 packages/qwen3_5/README.md。

## Supported models / 支持什么

**Positioned at Qwen3.5 and above / 定位在 Qwen3.5 及以上：**

- **Qwen3.5 hybrid attention**（Ornith / heretic family）：text-only, 248k vocab.
  9B 实测验收；同架构不同尺寸靠 GGUF 元数据自动适配（MTP 感知加载器）。
  文本，248k 词表；9B 实测验收。
- **Qwen3.5-MoE**（35B-A3B / 122B-A10B / 397B-A17B）：✅ 已支持，CPU 路径
  （softmax top-k 路由 + 共享专家 sigmoid 门控，逐位对拍验证）；Metal MoE
  内核待实现（进度见 packages/qwen3_5/README.md）。
- No MoE / multimodal / Qwen3-or-older / other architectures — a narrow scope,
  on purpose. 不支持 MoE / 多模态 / Qwen3 及更早 / 其他架构——窄定位是特性不是缺陷。
- The repo still ships the Qwen3-dense engine (`packages/infer`) for the CLI's
  historical compatibility path; it is **not part of the product story**.
  仓库仍保留 Qwen3 密集引擎（infer 包）供 CLI 历史兼容，但不属于产品定位。

## Product forms / 产品形态

| Form / 形态 | Artifact / 产物 | Use case / 场景 |
|---|---|---|
| **odin-serve**（primary / 主） | `bin/odin-serve` | HTTP (OpenAI/Ollama-compatible + dashboard) or `--mcp` / HTTP 或 `--mcp` |
| **odin-cli**（secondary / 辅） | `bin/odin-cli` | Terminal chat / 终端聊天 |
| **library / 库** | `packages/*` | Embed in any Odin project via `-collection` / `-collection` 嵌入任意 Odin 项目 |

## Platform adaptation / 平台适配

- **First-run kernel autotuning / 首启形状调优**：on first Metal start, the
  engine times every GEMV kernel variant on the real weight shapes, picks the
  best per shape and caches it (`~/.odin-infer/<model>.tune`) — adapts to
  M1/M2/M3/M4 automatically. 首次启动实测各 GEMV 内核变体、按矩阵形状选优并缓存，各代自动适配。
- **Memory budget / 内存预算**：startup prints `mem: ... (36%) OK`, warns
  above 85%; 8 GB guidance in [docs/MEMORY.md](docs/MEMORY.md). 启动打印 `mem:` 估算，>85% 告警。

## Repo layout / 仓库结构

```
packages/   ggml · infer(Qwen3) · qwen3_5 · qwen3_5_tokenizer · sampler · tokenizer
server/     odin-serve (HTTP + MCP + dashboard + integration tests / 集成测试)
cli/        odin-cli
tools/      gemm/delta/doctor/oracle harness + thermal_test.py
docs/       MCP · THERMAL · MEMORY · troubleshooting notes / 排障记录
```

## Docs / 文档索引

- [POSITIONING.md](POSITIONING.md) —— positioning, benchmark protocol, roadmap / 定位、对标口径、路线图
- [docs/MCP.md](docs/MCP.md) —— MCP tool schema and client setup / MCP 工具 schema 与客户端接入
- [docs/MEMORY.md](docs/MEMORY.md) —— 8 GB Air configuration / 8GB Air 配置
- [docs/THERMAL.md](docs/THERMAL.md) —— sustained-load comparison protocol / 持续负载对照协议
- [AGENTS.md](AGENTS.md) —— quickstart for AI coding agents / 给 AI agent 的构建/测试/使用速查

## License / 许可证

MIT
