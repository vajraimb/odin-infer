# odin-infer

**Apple Silicon（MacBook Air 优先）上 Qwen3 / Qwen3.5 的高效本地推理引擎。**

一个 7MB 的单文件二进制，把 9B 级 Qwen 模型跑在你的 Mac 上——无 Python、无
llama.cpp 依赖，首 token 73ms，原生 MCP，agent 就绪。

[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![language: Odin](https://img.shields.io/badge/language-Odin-9cf.svg)](https://odin-lang.org)

---

## 为什么是它

| 你的需求 | odin-infer 的回答 |
|---|---|
| 在 Air 上跑本地模型，不烫不卡 | 低线程内核专为被动散热设计，内存预算启动即显（[热测试协议](docs/THERMAL.md)） |
| 给 Cursor / Claude Code / pie 供模型 | `--mcp` 一键变 MCP 服务端，`odin.chat` / `odin.models` 开箱即用（[docs/MCP.md](docs/MCP.md)） |
| 不想装 Python / 编译链 | 单文件二进制，无需 Python 运行时（构建需 Odin 工具链） |
| 要 OpenAI / Ollama 兼容 | `/v1/chat/completions` + `/api/chat` + Web dashboard |
| 首 token 要快（交互体感） | 73ms（M3 实测，快过 ollama 的 ~90ms） |

## 30 秒上手（零编译）

Apple Silicon（M1–M4）直接下载预构建二进制：

```sh
curl -L -o odin-serve https://github.com/vajraimb/odin-infer/releases/latest/download/odin-serve
curl -L -o odin-cli   https://github.com/vajraimb/odin-infer/releases/latest/download/odin-cli
chmod +x odin-serve odin-cli
xattr -cr odin-serve odin-cli   # 仅当 macOS 拦截未签名二进制时需要
./odin-serve ./models/<你的模型>.gguf --metal --port 9748   # http://127.0.0.1:9748
```

MCP（给 agent 用）：

```sh
claude mcp add odin -- /path/to/odin-serve /path/to/model.gguf --mcp --metal
```

从源码构建（需要 [Odin](https://odin-lang.org) 工具链）：

```sh
git clone https://github.com/vajraimb/odin-infer && cd odin-infer
./build.sh                                  # 构建 + 单测，产出 bin/odin-serve bin/odin-cli
MODEL=./models/<你的模型>.gguf ./serve.sh   # http://127.0.0.1:9748
```

> 模型 GGUF 不在仓库里（gitignored）：从 Hugging Face 下载后放 `models/` 或
> 任意路径（见 models/README.md 的推荐布局）。

MCP（给 agent 用）：

```sh
claude mcp add odin -- \
  /path/to/odin-infer/bin/odin-serve \
  /path/to/model.gguf --mcp --metal
```

## 实测性能

M3 / 16GB，DeepSeek-V4-Pro-Qwen3.5-9B UNCENSORED heretic（Q4_K_M），单引擎驻留、冷却后：

| 指标 | odin-infer | ollama / llama.cpp |
|---|---|---|
| 解码吞吐（EN） | 13.3 tok/s（0.89×） | 14.9 tok/s |
| 128-token 回复墙钟 | 10.3–10.5 s | 7.1–9.1 s |
| 首 token | **73 ms** | ~90 ms |
| 二进制体积 | **7.1 MB** | 大二进制 + 运行时依赖 |

> 口径见 [POSITIONING.md](POSITIONING.md) 第五节；GEMV 内核的完整对照实验
> （llama.cpp 全家族扫描、形状调优、消融）在
> [packages/qwen3_5/README.md](packages/qwen3_5/README.md)。

## 支持什么

- **Qwen3 密集**（0.6B–32B）：CPU / Metal
- **Qwen3.5 混合注意力**（Ornith 同族）：文本，248k 词表
- 不支持 MoE / 多模态 / 其他架构 —— 窄定位是特性不是缺陷

## 产品形态

| 形态 | 产物 | 场景 |
|---|---|---|
| **odin-serve**（主） | `bin/odin-serve` | HTTP（OpenAI/Ollama 兼容 + dashboard）或 `--mcp` |
| **odin-cli**（辅） | `bin/odin-cli` | 终端聊天，自动识别 Qwen3/Qwen3.5 |
| **库** | `packages/*` | `-collection` 嵌入任意 Odin 项目 |

## 平台适配

- **首启形状调优**：首次启动实测各 GEMV 内核变体、按矩阵形状选优并缓存
  （`~/.odin-infer/<model>.tune`），M1/M2/M3/M4 各代自动适配
- **内存预算**：启动打印 `mem: ... (36%) OK`，>85% 告警；8GB 机型配置见
  [docs/MEMORY.md](docs/MEMORY.md)

## 仓库结构

```
packages/   ggml · infer(Qwen3) · qwen3_5 · qwen3_5_tokenizer · sampler · tokenizer
server/     odin-serve（HTTP + MCP + dashboard + 集成测试）
cli/        odin-cli
tools/      gemm/delta/doctor/oracle harness + thermal_test.py
docs/       MCP · THERMAL · MEMORY · 排障记录
```

## 文档索引

- [POSITIONING.md](POSITIONING.md) —— 定位、对标口径、路线图
- [docs/MCP.md](docs/MCP.md) —— MCP 工具 schema 与客户端接入
- [docs/MEMORY.md](docs/MEMORY.md) —— 8GB Air 配置
- [docs/THERMAL.md](docs/THERMAL.md) —— 持续负载对照协议
- [AGENTS.md](AGENTS.md) —— 给 AI agent 的构建/测试/使用速查

## License

MIT
