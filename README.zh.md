# Odinfer

**Apple Silicon（MacBook Air 优先）上 Qwen3.5 级模型的高效本地推理引擎。**

一个 7MB 的单文件二进制，把 Qwen3.5 级模型跑在你的 Mac 上——无 Python、无
llama.cpp 依赖，首 token 73ms，原生 MCP，agent 就绪。

[English version / 英文版](README.md) ·
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![language: Odin](https://img.shields.io/badge/language-Odin-9cf.svg)](https://odin-lang.org)

---

## 为什么是它

| 你的需求 | Odinfer 的回答 |
|---|---|
| 在 Air 上跑本地模型，不烫不卡 | 低线程内核专为被动散热设计，内存预算启动即显（[热测试协议](docs/THERMAL.zh.md)） |
| 给 Cursor / Claude Code / pie 供模型 | `--mcp` 一键变 MCP 服务端，`odin.chat` / `odin.models` 开箱即用（[docs/MCP.zh.md](docs/MCP.zh.md)） |
| 不想装 Python / 编译链 | 单文件二进制，GitHub Releases 零编译下载 |
| 要 OpenAI / Ollama 兼容 | `/v1/chat/completions` + `/api/chat` + Web dashboard |
| 首 token 要快（交互体感） | 73ms（M3 实测，快过 ollama 的 ~90ms） |

## 30 秒上手（零编译）

Apple Silicon（M1–M4）：

```sh
curl -L -o odin-serve https://github.com/vajraimb/odinfer/releases/latest/download/odin-serve
curl -L -o odin-cli   https://github.com/vajraimb/odinfer/releases/latest/download/odin-cli
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
git clone https://github.com/vajraimb/odinfer && cd odinfer
./build.sh                                  # 构建 + 单测，产出 bin/odin-serve bin/odin-cli
MODEL=./models/<你的模型>.gguf ./serve.sh   # http://127.0.0.1:9748
```

> 模型 GGUF 不在仓库里（gitignored）：从 Hugging Face 下载后放 `models/` 或
> 任意路径（推荐布局见 models/README.zh.md）。

## 实测性能

M3 / 16GB，DeepSeek-V4-Pro-Qwen3.5-9B UNCENSORED heretic（Q4_K_M），单引擎
驻留、冷却后：

| 指标 | Odinfer | ollama / llama.cpp |
|---|---|---|
| 解码吞吐（EN） | 13.3 tok/s（0.89×） | 14.9 tok/s |
| 128-token 回复墙钟 | 10.3–10.5 s | 7.1–9.1 s |
| 首 token | **73 ms** | ~90 ms |
| 二进制体积 | **7.1 MB** | 大二进制 + 运行时依赖 |

> 完整数字、Qwen3.8-27B 与 MoE 表格、满载降级形态见
> [docs/BENCHMARKS.zh.md](docs/BENCHMARKS.zh.md)；口径见
> [POSITIONING.zh.md](POSITIONING.zh.md) 第五节；GEMV 内核对照实验在
> [packages/qwen3_5/README.md](packages/qwen3_5/README.md)。

## 支持什么

**定位在 Qwen3.5 及以上：**

- **Qwen3.5 混合注意力**（Ornith / heretic 家族）：文本，248k 词表。9B 已
  端到端验收；同架构其他尺寸靠 GGUF 元数据自动加载（MTP 感知加载器）——
  包括 **Qwen3.8-27B**（GPU 解码已验证，ctx 8192）。
- **任意 llama.cpp 量化**：Q2_K..Q8_K + IQ 家族（IQ1_S..IQ4_XS）逐位一致
  实现——unsloth "Dynamic" 混合量化 GGUF 可加载并 GPU 解码。
- **Qwen3.5-MoE**（35B-A3B / 122B-A10B / 397B-A17B）：CPU + Metal 双路径
  已支持——softmax top-k 路由、逐专家 GEMV 分派（Q4_K/Q5_0/Q8_0/Q6_K）、
  sigmoid 门控共享专家；Metal 输出与 CPU 一致（~3e-3，fastMath），0.8B-MoE
  上 ~45× CPU 速度（37ms/token）。HF 上的社区小变体今天就能跑——包括
  llama.cpp 本身会拒绝的层数配置。
- 不回退支持 Qwen3.5 之前的旧架构、不支持多模态、不支持其他模型家族——窄定位是特性不是缺陷。

## 产品形态

| 形态 | 产物 | 场景 |
|---|---|---|
| **odin-serve**（主） | `bin/odin-serve` | HTTP（OpenAI/Ollama 兼容 + dashboard）或 `--mcp` |
| **odin-cli**（辅） | `bin/odin-cli` | 终端聊天 |
| **库** | `packages/*` | `-collection` 嵌入任意 Odin 项目 |

## 平台适配

- **首启形状调优**：首次启动实测各 GEMV 内核变体、按矩阵形状选优并缓存
  （`~/.odin-infer/<model>.tune`），M1/M2/M3/M4 各代自动适配
- **内存预算**：启动打印 `mem: ... (36%) OK`，>85% 告警；16GB 机型配置见
  [docs/MEMORY.zh.md](docs/MEMORY.zh.md)
- **Linux CUDA**（sm_70+，含 Tesla V100）：`-define:ODINFER_BACKEND="cuda"`
  编译。驱动和 NVRTC 运行时加载。见 [docs/CUDA.zh.md](docs/CUDA.zh.md)。
- **Windows Vulkan**（共享显存 Radeon）：`-define:ODINFER_BACKEND=vulkan`
  编译。Odin dev-2026-07 和驱动的 `vulkan-1.dll` 就够；SPIR-V 已嵌进二进制。
  见 [docs/VULKAN.zh.md](docs/VULKAN.zh.md)。
  macOS 构建仍走 Metal。

## 仓库结构

```
packages/   ggml · infer(Qwen3 历史兼容) · qwen3_5 · qwen3_5_tokenizer · sampler · tokenizer · cuda · vulkan
server/     odin-serve（HTTP + MCP + dashboard + 集成测试）
cli/        odin-cli
tools/      gemm/delta/doctor/oracle harness + thermal_test.py + MoE 测试
docs/       MCP · THERMAL · MEMORY · 排障记录
```

## 文档索引

- [POSITIONING.zh.md](POSITIONING.zh.md) —— 定位、对标口径、路线图
- [docs/MCP.zh.md](docs/MCP.zh.md) —— MCP 工具 schema 与客户端接入
- [docs/MEMORY.zh.md](docs/MEMORY.zh.md) —— 内存预算配置
- [docs/THERMAL.zh.md](docs/THERMAL.zh.md) —— 持续负载对照协议
- [docs/CUDA.zh.md](docs/CUDA.zh.md) —— Linux CUDA 后端（sm_70+，V100）
- [docs/VULKAN.zh.md](docs/VULKAN.zh.md) —— Windows Vulkan 后端（SPIR-V 已嵌入）
- [AGENTS.zh.md](AGENTS.zh.md) —— 给 AI agent 的构建/测试/使用速查

## 社区

- [CHANGELOG.md](CHANGELOG.md) —— 版本历史
- [CONTRIBUTING.zh.md](CONTRIBUTING.zh.md) —— 贡献规则与 PR 清单（[英文](CONTRIBUTING.md)）
- [SECURITY.md](SECURITY.md) —— 漏洞报告
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- Issues 与 PR：[GitHub](https://github.com/vajraimb/odinfer)

## License

MIT
