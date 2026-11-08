# odin-infer

**Apple Silicon（MacBook Air 优先）上 Qwen3 / Qwen3.5 的高效本地推理引擎。**

纯 Odin 实现，无 Python、无 llama.cpp 依赖。定位不是"最快的推理内核"，而是
**内存受限、被动散热、以 agent/工具为消费场景的个人 Mac** 上体验最好的本地
推理：~7MB 单文件服务端、73ms 首 token、前缀缓存、原生 tool-call grammar。

## 产品形态（一个仓库，三个交付物）

| 形态 | 构建产物 | 用途 |
|------|----------|------|
| **odin-serve**（主） | `bin/odin-serve`，`./serve.sh` 启动 | OpenAI 兼容 `/v1` + Ollama 兼容 `/api/chat` + Web dashboard；MCP 适配层在路线图上 |
| **odin-cli**（辅） | `bin/odin-cli` | 终端交互聊天，自动识别 Qwen3 / Qwen3.5 |
| **库** | `packages/*`（`-collection` 可嵌入） | ggml 解析、Qwen3/Qwen3.5 引擎、分词、采样 |

外部消费者（`pie` 等 agent）通过 OpenAI/Ollama 兼容端点接入，见 [POSITIONING.md](POSITIONING.md)。

## 快速开始

```sh
./build.sh                                  # 构建 bin/odin-serve + bin/odin-cli + 跑单测
MODEL=./models/<your>.gguf ./serve.sh       # 启动服务 http://127.0.0.1:9748
./codex.sh "your prompt"                    # Codex CLI 接入（responses wire API）
./pie.sh                                    # pie agent 接入
```

模型文件放 `models/`（gitignored，放软链或下载脚本即可；见 `models/README.md`）。

## 支持的模型

- **Qwen3 密集**（0.6B–32B）：`packages/infer`，CPU/Metal
- **Qwen3.5 混合注意力**（Ornith 同族，full + linear + gated delta）：`packages/qwen3_5`，文本
- 不支持：Qwen3 MoE、多模态、其他架构 —— 有意为之的窄定位（见 POSITIONING.md）

## 实测性能（M3 / 16GB，DeepSeek-V4-Pro-Qwen3.5-9B UNCENSORED heretic Q4_K_M）

| 指标 | odin-infer | ollama/llama.cpp |
|------|------------|------------------|
| 解码吞吐（EN，128 token） | 13.3 tok/s | 14.9 tok/s |
| 128-token 回复墙钟 | 10.3–10.5 s | 7.1–9.1 s |
| 首 token | **73 ms** | ~90 ms |
| prefill ~20 token | ~0.5 s | ~0.09 s |
| 二进制体积 | **7.1 MB** | ~30 MB + 依赖 |

> 单引擎驻留、冷却后测得。16GB 机器同时驻留两个 9B 引擎会互换内存双双 ~2×
> 变慢；Air 的 8GB 机型请用 4B/8B 模型（配置建议见 POSITIONING.md）。
> 性能工作的完整记录（GEMV 形状扫描、llama.cpp 全家族对照、消融分解）在
> [packages/qwen3_5/README.md](packages/qwen3_5/README.md)。

## 仓库结构

```
packages/   库：ggml / infer(Qwen3) / qwen3_5 / qwen3_5_tokenizer / sampler / tokenizer
server/     服务端（serve_harness：HTTP 路由 + dashboard + 推理编排）
cli/        终端聊天（原 odin-infer-mac 仓库，已并入）
tools/      开发工具：gemm_harness（内核验证/扫描）、doctor/oracle/delta harness
docs/       设计与排障记录
models/     模型软链（gitignored）
```

## 开发

```sh
./build.sh                         # 全量构建 + 单测
( cd packages/qwen3_5 && ./build.sh )   # qwen3_5 纯数学单测
( cd packages/tokenizer && odin test . ) # 分词器单测（内存紧张时可能 OOM 失败）
./serve.sh [port] [max_ctx]       # 开发模式（odin run）
```

内核 A/B 开关（env）：`QGEMV3=1`（llama.cpp 行切块移植）、`QGEMV2R=1`
（2 行/线程）、`QGEMVNORM=1`（norm 融合）——均为实测回归项，保留供剖析后复验。
消融开关：写 `/tmp/qprof_flags`（noout/nomlp/nosmall/nonorm/noattn/nodelta），
无需重启。

## 库 API（嵌入）

```odin
import infer "infer:infer"           // Qwen3 密集
import q35 "qwen3_5:qwen3_5"         // Qwen3.5 混合

Engine_Opts :: struct { max_ctx, num_threads: int, use_metal: bool }
engine_load    :: proc(path: string, opts: Engine_Opts) -> (Engine, bool)
engine_forward :: proc(e: ^Engine, token, pos: int) -> []f32
engine_forward_batch :: proc(e: ^Engine, tokens: []int, pos_start: int) -> []f32  // qwen3_5
engine_reset_state  :: proc(e: ^Engine)   // qwen3_5（gated-delta 状态不可回滚）
```

构建你的项目时挂 collection 标志：

```sh
odin build . \
  -collection:ggml=path/to/odin-infer/packages \
  -collection:infer=path/to/odin-infer/packages \
  -collection:qwen3_5=path/to/odin-infer/packages \
  -collection:qwen3_5_tokenizer=path/to/odin-infer/packages \
  -collection:sampler=path/to/odin-infer/packages \
  -collection:tokenizer=path/to/odin-infer/packages
```

## 平台

macOS / Apple Silicon（M1–M4）。CPU 前向与 Metal GPU 路径均可用；Metal 为
默认推荐（`--metal` / `-g 1`）。
