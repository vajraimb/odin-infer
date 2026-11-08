# POSITIONING.md — 定位与产品路线 / Positioning & product roadmap

> Decision record, 2026-11-08. 决策记录：2026-11-08。
> We deliberately do NOT chase llama.cpp-style generalization across all
> architectures, and we do NOT chase downwards compatibility either. The scope
> is: **a high-efficiency local inference engine for Qwen3.5-class models
> (3.5 and above, the hybrid-attention family) on Apple Silicon (MacBook Air
> first).**
> 不做 llama.cpp 式的全架构泛化，也不做向下兼容。定位：Apple Silicon
> （MacBook Air 优先）上 Qwen3.5 及以上（混合注意力家族）的高效本地推理引擎。

## 1. One-line positioning / 定位一句话

**An out-of-the-box local Qwen3.5-class engine for agents and tools on
personal Macs.** The selling points are power/memory/interactive latency and
agent-readiness — not peak throughput.
**个人 Mac 上、给 agent/工具用的、开箱即用的 Qwen3.5 级本地推理引擎**——卖点是功耗/内存/交互延迟与 agent 就绪度，不是纸面峰值吞吐。

## 2. Target hardware & users / 目标硬件与用户

- **MacBook Air (M1–M4), 16 GB and up** / MacBook Air（M1–M4），16GB 及以上：
  fanless (sustained load throttles), zero-config expectations. Qwen3.5 starts
  at 9B (~5.6 GB weights + KV), so the product floor is 16 GB; 8 GB machines
  are out of scope by design (see docs/MEMORY.md). / 无风扇、零配置期望。
  Qwen3.5 从 9B 起步（权重 ~5.6GB + KV），产品下限 16GB；8GB 机型有意不在范围内（见 docs/MEMORY.md）。
- Users: people running agent ecosystems (Cursor / Claude Code / pie / custom
  MCP) on a personal Mac, wanting privacy, offline, low-latency local models.
  用户画像：在个人 Mac 上跑 agent 生态的人，要隐私、离线、低延迟的本地模型。

## 3. Differentiation vs llama.cpp/ollama / 差异化（vs llama.cpp/ollama）

Measured on M3, 2026-11. 2026-11 实测于 M3：

| Dimension / 维度 | odin-infer | llama.cpp/ollama |
|---|---|---|
| Decode throughput (9B Q4) / 解码吞吐 | 13.3 tok/s (0.89×) | 14.9 tok/s |
| First token / 首 token | **73 ms** | ~90 ms |
| Binary / 二进制 | **7.1 MB single file** / 7.1MB 单文件 | large binary + runtime deps / 大二进制 + 运行时依赖 |
| Agent readiness / agent 就绪 | native tool-call grammar + prefix cache + OpenAI/Ollama/MCP / 原生 tool-call grammar + 前缀缓存 + 三协议 | generic, no built-in grammar / 通用但无内置 grammar |
| Power profile / 功耗倾向 | 8-lane low-thread kernels (passive-cooling friendly; pending thermal test) / 8-lane 低线程内核（待热测试验证） | peak-throughput oriented / 峰值吞吐导向 |

## 4. Product forms (one repo, three artifacts) / 产品形态（一仓库三交付物）

1. **odin-serve (primary / 主形态)**：HTTP server — OpenAI-compatible +
   Ollama-compatible + dashboard. **MCP adapter (stdio) shipped; SSE/streamable
   HTTP is a follow-up.** HTTP 服务端；MCP 适配层（stdio）已上线。
2. **odin-cli (secondary / 辅形态)**：terminal chat, dual-engine auto-detection.
   终端聊天，双引擎自动识别。
3. **library (embed form / 嵌入形态)**：`-collection` for other Odin projects.
   `-collection` 接入其他 Odin 项目。

**No native GUI app**: the dashboard covers casual use; a native shell is
months of work and does not serve the Air positioning.
**原生 GUI app 不做**：dashboard 已覆盖轻度使用；原生壳不服务 Air 定位。

## 5. Benchmark protocol / 对标口径（必须遵守）

- Same model & quantization (default heretic Q4_K_M), same prompt, **solo
  residency** / 同模型同量化、同 prompt、单引擎驻留
- Cooled machine, 3 runs each, average; report tps / wall / ttft / prefill.
  冷却后各 3 轮取均值；报告 tps / 墙钟 / ttft / prefill 四项。
- Two 9B engines co-resident on 16 GB thrash each other ~2× — such data is
  invalid for comparison. 16GB 机器双引擎共存互相拖垮 ~2×，禁止作为对比数据。

## 6. Air-specific TODOs / Air 专项待办

1. **Thermal test protocol / 热测试协议**：10-minute sustained generation,
   tps sampled every 30s; compare the droop curve against llama.cpp on the
   same machine. The "flatter curve" claim is NOT to be advertised until
   verified. 连续生成 10 分钟 tps 曲线与 llama.cpp 同机对比；未验证前不对外宣称。
2. ~~First-run shape autotuning / 首启形状调优~~ ✅（runtime measured pick +
   cache reuse, verified by test_tune.py / 运行时实测选优 + 缓存复用，test_tune.py 验证）
3. ~~Thermal protocol tooling / 热测试工具~~ ✅（tools/thermal_test.py +
   docs/THERMAL.md；实证待跑 / empirical run pending）
4. ~~Memory budget mode / 内存预算模式~~ ✅（startup `mem:` estimate + 85%
   warning + docs/MEMORY.md 8 GB table / 启动 mem: 估算 + 85% 告警 + 8GB 配置表）

## 7. Repo consolidation (done 2026-11-08) / 仓库整合（2026-11-08 完成）

- odin-infer is now the single repo: `packages/` + `server/` + `cli/` +
  `tools/` + `docs/` + `models/`. odin-infer 成为唯一仓库。
- **odin-infer-mac (GitHub) is retired**: its CLI merged into `cli/`; the old
  repo is archived with a pointer README. odin-infer-mac 已归档并入 cli/。
- The local `/Users/vajra/Claw/odin-infeer` directory keeps model files only;
  code there is frozen; `models/` symlinks point at its GGUFs.
  本地 odin-infeer 目录保留模型文件，代码不再更新。

## 8. Roadmap (in order) / 路线图（按序）

1. ~~repo consolidation + positioning docs~~ ✅ / 仓库整合 + 定位文档
2. ~~MCP adapter (stdio)~~ ✅ — `odin.chat` / `odin.models` tools + Claude
   Code / Cursor config samples + test_mcp.py full handshake; SSE/streamable
   HTTP later. MCP 适配层（stdio）已上线；SSE/streamable HTTP 后续。
3. ~~first-run autotuning + thermal protocol~~ ✅ / 首启调优 + 热测试协议
4. ~~memory budget mode~~ ✅ / 内存预算模式
5. ~~zero-compile release pipeline~~ ✅ — GitHub Actions builds and publishes
   `odin-serve` / `odin-cli` on every `v*` tag. 零编译发布管线（tag 触发 Release）。
6. ~~Qwen3.5-MoE CPU support~~ ✅（35B-A3B 系加载器 + softmax top-k 路由 +
   共享专家，合成 GGUF 对拍逐位一致；Metal MoE 内核与 24GB 机实测待做）/
   Qwen3.5-MoE CPU 支持已上线（Metal 内核待做）
7. (optional) perf round 2: needs full Xcode (`xcrun metal` + Instruments GPU
   counters) to chase the last ~11% decode gap. （可选）性能再攻坚：需完整 Xcode。
