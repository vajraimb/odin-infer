# POSITIONING.zh.md — 定位与产品路线

> 决策记录，2026-11-08。[English version / 英文版](POSITIONING.md)
>
> 不做 llama.cpp 式的全架构泛化，也不做向下兼容。定位：**Apple Silicon
> （MacBook Air 优先）上 Qwen3.5 及以上（混合注意力家族）的高效本地推理
> 引擎**。

## 1. 定位一句话

**个人 Mac 上、给 agent/工具用的、开箱即用的 Qwen3.5 级本地推理引擎**——
卖点是功耗/内存/交互延迟与 agent 就绪度，不是纸面峰值吞吐。

## 2. 目标硬件与用户

- **MacBook Air（M1–M4），16GB 及以上**：无风扇（持续负载热降频）、零配置
  期望。Qwen3.5 从 9B 起步（权重 ~5.6GB + KV），产品下限 16GB；8GB 机型有意
  不在范围内（见 docs/MEMORY.zh.md）。
- 用户画像：在个人 Mac 上跑 agent 生态（Cursor / Claude Code / pie / 自建
  MCP）的人，要隐私、离线、低延迟的本地模型。

## 3. 差异化（vs llama.cpp/ollama）

2026-11 实测于 M3：

| 维度 | odin-infer | llama.cpp/ollama |
|---|---|---|
| 解码吞吐（9B Q4） | 13.3 tok/s (0.89×) | 14.9 tok/s |
| 首 token | **73 ms** | ~90 ms |
| 二进制 | **7.1 MB 单文件** | 大二进制 + 运行时依赖 |
| agent 就绪度 | 原生 tool-call grammar + 前缀缓存 + OpenAI/Ollama/MCP | 通用但无内置 grammar |
| 功耗倾向 | 8-lane 低线程内核（待热测试验证） | 峰值吞吐导向 |
| Qwen3.5-MoE 小变体 | 元数据驱动、任意层数（llama.cpp 只认 40/48/60 层） | 仅 40/48/60 |

## 4. 产品形态（一仓库三交付物）

1. **odin-serve（主形态）**：HTTP 服务端——OpenAI 兼容 + Ollama 兼容 +
   dashboard。**MCP 适配层（stdio）已上线；SSE/streamable HTTP 后续。**
2. **odin-cli（辅形态）**：终端聊天，双引擎自动识别。
3. **库（嵌入形态）**：`-collection` 接入其他 Odin 项目。

**原生 GUI app 不做**：dashboard 已覆盖轻度使用；原生壳不服务 Air 定位。

## 5. 对标口径（必须遵守）

- 同模型同量化（默认 heretic Q4_K_M）、同 prompt、**单引擎驻留**。
- 冷却后各 3 轮取均值；报告 tps / 墙钟 / ttft / prefill 四项。
- 16GB 机器双引擎共存互相拖垮 ~2×，禁止作为对比数据。

## 6. Air 专项待办

1. **热测试协议**：连续生成 10 分钟 tps 曲线与 llama.cpp 同机对比；未验证前
   不对外宣称"不掉速"。
2. ~~首启形状调优~~ ✅（运行时实测选优 + 缓存复用，test_tune.py 验证）
3. ~~热测试工具~~ ✅（tools/thermal_test.py + docs/THERMAL.zh.md；实证待跑）
4. ~~内存预算模式~~ ✅（启动 mem: 估算 + 85% 告警 + docs/MEMORY.zh.md）

## 7. 仓库整合（2026-11-08 完成）

- odin-infer 成为唯一仓库：`packages/` + `server/` + `cli/` + `tools/` +
  `docs/` + `models/`。
- **odin-infer-mac（GitHub）已归档**：其 CLI 并入 `cli/`，旧仓库留指针
  README。
- 本地 `/Users/vajra/Claw/odin-infeer` 目录只保留模型文件，代码不再更新；
  `models/` 软链指向其 GGUF。

## 8. 路线图（按序）

1. ~~仓库整合 + 定位文档~~ ✅
2. ~~MCP 适配层（stdio）~~ ✅ —— `odin.chat` / `odin.models` 工具 + Claude
   Code / Cursor 配置样例 + test_mcp.py 全握手测试；SSE/streamable HTTP 后续。
3. ~~首启调优 + 热测试协议~~ ✅
4. ~~内存预算模式~~ ✅
5. ~~零编译发布管线~~ ✅ —— GitHub Actions 在 `v*` tag 自动发布两个二进制。
6. ~~Qwen3.5-MoE CPU 支持~~ ✅ —— 35B-A3B 系加载器 + softmax top-k 路由 +
   门控共享专家；与 Python 参考逐位一致；真实 0.8B-MoE GGUF 已验证。Metal
   MoE 内核与 24GB 机实测待做。
7. **Metal MoE 内核**（下一步）——把 top-k 专家的 GEMV 路由到现有内核
   （权重偏移量复用），在 0.8B-MoE 上对拍 CPU 路径；这是 35B-A3B 实用化的
   关键。
8. （可选）性能再攻坚：需完整 Xcode（`xcrun metal` + Instruments GPU 计数
   器）追最后 ~11% 解码差距。
