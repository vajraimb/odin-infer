# POSITIONING.md — odin-infer 定位与产品路线

> 决策记录：2026-11-08。不做 llama.cpp 式的全架构泛化，只做
> **Apple Silicon（MacBook Air 优先）上 Qwen3/Qwen3.5 的高效本地推理引擎**。

## 一、定位一句话

**个人 Mac 上、给 agent/工具用的、开箱即用的 Qwen 本地推理引擎** ——
卖点是功耗/内存/交互延迟与 agent 就绪度，不是纸面峰值吞吐。

## 二、目标硬件与用户

- **MacBook Air（M1–M4）**：无风扇（持续负载热降频）、8–24GB 统一内存、
  用户期望零配置
- 用户画像：在个人 Mac 上跑 agent 生态（Cursor / Claude Code / pie / 自定义
  MCP）的人，要隐私、离线、低延迟的本地模型

## 三、差异化（vs llama.cpp/ollama，2026-11 实测于 M3）

| 维度 | odin-infer | llama.cpp/ollama |
|------|-----------|------------------|
| 解码吞吐（9B Q4） | 13.3 tok/s（0.89×） | 14.9 tok/s |
| 首 token | **73ms** | ~90ms |
| 二进制 | **7.1MB 单文件** | 大二进制 + 运行时依赖 |
| agent 就绪 | 原生 tool-call grammar + 前缀缓存 + OpenAI/Ollama 双协议 | 通用但无内置 grammar |
| 功耗倾向 | 8-lane 低线程内核（Air 被动散热友好，待热测试验证） | 峰值吞吐导向 |

## 四、产品形态（一仓库三交付物）

1. **odin-serve（主形态）**：HTTP 服务端，OpenAI 兼容 + Ollama 兼容 +
   dashboard。**MCP 适配层（stdio/SSE）为下一优先项** —— 本机 agent 生态
   （Cursor worker、pie、memory-mcp）是明确的消费场景，且 serve_harness 已
   内置 tool-call grammar（`TOOLCALL_GRAMMAR` + `parse_tool_calls`），MCP 只是
   协议适配。
2. **odin-cli（辅形态）**：终端聊天，双引擎自动识别。
3. **库（嵌入形态）**：`-collection` 接入其他 Odin 项目。

**原生 GUI app 不做**：dashboard 已覆盖轻度使用；原生壳是月级投入且不服务
Air 定位。

## 五、对标口径（必须遵守）

- 同模型同量化（默认 heretic Q4_K_M）、同 prompt、**单引擎驻留**
- 冷却后各 3 轮取均值；报告 tps / 墙钟 / ttft / prefill 四项
- 16GB 机器双引擎共存会互相拖垮 ~2×，禁止作为对比数据

## 六、Air 专项待办

1. **热测试协议**：连续生成 10 分钟，每 30s 记录 tps 曲线；与 llama.cpp 同机
   对比。假设：低线程内核曲线更平（被动散热红利）——这是定位的核心卖点，
   未验证前不对外宣称。
2. **首启形状调优**：`tools/gemm_harness` 的扫描（26 内核 × 形状）做成首次
   运行 30s 自动调优 + 缓存（llama.cpp 同款机制；8-lane 最优点是在 M3 上
   找到的，M1/M2/M4 各代需实测）。
3. **8GB 配置建议**：4B/8B 模型、`--max-ctx 2048`、KV 预算上限；文档给出
   矩阵。

## 七、仓库整合（2026-11-08 完成）

- odin-infer 成为唯一仓库：`packages/` + `server/` + `cli/` + `tools/` +
  `docs/` + `models/`
- **odin-infer-mac（GitHub）已停更，功能并入 `cli/`**；请在 GitHub 上将旧
  repo 标记 Archive，README 指向本仓库
- 本地 `/Users/vajra/Claw/odin-infeer`（历史目录）保留模型文件，代码不再
  更新；`models/` 内的软链已指向其 GGUF

## 八、路线图（按序）

1. ~~repo 整合 + 定位文档~~ ✅
2. **MCP 适配层**（stdio + SSE，复用 tool-call grammar；配 Claude Code /
   Cursor / pie 三个接入样例与测试）
3. 首启形状调优 + 热测试协议（Air 实证）
4. 内存预算模式（8GB 机型默认配置）
5. （可选）性能再攻坚：需完整 Xcode（`xcrun metal` + Instruments GPU 计数
   器），目标补齐最后 ~11% 解码差距
