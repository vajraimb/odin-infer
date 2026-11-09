# BENCHMARKS.zh.md · 基准数据

[English version / 英文版](BENCHMARKS.md)

以下数字全部实测于作者的 **M3 / 16GB MacBook Pro**，Odinfer 官方 release
二进制 vs ollama（内置 llama.cpp）。口径遵循 POSITIONING.zh.md 第五节：
同模型、同 prompt、注明单引擎驻留与冷却条件。"满载机器"标记的数字是在
无关后台进程运行下测得，仅用于展示降级形态。

## 解码（9B dense，heretic Q4_K_M）

单引擎驻留、冷却后：

| 指标 | Odinfer | ollama（llama.cpp） |
|---|---|---|
| 解码吞吐（EN） | 13.3 tok/s | 14.9 tok/s |
| 128-token 回复墙钟 | 10.3–10.5 s | 7.1–9.1 s |
| 首 token | **73 ms** | ~90 ms |
| 二进制体积 | **7.1 MB** | 大二进制 + 运行时 |

## Qwen3.8-27B（UD-IQ2_XXS，7.3GB，2-bit Dynamic 混合量化）

ollama 目前无法服务的模型（Unsloth Dynamic 混合量化 + 逐层量化混排）；
Odinfer 端到端可跑：

| 指标 | 数值 |
|---|---|
| 加载 | ~20ms mmap + 配置（13 种量化、851 张量） |
| GPU 解码 | 0.2–1.3 s/tok（M3；热态实测 223ms/tok，满载 1.3s/tok） |
| CPU 解码 | 15–50 s/tok（满载机器） |
| GPU 批量 prefill | ~227 ms/tok（30-token prompt）vs 逐 token ~420 ms/tok |
| CPU vs Metal logits | max_abs 1.4e-3（2 token），0 个 >1e-2 |
| 批量 vs 逐 token prefill | max_abs 5.7e-3，解码步 6.6e-3，0 个 >1e-2 |
| ctx 8192 | kv 1.07GB，16GB 机器总占用 8.5GB（49%） |
| 质量说明 | 2-bit：短答连贯；上下文消歧有限（"transformer"答成电气变压器） |

## MoE（0.8B-MoE，社区 qwen35moe 变体）

| 指标 | 数值 |
|---|---|
| Metal 解码 | 37 ms/tok（vs CPU 1.7s/tok，~45×） |
| CPU vs Metal logits | max_abs ~3e-3 |
| 批量 prefill（分组 GEMM） | 比逐 token 快 ~40%；max_abs 8e-4 |

## 量化反量化（全类型）

Odin == Python 参考 == 编译后的 llama.cpp C，逐位一致（23040 项检查，
worst diff 0.0）：F32/F16、Q4_0/Q4_1、Q5_0/Q5_1、Q8_0、Q2_K..Q8_K、
IQ1_S/IQ1_M/IQ2_XXS/XS/S/IQ3_XXS/S/IQ4_NL/XS。逐内核 Metal GEMV vs CPU：
max_rel ≤ 1.8e-5（两个逐位一致）。

## 满载机器的现实（降级形态）

后台 agent/交易机器人运行时，两个引擎都降级约 2×（Odinfer 13.3 → ~3
tok/s，ollama ~15 → ~2.3 tok/s）——见 docs/THERMAL.zh.md 的单引擎驻留协议。
不要在机器忙于其他任务时跑基准。

## 复现方法

```sh
# 9B 对照（thermal_test.py，双引擎）
python3 tools/thermal_test.py --engine odin --duration 600 --out odin.csv
python3 tools/thermal_test.py --engine ollama --duration 600 --out ollama.csv

# 27B 验证
odin run tools/q38_load_test.odin -file <collections> -- <qwen3.8.gguf> 256 --compare
odin run tools/q38_load_test.odin -file <collections> -- <qwen3.8.gguf> 256 --batch-compare
```
