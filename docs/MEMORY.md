# Memory budget / 内存预算（8GB MacBook Air 配置指南）

odin-serve / odin-cli print a memory budget estimate at startup:
odin-serve / odin-cli 在启动时打印内存预算估算：

```
mem: model=5.63 GB kv=0.54 GB lin=0.05 GB total=6.22 GB / physical=16.00 GB (39%) OK
```

A `WARNING: tight for this machine` is emitted when the total exceeds 85% of
physical RAM. `kv` grows linearly with `--max-ctx` (9B: ~0.54 GB @ 8k,
~0.27 GB @ 4k). total 超过物理内存 85% 时会输出 `WARNING: tight for this
machine`。`kv` 随 `--max-ctx` 线性增长。

## Product floor: 16 GB / 产品下限：16GB

The product is positioned at Qwen3.5 and above; the Qwen3.5 hybrid family
starts at 9B (~5.6 GB weights at Q4_K_M). An 8 GB Air leaves ~3 GB for the
engine after macOS + apps, so **8 GB machines are out of product scope by
design** — no Qwen3 fallback story.
产品定位为 Qwen3.5 及以上；Qwen3.5 家族从 9B 起步（Q4_K_M 权重 ~5.6GB）。
8GB Air 扣除系统后引擎预算仅 ~3GB，**8GB 机型有意不在产品范围内**。

| Model / 模型 | Quant / 量化 | Weights / 权重 | Suggested max-ctx / 推荐 max-ctx | Verdict / 结论 |
|---|---|---|---|---|
| Qwen3.5-9B (Ornith/heretic family) | Q4_K_M | ~5.6 GB | 4096–8192 | ✅ 16 GB floor / 16GB 下限 |
| 同架构更大尺寸（若发布） | Q4_K_M | 按元数据适配 | 按 `mem:` 行调 | 32GB+ / 24GB+ |

> The engine is config-driven: any same-architecture Qwen3.5 size loads via
> GGUF metadata (MTP-aware loader) — the table grows as the family does.
> 引擎由 GGUF 元数据驱动：同架构任意尺寸直接加载（MTP 感知加载器）。

## Manual budgeting / 手动预算方法

1. Check the `mem:` line at startup. 启动时看 `mem:` 行；
2. Halve `--max-ctx` → kv halves. `--max-ctx` 减半 → kv 减半；
3. Drop one model size (weights dominate). 模型换小一档（权重是最大头）；
4. `KV_PERSIST=0` avoids snapshot-write peak memory. 关闭 KV 持久化可省峰值内存；
5. Never run ollama / another inference engine concurrently — unified-memory
   swap kills both. 不要同时运行其他推理引擎（统一内存互换双杀吞吐）。
