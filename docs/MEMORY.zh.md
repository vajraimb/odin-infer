# 内存预算

odin-serve / odin-cli 在启动时打印内存预算估算：

```
mem: model=5.63 GB kv=0.54 GB lin=0.05 GB total=6.22 GB / physical=16.00 GB (39%) OK
```

total 超过物理内存 85% 时会输出 `WARNING: tight for this machine`。
`kv` 随 `--max-ctx` 线性增长（9B：8k 约 0.54GB，4k 约 0.27GB）。

[English version / 英文版](MEMORY.md)

## 产品下限：16GB

产品定位为 Qwen3.5 及以上；Qwen3.5 混合注意力家族从 9B 起步（Q4_K_M 权重
~5.6GB）。8GB Air 扣除 macOS + 应用后引擎预算仅 ~3GB，**8GB 机型有意不在产品
范围内**。

| 模型 | 量化 | 权重 | 推荐 max-ctx | 结论 |
|---|---|---|---|---|
| Qwen3.5-9B（Ornith/heretic 家族） | Q4_K_M | ~5.6 GB | 4096–8192 | ✅ 16GB 下限 |
| Qwen3.5-27B | Q4_K_M | ~17 GB | 按 `mem:` 行调 | 32GB+ |
| Qwen3.5-MoE 35B-A3B | Q3_K_S/Q4_K_M | ~14.5/21 GB | 小 ctx | 24GB+（当前走 CPU 路径） |
| 同架构更大尺寸（若发布） | Q4_K_M | 按元数据 | 按 `mem:` 行调 | 32GB+ |

> 引擎由 GGUF 元数据驱动：同架构任意尺寸直接加载（MTP 感知加载器）——表格
> 随家族扩展而增长。

## 手动预算方法

1. 启动时看 `mem:` 行；
2. `--max-ctx` 减半 → kv 减半；
3. 模型换小一档（权重是最大头）；
4. `KV_PERSIST=0` 可省去快照写入时的峰值内存；
5. 不要同时运行 ollama / 其他推理引擎（统一内存互换双杀吞吐）。
