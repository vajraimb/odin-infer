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

## Recommended configs for 8 GB / 8GB 机型推荐配置

After macOS + browser/editor, an 8 GB Air leaves ~3 GB for the engine:
8GB Air 扣除 macOS + 浏览器/编辑器后，引擎可用预算 ≈ 3GB：

| Model / 模型 | Quant / 量化 | Weights / 权重 | Suggested max-ctx / 推荐 max-ctx | Verdict / 结论 |
|---|---|---|---|---|
| Qwen3-4B | Q4_K_M | ~2.5 GB | 2048 | default choice / 默认选择 |
| Qwen3-8B | Q4_K_M | ~5.2 GB | — | ❌ over budget (needs ≥16 GB) / 超出预算（≥16GB 才用） |
| Qwen3.5-9B (Ornith/heretic family) | Q4_K_M | ~5.6 GB | — | ❌ over budget (needs ≥16 GB) / 超出预算（≥16GB 才用） |

> The Qwen3 line (infer engine + odin-cli / HTTP) covers 4B; the Qwen3.5
> hybrid architecture currently starts at 9B — on 8 GB machines stay on
> Qwen3. Qwen3 系列覆盖 4B；Qwen3.5 目前只有 9B 档，8GB 机型请留在 Qwen3。

## Manual budgeting / 手动预算方法

1. Check the `mem:` line at startup. 启动时看 `mem:` 行；
2. Halve `--max-ctx` → kv halves. `--max-ctx` 减半 → kv 减半；
3. Drop one model size (weights dominate). 模型换小一档（权重是最大头）；
4. `KV_PERSIST=0` avoids snapshot-write peak memory. 关闭 KV 持久化可省峰值内存；
5. Never run ollama / another inference engine concurrently — unified-memory
   swap kills both. 不要同时运行其他推理引擎（统一内存互换双杀吞吐）。
