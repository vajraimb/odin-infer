# 内存预算（8GB MacBook Air 配置指南）

odin-serve / odin-cli 在启动时打印内存预算估算：

```
mem: model=5.63 GB kv=0.54 GB lin=0.05 GB total=6.22 GB / physical=16.00 GB (39%) OK
```

total 超过物理内存 85% 时会输出 `WARNING: tight for this machine`。
`kv` 随 `--max-ctx` 线性增长（9B 模型 ≈ 0.54GB @ 8k、0.27GB @ 4k）。

## 8GB 机型推荐配置

8GB Air 扣除 macOS + 浏览器/编辑器后，引擎可用预算 ≈ 3GB：

| 模型 | 量化 | 权重 | 推荐 max-ctx | 适合 |
|------|------|------|--------------|------|
| Qwen3-4B | Q4_K_M | ~2.5GB | 2048 | 默认选择 |
| Qwen3-8B | Q4_K_M | ~5.2GB | — | ❌ 超出预算（≥16GB 才用） |
| Qwen3.5-9B（Ornith/heretic 系） | Q4_K_M | ~5.6GB | — | ❌ 超出预算（≥16GB 才用） |

> Qwen3 系列（`infer` 引擎 + `odin-cli` / HTTP 服务）覆盖 4B；Qwen3.5
> 混合架构目前只有 9B 档，8GB 机型请留在 Qwen3。

## 手动预算方法

1. 启动时看 `mem:` 行；
2. `--max-ctx` 减半 → kv 减半（9B：8k→0.54GB，4k→0.27GB）；
3. 模型换小一档（权重是最大头）；
4. 关闭 KV 持久化（`KV_PERSIST=0`）可省去快照写入时的峰值内存；
5. 不要同时运行 ollama / 其他推理引擎（统一内存互换会双杀吞吐）。
