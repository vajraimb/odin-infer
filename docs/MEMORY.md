# Memory budget

odin-serve / odin-cli print a memory budget estimate at startup:

```
mem: model=5.63 GB kv=0.54 GB lin=0.05 GB total=6.22 GB / physical=16.00 GB (39%) OK
```

A `WARNING: tight for this machine` is emitted when the total exceeds 85% of
physical RAM. `kv` grows linearly with `--max-ctx` (9B: ~0.54 GB @ 8k,
~0.27 GB @ 4k).

[中文版 / Chinese version](MEMORY.zh.md)

## Product floor: 16 GB

The product is positioned at Qwen3.5 and above; the Qwen3.5 hybrid family
starts at 9B (~5.6 GB weights at Q4_K_M). An 8 GB Air leaves ~3 GB for the
engine after macOS + apps, so **8 GB machines are out of product scope by
design**.

| Model | Quant | Weights | Suggested max-ctx | Verdict |
|---|---|---|---|---|
| Qwen3.5-9B (Ornith/heretic family) | Q4_K_M | ~5.6 GB | 4096–8192 | ✅ 16 GB floor |
| Qwen3.5-27B | Q4_K_M | ~17 GB | per `mem:` line | 32 GB+ |
| Qwen3.5-MoE 35B-A3B | Q3_K_S/Q4_K_M | ~14.5/21 GB | small | 24 GB+ (CPU path for now) |
| Larger same-arch sizes (if released) | Q4_K_M | per metadata | per `mem:` line | 32 GB+ |

> The engine is config-driven: any same-architecture Qwen3.5 size loads via
> GGUF metadata (MTP-aware loader) — the table grows as the family does.

## Manual budgeting

1. Check the `mem:` line at startup.
2. Halve `--max-ctx` → kv halves.
3. Drop one model size (weights dominate).
4. `KV_PERSIST=0` avoids snapshot-write peak memory.
5. Never run ollama / another inference engine concurrently — unified-memory
   swap kills both.
