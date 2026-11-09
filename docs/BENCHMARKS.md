# Benchmarks · 基准数据

[中文版 / Chinese version](BENCHMARKS.zh.md)

All numbers measured on the author's **M3 / 16 GB MacBook Pro**, Odinfer
releases vs ollama (bundled llama.cpp). Protocol follows POSITIONING.md §5:
same model, same prompt, solo residency where stated, cooled machine. Numbers
marked "loaded machine" were taken with unrelated background processes running
and are included only to show the degradation mode.

## Decode (9B dense, heretic Q4_K_M)

Solo residency, cooled:

| Metric | Odinfer | ollama (llama.cpp) |
|---|---|---|
| Decode throughput (EN) | 13.3 tok/s | 14.9 tok/s |
| Wall, 128-token reply | 10.3–10.5 s | 7.1–9.1 s |
| First token | **73 ms** | ~90 ms |
| Binary size | **7.1 MB** | large binary + runtime |

## Qwen3.8-27B (UD-IQ2_XXS, 7.3 GB, 2-bit Dynamic mix)

A model ollama cannot currently serve (Unsloth Dynamic mixed quant + qwen35
arch with per-layer quant mix); Odinfer runs it end-to-end:

| Metric | Value |
|---|---|
| Load | ~20 ms mmap + config (13 quant kinds, 851 tensors) |
| GPU decode | 0.2–1.3 s/tok (M3; 223 ms/tok measured warm, 1.3 s/tok under heavy load) |
| CPU decode | 15–50 s/tok (loaded machine) |
| GPU batched prefill | ~227 ms/tok (30-token prompt) vs ~420 ms/tok per-token |
| CPU-vs-Metal logits | max_abs 1.4e-3 (2 tokens), 0 diffs > 1e-2 |
| Batch-vs-per-token prefill | max_abs 5.7e-3, decode-step 6.6e-3, 0 diffs > 1e-2 |
| ctx 8192 | kv 1.07 GB, total 8.5 GB on 16 GB (49%) |
| Quality note | 2-bit: coherent short answers; context disambiguation limited ("transformer" → electrical transformer) |

## MoE (0.8B-MoE, community qwen35moe variant)

| Metric | Value |
|---|---|
| Metal decode | 37 ms/tok (vs 1.7 s/tok CPU, ~45×) |
| CPU-vs-Metal logits | max_abs ~3e-3 |
| Batched prefill (grouped GEMM) | ~40% faster than per-token; max_abs 8e-4 |

## Quant dequant (all types)

Odin == Python reference == compiled llama.cpp C, bit-exact (23040 checks,
worst diff 0.0): F32/F16, Q4_0/Q4_1, Q5_0/Q5_1, Q8_0, Q2_K..Q8_K,
IQ1_S/IQ1_M/IQ2_XXS/XS/S/IQ3_XXS/S/IQ4_NL/XS. Per-kind Metal GEMV vs CPU:
max_rel <= 1.8e-5 (two bit-exact).

## Loaded-machine reality (degradation mode)

With background agents/trading bots running, BOTH engines degrade ~2× (Odinfer
13.3 → ~3 tok/s, ollama ~15 → ~2.3 tok/s) — see docs/THERMAL.md for the
solo-residency protocol. Never benchmark a machine that is doing other work.

## How to reproduce

```sh
# 9B comparison (thermal_test.py, both engines)
python3 tools/thermal_test.py --engine odin --duration 600 --out odin.csv
python3 tools/thermal_test.py --engine ollama --duration 600 --out ollama.csv

# 27B validation
odin run tools/q38_load_test.odin -file <collections> -- <qwen3.8.gguf> 256 --compare
odin run tools/q38_load_test.odin -file <collections> -- <qwen3.8.gguf> 256 --batch-compare
```
