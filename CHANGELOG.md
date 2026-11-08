# Changelog · 更新日志

## Unreleased

- Full quant dequant coverage: Q2_K/Q3_K/Q5_K/Q8_K + IQ family (IQ1_S/IQ1_M/
  IQ2_XXS/XS/S/IQ3_XXS/S/IQ4_NL/XS), bit-exact vs llama.cpp. Unsloth Dynamic
  GGUFs load — Qwen3.8-27B-UD-IQ2_XXS validated end-to-end (CPU).

All notable changes are recorded here. Format: keep it human-readable.

## v0.2.0 — 2026-11-08

- **Qwen3.5-MoE (qwen35moe) support**: 35B-A3B family loader, CPU + Metal
  paths, softmax top-k routing, gated shared expert. Works with community
  MoE variants of ANY layer count (llama.cpp only accepts 40/48/60).
- Metal MoE: GPU router + per-expert GEMV dispatch (Q4_K/Q5_0/Q8_0/Q6_K),
  ~45× faster than the CPU path on the 0.8B-MoE (37 ms/tok).
- MoE batched prefill: batched router/top-k, one sync per layer, grouped
  GEMM (~6 dispatches/layer vs T×used×4) — bit-exact vs per-token.
- Server/MCP verified end-to-end with a MoE model ("2+2 is 4.").
- Validation additions: synthetic MoE GGUF + Python reference (bit-exact),
  CPU-vs-GPU and batch-vs-per-token compare modes.

## v0.1.0 — 2026-11-08

- Initial public release: odin-serve (HTTP + MCP + dashboard) and odin-cli,
  zero-compile arm64 binaries via GitHub Actions.
- Qwen3.5 dense engine: hybrid attention (full + gated delta), 248k vocab,
  first-run kernel autotuning, memory budget reporting.
- Bilingual (EN + ZH) documentation: README, AGENTS, POSITIONING, docs/.
