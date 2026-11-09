# Changelog · 更新日志

## v0.4.0 — 2026-11-09

- Batched MMQ for IQ/k-quant types (11 new kernels): Qwen3.8-27B Dynamic
  mixed-quant prompts prefill batched (~1.8x GPU); batch vs per-token
  max_abs 5.7e-3 prefill / 6.6e-3 decode, argmax agrees. Fixed a latent
  m_b_proj_half overflow on models where att_head_dim > dim.
- OpenAI-path prefix cache / conversation continuity (also covers MCP
  odin.chat): continuation, system-boundary snapshot, full-reset branches.
- docs/BENCHMARKS(.zh).md: consolidated measured data (9B vs ollama,
  Qwen3.8-27B, MoE, quant bit-exactness, loaded-machine degradation).

## v0.3.0 — 2026-11-09


- Full quant dequant coverage: Q2_K/Q3_K/Q5_K/Q8_K + IQ family (IQ1_S/IQ1_M/
  IQ2_XXS/XS/S/IQ3_XXS/S/IQ4_NL/XS), bit-exact vs llama.cpp. Unsloth Dynamic
  GGUFs load — Qwen3.8-27B-UD-IQ2_XXS validated end-to-end.
- Metal IQ/k-quant GEMV kernels: Qwen3.8-27B decodes on GPU (223 ms/tok,
  CPU-vs-Metal max_abs ~1.4e-3).
- FIXED: shader library failed to compile since v0.2.0 ('half' reserved word +
  uint3 field misuse) — every model silently fell back to CPU.
- FIXED: MoE grouped-GEMM batch prefill — three stacked bugs (zero-grid MMQ
  for M=16 ssm tensors, top-k threadgroup race across tokens, MMQ ordering
  amplification); batch prefill is now the default Metal path (max_abs 8e-4
  vs per-token, ~40% faster).
- Qwen3.8-27B validated at ctx 8192 (kv 1.07 GB, coherent long-prompt answer).

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
