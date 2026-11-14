# Changelog · 更新日志

## Unreleased

- Vulkan backend for Windows x64 (`-define:ODINFER_BACKEND=vulkan`). SPIR-V is
  embedded; run time loads the driver's `vulkan-1.dll` and does not link
  `vulkan-1.lib`. Default CPU/Metal/CUDA binaries are unchanged. Lavapipe:
  exact GEMV max_abs 0 on the native quant kinds, greedy tokens match the CPU
  reference on a synthetic hybrid model and a synthetic MoE model. See
  docs/VULKAN.md.
- Load-time Metal shader specialization: per-model constants baked into the
  MSL prelude (QCONST=0 for legacy A/B). 27B decode −0.8% (best −2.5%);
  9B flat (MLP-bound). QPROF ablation switches now compile-time only
  (#config ODIN_PROF) — release binaries contain zero profiling code.
- MSL codebook tables generated from tables.odin (gen_msl_tables.py;
  CI --check).
- Q8_0 KV cache is now the default (~1.88x smaller than f16, 3.76x vs f32):
  Qwen3.8-27B ctx 8192 KV 0.27 GB, ctx 32768 1.14 GB. `.oikv` bumped to v2
  (on-disk body stays f32; v1 files are refused). KV_Q8=0 restores f16 KV.
  Known cost: on the 27B, CPU-vs-Metal logits max_abs grows from 1.4e-3 (f16)
  to ~1.1e-1 because of V-cache outliers (amax ~85); argmax agrees.
- Metal init failure is now a hard error when `--metal` / `-g 1` is requested
  (previously: silent CPU fallback, which hid the v0.2.0 shader-compile bug for
  a whole release). METAL_FALLBACK=1 restores the old fallback. odin-cli now
  checks engine_load's result.
- build.sh runs `tools/metal_smoke.odin` on macOS (incl. the release CI
  runner): compiles the MSL library with fastMath on/off, builds a pipeline for
  every kernel and checks that every `make_pso` name exists. No model needed.
- Docs: KV_CACHE/ENVIRONMENT/GRAMMAR synced to odin-cli + Q8 KV v2; AGENTS
  MoE batch-prefill notes updated (default since v0.3.0); README scope line
  fixed (EN/ZH mismatch).
- Repo: stop tracking `.tmp-cache/` CI log zips (already gitignored).

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
