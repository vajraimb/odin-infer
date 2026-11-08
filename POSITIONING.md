# POSITIONING.md — Positioning & product roadmap

> Decision record, 2026-11-08. [中文版 / Chinese version](POSITIONING.zh.md)
>
> We deliberately do NOT chase llama.cpp-style generalization across all
> architectures, and we do NOT chase downwards compatibility either. The scope
> is: **a high-efficiency local inference engine for Qwen3.5-class models
> (3.5 and above, the hybrid-attention family) on Apple Silicon (MacBook Air
> first).**

## 1. One-line positioning

**An out-of-the-box local Qwen3.5-class engine for agents and tools on
personal Macs.** The selling points are power/memory/interactive latency and
agent-readiness — not peak throughput.

## 2. Target hardware & users

- **MacBook Air (M1–M4), 16 GB and up**: fanless (sustained load throttles),
  zero-config expectations. Qwen3.5 starts at 9B (~5.6 GB weights + KV), so
  the product floor is 16 GB; 8 GB machines are out of scope by design (see
  docs/MEMORY.md).
- Users: people running agent ecosystems (Cursor / Claude Code / pie / custom
  MCP) on a personal Mac, wanting privacy, offline, low-latency local models.

## 3. Differentiation vs llama.cpp/ollama

Measured on M3, 2026-11:

| Dimension | odin-infer | llama.cpp/ollama |
|---|---|---|
| Decode throughput (9B Q4) | 13.3 tok/s (0.89×) | 14.9 tok/s |
| First token | **73 ms** | ~90 ms |
| Binary | **7.1 MB single file** | large binary + runtime deps |
| Agent readiness | native tool-call grammar + prefix cache + OpenAI/Ollama/MCP | generic, no built-in grammar |
| Power profile | 8-lane low-thread kernels (passive-cooling friendly; pending thermal test) | peak-throughput oriented |
| Qwen3.5-MoE small variants | any layer count via metadata (llama.cpp rejects non-40/48/60) | 40/48/60 only |

## 4. Product forms (one repo, three artifacts)

1. **odin-serve (primary)**: HTTP server — OpenAI-compatible +
   Ollama-compatible + dashboard. **MCP adapter (stdio) shipped; SSE/streamable
   HTTP is a follow-up.**
2. **odin-cli (secondary)**: terminal chat, dual-engine auto-detection.
3. **library (embed form)**: `-collection` for other Odin projects.

**No native GUI app**: the dashboard covers casual use; a native shell is
months of work and does not serve the Air positioning.

## 5. Benchmark protocol (mandatory)

- Same model & quantization (default heretic Q4_K_M), same prompt, **solo
  residency**.
- Cooled machine, 3 runs each, average; report tps / wall / ttft / prefill.
- Two 9B engines co-resident on 16 GB thrash each other ~2× — such data is
  invalid for comparison.

## 6. Air-specific TODOs

1. **Thermal test protocol**: 10-minute sustained generation, tps sampled
   every 30s; compare the droop curve against llama.cpp on the same machine.
   The "flatter curve" claim is NOT to be advertised until verified.
2. ~~First-run shape autotuning~~ ✅ (runtime measured pick + cache reuse,
   verified by test_tune.py)
3. ~~Thermal protocol tooling~~ ✅ (tools/thermal_test.py + docs/THERMAL.md;
   empirical run pending)
4. ~~Memory budget mode~~ ✅ (startup `mem:` estimate + 85% warning +
   docs/MEMORY.md)

## 7. Repo consolidation (done 2026-11-08)

- odin-infer is now the single repo: `packages/` + `server/` + `cli/` +
  `tools/` + `docs/` + `models/`.
- **odin-infer-mac (GitHub) is retired**: its CLI merged into `cli/`; the old
  repo is archived with a pointer README.
- The local `/Users/vajra/Claw/odin-infeer` directory keeps model files only;
  code there is frozen; `models/` symlinks point at its GGUFs.

## 8. Roadmap (in order)

1. ~~repo consolidation + positioning docs~~ ✅
2. ~~MCP adapter (stdio)~~ ✅ — `odin.chat` / `odin.models` tools + Claude
   Code / Cursor config samples + test_mcp.py full handshake; SSE/streamable
   HTTP later.
3. ~~first-run autotuning + thermal protocol~~ ✅
4. ~~memory budget mode~~ ✅
5. ~~zero-compile release pipeline~~ ✅ — GitHub Actions builds and publishes
   `odin-serve` / `odin-cli` on every `v*` tag.
6. ~~Qwen3.5-MoE CPU support~~ ✅ — 35B-A3B-family loader + softmax top-k
   routing + gated shared expert; bit-exact vs a Python reference; validated
   on a real 0.8B-MoE GGUF. Metal MoE kernels + a 24 GB-machine real-model run
   remain.
7. ~~Metal MoE kernels~~ ✅ — top-k experts dispatched to the GEMV kernels
   (Q4_K/Q5_0/Q8_0) via tensor-base + expert offsets; GPU router (gemv_f32 +
   softmax-topk) with one sync per layer; validated against the CPU path on
   the 0.8B-MoE (max_abs ~3e-3, 37 ms/tok vs 1.7 s/tok CPU). The 35B-A3B
   real-model run still needs a 24 GB machine; indirect-dispatch (no per-layer
   sync) is the next optimization.
8. (optional) perf round 2: needs full Xcode (`xcrun metal` + Instruments GPU
   counters) to chase the last ~11% decode gap.
