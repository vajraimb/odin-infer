# Odinfer

**A high-efficiency local inference engine for Qwen3.5-class models on Apple Silicon (MacBook Air first).**

A single 7 MB binary that runs Qwen3.5-class models on your Mac — no Python,
no llama.cpp dependency, 73 ms first token, native MCP, agent-ready.

[中文版 / Chinese version](README.zh.md) ·
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![language: Odin](https://img.shields.io/badge/language-Odin-9cf.svg)](https://odin-lang.org)

---

## Why Odinfer

| Your need | Odinfer's answer |
|---|---|
| Run local models on an Air without heat issues | Low-thread kernels tuned for passive cooling; memory budget shown at startup ([thermal protocol](docs/THERMAL.md)) |
| Serve models to Cursor / Claude Code / pie | `--mcp` turns it into an MCP server; `odin.chat` / `odin.models` out of the box ([docs/MCP.md](docs/MCP.md)) |
| No Python / toolchain install | Single-file binary from GitHub Releases, zero compile |
| OpenAI / Ollama compatibility | `/v1/chat/completions` + `/api/chat` + Web dashboard |
| Fast first token (interactive feel) | 73 ms measured on M3, faster than ollama's ~90 ms |

## Quick start (zero compile)

Apple Silicon (M1–M4):

```sh
curl -L -o odin-serve https://github.com/vajraimb/odinfer/releases/latest/download/odin-serve
curl -L -o odin-cli   https://github.com/vajraimb/odinfer/releases/latest/download/odin-cli
chmod +x odin-serve odin-cli
xattr -cr odin-serve odin-cli   # only if macOS blocks the unsigned binaries
./odin-serve ./models/<your-model>.gguf --metal --port 9748   # http://127.0.0.1:9748
```

MCP (for agents):

```sh
claude mcp add odin -- /path/to/odin-serve /path/to/model.gguf --mcp --metal
```

Build from source (needs the [Odin](https://odin-lang.org) toolchain):

```sh
git clone https://github.com/vajraimb/odinfer && cd odinfer
./build.sh                                  # build + unit tests, produces bin/odin-serve bin/odin-cli
MODEL=./models/<your-model>.gguf ./serve.sh # http://127.0.0.1:9748
```

> Model GGUFs are not in the repo (gitignored): download from Hugging Face into
> `models/` or any path (see models/README.md for the recommended layout).

## Measured performance

M3 / 16 GB, DeepSeek-V4-Pro-Qwen3.5-9B UNCENSORED heretic (Q4_K_M), solo
residency, cooled machine:

| Metric | Odinfer | ollama / llama.cpp |
|---|---|---|
| Decode throughput (EN) | 13.3 tok/s（0.89×） | 14.9 tok/s |
| Wall, 128-token reply | 10.3–10.5 s | 7.1–9.1 s |
| First token | **73 ms** | ~90 ms |
| Binary size | **7.1 MB** | large binary + runtime deps |

> Benchmark protocol in [POSITIONING.md](POSITIONING.md) §5; the full GEMV
> kernel comparison experiments (llama.cpp family sweep, shape tuning,
> ablations) live in [packages/qwen3_5/README.md](packages/qwen3_5/README.md).

## Supported models

**Positioned at Qwen3.5 and above:**

- **Qwen3.5 hybrid attention** (Ornith / heretic family): text-only, 248k vocab.
  The 9B is validated end-to-end; other sizes of the same architecture load
  automatically from GGUF metadata (MTP-aware loader) — including
  **Qwen3.8-27B** (GPU decode verified, ctx 8192).
- **Any llama.cpp quant**: Q2_K..Q8_K + the IQ family (IQ1_S..IQ4_XS) are
  implemented bit-exact vs llama.cpp — Unsloth "Dynamic" mixed-quant GGUFs
  load and decode on GPU.
- **Qwen3.5-MoE** (35B-A3B / 122B-A10B / 397B-A17B): CPU + Metal paths
  supported — softmax top-k routing, per-expert GEMV dispatch (Q4_K/Q5_0/Q8_0/
  Q6_K), sigmoid-gated shared expert; Metal output matches CPU to ~3e-3
  (fastMath) at ~45× CPU speed on the 0.8B-MoE (37 ms/tok). Small community
  qwen35moe variants work today, including layer counts that llama.cpp itself
  rejects.
- No fallback to pre-Qwen3.5 architectures, no multimodal, no other model
  families — a narrow scope, on purpose.

## Product forms

| Form | Artifact | Use case |
|---|---|---|
| **odin-serve** (primary) | `bin/odin-serve` | HTTP (OpenAI/Ollama-compatible + dashboard) or `--mcp` |
| **odin-cli** (secondary) | `bin/odin-cli` | Terminal chat |
| **library** | `packages/*` | Embed in any Odin project via `-collection` |

## Platform adaptation

- **First-run kernel autotuning**: on first Metal start, the engine times every
  GEMV kernel variant on the real weight shapes, picks the best per shape and
  caches it (`~/.odin-infer/<model>.tune`) — adapts to M1/M2/M3/M4
  automatically.
- **Memory budget**: startup prints `mem: ... (36%) OK`, warns above 85%;
  16 GB guidance in [docs/MEMORY.md](docs/MEMORY.md).
- **Linux CUDA** (sm_70+, including Tesla V100): build with
  `-define:ODINFER_BACKEND="cuda"`. The driver and NVRTC are loaded at
  runtime. See [docs/CUDA.md](docs/CUDA.md). macOS builds stay on Metal.
- **Windows Vulkan** (shared-memory Radeon): build with
  `-define:ODINFER_BACKEND=vulkan`. Odin dev-2026-07 and the driver's
  `vulkan-1.dll` are enough; SPIR-V is embedded. See [docs/VULKAN.md](docs/VULKAN.md).

## Repo layout

```
packages/   ggml · infer(Qwen3, legacy) · qwen3_5 · qwen3_5_tokenizer · sampler · tokenizer · cuda · vulkan
server/     odin-serve (HTTP + MCP + dashboard + integration tests)
cli/        odin-cli
tools/      gemm/delta/doctor/oracle harness + thermal_test.py + MoE tests
docs/       MCP · THERMAL · MEMORY · troubleshooting notes
```

## Docs

- [POSITIONING.md](POSITIONING.md) — positioning, benchmark protocol, roadmap
- [docs/MCP.md](docs/MCP.md) — MCP tool schema and client setup
- [docs/MEMORY.md](docs/MEMORY.md) — memory budget guidance
- [docs/THERMAL.md](docs/THERMAL.md) — sustained-load comparison protocol
- [docs/CUDA.md](docs/CUDA.md) — Linux CUDA backend (sm_70+, V100)
- [docs/VULKAN.md](docs/VULKAN.md) — Windows Vulkan backend (embedded SPIR-V)
- [AGENTS.md](AGENTS.md) — quickstart for AI coding agents

## Community / 社区

- [CHANGELOG.md](CHANGELOG.md) — release history
- [CONTRIBUTING.md](CONTRIBUTING.md) — ground rules & PR checklist ([中文](CONTRIBUTING.zh.md))
- [SECURITY.md](SECURITY.md) — reporting vulnerabilities
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- Issues and PRs: [GitHub](https://github.com/vajraimb/odinfer)

## License

MIT
