# AGENTS.md — AI agent quickstart (build / test / use)

This file is for AI coding agents (Cursor / Claude Code etc.). Read it once and
you can build, test, run and modify this project independently.

[中文版 / Chinese version](AGENTS.zh.md)

## Architecture in one line

A Qwen3.5-class inference engine for Apple Silicon in pure Odin (no Python
runtime dependency). **Product positioning: Qwen3.5 and above** — the server
(`server/serve_harness/serve.odin`, the primary product form) runs the
`packages/qwen3_5` engine (hybrid attention, gated delta) only, by design.
`packages/infer` (Qwen3 dense) exists as a legacy compatibility path for the
CLI only; do NOT extend it in the product docs. Shared: `packages/ggml`
(GGUF parsing / quant dequant). CLI is `cli/main.odin`.

## Releases

- Zero-compile binaries are built by GitHub Actions:
  `.github/workflows/release.yml` (macos-15 arm64 runner; Odin pinned to
  dev-2026-07 via tarball; `./build.sh`; pushing a `v*` tag runs
  `gh release create` publishing bin/odin-serve + bin/odin-cli).
- After changing the release flow: push a tag and watch with `gh run watch`.

## Install & run (zero compile — agents read this first)

No toolchain needed: the installer downloads the release binaries (and
optionally a model) and prints ready-to-run commands.

```sh
bash scripts/install.sh --model moe   # binaries + 524 MB MoE test model
export PATH="$PATH:$HOME/.odinfer/bin"
odin-serve "$HOME/.odinfer/models/Qwen3.5-0.8B-MoE-A0.6B-Q4_K_M.gguf" --mcp --metal
```

- `--model qwen9b` downloads the 5.6 GB heretic model instead.
- `--prefix` / `MODEL_DIR` override install paths.
- Manual equivalent (no script): curl the two binaries from
  `https://github.com/vajraimb/odinfer/releases/latest/download/{odin-serve,odin-cli}`.

The source build below is for development only.

## Build & test (run in this order)

```sh
./build.sh        # everything: bin/odin-serve + bin/odin-cli + all unit tests (~1-2 min)
```

Unit tests individually:

```sh
( cd packages/qwen3_5 && ./build.sh )        # 11 unit tests (incl. 3 MoE math tests)
( cd packages/tokenizer && odin test . -define:ODIN_TEST_THREADS=1 )
# ⚠️ tokenizer tests MUST be single-threaded: 5 parallel runners each build the
#    248k-entry BPE map; under memory pressure (<100MB free) they OOM/segfault.
#    This is a known environmental issue, not a code bug.
```

Integration tests (load a real model, ~30-60s each):

```sh
python3 server/serve_harness/test_mcp.py    # full MCP handshake (needs a Qwen3.5 GGUF)
python3 server/serve_harness/test_tune.py   # first-run autotune + cache reuse
# Common env var: MODEL=<path> overrides the default model
```

> Integration tests cache their binary at
> `server/serve_harness/.test-serve.bin`; **they rebuild only when serve.odin
> or packages/*.odin is newer** (max-mtime check).

Linux CPU builds use the same `./build.sh` (Metal sources are `#+build darwin`).
The CUDA backend is a separate define, not the default:

```sh
# Linux, driver + NVRTC present, no nvcc needed at compile time
./scripts/cuda_check.sh
# or: -define:ODINFER_BACKEND="cuda"  -collection:cuda=packages
```

See [docs/CUDA.md](docs/CUDA.md).

Windows Vulkan is the same kind of switch (`-define:ODINFER_BACKEND=vulkan`,
`-collection:vulkan=packages`). Odin dev-2026-07 is enough to build: SPIR-V is
committed and embedded, and `vulkan-1.dll` is loaded at run time. The SDK is
only for `./scripts/compile_spirv.sh`. Checks: `./scripts/vulkan_check.sh`.
See [docs/VULKAN.md](docs/VULKAN.md). Hot path is `when GPU_METAL` /
`when GPU_CUDA` / `when GPU_VULKAN`; a binary never contains more than one.

## Run

```sh
# HTTP server (dev mode)
MODEL=./models/<qwen3.5-gguf>.gguf ./serve.sh 9748 8192
# Endpoints: POST /v1/chat/completions, /api/chat, /v1/responses (SSE), GET /api/state

# MCP mode (stdout is the protocol channel — never log to stdout)
bin/odin-serve <model.gguf> --mcp --metal --max-ctx 4096

# CLI (auto-detects Qwen3 / Qwen3.5)
bin/odin-cli <model.gguf> -g 1 -c 2048
```

## Model files

- `models/` is a gitignored symlink directory (see models/README.md).
- Default benchmark model: `models/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf`
  (→ symlink to an ollama blob, Qwen3.5, 5.6 GB).
- MoE test model: `models/Qwen3.5-0.8B-MoE-A0.6B-Q4_K_M.gguf` (524 MB,
  bayliner1980) — used by tools/moe_smoke.odin.
- Quick smoke test with 0.6B: any `Qwen3-0.6B-Q4_K_M.gguf` (Qwen3, legacy
  infer path, loads in ~1s) — CLI only; **serve accepts Qwen3.5 only, by
  design (positioning is 3.5+)**.

## Key conventions (read before editing)

1. **Diagnostics always on stderr**: `fmt.eprint*`. In `--mcp` mode stdout may
   only carry protocol messages; library startup prints already moved to
   stderr — do not revert.
2. **fmt.tprintf braces are verbs**: JSON templates must use `{{`/`}}`.
3. **json_escape emits its own outer quotes** (outputs `"escaped"`); callers
   must not add quotes around it.
4. **Collection imports**: `import q35 "qwen3_5:qwen3_5"`; builds must pass
   `-collection:<name>=packages` for ggml, infer, qwen3_5, qwen3_5_tokenizer,
   sampler, tokenizer, cuda, and vulkan. The CUDA and Vulkan packages are
   imported even when that backend is not selected. See COLLECTIONS in build.sh.
5. **Kernel A/B switches** (env, all off by default, all measured regressions):
   `QGEMV3=1` (llama.cpp row-split port), `QGEMV2R=1` (2 rows/lane),
   `QGEMVNORM=1` (norm fusion). Ablation switches: write `/tmp/qprof_flags`
   (noout/nomlp/nosmall/nonorm/noattn/nodelta), read per token, no restart.
6. **Tune cache**: `~/.odin-infer/<model>.tune` (`ODIN_TUNE_DIR` overrides,
   `ODIN_TUNE=0` disables); delete the cache after changing kernel
   performance characteristics to force a retune.
7. **KV persistence**: `KV_PERSIST=0` disables, `KV_PERSIST_DIR` changes the
   dir; a seq_len mismatch across max-ctx is expected in logs.
8. **Never commit binaries**: build products go to `bin/` (ignored); stray
   compiled artifacts `tokenizer`/`serve` were committed by mistake once and
   are now gitignored via `/tokenizer`, `/serve`.

## MoE (qwen35moe)

- CPU + Metal paths supported. Metal MoE: GPU router (moe_gemv_f32 +
  moe_softmax_topk, one sync per layer), expert GEMVs reuse the production
  kernels via tensor-base + expert offsets, moe_combine does the weighted sum;
  shared expert runs via moe_gemv_f32 on F32 uploads. MOE_CPU=1 forces CPU.
  MoE batched prefill (forward_gpu_batch_moe, grouped GEMM) is the default
  Metal path since v0.3.0 (max_abs 8e-4 vs per-token); MOE_BATCH=0 switches
  back to per-token prefill for A/B.
- Expert tensor layout (verified against the real unsloth 35B-A3B GGUF header):
  `ffn_gate_inp` [dim, n_expert], `ffn_up_exps`/`ffn_gate_exps`
  [dim, n_ff_exp, n_expert], `ffn_down_exps` [n_ff_exp, dim, n_expert] —
  ne0 is the contraction dim; quantization granularity = one ne0 row.
- Routing math mirrors llama.cpp build_moe_ffn (softmax → top-k → renormalize
  with divisor clamp 6.1035e-5 → expert_weights_scale → silu(gate)×up per
  expert → shared expert with sigmoid gate).
- Validation: `tools/make_moe_gguf.py` writes a tiny all-F32 MoE GGUF +
  Python reference; `tools/moe_load_test.odin` compares logits (currently
  bit-exact). Run via `./build.sh`.
- Real-model smoke: `tools/moe_smoke.odin <model.gguf> [steps]` (CPU).
  Validated on `bayliner1980/Qwen3.5-0.8B-MoE-A0.6B-GGUF` Q4_K_M (524MB,
  24 layers, 28 experts, top-14, 248k vocab) — loads, prefills, decodes
  coherently (~1.8 s/tok CPU). Note: our loader accepts ANY layer count
  (llama.cpp's qwen35moe type switch only allows 40/48/60).

## Verified traps (do not re-learn)

- GEMV kernel shape: on this M3 the optimum is 256 threads / 32 rows / 8 lanes;
  every multi-row/multi-thread/fusion variant measured as a regression (data in
  packages/qwen3_5/README.md and gemm_harness).
- Kernel timing must be **wall-clock + correctness-checked**: GPU timers
  (GPUEndTime) lie for short dispatches (once reported a 22.5 ms output layer
  that was actually 6.4 ms).
- Odin syntax gotchas: array literals cannot appear directly in a for loop
  (hoist to a named variable); taking the address of a proc parameter needs a
  local copy first; `os.to_stream(f)` for writing file streams.
- MoE SwiGLU: silu applies to the GATE projection (silu(gate)×up), not up —
  symmetric test data hides this bug.
- GGUF tensor dims are ggml ne order (ne0 fastest) = REVERSED of numpy shape;
  hand-written GGUF writers must reverse dims and pad the data section to
  32 bytes.
- MTL setBuffer signature is `(buffer, offset, index)` — passing the index as
  the second arg silently rebinds index 0 (debugged: router weights landed in
  the logits buffer). / MTL setBuffer 签名是 (buffer, offset, index)，第二个参数当 index 写会静默重绑 index 0。
- Expert GEMV offsets are relative to EACH tensor; m_weights is the whole-model
  mmap — add woff(tensor.data) as the base. / 专家偏移是张量内相对偏移，需加 woff(张量) 基址。
- Q5_0 high bits: element j<16 uses qh bit j; element j+16 uses qh bit j+16
  (CPU dequant masks 0x10 after >>(j+12)). / Q5_0 高比特：元素 j+16 用的是 qh 的第 j+16 位。
- Community MoE GGUFs mix per-layer quant kinds (0.8B-MoE: down_exps alternates
  Q8_0/Q5_0) — dispatch by tensor kind, never assume. / 社区 MoE GGUF 逐层混合量化种类，按张量 kind 分派。

- A failed Metal init used to silently fall back to CPU (v0.2.0 shipped a
  shader library that never compiled — every model ran on CPU). Now
  `--metal`/`-g 1` + failed init is a hard error (METAL_FALLBACK=1 restores
  the old behaviour), and build.sh runs tools/metal_smoke.odin on macOS to
  compile every kernel. / Metal 初始化失败现为硬错误，build.sh 在 macOS 上跑 shader 冒烟。

## Post-change self-check

```sh
./build.sh                                          # build + all unit tests green
python3 server/serve_harness/test_mcp.py            # MCP handshake passes
MODEL=./models/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf ./serve.sh &
curl -s http://127.0.0.1:9748/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"odinfer","messages":[{"role":"user","content":"What is 2+2? Answer in one word."}],"max_tokens":8}'
# Expect: content is "4"; stderr has a mem: line and (on first run) tune: lines.
