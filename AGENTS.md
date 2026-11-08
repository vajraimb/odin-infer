# AGENTS.md — AI agent quickstart (build / test / use) · AI agent 速查（构建 / 测试 / 使用）

This file is for AI coding agents (Cursor / Claude Code etc.). Read it once and
you can build, test, run and modify this project independently.
这是给 AI coding agent（Cursor / Claude Code 等）的项目速查。读完即可独立构建、测试、运行和改动本项目。

## Architecture in one line / 一句话架构

A Qwen3.5-class inference engine for Apple Silicon in pure Odin (no Python
runtime dependency). **Product positioning: Qwen3.5 and above** — the server
(`server/serve_harness/serve.odin`, the primary product form) runs the
`packages/qwen3_5` engine (hybrid attention, gated delta) only, by design.
`packages/infer` (Qwen3 dense) exists as a legacy compatibility path for the
CLI only; do NOT extend it in the product docs. Shared: `packages/ggml`
(GGUF parsing / quant dequant). CLI is `cli/main.odin`.
Apple Silicon 上的 Qwen3.5 级推理引擎（纯 Odin）。**产品定位：Qwen3.5 及以上**——
服务端（主形态）只跑 `packages/qwen3_5` 引擎（混合注意力 + gated delta），这是有意为之。
`packages/infer`（Qwen3 密集）仅作 CLI 的历史兼容路径，不要在产品文档中扩展。
共享 `packages/ggml`。服务端在 `server/serve_harness/serve.odin`，CLI 在 `cli/main.odin`。

## Releases / 发布

- Zero-compile binaries are built by GitHub Actions:
  `.github/workflows/release.yml` (macos-15 arm64 runner; Odin pinned to
  dev-2026-07 via tarball; `./build.sh`; pushing a `v*` tag runs
  `gh release create` publishing bin/odin-serve + bin/odin-cli).
  零编译二进制由 GitHub Actions 构建：macos-15 arm64 runner，Odin 锁定 dev-2026-07，推 `v*` tag 自动发布 bin/odin-serve + bin/odin-cli。
- After changing the release flow: push a tag and watch with `gh run watch`.
  改发布流程后：push tag 触发；`gh run watch` 观察。

## Build & test (run in this order) / 构建与测试（必跑顺序）

```sh
./build.sh        # everything: bin/odin-serve + bin/odin-cli + all unit tests (~1-2 min)
                  # 全量：bin/odin-serve + bin/odin-cli + 全部单测（约 1-2 分钟）
```

Unit tests individually / 单测单独跑：

```sh
( cd packages/qwen3_5 && ./build.sh )        # 8 pure-math unit tests / 8 个纯数学单测
( cd packages/tokenizer && odin test . -define:ODIN_TEST_THREADS=1 )
# ⚠️ tokenizer tests MUST be single-threaded: 5 parallel runners each build the
#    248k-entry BPE map; under memory pressure (<100MB free) they OOM/segfault.
#    This is a known environmental issue, not a code bug.
# ⚠️ tokenizer 测试必须单线程：5 个并行 runner 各建 248k 词表映射，内存紧张时会 OOM 段错误——环境性已知问题
```

Integration tests (load a real model, ~30-60s each) / 集成测试（会加载真实模型）：

```sh
python3 server/serve_harness/test_mcp.py    # full MCP handshake (needs a Qwen3.5 GGUF) / MCP 全握手
python3 server/serve_harness/test_tune.py   # first-run autotune + cache reuse / 首启调优 + 缓存复用
# Common env var: MODEL=<path> overrides the default model / 通用环境变量：MODEL=<路径> 覆盖默认模型
```

> Integration tests cache their binary at
> `server/serve_harness/.test-serve.bin`; **they rebuild only when serve.odin
> or packages/*.odin is newer** (max-mtime check). 测试二进制缓存于
> `server/serve_harness/.test-serve.bin`；**只在 serve.odin 或 packages/*.odin 更新时自动重建**。

## Run / 运行

```sh
# HTTP server (dev mode) / HTTP 服务（开发模式）
MODEL=./models/<qwen3.5-gguf>.gguf ./serve.sh 9748 8192
# Endpoints: POST /v1/chat/completions, /api/chat, /v1/responses (SSE), GET /api/state
# 端点：POST /v1/chat/completions、/api/chat、/v1/responses(SSE)、GET /api/state

# MCP mode (stdout is the protocol channel — never log to stdout) / MCP 模式（stdout 是协议通道，禁止往 stdout 打日志）
bin/odin-serve <model.gguf> --mcp --metal --max-ctx 4096

# CLI (auto-detects Qwen3 / Qwen3.5) / CLI（自动识别 Qwen3 / Qwen3.5）
bin/odin-cli <model.gguf> -g 1 -c 2048
```

## Model files / 模型文件

- `models/` is a gitignored symlink directory (see models/README.md) / gitignored 软链目录
- Default benchmark model: `models/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf`
  (→ symlink to an ollama blob, Qwen3.5, 5.6 GB) / 默认基准模型（ollama blob 软链，Qwen3.5，5.6GB）
- Quick smoke test with 0.6B: `/Users/vajra/Claw/odin-infeer/Qwen3-0.6B-Q4_K_M.gguf`
  (Qwen3, legacy infer path, loads in ~1s) — CLI only; **serve accepts Qwen3.5
  only, by design (positioning is 3.5+)** / 快速冒烟用 0.6B（Qwen3，历史兼容路径，加载 ~1s）——仅 CLI；serve 只认 Qwen3.5，产品定位即 3.5+

## Key conventions (read before editing) / 关键约定（改代码前必读）

1. **Diagnostics always on stderr**: `fmt.eprint*`. In `--mcp` mode stdout may
   only carry protocol messages; library startup prints already moved to
   stderr — do not revert. / 诊断一律 stderr：`fmt.eprint*`。`--mcp` 模式下 stdout 只允许协议消息；库层打印已迁到 stderr，不要回退。
2. **fmt.tprintf braces are verbs**: JSON templates must use `{{`/`}}`. / `fmt.tprintf` 的 `{`/`}` 是动词：JSON 模板必须写 `{{`/`}}`。
3. **json_escape emits its own outer quotes** (outputs `"escaped"`); callers
   must not add quotes around it. / `json_escape` 自带外层引号，调用方不要再加引号。
4. **Collection imports**: `import q35 "qwen3_5:qwen3_5"`; builds must pass
   `-collection:<name>=packages` for all 6 collections (ggml/infer/qwen3_5/
   qwen3_5_tokenizer/sampler/tokenizer), see COLLECTIONS in build.sh. / 集合导入：构建必须挂 6 个 `-collection:<name>=packages`，见 build.sh 的 COLLECTIONS。
5. **Kernel A/B switches** (env, all off by default, all measured regressions):
   `QGEMV3=1` (llama.cpp row-split port), `QGEMV2R=1` (2 rows/lane),
   `QGEMVNORM=1` (norm fusion). Ablation switches: write `/tmp/qprof_flags`
   (noout/nomlp/nosmall/nonorm/noattn/nodelta), read per token, no restart. /
   内核 A/B 开关（env，默认全关，均为实测回归项）。消融开关：写 `/tmp/qprof_flags`，每 token 读一次，免重启。
6. **Tune cache**: `~/.odin-infer/<model>.tune` (`ODIN_TUNE_DIR` overrides,
   `ODIN_TUNE=0` disables); delete the cache after changing kernel
   performance characteristics to force a retune. / 调优缓存：改了内核性能特征后删除缓存强制重测。
7. **KV persistence**: `KV_PERSIST=0` disables, `KV_PERSIST_DIR` changes the
   dir; a seq_len mismatch across max-ctx is expected in logs. / KV 持久化：跨 max-ctx 不兼容（seq_len mismatch 属正常日志）。
8. **Never commit binaries**: build products go to `bin/` (ignored); stray
   compiled artifacts `tokenizer`/`serve` were committed by mistake once and
   are now gitignored via `/tokenizer`, `/serve`. / 不要提交二进制：构建产物在 `bin/`（已 ignore）。

## MoE (qwen35moe) / MoE 支持

- Supported on the CPU path (Metal MoE kernels are a TODO; engine_load falls
  back to CPU with a warning). 已支持 CPU 路径（Metal MoE 内核为 TODO，加载时自动回退 CPU 并告警）。
- Expert tensor layout (verified against the real unsloth 35B-A3B GGUF header):
  `ffn_gate_inp` [dim, n_expert], `ffn_up_exps`/`ffn_gate_exps`
  [dim, n_ff_exp, n_expert], `ffn_down_exps` [n_ff_exp, dim, n_expert] —
  ne0 is the contraction dim; quantization granularity = one ne0 row.
  专家张量布局（已对真实 35B-A3B 头部验证）：ne0 是收缩维，量化粒度 = 一行 ne0。
- Routing math mirrors llama.cpp build_moe_ffn (softmax → top-k → renormalize
  with divisor clamp 6.1035e-5 → expert_weights_scale → silu(gate)×up per
  expert → shared expert with sigmoid gate). 路由数学与 llama.cpp 一致。
- Validation: `tools/make_moe_gguf.py` writes a tiny all-F32 MoE GGUF +
  Python reference; `tools/moe_load_test.odin` compares logits (currently
  bit-exact). Run via `./build.sh`. 验证：合成 GGUF + Python 参考对拍（当前逐位一致），./build.sh 自动执行。
- Real-model smoke: `tools/moe_smoke.odin <model.gguf> [steps]` (CPU). Validated
  on `bayliner1980/Qwen3.5-0.8B-MoE-A0.6B-GGUF` Q4_K_M (524MB, 24 layers, 28
  experts, top-14, 248k vocab) — loads, prefills, decodes coherently
  (~1.8 s/tok CPU). Note: our loader accepts ANY layer count (llama.cpp's
  qwen35moe type switch only allows 40/48/60). 真实模型冒烟：bayliner1980 的
  0.8B MoE（24 层/28 专家/top-14）加载+解码连贯；加载器不限层数（llama.cpp 只认 40/48/60 层）。

## Verified traps (do not re-learn) / 已验证的坑（不要再踩）

- GEMV kernel shape: on this M3 the optimum is 256 threads / 32 rows / 8 lanes;
  every multi-row/multi-thread/fusion variant measured as a regression (data in
  packages/qwen3_5/README.md and gemm_harness). / 本机（M3）最优 = 256 线程/32 行/8 lanes；其余变体实测回归。
- Kernel timing must be **wall-clock + correctness-checked**: GPU timers
  (GPUEndTime) lie for short dispatches (once reported a 22.5 ms output layer
  that was actually 6.4 ms). / 内核计时必须墙钟 + 正确性验证：GPU 计时器对短 dispatch 会给出假数据。
- Odin syntax gotchas: array literals cannot appear directly in a for loop
  (hoist to a named variable); taking the address of a proc parameter needs a
  local copy first; `os.to_stream(f)` for writing file streams. / Odin 语法小坑：for 循环里数组字面量要先命名变量；proc 参数取地址先拷局部；`os.to_stream(f)` 写文件流。
- MoE SwiGLU: silu applies to the GATE projection (silu(gate)×up), not up —
  symmetric test data hides this bug. / MoE SwiGLU：silu 作用于 gate 投影；对称测试数据会掩盖此错。
- GGUF tensor dims are ggml ne order (ne0 fastest) = REVERSED of numpy shape;
  hand-written GGUF writers must reverse dims and pad the data section to
  32 bytes. / GGUF 维度是 ggml ne 序（ne0 最快）= numpy shape 的反转；手写 GGUF 写入器要反转 dims 且数据段 32 字节对齐。

## Post-change self-check / 验证清单（改动后自检）

```sh
./build.sh                                          # build + all unit tests green / 构建 + 单测全绿
python3 server/serve_harness/test_mcp.py            # MCP handshake passes / MCP 握手全过
MODEL=./models/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf ./serve.sh &
curl -s http://127.0.0.1:9748/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"odin-infer","messages":[{"role":"user","content":"What is 2+2? Answer in one word."}],"max_tokens":8}'
# Expect: content is "4"; stderr has a mem: line and (on first run) tune: lines.
# 期望：content 为 "4"，stderr 有 mem: 行与（首次）tune: 行
```
