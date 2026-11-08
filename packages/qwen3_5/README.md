# qwen3_5 · Qwen3.5 推理包

Independent Qwen3.5 inference package for the odin-infer library. Targets the
**Ornith-1.0-9B** family (Qwen3.5 text model, hybrid full + linear attention),
text-only. Sibling to the Qwen3-only `infer` package; reuses the shared `ggml`
collection for GGUF parsing and quantized dot products.
odin-infer 的 Qwen3.5 独立推理包：面向 Ornith-1.0-9B 家族（Qwen3.5 文本模型，
full + linear 混合注意力），纯文本。与仅支持 Qwen3 的 `infer` 包并列，复用
共享的 `ggml`（GGUF 解析与量化点积）。

> **Porting a new architecture?** Read [`PITFALLS.md`](./PITFALLS.md) first —
> it logs every bug hit while building this package (RMSNorm baking, `ssm_a`
> precompute, V-head tiled reorder, Metal buffer zeroing, etc.) with symptoms,
> root causes, and a checklist.

> **Next milestone: batched prefill.** [`BATCHED_PREFILL.md`](./BATCHED_PREFILL.md)
> tracks the simdgroup-GEMM work that fixes slow prompt/tool-result prefill
> (the agent/tool bottleneck). Stage 1a performance is verified (~50× potential),
> blocked on simdgroup matrix layout correctness.

## Status / 状态

**Validated end-to-end against the real `deepreinforce-ai/Ornith-1.0-9B-GGUF`
(Q4_K_M)** — loads, runs forward, and produces coherent on-topic English
(verified: prompt "What is 2+2?" yields reasoning text like "The user asked a
\"2+\"" / "The user wants" with proper `<think>`/`</think>` structure).
已对真实 Ornith GGUF 做过端到端验收：加载、前向、输出连贯且切题的英文
（"What is 2+2?" 产生带 `<think>`/`</think>` 结构的推理文本）。

Also smoke-tested against `mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-GGUF`
(Q4_K_M): loads and produces coherent EN/ZH replies ("2+2 equals 4.", 中文介绍北京).
That conversion writes `qwen35.block_count = num_hidden_layers + nextn_predict_layers`
(33 = 32 trunk + 1 MTP) while shipping only the 32 trunk blocks, so the loader
subtracts `qwen35.nextn_predict_layers` (llama.cpp's convention) and runs the
trunk only — MTP is ignored on inference.
另对 mradermacher 的 heretic GGUF 做过冒烟验证（中英文输出均连贯）。该转换
把 MTP 层计入 `block_count`（33 = 32 主干 + 1 MTP）但只发布 32 块主干权重，
加载器按 llama.cpp 约定减去 `nextn_predict_layers`，只跑主干。

## Decode-path performance / 解码路径性能 (Apple M3, heretic Q4_K_M, Metal)

Benchmarked solo (one engine resident at a time; with two 9B engines resident
the 16 GB machine swaps and both slow down ~2×):

| step | before | after |
|------|--------|-------|
| decode throughput (EN 128 tok) | 8.8 tok/s | **12.0–13.7 tok/s** (ollama/llama.cpp: 14.9) |
| wall, 128-token reply | 26.4 s | **10.0–11.5 s** (ollama: 8.7 s) |
| prefill 19-23 tok | 1.2–3.7 s | **0.5 s** |
| output-layer GEMV share/token | 25 ms | **~8 ms** |
| per-token GPU time | 100 ms | **~73–77 ms** |

> Numbers are best-of-session on a cool machine. With the M3 run continuously for
> hours, thermal throttling + the 16 GB RAM ceiling cut both engines ~2×; always
> benchmark solo (one engine resident at a time).
>
> Final clean A/B (same prompt, solo residency, cooled machine, 3 runs):
> decode 13.1–13.4 vs ollama 14.4–15.5 tok/s (~0.88×); wall for a 128-token
> reply 10.3–10.5 s vs 7.1–9.1 s.

What moved the needle:
- **GEMV threadgroup shape tuning** (the single biggest lever): the output
  projection (248k rows = 31k TGs at 8 rows/TG) was launch-overhead bound.
  Swept rows-per-TG x lanes-per-row on the M3:
  | config | per-token GPU |
  |---|---|
  | 256 thr / 8 rows / 32 lanes (original) | ~100 ms |
  | 1024 thr / 32 rows / 32 lanes | ~89 ms |
  | 512 thr / 32 rows / 16 lanes | ~80 ms |
  | **256 thr / 32 rows / 8 lanes (final)** | **~73–77 ms** |
  | 128 thr / 32 rows / 4 lanes | ~92 ms |
  | 512 thr / 64 rows / 8 lanes | ~130 ms |
  | 256 thr / 64 rows / 2 rows-per-lane x 8 lanes | ~105 ms |
  | 256 thr / 64 rows / 4 rows-per-lane x 32 lanes | ~165 ms |
  Since rows share simdgroups at <32 lanes, the GEMV reduction switched from
  `simd_sum` (which would mix two rows per simdgroup) to
  `simd_shuffle_xor` butterflies (8-lane groups) with `lane % 8 == 0` writes.
- **Padded-tail batched prefill**: prompt tails of 3–7 tokens were running
  through the per-token GEMV path (~90 ms each); `engine_forward_batch` now
  pads the tail to a multiple of 8 with filler tokens and `forward_gpu_batch`
  takes `n_real`, so stateful per-token loops skip the filler slots. Tails of
  1–2 tokens still fall back to per-token (conv-state update needs n≥3).
- **Dropped the wasted final forward** per request (one unsampled logits pass).

What did NOT help (reverted, kept as regression notes): 4-rows-per-thread
vectorized GEMV (2× slower), uint-packed nibble loads (~8% slower), float4-dot
accumulation (neutral), and a faithful port of llama.cpp's current
`kernel_mul_mv_ext_q4x4` row-split structure (~108 ms — their tuned nxpsg/r1ptg
combos do not transfer to this engine). The port is preserved as validated-but-
disabled `gemv_q4_k_v3`/`gemv_q6_k_v3` (env `QGEMV3=1`), with a standalone
correctness/perf harness at `gemm_harness/gemv3_test.odin` (both kernels match
CPU references to <0.03 max_abs on synthetic weights). The remaining gap to
llama.cpp (~65 vs ~84 GB/s effective weight-read bandwidth at 32 lanes) narrowed
to ~10% at the 8-lane sweet spot; closing it fully likely needs Xcode GPU
profiling, which this machine lacks (`xcrun metal` unavailable).

Decode composition at the 8-lane sweet spot (per-token GPU, flag-file ablation
`/tmp/qprof_flags`, no restart needed):

| part | ms/token | eff. BW |
|---|---|---|
| MLP GEMVs (w1/w3/w2, 3.1 GB) | ~41 | ~76 GB/s |
| attention+linear GEMVs (1.9 GB) | ~21 | ~90 GB/s |
| output GEMV (572 MB, 7760 TGs) | ~9 | ~64 GB/s |
| small kernels (norms/attn/conv/delta, ~330 dispatches) | ~3 | — |

The 8-lane sweet spot already puts the big GEMVs near the M3's practical
bandwidth; every multi-row-per-lane or dispatch-shape alternative measured so
far regresses. A/B switches kept for future profiling: `QGEMV3=1` (llama.cpp
row-split port, ~108 ms) and `QGEMV2R=1` (2 rows/lane, ~105 ms).

### llama.cpp full-family comparison / llama.cpp 全家族对照 (sweep, wall-clock + CPU-reference verified)

Systematic port of llama.cpp's parameterized `kernel_mul_mv_ext` GEMV family
(nxpsg x r1ptg grid: 4-32 lanes/row x 1-4 rows/thread, 26 kernels incl. the
production 8-lane baseline), timed in isolation per matrix shape in
`gemm_harness/gemv_sweep.odin` (correctness rel_err ~2e-6 on every kernel):

| shape (Q4_K) | production | best llama-variant |
|---|---|---|
| 248320 x 4096 (output) | **6.4 ms / 90 GB/s** | 7.4 ms (n4r2) |
| 12288 x 4096 (w1/w3) | **0.70 ms** | 0.67 ms (n8r1, within noise) |
| 8192 x 4096 (wq) | **0.25 ms / 76 GB/s** | 0.32 ms (n4r1) |
| Q6_K shapes (w2, in_qkv) | **at parity with best variant** | — |

Verdict: on this M3 the production 8-lane kernels already match or beat the
llama.cpp family on every shape — the remaining gap to llama.cpp end-to-end is
NOT in GEMV structure. `gemv_q4_k_o` (llama.cpp default 32x4 shape) is kept
compiled but unselected as a live counter-example.

### Small-kernel fusion attempts / 小内核融合尝试 (both regressed, reverted)

- **Norm-fused GEMV** (`QGEMVNORM=1`, kernels `gemv_q4_k_norm`/`gemv_q6_k_norm`):
  each GEMV recomputes rms(x) cooperatively (256-thread threadgroup reduce with
  8 barriers) and folds the norm weight in, eliminating 64 standalone norm
  dispatches + the xb round trip per token. Result: correct, but ~134 ms/token
  (1.9x slower) — the per-GEMV barrier cost dwarfs the saved dispatch.
- **Widened rmsnorm** (32 -> 256 threads with the same barrier reduce): correct,
  ~120 ms/token. On this M3, 32-thread `simd_sum` mini-kernels beat every
  barrier-based cooperative variant; reverted to the 32-thread version.

Conclusion after 10 structural variants: the production configuration (8-lane
GEMV + 32-thread simd_sum norms) is a measured local optimum for every kernel
on this device; the residual ~11% vs llama.cpp needs GPU-profiler guidance.

Also: the server now retries `listen_tcp` with backoff (TIME_WAIT after a kill
used to make restarts flaky ~50% of the time).

| Component | State |
|-----------|-------|
| GGUF config loader | done |
| Hybrid layer dispatch (full every 4th, linear otherwise) | done |
| Full attention (per-head q/k RMSNorm, partial-rotary MRoPE, attn output gate) | done |
| Linear attention (gated delta rule, per-token recurrent path) | done |
| Depthwise causal conv1d | done |
| MLP (SwiGLU) | done |
| CPU forward + quantized GEMV | done (~0.1–0.6 tok/s on a 9B on CPU; needs Metal for usable speed) |
| Metal GPU path | not yet |
| Tokenizer (248k BPE) | see sibling `qwen3_5_tokenizer` package |

Pure-math unit tests cover the novel pieces (delta recurrent, conv1d, partial
MRoPE, rmsnorm, l2norm) and pass: `./build.sh`.

## Architecture notes / 架构说明 (from transformers v5.8.1 `Qwen3_5TextModel`)

- **layer_types**: `[linear, linear, linear, full]` repeated. Full attention at
  indices 3, 7, 11, ..., 31 (8 full + 24 linear layers).
- **MRoPE** (`mrope_section [11,11,10]`, `partial_rotary_factor 0.25`): for
  text-only inputs the three grids (T, H, W) share one position id, so the
  interleaved layout in `apply_interleaved_mrope` collapses to plain 1D RoPE
  over the first `head_dim * 0.25 = 64` elements of each head; the remaining
  192 dims pass through. See `ops.odin: apply_mrope_text_head`.
- **Full attention gate**: `q_proj` outputs `n_heads * head_dim * 2`, reshaped
  `[..., n_heads, head_dim*2]` and chunked into per-head `q` and `gate`. The
  attention output is multiplied by `sigmoid(gate)` elementwise per head before
  `o_proj`. The gate is interleaved per head in the flat buffer
  (`qproj[h*512 : h*512+256]` = q, `[h*512+256 : (h+1)*512]` = gate).
- **Linear attention** = gated delta net (`Qwen3_5GatedDeltaNet`):
  - 16 key heads x 128, 32 value heads x 128 (each key head serves 2 value heads)
  - depthwise causal conv1d (kernel 4) over the fused QKV projection
  - l2-normalized q (scaled by `1/sqrt(128)`) and k
  - forget gate `g = -exp(A_log) * softplus(a + dt_bias)`, recurrent decay
    `exp(g)`, delta update `state += k (v - state.k) beta`, output `state.q`
  - `RMSNormGated`: standard rmsnorm then `* silu(z)`
- **RMSNorm**: `Qwen3.5RMSNorm` is zero-initialised and applies `(1+w) * x_normed`
  (input/post/final/q/k norms); `RMSNormGated` is ones-initialised and applies
  plain `w * x_normed`. The loader bakes `+1` (toggle `RMSNORM_BAKE_PLUS_ONE`)
  so every call site uses the standard form.
- **MTP** head is ignored on inference (`_keys_to_ignore_on_load = ["^mtp.*"]`).

## Known unknowns / 已解开的疑点 (resolve when a GGUF lands)

These were all **resolved** by inspecting the real Ornith GGUF + llama.cpp
source (`src/models/qwen35.cpp`, `src/models/delta-net-base.cpp`,
`conversion/qwen.py`):

- **Architecture name**: `general.architecture = "qwen35"` (no underscore);
  metadata keys use the `qwen35.*` prefix.
- **Linear-attn tensor names**: `attn_qkv` (in_proj_qkv), `attn_gate` (in_proj_z
  / the gate), `ssm_beta` (in_proj_b), `ssm_alpha` (in_proj_a), `ssm_conv1d`,
  `ssm_dt.bias`, `ssm_norm`, `ssm_out`, and `ssm_a` (**no `.weight` suffix** =
  the per-head A_log parameter). MLP-norm tensor is `post_attention_norm` (not
  `ffn_norm`).
- **`ssm_a` stores `-exp(A_log)`** precomputed at conversion — use it directly
  in the forget gate, do NOT apply another `-exp()`.
- **RMSNorm `+1`**: llama.cpp bakes `(1+w)` into the stored weight for all main
  norms (`output_norm`, `attn_norm`, `post_attention_norm`, q/k norms) but NOT
  for `ssm_norm`. Runtime applies plain `weight * x_rmsnorm`. Hence
  `RMSNORM_BAKE_PLUS_ONE = false` here.
- **conv1d weight** is stored `[kernel=4, conv_dim]` (kernel inner/contiguous,
  oldest tap first) — `conv1d_step` indexes `weight[c*kernel + k]` which
  matches; silu is applied to the conv output before splitting into q/k/v.

## Usage / 用法

```odin
import q35 "qwen3_5:qwen3_5"

e, _ := q35.engine_load("model.gguf", q35.Engine_Opts{
    max_ctx     = 4096,
    num_threads = 8,
})
defer q35.engine_destroy(&e)

logits := q35.engine_forward(&e, token_id, position)
```

Build a consumer with both collections:

```sh
LIB=/path/to/odin-infer
odin build . \
  -collection:ggml=$LIB \
  -collection:qwen3_5=$LIB
```

## Files / 文件

| File | Purpose |
|------|---------|
| `model.odin` | Config, hybrid weight structs, GGUF loader, `(1+w)` bake |
| `forward.odin` | Layer dispatch, full attn, linear attn, MLP |
| `ops.odin` | rmsnorm / l2norm / partial MRoPE / conv1d_step / delta_recurrent_step |
| `matmul.odin` | Self-contained multithreaded quantized GEMV |
| `engine.odin` | Public `Engine` / `engine_load` / `engine_forward` API |
| `tests.odin` | Pure-math unit tests (no GGUF needed) |
