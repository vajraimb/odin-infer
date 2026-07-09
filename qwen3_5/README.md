# qwen3_5

Independent Qwen3.5 inference package for the odin-infer library. Targets the
**Ornith-1.0-9B** family (Qwen3.5 text model, hybrid full + linear attention),
text-only. Sibling to the Qwen3-only `infer` package; reuses the shared `ggml`
collection for GGUF parsing and quantized dot products.

> **Porting a new architecture?** Read [`PITFALLS.md`](./PITFALLS.md) first —
> it logs every bug hit while building this package (RMSNorm baking, `ssm_a`
> precompute, V-head tiled reorder, Metal buffer zeroing, etc.) with symptoms,
> root causes, and a checklist.

> **Next milestone: batched prefill.** [`BATCHED_PREFILL.md`](./BATCHED_PREFILL.md)
> tracks the simdgroup-GEMM work that fixes slow prompt/tool-result prefill
> (the agent/tool bottleneck). Stage 1a performance is verified (~50× potential),
> blocked on simdgroup matrix layout correctness.

## Status

**Validated end-to-end against the real `deepreinforce-ai/Ornith-1.0-9B-GGUF`
(Q4_K_M)** — loads, runs forward, and produces coherent on-topic English
(verified: prompt "What is 2+2?" yields reasoning text like "The user asked a
\"2+\"" / "The user wants" with proper `<think>`/`</think>` structure).

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

## Architecture notes (from transformers v5.8.1 `Qwen3_5TextModel`)

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

## Known unknowns (resolve when a GGUF lands)

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

## Usage

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

## Files

| File | Purpose |
|------|---------|
| `model.odin` | Config, hybrid weight structs, GGUF loader, `(1+w)` bake |
| `forward.odin` | Layer dispatch, full attn, linear attn, MLP |
| `ops.odin` | rmsnorm / l2norm / partial MRoPE / conv1d_step / delta_recurrent_step |
| `matmul.odin` | Self-contained multithreaded quantized GEMV |
| `engine.odin` | Public `Engine` / `engine_load` / `engine_forward` API |
| `tests.odin` | Pure-math unit tests (no GGUF needed) |
