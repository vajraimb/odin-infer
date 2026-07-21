# KV + SSM state persistence

`odin-infer` can save the position-evolving state of a Qwen3.5 (Ornith) chat
to disk and reload it later, so a conversation resumes without re-prefilling
the entire history.

This mirrors colibri's `.coli_kv` feature, adapted for our dense-attention +
gated-delta hybrid architecture.

## What's persisted

Four state buffers, all position-evolving:

| Buffer             | Layers           | Shape (per layer)                  | Notes                                         |
|--------------------|------------------|------------------------------------|-----------------------------------------------|
| `key_cache`        | full-attention   | `[seq_len, kv_dim]` (trimmed)      | f16 head-major on Metal, f32 layer-major on CPU |
| `value_cache`      | full-attention   | `[seq_len, kv_dim]` (trimmed)      | same as K                                     |
| `conv_states`      | linear-attention | `[conv_dim, kernel-1]`             | causal conv1d state                            |
| `recurrent_states` | linear-attention | `[v_heads, head_k_dim, head_v_dim]`| gated-delta recurrent state                   |

**Not persisted** (and don't need to be):
- All weights (loaded fresh from the GGUF each session)
- Per-token activation temporaries (`x`, `xb`, `hb`, ...)

The on-disk format is **CPU f32 layer-major** regardless of which backend
saved it. The Metal backend does f16↔f32 + head-major↔layer-major conversion
at save/load time. A round-trip is bit-exact (verified by `kv_smoke.odin`).

## File format (`.oikv`, all little-endian)

```
magic            : u32 = 0x4F494B56   ('OIKV')
version          : u32 = 1
arch             : u32 = 2            (qwen3_5)
fingerprint      : u64                FNV-1a of identifying Config fields
n_valid_pos      : u32                positions actually populated
seq_len          : u32                max_ctx at save time
n_full_layers    : u32
n_linear_layers  : u32
kv_dim           : u32
conv_dim         : u32
lin_conv_kernel  : u32
lin_n_v_heads    : u32
lin_head_k_dim   : u32
lin_head_v_dim   : u32
key_cache        : [n_full × n_valid_pos × kv_dim] f32
value_cache      : [n_full × n_valid_pos × kv_dim] f32
conv_states      : [n_linear × conv_dim × (kernel-1)] f32  (full)
recurrent_states : [n_linear × lin_n_v_heads × lin_head_k_dim × lin_head_v_dim] f32  (full)
crc32            : u32                IEEE CRC over header+body
```

K/V is trimmed to `n_valid_pos` — a 100-token conversation at Ornith-9B
dimensions (`n_full=8, kv_dim=1024`) saves ~6 MB of K/V plus ~50 MB of
fixed conv/recurrent state, vs the 1.8 GB a full `seq_len=4096` snapshot
would cost.

## CLI usage

Two new flags on `odin-infer-mac`:

```sh
# Save state on exit:
./odin-infer-mac ornith-1.0-9b-Q4_K_M.gguf -g 1 -S /path/to/chat.oikv

# Resume from saved state:
./odin-infer-mac ornith-1.0-9b-Q4_K_M.gguf -g 1 -L /path/to/chat.oikv

# Both (resume, then save updated state on exit):
./odin-infer-mac ornith-1.0-9b-Q4_K_M.gguf -g 1 \
  -L /path/to/chat.oikv -S /path/to/chat.oikv
```

Behavior:

- `-L <path>`: Load state. `pos` is restored to the saved `n_valid_pos`.
  The system-prompt prompt is skipped (the loaded KV already includes it).
  If the file is missing/corrupt/model-mismatched, a warning is printed
  and the chat starts fresh from `pos=0`.
- `-S <path>`: Save state on exit (when user presses Enter on an empty
  prompt line). Overwrites any existing file.

**Qwen3.5 only.** `-L`/`-S` on a Qwen3 dense model print a warning and are
ignored.

The same `-c <max_ctx>` MUST be used on save and load (the file's `seq_len`
must match the engine's `seq_len`). The fingerprint catches any other model
identity mismatch.

## Library API

For Odin programs using `qwen3_5` directly:

```odin
import q35 "qwen3_5:qwen3_5"

// save: caller passes the current `pos` (next position to be filled)
ok := q35.engine_save_kv(&engine, "chat.oikv", current_pos)

// load: returns the restored n_valid_pos (set initial `pos` to this)
pos, ok := q35.engine_load_kv(&engine, "chat.oikv")
```

Both dispatch internally to the right backend (CPU f32 standard layout vs
Metal f16 head-major with conversion).

## Verification

The `kv_smoke.odin` standalone test (in `/tmp` while prototyping; candidate
for `delta_harness/` or similar permanent home) does:

1. Load Ornith 9B
2. Forward 10 tokens to populate state
3. Save → re-forward 1 token → capture reference logits
4. Reset state → load → re-forward 1 token
5. Compare logits — **max_abs_diff = 0** (bit-exact round-trip)

Verified on both CPU and Metal backends.

## Future work (v2)

- Auto-save every N tokens in a background thread (unexpected-exit safety net)
- Compress on save (zstd/LZ4 would shrink 52 MB → ~10 MB)
- Speculative loading (start chat immediately, load KV in background)
- Cross-architecture: also support dense Qwen3 (`infer/` package)
