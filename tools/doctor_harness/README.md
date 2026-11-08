# doctor_harness

Read-only diagnostic + load-planner for `odin-infer`. Mirrors colibri's
`coli doctor` / `coli plan` UX.

## Subcommands

```sh
odin run doctor_harness/doctor.odin -file -- <subcommand> [args] \
  -collection:ggml=. -collection:infer=. \
  -collection:tokenizer=. -collection:sampler=.
```

### `doctor <model.gguf>`

Lightweight readiness check. Parses GGUF only (no `engine_load`), prints:

- **System**: OS, arch, CPU cores, Metal device name + threadgroup memory limit
- **Environment**: every recognized env var and its current value
- **GGUF file**: size, magic check, architecture, name, file_type,
  tensor count, metadata KV count, quant-type histogram, model dims
- **Verdict**: `OK ✓` / warnings / `FAIL`

### `plan <model.gguf> [--max-ctx N] [--metal] [--threads N]`

Full `engine_load`, prints the planned runtime layout:

- Everything from `doctor`'s system section
- Effective `max_ctx`, backend, threads
- **Memory budget**:
  - Weights (mmap'd): file size
  - KV cache: `2 × n_layers × seq_len × n_kv_heads × head_dim × 4 bytes`
  - Activations: `x, xb, xb2, hb, hb2, q, att, logits`
  - **Total resident** footprint
- `metal_ready` status

### `env`

Just the env-var dump (subset of `doctor` output).

## Exit codes

- `0` OK
- `1` WARN
- `2` FAIL (bad magic / truncated / unsupported arch)

## Platform

macOS-only in v1 (the Metal probe uses `MTL.CreateSystemDefaultDevice`).
For Linux, factor the Metal block into a `#+build !darwin` stub.

## Output example (truncated)

```
══ System ════════════════════════════════════════════════════
  os            : Darwin (macOS)
  arch          : arm64
  cpu cores     : 8
  metal device  : Apple M3
  tg mem max    : 32768 bytes (32.0 KB)

══ GGUF file ════════════════════════════════════════════════
  path          : Qwen3-0.6B-Q4_K_M.gguf
  size          : 378.33 MB
  magic         : GGUF ✓
  architecture  : qwen3 (dense)
  tensors       : 310
  quant types   :
    Q6_K  × 29
    Q4_K  × 168
    F32   × 113
  verdict       : OK ✓
```

See `docs/ENVIRONMENT.md` for the full env-var reference.
