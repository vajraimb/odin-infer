# Environment variables

All tunables recognized by `odin-infer` (library) and `odin-infeer` (CLI).
Defaults apply when the variable is unset.

Inspect current values on your machine:

```sh
odin run doctor_harness/doctor.odin -file -- env \
  -collection:ggml=. -collection:infer=. \
  -collection:tokenizer=. -collection:sampler=.
```

## Inference / Metal backend

Sourced from `qwen3_5/metal.odin` (read once during `metal_init`). The Qwen3
Metal backend (`infer/metal.odin`) currently has no env-var knobs.

| Variable       | Default | Effect                                                                 |
|----------------|---------|------------------------------------------------------------------------|
| `QFASTMATH`    | `ON`    | Metal shader fast-math enable. Set `=0` to disable for precision debug |
| `QTIMING`      | off     | Set `=1` to print per-prefill GPU ms                                  |
| `QPROF_NOLIN`  | off     | Ablation: skip linear-layer per-token stateful loop (output garbage)  |
| `QPROF_NOFULL` | off     | Ablation: skip full-attention per-token block                          |

`QPROF_*` are GPU-kernel timing ablations — they produce wrong output but let
you isolate per-stage GPU time. Never enable in production.

## CLI (`odin-infeer`)

| Variable | Default | Effect                                  |
|----------|---------|-----------------------------------------|
| `QDBG`   | off     | Set `=1` to dump token encode/decode to stderr |

## Compile-time constants (not env vars, but adjacent)

These are baked at build time; changing them requires a rebuild.

| Constant                 | Where                  | Default | Notes                                            |
|--------------------------|------------------------|---------|--------------------------------------------------|
| `DEFAULT_MAX_CONTEXT`    | `infer/model.odin`     | 4096    | Cap on KV cache; CLI overrides with `-c`         |
| `MAX_MATMUL_N`           | `infer/matmul.odin`    | 65536   | Per-thread dequant scratch length                |
| `MAX_BATCH_T`            | `qwen3_5/metal.odin`   | 512     | Max tokens per batched-prefill chunk (Stage 1a)  |
| `MAX_VOCAB` / `MAX_MERGES` | `tokenizer/tokenizer.odin` | 151936 / 151386 | Tokenizer buffer ceilings  |

## CLI flags (not env, but worth documenting here)

`odin-infer-mac <model.gguf> [flags]`:

| Flag | Meaning                                  | Default                |
|------|------------------------------------------|------------------------|
| `-t` | Temperature                              | 0.6                    |
| `-p` | Top-p                                    | 0.95                   |
| `-s` | RNG seed (0 = time-based)                | 0                      |
| `-m` | Multi-turn (1 = on)                      | off                    |
| `-k` | Thinking on (1 = on, Qwen3 reasoning)    | off                    |
| `-x` | Repetition penalty                       | 1.0 (off)              |
| `-r` | Print tokens-per-sec                     | off                    |
| `-f` | Print TTFT                               | off                    |
| `-j` | Thread count                             | physical core count    |
| `-c` | Max context                              | `DEFAULT_MAX_CONTEXT`   |
| `-g` | Metal GPU (1 = on)                       | off                    |

Special: `<model.gguf> --dump` runs `dump_gguf` (metadata dump) instead of chat.

## Adding a new env var

1. Read it in the relevant package with `os.get_env(NAME, context.temp_allocator)`.
2. Add a row to the table above.
3. Add an entry to `ENV_VARS` in `doctor_harness/doctor.odin` so `doctor env`
   picks it up automatically.
