# odin-infer

Reusable Odin inference library for **Qwen3** GGUF models. Provides GGUF parsing,
quantized GEMV, CPU/Metal forward pass, BPE tokenizer, and sampling — consumed by
the [odin-infer-mac](https://github.com/vajraimb/odin-infer-mac) CLI and embeddable
in other Odin projects.

## Packages

| Collection | Path | Description |
|------------|------|-------------|
| `ggml` | `ggml/` | GGUF v2/v3 parser, GGML quant dequant + SIMD dot products |
| `infer` | `infer/` | Qwen3 model, CPU/Metal forward, public `Engine` API |
| `tokenizer` | `tokenizer/` | BPE tokenizer with embedded vocab/merges |
| `sampler` | `sampler/` | Temperature / top-p sampling |

## Engine API

```odin
import infer "infer"

Engine_Opts :: struct { max_ctx: int, use_metal: bool, num_threads: int }
Engine :: struct { transformer: Transformer, metal_ready: bool }

engine_load    :: proc(path: string, opts: Engine_Opts) -> (Engine, bool)
engine_destroy :: proc(e: ^Engine)
engine_forward :: proc(e: ^Engine, token, pos: int) -> []f32
engine_config    :: proc(e: ^Engine) -> ^Config
engine_metal_ready :: proc(e: ^Engine) -> bool
matmul_set_threads :: proc(n: int)
destroy_matmul_pool :: proc()
```

Also exported from `infer`: `Config`, `Transformer`, `DEFAULT_MAX_CONTEXT`,
`build_transformer`, `free_transformer`, `softmax`.

## Usage in your project

Clone this repo alongside your project (or add as submodule), then build with
`-collection` flags:

```sh
LIB="../odin-infer"
odin build . \
  -collection:ggml=$LIB \
  -collection:infer=$LIB \
  -collection:tokenizer=$LIB \
  -collection:sampler=$LIB
```

Imports use Odin collection syntax (`collection:package`):

```odin
import ggml "ggml:ggml"
import infer "infer:infer"
import tokenizer "tokenizer:tokenizer"
import sampler "sampler:sampler"
```

Example:

```odin
import infer "infer:infer"

main :: proc() {
    e, ok := infer.engine_load("model.gguf", infer.Engine_Opts{
        max_ctx     = 4096,
        use_metal   = true,
        num_threads = 8,
    })
    defer infer.engine_destroy(&e)

    logits := infer.engine_forward(&e, token_id, position)
    // ...
    infer.destroy_matmul_pool()
}
```

## Build / test

```sh
./build.sh   # runs tokenizer tests
```

Manual test:

```sh
cd tokenizer && odin test .
```

## Platform

- **CPU forward + matmul**: any platform Odin supports
- **Metal GPU**: macOS Apple Silicon only (`#+build darwin` in `infer/metal.odin`)
