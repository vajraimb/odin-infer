<!--
Internal engineering log, kept in its original language (中文内部工程记录，保留原文).
The user-facing documentation (README / AGENTS / POSITIONING / docs/MCP·MEMORY·THERMAL) is bilingual.
面向用户的文档（README/AGENTS/POSITIONING/docs 下三篇）为中英双语；本文件为内部工程日志。
-->

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
| `KV_Q8`        | `ON`    | Q8_0 KV cache. Set `=0` for the legacy f16 Metal KV (A/B, precision)   |
| `KV_V`         | `mix`   | V cache format when `KV_Q8` is on. Default stores the peak value of each 32-block as f16 and Q8-quantizes the rest. Set `=q8` for symmetric Q8 V |
| `QGEMVSPEC`    | `ON`    | Per-tensor GEMV shape kernels for Q4_K/Q6_K and the IQ family. Set `=0` to use the generic kernels |
| `QMMQN`        | `ON`    | Wide-N prefill MMQ. Q4_K is 32×32×64, Q6_K is 16×64×64, and the eight IQ kinds are 32×32×64, when M/N/K divide. Set `=0` for the BN=8 tiles |
| `QGEMVEXACT`   | off     | Dense prefill only. Set `=1` to recompute attention and linear projections with per-token GEMV so they match decode bit for bit. MoE prefill always does that overwrite |
| `METAL_FALLBACK` | off   | Set `=1` to let a failed Metal init fall back to CPU. Default: `--metal`/`-g 1` + failed init = hard error |

`QPROF_*` are GPU-kernel timing ablations — they produce wrong output but let
you isolate per-stage GPU time. Never enable in production.

## CUDA backend (Linux, `-define:ODINFER_BACKEND="cuda"`)

| Variable | Default | Effect |
|----------|---------|--------|
| `ODINFER_NVRTC_PATH` | unset | Exclusive path to `libnvrtc.so.12` (sm_70) or `.13`, or the directory that contains it. CUDA 13 NVRTC cannot target compute_70; a V100 needs NVRTC 12.x (`pip install nvidia-cuda-nvrtc-cu12`). |
| `NVRTC_DIR` | unset | Directory or soname path searched first when `ODINFER_NVRTC_PATH` is unset. For sm_70 a 13.x library there is skipped and the search continues to pip `libnvrtc.so.12` and the loader. |
| `CUDA_DEVICE` | `0` | Device ordinal. |
| `ODINFER_CUDA_TC` | auto | Prefill GEMM. Unset: Volta WMMA when `T≥16`. `1`: WMMA for every `T>1`. `0`: MMQ. Decode (`T==1`) stays GEMV. |
| `ODINFER_CUDA_GEMV` | auto | Decode GEMV. Unset: dp4a bandwidth kernel; Q5_0 and Q8_0 under 16 MiB stay on the scalar kernel. `new`: bandwidth kernel for those small shapes too. `old` or `0`: scalar GEMV for everything. |
| `ODINFER_CUDA_GEMV_ROWS` | `4` | Q4_K / Q6_K rows per warp. Unset is four. `1` selects the one-row kernel, which was slower on the V100S. |
| `ODINFER_CUDA_GEMV_WARPS` | `8` | Warps per bandwidth-GEMV block, 1..16. V100S tuning knob. |
| `ODINFER_CUDA_GEMV_SPLIT` | on | Q4_K / Q6_K decode with `rows≤8192` splits K four ways and `atomicAdd`s the partials. `=0` is one K range. |
| `ODINFER_CUDA_TC2` | on | Q4_K / Q6_K prefill (K multiple of 256). `T<=64` uses the 32-token fp16 tile. `T>64` uses the scalar-dequant tile. `=0` uses the scalar tile at every T. |
| `ODINFER_CUDA_TC2_WIDE` | off | `=1` uses the 128-token tile for `T>64`. Slower than the scalar tile at 581 and 2000 tokens on the V100S. |
| `ODINFER_CUDA_ATTN_BATCH` | on | Prefill full attention launches once per chunk (heads by tokens). `=0` launches once per token. |
| `ODINFER_CUDA_FUSE_NORM` | off | `=1` writes the projection to a side buffer and fuses the add with the next RMSNorm and its q8 store. Slower than the accum GEMV plus `rmsnorm_q8` on the V100S. |
| `ODINFER_CUDA_Q4_NC` | off | `=1` uses `ld.global.nc` for the Q4 GEMV. Slower than plain `ld.global` on the V100S. |
| `ODINFER_CUDA_Q4_ILP` | off | `=1` uses `gemv2_q4x` (8 rows/warp, coherent 128-bit loads, two super-blocks per iteration). `--bench-gemv` prints it as `r8`. Not the decode default. |
| `ODINFER_CUDA_Q6_STAGE` | auto | Unset stages a Q6 GEMV only when K≥16384 or rows>8192. `=1` stages every Q6 GEMV. `=0` uses direct loads. |
| `ODINFER_CUDA_SMOKE_PREFILL` | unset | `cuda_smoke` only. `=N` prefills N tokens of a fixed repeated sentence instead of the short 2+2 prompt. |
| `ODINFER_CUDA_GRAPH` | off | `=1` captures dense `T==1` decode as one CUDA graph. MoE and prefill stay eager. Off while `ODINFER_CUDA_PROFILE=1`. |
| `ODINFER_CUDA_PROFILE` | off | `=1` prints per-kernel GPU time for each forward (disables the graph). The 8192-event ring is aggregated by kernel name as it fills, so a long prefill is complete. |
| `ODINFER_CUDA_LAUNCHES` | off | `=1` prints the kernel launch count of each CUDA forward on stderr. A graph replay prints the captured count. |
| `ODINFER_CUDA_PEAK_GBS` | `1134` | HBM peak for `cuda_kernel_test --bench-gemv` (percent of peak). |
| `QTIMING` | off | `=1` prints per-chunk CUDA wall time on stderr. |
| `KV_Q8` / `KV_V` | Q8 K + mixed V | Same switches as Metal. The on-disk KV snapshot format is shared. |
| `METAL_FALLBACK` | off | `=1` lets a failed CUDA init fall back to CPU. `--cuda` / `--metal` / `-g 1` otherwise hard-error. |
| `MOE_CPU` | off | `=1` forces the MoE path onto CPU. |

`QGEMVSPEC`, `QMMQN`, `QFASTMATH`, and the tune cache are Metal-only. The CUDA build does not read them.

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

`odin-cli <model.gguf> [flags]`:

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
