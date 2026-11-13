# CUDA backend

[中文](CUDA.zh.md)

Linux x86_64 backend for the Qwen3.5 engine. It targets **sm_70 and newer**
(validated for Tesla V100S, compute capability 7.0, driver CUDA 13.0).
Accumulation is fp32. There is no bf16. Prefill GEMM (`T>1`) can use Volta
tensor cores (WMMA 16×16×16, fp16 operands, fp32 accumulate). Decode (`T==1`)
stays on the GEMV kernels. dp4a is not required.

macOS builds are unchanged: the default backend is still Metal. A single
binary never contains both hot paths.

## Build

Odin compile does not need `nvcc`, the CUDA toolkit headers, or a GPU.
`libcuda.so.1` and NVRTC are `dlopen`'d at runtime.

```sh
# CPU Linux build (no Metal frameworks). Same entry point as macOS.
./build.sh

# CUDA engine
odin build server/serve_harness/serve.odin -file -out:bin/odin-serve \
  -o:speed -no-bounds-check -disable-assert \
  -define:ODINFER_BACKEND="cuda" \
  -collection:ggml=packages -collection:infer=packages \
  -collection:qwen3_5=packages -collection:qwen3_5_tokenizer=packages \
  -collection:sampler=packages -collection:tokenizer=packages \
  -collection:cuda=packages

bin/odin-serve model.gguf --cuda --max-ctx 2048
# --metal is accepted as the same "use the GPU" switch on a CUDA build.
```

`-define:ODINFER_BACKEND="cuda"` must be quoted. The string is compared to
`"cuda"`; the bare word `default` (the unset value) keeps Metal on Darwin and
CPU everywhere else.

The define is read in `packages/qwen3_5/backend.odin` (`GPU_METAL` /
`GPU_CUDA`). Forward, KV save/load, and reset are `when` branches, so the
compiled hot path has no Metal-vs-CUDA test.

## NVRTC

`ODINFER_NVRTC_PATH` is an exclusive override: a full path to
`libnvrtc.so.12` (or `.13`), or a directory that contains it. When it is
set, nothing else is searched. `NVRTC_DIR` is searched first when the
override is unset, and the search then continues.

Otherwise the loader tries versioned sonames before the bare name
(`libnvrtc.so.13`, then `.12`, then `libnvrtc.so`), then a pip
`nvidia-cuda-nvrtc-cu12` tree
(`~/.local/lib/python3.*/site-packages/nvidia/cuda_nvrtc/lib`). The matching
`libnvrtc-builtins.so.<major>.<minor>` (the wheel ships
`libnvrtc-builtins.so.12.9`, not `.so.12`) is preloaded with `RTLD_GLOBAL`.
The `.alt.` copy is skipped.

CUDA 13 NVRTC cannot target `compute_70`. Archs below sm_75 were removed,
and the bare `libnvrtc.so` on a CUDA 13 machine is that 13.x library. A
13.x driver can still JIT PTX produced by NVRTC 12.9. When the device is
sm_70 / sm_72, or a `compute_70` compile fails on a 13.x library, the
runtime loads NVRTC 12.x only and prints the pip path if none is found.
NVRTC 12.9 warns that archs before sm_75 are deprecated. That warning is
expected on a V100.

```sh
python3 -m pip install --user nvidia-cuda-nvrtc-cu12
export ODINFER_NVRTC_PATH="$HOME/.local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12"
```

Every compile passes `--gpu-architecture=compute_XX` from the device.
`CUDA_DEVICE` selects the ordinal (default 0).

## Tensor cores

Prefill and other batched GEMMs (`Y[T, rows] = X[T, cols] · W[rows, cols]`)
dequantize a 64×32 fp16 tile of `W` into shared memory, narrow `X` to fp16,
and issue `wmma.mma.sync.aligned.m16n16k16` (fp16 in, fp32 accumulate). The
PTX is inline. NVRTC does not ship `mma.h`, and `mma.sync.m16n16k16` is not
the Volta instruction; the WMMA form is what sm_70 JITs. `ldmatrix` is not
used.

The quant kinds with a tensor-core kernel are the same set as MMQ: F32, F16,
Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K. The 9B
Q4_K_M prefill is Q4_K (up/gate/attention) and Q6_K (down). The 0.8B MoE
projections that are batched use Q4_K, Q8_0, and Q5_0. Expert GEMVs stay on
the GEMV kernels.

Selection, read at `cuda_init`:

| `ODINFER_CUDA_TC` | Behaviour |
|-------------------|-----------|
| unset | tensor cores when compute capability ≥ 7.0 and `T ≥ 16` |
| `1` | tensor cores for every `T > 1` (still GEMV at `T == 1`) |
| `0` | MMQ for `T > 1`, GEMV for decode |

`T == 1` is always GEMV. A missing tensor-core kernel falls back to MMQ.

Correctness (`max_abs`, the tolerance at that element, `atol`, `rtol`) and
the microbenchmark:

```sh
# build is the same command as bin/cuda_kernel_test in the check script
bin/cuda_kernel_test
bin/cuda_kernel_test --bench
```

`cuda_selftest` compares tensor-core GEMM to the CPU fp32 dot for F32, F16,
Q8_0, Q4_0, Q5_0, Q4_K, Q6_K, Q5_K, Q8_K at `T=32`, `rows=64`, `cols=256`,
plus partial tiles (`T=20`, `rows=40`) for Q4_K, Q8_0, and Q5_0. fp16
rounding uses `atol=2e-2 rtol=1e-2` for F32/F16 and `atol=5e-2 rtol=5e-2`
for the quantized kinds. A wrong tile mapping is far outside that.

`--bench` prints milliseconds and TFLOPS (2·T·rows·cols) for tensor cores
and MMQ:

| Name | Kind | rows × T × cols |
|------|------|-----------------|
| 9B w1/w3 | Q4_K | 12288 × 64 × 4096 |
| 9B wq | Q4_K | 8192 × 64 × 4096 |
| 9B wo | Q4_K | 4096 × 64 × 4096 |
| 9B w2 | Q6_K | 4096 × 64 × 12288 |
| 0.8B Q8_0 | Q8_0 | 1024 × 32 × 1024 |
| 0.8B Q5_0 | Q5_0 | 1024 × 32 × 1024 |

End-to-end prefill, same prompt, tensor cores on and off. The smoke line
`prefill N tok` is the number that has to be at least 16 for the default
(auto) policy; `ODINFER_CUDA_TC=1` uses tensor cores for any `N > 1`.

```sh
ODINFER_CUDA_TC=0 bin/cuda_smoke "$MODEL" 8 2048
bin/cuda_smoke "$MODEL" 8 2048
ODINFER_CUDA_TC=1 bin/cuda_smoke "$MODEL" 8 2048
```

Decode tok/s should match across those three. The init line says
`tensor cores auto (T>=16)`, `off`, or `on for T>1`.

## Check script

`scripts/cuda_check.sh` builds the CUDA test and smoke binaries, compiles the
kernel source for `compute_70` (no GPU required for that step), and, when a
GPU is visible, runs the kernel self-test and an end-to-end generation.

```sh
./scripts/cuda_check.sh
```

Model search, first hit wins (`MODEL` overrides):

| Path | Size | What it exercises |
|------|------|-------------------|
| `$MODEL` | any Qwen3.5 GGUF the loader accepts | whatever you point at |
| `models/Qwen3.5-0.8B-MoE-A0.6B-Q4_K_M.gguf` | 524 MB | MoE, mixed Q4_K / Q8_0 / Q5_0 |
| `$HOME/.odinfer/models/Qwen3.5-0.8B-MoE-A0.6B-Q4_K_M.gguf` | 524 MB | same, installer layout |
| `models/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf` | Mac benchmark, sha256 prefix `a692de95` | dense Q4_K_M (Q4_K + Q6_K + F32 norms) |

Smoke arguments (overridable): `SMOKE_TOKENS` default 16, `SMOKE_CTX` default
2048. If no model file is present the kernel test still runs and smoke is
skipped, unless `CUDA_CHECK_REQUIRE_SMOKE=1`.

The script prints `sha256 <hex> <path>` for the file it found. The Mac
benchmark 9B starts with `a692de95`. Current Hugging Face `main` of
`mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-GGUF`
(commits `29319295d35e` through `a713868aa6cd`) is a different quant:
sha256 prefix `b4363b57`, 5780092768 bytes. That revision is not pinned,
because pinning it would not match the Mac file, and the `a692de95` blob
is not in the surviving history of that repo. `cuda_check.sh` refuses to
smoke a heretic 9B whose hash does not start with `a692de95` unless
`CUDA_CHECK_ALLOW_REQUANT=1`. `scripts/install.sh --model qwen9b` still
fetches HF `main`, prints the sha256, and warns when it is not
`a692de95`.

Direct commands the script wraps:

```sh
odin build tools/cuda_kernel_test.odin -file -out:bin/cuda_kernel_test \
  -define:ODINFER_BACKEND="cuda" \
  -collection:ggml=packages -collection:infer=packages \
  -collection:qwen3_5=packages -collection:qwen3_5_tokenizer=packages \
  -collection:sampler=packages -collection:tokenizer=packages \
  -collection:cuda=packages

bin/cuda_kernel_test --nvrtc-only compute_70
bin/cuda_kernel_test

odin build tools/cuda_smoke.odin -file -out:bin/cuda_smoke \
  -define:ODINFER_BACKEND="cuda" \
  -collection:ggml=packages -collection:infer=packages \
  -collection:qwen3_5=packages -collection:qwen3_5_tokenizer=packages \
  -collection:sampler=packages -collection:tokenizer=packages \
  -collection:cuda=packages

bin/cuda_smoke models/Qwen3.5-0.8B-MoE-A0.6B-Q4_K_M.gguf 16 2048
```

`cuda_kernel_test` compares GEMV/MMQ, RMSNorm, RoPE, SwiGLU, mixed-V KV
store + attention, and the gated-delta step to the CPU reference in
`packages/ggml` and `packages/qwen3_5/ops.odin`. `cuda_smoke` prints prefill
and decode tok/s. Sampling stays on the existing CPU sampler (temperature 0
in the smoke tool).

## What is ported

GEMV (decode, one warp per row) and MMQ (prefill, tile of up to 8 tokens,
any `T`, token-major output), fp32 accumulate:

F32, F16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K.

Also: embedding via the CPU dequantizer plus an upload, RMSNorm, per-head
Q RMSNorm, partial RoPE, attention decode and prefill, KV store and attention
for f16, symmetric Q8_0, and the 40-byte mixed V block, attention gate,
SwiGLU, residual, depthwise conv1d+SiLU, l2norm+scale, per-token gated delta,
RMSNormGated, MoE softmax top-k plus per-expert GEMV and the optional shared
expert.

The default 9B Q4_K_M file (Q4_K, Q6_K, F32 norms, often Q8_0 or Q6_K
embeddings) and the 0.8B MoE file (Q4_K, Q8_0, Q5_0, F32) are in that set.
Init scans every projection and refuses a kind that is not in the list,
naming the tensor. Embeddings may be any kind `ggml` can dequant on the CPU,
including IQ; the output projection may not.

## What is not ported

IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q8_1.

Also not in this backend: Metal's norm-fused GEMV, shape-specialized GEMV
(`QGEMVSPEC`), wide-N autotune and the tune cache, fast-math, MoE grouped
GEMM (`forward_gpu_batch_moe`), and `engine_verify_batch` (speculative
verify). MoE prefill runs the attention projections as MMQ and the experts
as per-token GEMV. Dense prefill chunks by `CUDA_MAX_BATCH` (128, smaller if
the activation alloc fails). Decode is GEMV, not MMQ.

Weights are copied to the device once (`cuMemcpyHtoD` of the GGUF mmap, plus
the host-dequantized MoE router and shared-expert matrices). A 5.6 GB Q4_K_M
9B fits in a 32 GB V100S; the resident KV cache is extra and follows the same
`KV_Q8` / `KV_V` layout as Metal, so snapshots interchange.

Sigmoid and softplus in the CUDA kernels match `ops.odin` (stable sigmoid,
softplus clamped at ±20). Metal's fast-math `1/(1+exp(-x))` can differ by a
few ulps. Kernel tests compare against the CPU reference, not against Metal
golden files. GEMV tolerance is about `2e-2 + 1e-3 * |ref|` because the warp
reduction order is not the CPU's left-to-right sum.
