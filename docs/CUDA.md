# CUDA backend

[中文](CUDA.zh.md)

Linux x86_64 backend for the Qwen3.5 engine. It targets **sm_70 and newer**
(validated for Tesla V100S, compute capability 7.0, driver CUDA 13.0).
Accumulation is fp32. There is no bf16. Prefill GEMM (`T>1`) can use Volta
tensor cores (WMMA 16×16×16, fp16 operands, fp32 accumulate). Decode (`T==1`)
uses a bandwidth GEMV: 128-bit loads, `dp4a`, and the activation quantized to
q8 once per vector. Short Q4_K / Q6_K matrices use one row per warp; taller
ones keep four. `ODINFER_CUDA_GEMV=old` restores the scalar kernel. Dense
decode is one CUDA graph unless `ODINFER_CUDA_GRAPH=0`.

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

## Decode GEMV

`T==1` is a memory-bound GEMV. The scalar kernel (one warp per row, byte
loads) is the `ODINFER_CUDA_GEMV=old` fallback. The bandwidth kernel has two
shapes. Four rows per warp, eight threads per row, is the tall-matrix path
(FFN up, the 27B Q projection, the logits row). Those eight threads issue
one `ld.global.v4` each, so a row's quant payload is a single coalesced
128-byte read, and the dot is `dp4a.s32.s32`. Q6_K blocks are 210 bytes, so
the 16-byte load is a shift of two aligned windows; the weight allocation
is padded by 64 bytes and the load never emits a misaligned `v4` on sm_70.
Q6 codes are unpacked per byte. There is no `vsub4` in the PTX: NVRTC emits
`vsub4.s32.s32.s32`, and the sm_70 JIT rejects that opcode, which rejects
the whole module. Scales stay in registers.

Four rows per warp is the default for every Q4_K and Q6_K shape. One row
per warp was slower on the V100S at every size that was timed (9B decode
33 tok/s versus 62 for four rows; bench r1 was 8% of peak where r4 was
39%). `ODINFER_CUDA_GEMV_ROWS=1` still selects it. Total warps are
`rows / 4`. Changing warps per block does not change that, so split-K was
not added. A second sequential launch would not fill more SMs, and a fused
split would need atomics.

Q8_0 and Q5_0 matrices under 16 MiB stay on the scalar kernel. At
2048×2048 the new kernel was slower (Q8_0 118.5 vs 121.8 GB/s, Q5_0 49.2
vs 69.7). The 9B and 27B hot weights are Q4_K and Q6_K, so they are not
on this path. `ODINFER_CUDA_GEMV=new` keeps the new kernel for those small
shapes too.

The activation is quantized to q8 (groups of 32, signed int8, scale
`amax/127`) once per vector. Q4_K and Q5_K also keep the exact sum of each
16-float half so the `dmin` term is not quantized. A one-row RMSNorm writes
that q8 in the same kernel; the Q, K, and V projections, and the two FFN
up projections, reuse it. SwiGLU and a multi-row norm drop the cache.
Kinds without a bandwidth kernel (F32, Q2_K, Q3_K, Q8_K, the Q4_0 family)
stay on the scalar GEMV. So does a K that is not a multiple of the quant
block.

Block size is `ODINFER_CUDA_GEMV_WARPS` warps (default 8, range 1..16).
With four rows per warp that is 32 rows per block. With one row per warp
it is one row per warp in the block. V100S has 80 SMs and 64 warp slots
per SM.

On a dense `T==1` step the attention out, the FFN down, and the linear
out add their GEMV straight into the residual, so that token skips the
separate residual kernel. Prefill (`T>1`) and the scalar fallback still
add with their own kernel. The logits GEMV does not accumulate.

| Variable | Default | Effect |
|----------|---------|--------|
| `ODINFER_CUDA_GEMV` | auto | `old` or `0`: scalar GEMV. `new`: bandwidth kernel even for small Q5_0 / Q8_0. Unset: bandwidth, except Q5_0 / Q8_0 under 16 MiB |
| `ODINFER_CUDA_GEMV_ROWS` | `4` | Rows per warp for Q4_K and Q6_K. Unset is four. `1` selects the slower one-row kernel |
| `ODINFER_CUDA_GEMV_WARPS` | `8` | Warps per bandwidth-GEMV block (1..16) |
| `ODINFER_CUDA_LAUNCHES` | off | `=1` prints the kernel launch count for each forward |
| `ODINFER_CUDA_PEAK_GBS` | `1134` | Peak used by `--bench-gemv` to print % of HBM |
| `ODINFER_CUDA_GRAPH` | on | Dense `T==1` is one CUDA graph. `=0` runs every kernel eager. MoE and prefill stay eager. `ODINFER_CUDA_PROFILE=1` turns the graph off |
| `ODINFER_CUDA_PROFILE` | off | `=1` prints per-kernel GPU time for that forward, slowest first |

`cuda_selftest` runs the four-row kernel against the CPU fp32 dot for Q4_K,
Q6_K, Q8_0, Q5_0, and Q5_K at 64×512 and at 6×256 (the tail that is not a
multiple of four rows). Q4_K and Q6_K also run the one-row kernel, including
`accumulate=1` (output starts at 0.25, expect `ref+0.25`). Allowance is
`1 + 0.05*|ref|`. The gap against the CPU is the q8 of `x`; a swapped
nibble or scale is on the order of `|ref|`.

```sh
bin/cuda_kernel_test
bin/cuda_kernel_test --bench-gemv
ODINFER_CUDA_GEMV_WARPS=4 bin/cuda_kernel_test --bench-gemv
```

`--bench` still prints the tensor-core TFLOPS table, then the same GEMV
table. `--bench-gemv` prints only the GEMV table: milliseconds, GB/s, and
percent of peak, for the four-row kernel (`r4`), the one-row kernel where
it exists (`r1`), and the scalar kernel (`old`).

| Name | Kind | rows × cols | Where it shows up |
|------|------|-------------|-------------------|
| 9B w1 | Q4_K | 12288 × 4096 | FFN up / gate |
| 9B wq | Q4_K | 8192 × 4096 | Q projection |
| 9B wo | Q4_K | 4096 × 4096 | output projection of attention |
| 9B w2 | Q6_K | 4096 × 12288 | FFN down |
| 9B out | Q6_K | 248320 × 4096 | logits |
| 27B w1 | Q4_K | 17408 × 5120 | FFN up / gate |
| 27B wq | Q4_K | 12288 × 5120 | Q projection |
| 27B wo | Q4_K | 5120 × 6144 | attention out |
| 27B w2 | Q6_K | 5120 × 17408 | FFN down |
| 27B out | Q6_K | 248320 × 5120 | logits |
| q8_0 / q5_0 | | 2048 × 2048 | 0.8B MoE kinds |
| q5_k | Q5_K | 5120 × 5120 | mixed Q4_K_M tensors |

GB/s counts the weight bytes of one launch. The q8 of `x` is outside the
timed loop, which is how a cached activation behaves. The two output rows
are about 1 GB; a failed alloc is printed and the rest of the table still
runs. `ODINFER_CUDA_PEAK_GBS` overrides 1134 when the device is not a V100S.

End-to-end decode, scalar kernel against the default. The smoke prompt is
short, so this is the short-context number (the 9B figure that was 32.7
tok/s before this kernel). A 990-token prefill is a separate measurement:
the KV read at that length is a few MB per full-attention layer, and the
weight stream is still the whole model.

```sh
ODINFER_CUDA_GEMV=old bin/cuda_smoke "$MODEL9B" 16 2048
bin/cuda_smoke "$MODEL9B" 16 2048
ODINFER_CUDA_GEMV=old bin/cuda_smoke "$MODEL27B" 16 2048
bin/cuda_smoke "$MODEL27B" 16 2048
ODINFER_CUDA_LAUNCHES=1 bin/cuda_smoke "$MODEL9B" 8 2048
```

Init prints the GEMV row policy and either `graph on for dense T=1` or
`gemv old`.

Dense decode (`T==1`, not MoE) is captured once into a CUDA graph and
replayed. Capture does not run the kernels; the first token pays for
capture and then launches the graph. The token position is a device
symbol (`g_decode_pos`), written before each replay, because a kernel
argument would be baked in at capture. Prefill passes the real position
and stays eager, so a graph cannot change a `T>1` result. If capture or
instantiate fails, that token runs eager and graphs stay off.
`ODINFER_CUDA_GRAPH=0` is the eager path. `ODINFER_CUDA_PROFILE=1`
disables the graph and prints each kernel's GPU time (name, count,
milliseconds), slowest first, plus wall time against the sum. Sampling
stays on the CPU, after the logits copy. The graph and event symbols are
looked up on their own; a driver without them disables graphs and the
profiler and still loads the GEMV.

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

GEMV (decode) and MMQ (prefill, tile of up to 8 tokens, any `T`,
token-major output), fp32 accumulate. The default decode GEMV is the
bandwidth kernel for Q4_K, Q5_K, Q6_K, Q8_0, and Q5_0; every kind still
has the scalar one-warp-per-row kernel:

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

Also not in this backend: Metal's shape-specialized GEMV (`QGEMVSPEC`),
wide-N autotune and the tune cache, fast-math, MoE grouped GEMM
(`forward_gpu_batch_moe`), and `engine_verify_batch`
(speculative verify). A one-row RMSNorm on CUDA does feed the q8 cache
the next GEMV reads. MoE prefill runs the attention projections as MMQ
and the experts as per-token GEMV. Dense prefill chunks by
`CUDA_MAX_BATCH` (128, smaller if the activation alloc fails). Decode is
GEMV, not MMQ.

Weights are copied to the device once (`cuMemcpyHtoD` of the GGUF mmap, plus
the host-dequantized MoE router and shared-expert matrices). A 5.6 GB Q4_K_M
9B fits in a 32 GB V100S; the resident KV cache is extra and follows the same
`KV_Q8` / `KV_V` layout as Metal, so snapshots interchange.

Sigmoid and softplus in the CUDA kernels match `ops.odin` (stable sigmoid,
softplus clamped at ±20). Metal's fast-math `1/(1+exp(-x))` can differ by a
few ulps. Kernel tests compare against the CPU reference, not against Metal
golden files. The scalar GEMV tolerance is about `2e-2 + 1e-3 * |ref|`
because the warp reduction order is not the CPU's left-to-right sum. The
bandwidth GEMV allowance is `1 + 0.05 * |ref|` because `x` is q8.
