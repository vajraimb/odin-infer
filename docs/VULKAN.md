# Vulkan backend

[中文](VULKAN.zh.md)

Windows x64 backend for the Qwen3.5 engine. The target machine is a shared-memory
Radeon (RDNA 3.5 iGPU): weights and activations stay in host-visible memory,
preferring `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` when the driver offers
that heap. The default binary is unchanged (Metal on macOS, CPU elsewhere).
A Vulkan build never contains the Metal or CUDA hot path.

SPIR-V is committed under `packages/qwen3_5/spirv` and embedded with `#load`.
Building and running needs Odin plus the driver's `vulkan-1.dll`. The Vulkan
SDK and `glslangValidator` are only for regenerating shaders.

## What the forward runs

Token generation uses one command buffer per forward (MoE submits once mid-layer
to read the router, then continues). GEMV defaults to one thread per row so the
dot matches the CPU association. `ODINFER_VULKAN_PARALLEL=1` selects a 64-wide
kernel: subgroup add when the subgroup size divides 64 and the device supports
subgroup arithmetic, otherwise a shared-memory reduction. `ODINFER_VULKAN_EXACT=1`
forces the exact kernel; `ODINFER_VULKAN_EXACT=0` forces the parallel kernel.
When both are set, `EXACT` wins.

Native GEMV kinds: F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K,
Q5_K, Q6_K, Q8_K. Other supported kinds (IQ*) are dequantized to f32 on upload
and run as F32. Norms stay f32. KV cache on device is f32. Prefill batches up
to 8 tokens.

Element kernels: RMSNorm, L2 norm, partial RoPE, gated RMSNorm, depthwise causal
conv, gated-delta, attention (one workgroup per head), SwiGLU, residual axpy.
Dense MLP and qwen35moe (router on the host, then expert GEMVs, optional shared
expert) both go through this path.

Integer-dot (Q8_0 / Q4_0) and cooperative-matrix (16×16×16 f16, f32 accumulate)
pipelines are created only when the device exposes them. They are exercised by
`vulkan_kernel_test` and are not used for token generation. A missing feature,
or a pipeline that fails to create, is a skip, not a failed init.

`ODINFER_VULKAN_DEVICE` is a device index. Unset, the first non-CPU device whose
name is not llvmpipe is used, else device 0. Startup prints every device to stderr.

## Build on Windows

Odin **dev-2026-07** (the same pin as `.github/workflows/release.yml`) is
already on the laptop and on `PATH`. From the repo root:

```bat
odin build server\serve_harness\serve.odin -file -out:bin\odin-serve.exe -o:speed ^
  -define:ODINFER_BACKEND=vulkan ^
  -collection:ggml=packages -collection:infer=packages ^
  -collection:qwen3_5=packages -collection:qwen3_5_tokenizer=packages ^
  -collection:sampler=packages -collection:tokenizer=packages ^
  -collection:cuda=packages -collection:vulkan=packages

odin build cli -out:bin\odin-cli.exe -o:speed ^
  -define:ODINFER_BACKEND=vulkan ^
  -collection:ggml=packages -collection:infer=packages ^
  -collection:qwen3_5=packages -collection:qwen3_5_tokenizer=packages ^
  -collection:sampler=packages -collection:tokenizer=packages ^
  -collection:cuda=packages -collection:vulkan=packages

odin build tools\vulkan_kernel_test.odin -file -out:bin\vulkan_kernel_test.exe ^
  -define:ODINFER_BACKEND=vulkan ^
  -collection:ggml=packages -collection:infer=packages ^
  -collection:qwen3_5=packages -collection:qwen3_5_tokenizer=packages ^
  -collection:sampler=packages -collection:tokenizer=packages ^
  -collection:cuda=packages -collection:vulkan=packages

odin build tools\vulkan_smoke.odin -file -out:bin\vulkan_smoke.exe ^
  -define:ODINFER_BACKEND=vulkan ^
  -collection:ggml=packages -collection:infer=packages ^
  -collection:qwen3_5=packages -collection:qwen3_5_tokenizer=packages ^
  -collection:sampler=packages -collection:tokenizer=packages ^
  -collection:cuda=packages -collection:vulkan=packages
```

`vulkan-1.dll` is loaded at run time with `core:dynlib` (`LoadLibrary`). Nothing
links `vulkan-1.lib`. The exe still imports the usual system libraries
(`kernel32.dll` and the rest of the Odin Windows runtime). A CPU build of the
same commit omits `-define:ODINFER_BACKEND=vulkan` and is the reference.

Git Bash can also use `ODINFER_BACKEND=vulkan ./serve.sh 9748 4096`. That script
passes the define and `--vulkan`.

Init failure with `--vulkan` or `-g 1` is a hard error. `METAL_FALLBACK=1`
restores the CPU fallback.

## Regenerate shaders

Only after editing `packages/qwen3_5/shaders`. `./scripts/compile_spirv.sh`
needs `glslangValidator` new enough for `GL_EXT_integer_dot_product` (tested
with glslang 16.6; 15.1 cannot compile the integer-dot shader). Most shaders
target Vulkan 1.2; `q8_dp4a.comp` and `coopmat.comp` target Vulkan 1.3. Commit
the `.spv` files with the GLSL change.

## What was run, and what was only compiled

Run on lavapipe (Mesa 25.2.8, `llvmpipe (LLVM 20.1.2, 256 bits)`, API 1.4,
`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json`). There is no AMD GPU
and no Windows machine in this environment.

`./scripts/vulkan_check.sh` builds the tools and runs the checks below.
`bin/vulkan_smoke` is compiled and not executed: it needs a real Qwen3.5 GGUF.

Kernel self-test, 3×256 GEMV against `ggml.dequant_row`:

| Kernel | Result |
|---|---|
| exact GEMV, all 13 kinds, T=1 and T=2 | max_abs = 0 |
| parallel / subgroup GEMV, worst | F32 T=2 subgroup 1.937151e-07; Q8_0 3.5762787e-07; Q4_0/Q5_0 T=2 7.4505806e-08. K-quants 0 |
| rmsnorm, l2norm, rope, conv | max_abs = 0 (conv state 0) |
| attn | 2.9802322e-08 |
| delta | 1.8626451e-09, state 1.4901161e-08 |
| integer dot vs the integer reference | Q8_0 and Q4_0 max_abs = 0 |
| integer dot vs the f32 dot | Q8_0 0.0037726164, Q4_0 6.301701e-05 |
| cooperative matrix | skip (lavapipe has no coopmat) |

Synthetic hybrid GGUF (`tools/make_vulkan_gguf.py`, dim 256, 2 layers, linear
then full attention, every native quant kind in the matmul). Greedy prefill of
tokens `1 2 3 4` plus 4 decode steps:

```
cpu: 1 2 3 4 43 58 45 58
vk:  1 2 3 4 43 58 45 58
```

Exact prefill logits max_abs = 0.00085449219, argmax 43/43.
`ODINFER_VULKAN_PARALLEL=1` (subgroup kernel on this device, subgroup size 8)
produced the same tokens. Parallel prefill logits max_abs = 0.00067138672,
argmax 43/43.

Synthetic F32 qwen35moe (`tools/make_moe_gguf.py`, both layers full attention),
prompt `3 5 7` plus 4 decode steps: `3 5 7 31 54 2 2` on CPU and Vulkan.

Default `./build.sh` (CPU backend, Vulkan package only imported) exited 0.
`packages/qwen3_5` finished 19 tests, all successful. Tokenizer tests and
`moe_load_test` are part of that script.

Windows: `odin check -target:windows_amd64 -define:ODINFER_BACKEND=vulkan` passed
for `server/serve_harness/serve.odin`, `cli`, and `tools/vulkan_kernel_test.odin`.
`odin build -build-mode:obj -target:windows_amd64` emitted COFF objects. The
string `vulkan-1.dll` is in the `vkr` object; `vulkan-1.lib` is not. This Odin
(dev-2026-07) prints `Linking for cross compilation for this platform is not yet
supported (windows amd64)` and does not write a PE, so no Windows exe was run.

## Laptop check

1. Build the four programs in [Build on Windows](#build-on-windows). No SDK.
2. `bin\vulkan_kernel_test.exe`
   Stderr should list the Radeon (not only a CPU device) and end with
   `vulkan_selftest: PASS`. If several devices are printed, rerun with
   `ODINFER_VULKAN_DEVICE` set to the Radeon index. `uma=true` means a buffer
   landed in a device-local host-visible heap, which is the shared-memory path.
   Integer-dot and coopmat lines are optional; `skip` there is fine.
3. Text match against a CPU build of the same commit. With Python and numpy:

   ```bat
   python tools\make_vulkan_gguf.py %TEMP%\odinfer_vulkan_tiny.gguf
   odin build tools\greedy_dump.odin -file -out:bin\greedy_cpu.exe -collection:ggml=packages -collection:infer=packages -collection:qwen3_5=packages -collection:qwen3_5_tokenizer=packages -collection:sampler=packages -collection:tokenizer=packages -collection:cuda=packages -collection:vulkan=packages
   odin build tools\greedy_dump.odin -file -out:bin\greedy_vk.exe -define:ODINFER_BACKEND=vulkan -collection:ggml=packages -collection:infer=packages -collection:qwen3_5=packages -collection:qwen3_5_tokenizer=packages -collection:sampler=packages -collection:tokenizer=packages -collection:cuda=packages -collection:vulkan=packages
   bin\greedy_cpu.exe %TEMP%\odinfer_vulkan_tiny.gguf 32 4 1 2 3 4
   bin\greedy_vk.exe %TEMP%\odinfer_vulkan_tiny.gguf 32 4 1 2 3 4
   ```

   The two stdout lines should be `1 2 3 4 43 58 45 58`. Then
   `set ODINFER_VULKAN_PARALLEL=1` and run `greedy_vk` again.
4. Real model. `bin\vulkan_smoke.exe <model.gguf> 32 2048` prints prefill and
   decode tok/s and the decoded text (temperature 0, prompt asks for 2+2).
   `bin\odin-serve.exe <model.gguf> --vulkan --max-ctx 4096` and
   `bin\odin-cli.exe <model.gguf> -g 1 -c 2048` are the product entry points.
   Build `odin-cli` again without the define and run `-g 0` on the same prompt
   to compare the text.
5. Benchmark the iGPU with the same smoke command twice: the default exact GEMV,
   then `set ODINFER_VULKAN_PARALLEL=1`. The parallel kernel is the one that
   uses the GPU's subgroups. Record both tok/s lines. Exact mode is the
   correctness reference; it is one thread per output row.
