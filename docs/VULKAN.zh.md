# Vulkan 后端

[English](VULKAN.md)

Qwen3.5 引擎的 Windows x64 后端。目标机器是共享显存的 Radeon（RDNA 3.5 核显）：
权重和激活放在 host-visible 内存里，驱动提供
`DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` 时优先用这块堆。默认二进制不变
（macOS 为 Metal，其他平台为 CPU）。Vulkan 构建不包含 Metal 或 CUDA 热路径。

SPIR-V 提交在 `packages/qwen3_5/spirv`，用 `#load` 嵌进二进制。编译和运行只需要
Odin 和驱动自带的 `vulkan-1.dll`。Vulkan SDK 和 `glslangValidator` 只在重新生成
着色器时才需要。

## 前向实际跑什么

生成 token 时，一次前向记录一条命令缓冲（MoE 在层中途提交一次，读回 router，
再继续）。GEMV 默认一行一个线程，点积顺序与 CPU 一致。
`ODINFER_VULKAN_PARALLEL=1` 选择 64 宽内核：设备支持 subgroup 算术且 subgroup
大小整除 64 时用 subgroup 求和，否则用共享内存归约。`ODINFER_VULKAN_EXACT=1`
强制精确内核；`ODINFER_VULKAN_EXACT=0` 强制并行内核。两个都设置时 `EXACT` 优先。

原生 GEMV 类型：F32、F16、Q4_0、Q4_1、Q5_0、Q5_1、Q8_0、Q2_K、Q3_K、Q4_K、
Q5_K、Q6_K、Q8_K。其余已支持的类型（IQ*）上传时反量化成 f32，再按 F32 计算。
Norm 保持 f32。设备上的 KV 是 f32。Prefill 一批最多 8 个 token。

逐元素内核：RMSNorm、L2 norm、部分 RoPE、gated RMSNorm、深度可分离因果卷积、
gated delta、注意力（每个 head 一个 workgroup）、SwiGLU、残差 axpy。稠密 MLP
和 qwen35moe（router 在主机上算，然后专家 GEMV，可选共享专家）都走这条路径。

整数点积（Q8_0 / Q4_0）和 cooperative matrix（16×16×16 f16，f32 累加）只在设备
提供对应特性时创建管线。它们由 `vulkan_kernel_test` 覆盖，不参与生成 token。
设备没有该特性，或管线创建失败，记为 skip，不是初始化失败。

`ODINFER_VULKAN_DEVICE` 是设备序号。不设置时，选用第一个非 CPU 且名字不是
llvmpipe 的设备，否则用 0 号。启动时把每个设备打到 stderr。

## 在 Windows 上编译

笔记本上已经装好 Odin **dev-2026-07**（与 `.github/workflows/release.yml` 的
钉版本相同），并且在 `PATH` 里。在仓库根目录：

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

`vulkan-1.dll` 在运行时用 `core:dynlib`（`LoadLibrary`）加载。不链接
`vulkan-1.lib`。exe 仍会导入 Odin Windows 运行时需要的系统库（`kernel32.dll`
等）。同一提交的 CPU 构建去掉 `-define:ODINFER_BACKEND=vulkan`，作为对照。

Git Bash 也可以用 `ODINFER_BACKEND=vulkan ./serve.sh 9748 4096`。脚本会带上
define 和 `--vulkan`。

`--vulkan` 或 `-g 1` 时初始化失败是硬错误。`METAL_FALLBACK=1` 恢复退回 CPU。

## 重新生成着色器

只在改了 `packages/qwen3_5/shaders` 之后。`./scripts/compile_spirv.sh` 需要能编译
`GL_EXT_integer_dot_product` 的 `glslangValidator`（用 glslang 16.6 测过；15.1
编不过整数点积着色器）。大多数着色器目标是 Vulkan 1.2；`q8_dp4a.comp` 和
`coopmat.comp` 目标是 Vulkan 1.3。改完 GLSL 后把 `.spv` 一起提交。

## 实际跑过的，和只编译过的

在 lavapipe 上跑过（Mesa 25.2.8，`llvmpipe (LLVM 20.1.2, 256 bits)`，API 1.4，
`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json`）。这个环境没有 AMD GPU，
也没有 Windows 机器。

`./scripts/vulkan_check.sh` 会构建工具并跑下面的检查。`bin/vulkan_smoke` 只编译
不运行：它需要一份真实的 Qwen3.5 GGUF。

内核自测，3×256 的 GEMV 对照 `ggml.dequant_row`：

| 内核 | 结果 |
|---|---|
| 精确 GEMV，13 种类型，T=1 和 T=2 | max_abs = 0 |
| 并行 / subgroup GEMV，最差值 | F32 T=2 subgroup 1.937151e-07；Q8_0 3.5762787e-07；Q4_0/Q5_0 T=2 7.4505806e-08。K 量化为 0 |
| rmsnorm、l2norm、rope、conv | max_abs = 0（conv 状态 0） |
| attn | 2.9802322e-08 |
| delta | 1.8626451e-09，状态 1.4901161e-08 |
| 整数点积对照整数参考 | Q8_0 和 Q4_0 max_abs = 0 |
| 整数点积对照 f32 点积 | Q8_0 0.0037726164，Q4_0 6.301701e-05 |
| cooperative matrix | skip（lavapipe 没有 coopmat） |

合成混合 GGUF（`tools/make_vulkan_gguf.py`，dim 256，2 层，先线性注意力再全注意力，
矩阵乘覆盖每一种原生量化）。贪心 prefill token `1 2 3 4` 再解码 4 步：

```
cpu: 1 2 3 4 43 58 45 58
vk:  1 2 3 4 43 58 45 58
```

精确模式 prefill logits max_abs = 0.00085449219，argmax 43/43。
`ODINFER_VULKAN_PARALLEL=1`（这台设备上是 subgroup 内核，subgroup 大小 8）
token 相同。并行 prefill logits max_abs = 0.00067138672，argmax 43/43。

合成 F32 qwen35moe（`tools/make_moe_gguf.py`，两层都是全注意力），提示 `3 5 7`
再解码 4 步：CPU 和 Vulkan 都是 `3 5 7 31 54 2 2`。

默认 `./build.sh`（CPU 后端，Vulkan 包只被 import）退出码 0。`packages/qwen3_5`
跑完 19 个测试，全部成功。tokenizer 测试和 `moe_load_test` 也在这个脚本里。

Windows：`odin check -target:windows_amd64 -define:ODINFER_BACKEND=vulkan` 对
`server/serve_harness/serve.odin`、`cli` 和 `tools/vulkan_kernel_test.odin`
通过。`odin build -build-mode:obj -target:windows_amd64` 生成了 COFF 目标文件。
`vkr` 目标文件里有字符串 `vulkan-1.dll`，没有 `vulkan-1.lib`。这份 Odin
（dev-2026-07）会打印 `Linking for cross compilation for this platform is not yet
supported (windows amd64)`，不写出 PE，所以没有在 Windows 上运行 exe。

## 笔记本上怎么验

1. 按上面的 [在 Windows 上编译](#在-windows-上编译) 编出四个程序。不需要 SDK。
2. `bin\vulkan_kernel_test.exe`
   stderr 应列出 Radeon（不能只有 CPU 设备），并以 `vulkan_selftest: PASS` 结束。
   如果列出多块设备，把 `ODINFER_VULKAN_DEVICE` 设成 Radeon 的序号再跑一次。
   `uma=true` 表示有缓冲落在 device-local 且 host-visible 的堆上，也就是共享显存
   那条路径。整数点积和 coopmat 那几行是可选的，`skip` 没问题。
3. 和同一提交的 CPU 构建对文本。有 Python 和 numpy 时：

   ```bat
   python tools\make_vulkan_gguf.py %TEMP%\odinfer_vulkan_tiny.gguf
   odin build tools\greedy_dump.odin -file -out:bin\greedy_cpu.exe -collection:ggml=packages -collection:infer=packages -collection:qwen3_5=packages -collection:qwen3_5_tokenizer=packages -collection:sampler=packages -collection:tokenizer=packages -collection:cuda=packages -collection:vulkan=packages
   odin build tools\greedy_dump.odin -file -out:bin\greedy_vk.exe -define:ODINFER_BACKEND=vulkan -collection:ggml=packages -collection:infer=packages -collection:qwen3_5=packages -collection:qwen3_5_tokenizer=packages -collection:sampler=packages -collection:tokenizer=packages -collection:cuda=packages -collection:vulkan=packages
   bin\greedy_cpu.exe %TEMP%\odinfer_vulkan_tiny.gguf 32 4 1 2 3 4
   bin\greedy_vk.exe %TEMP%\odinfer_vulkan_tiny.gguf 32 4 1 2 3 4
   ```

   两行 stdout 都应该是 `1 2 3 4 43 58 45 58`。然后 `set ODINFER_VULKAN_PARALLEL=1`，
   再跑一次 `greedy_vk`。
4. 真实模型。`bin\vulkan_smoke.exe <model.gguf> 32 2048` 会打印 prefill 和 decode
   的 tok/s，以及解码文本（温度 0，提示要求回答 2+2）。
   `bin\odin-serve.exe <model.gguf> --vulkan --max-ctx 4096` 和
   `bin\odin-cli.exe <model.gguf> -g 1 -c 2048` 是产品入口。去掉 define 再编一次
   `odin-cli`，用 `-g 0` 跑同一句提示，对照文本。
5. 在核显上计时：同一条 smoke 命令跑两遍，先默认精确 GEMV，再
   `set ODINFER_VULKAN_PARALLEL=1`。并行内核才会用 GPU 的 subgroup。把两行
   tok/s 都记下来。精确模式是正确性对照，每个输出行只有一个线程。
