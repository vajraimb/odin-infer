# CUDA 后端

[English](CUDA.md)

Qwen3.5 引擎的 Linux x86_64 后端。目标是 **sm_70 及以上**（按 Tesla V100S、
计算能力 7.0、驱动 CUDA 13.0 来写）。累加是 fp32，没有 bf16。预填充 GEMM
（`T>1`）可以用 Volta tensor core（WMMA 16×16×16，fp16 乘、fp32 累加）。
解码（`T==1`）走带宽 GEMV：128 位加载、`dp4a`，激活每条向量只量化成 q8
一次。短的 Q4_K / Q6_K 每个 warp 一行，更高的矩阵仍是四行。
`ODINFER_CUDA_GEMV=old` 回到标量内核。稠密解码默认是一张 CUDA graph，
`ODINFER_CUDA_GRAPH=0` 关掉。

macOS 构建不变，默认仍是 Metal。同一个二进制不会同时带上两条热路径。

## 构建

Odin 编译不需要 `nvcc`、CUDA toolkit 头文件，也不需要 GPU。`libcuda.so.1`
和 NVRTC 在运行时 `dlopen`。

```sh
# Linux CPU 构建（不链 Metal）。入口和 macOS 一样。
./build.sh

# CUDA 引擎
odin build server/serve_harness/serve.odin -file -out:bin/odin-serve \
  -o:speed -no-bounds-check -disable-assert \
  -define:ODINFER_BACKEND="cuda" \
  -collection:ggml=packages -collection:infer=packages \
  -collection:qwen3_5=packages -collection:qwen3_5_tokenizer=packages \
  -collection:sampler=packages -collection:tokenizer=packages \
  -collection:cuda=packages

bin/odin-serve model.gguf --cuda --max-ctx 2048
# CUDA 构建里 --metal 和 --cuda 都表示「用 GPU」。
```

`-define:ODINFER_BACKEND="cuda"` 要带引号。未设置时是 `"default"`：Darwin
上仍是 Metal，其它系统是 CPU。

开关在 `packages/qwen3_5/backend.odin`（`GPU_METAL` / `GPU_CUDA`）。前向、
KV 存取、复位都是 `when`，编译出来的热路径没有 Metal/CUDA 分支。

## NVRTC

`ODINFER_NVRTC_PATH` 是独占覆盖：指向 `libnvrtc.so.12`（或 `.13`）的完整路径，
或包含它的目录。设了它就不再搜索其它位置。没设覆盖时先找 `NVRTC_DIR`，然后
继续往下找。

否则加载器先试带版本的 soname，再试不带版本的名字（`libnvrtc.so.13`、`.12`、
`libnvrtc.so`），然后找 pip 的 `nvidia-cuda-nvrtc-cu12`
（`~/.local/lib/python3.*/site-packages/nvidia/cuda_nvrtc/lib`）。对应的
`libnvrtc-builtins.so.<主版本>.<次版本>`（wheel 里是
`libnvrtc-builtins.so.12.9`，没有 `.so.12`）用 `RTLD_GLOBAL` 预加载。
`.alt.` 那份会跳过。

CUDA 13 的 NVRTC 不能编译 `compute_70`。低于 sm_75 的架构被删了，CUDA 13
机器上不带版本的 `libnvrtc.so` 就是这个 13.x 库。13.x 的驱动仍然可以 JIT
NVRTC 12.9 产出的 PTX。设备是 sm_70 / sm_72，或者 `compute_70` 在 13.x 库上
编译失败时，运行时只加载 NVRTC 12.x，找不到就打印 pip 路径。NVRTC 12.9 会
警告 sm_75 之前的架构已弃用，在 V100 上这是预期的。

```sh
python3 -m pip install --user nvidia-cuda-nvrtc-cu12
export ODINFER_NVRTC_PATH="$HOME/.local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12"
```

每次编译都带上从设备算力得到的 `--gpu-architecture=compute_XX`。
`CUDA_DEVICE` 选设备序号，默认 0。

## Tensor core

预填充和其它批量 GEMM（`Y[T, rows] = X[T, cols] · W[rows, cols]`）把 `W`
反量化成 shared memory 里的 fp16 小块（64×32），`X` 收成 fp16，再发
`wmma.mma.sync.aligned.m16n16k16`（fp16 进、fp32 累加）。PTX 是内联的。
NVRTC 不带 `mma.h`。`mma.sync.m16n16k16` 不是 Volta 的指令；sm_70 能 JIT 的
是 WMMA 这条。不用 `ldmatrix`。

有 tensor core 内核的量化种类和 MMQ 相同：F32、F16、Q4_0、Q4_1、Q5_0、Q5_1、
Q8_0、Q2_K、Q3_K、Q4_K、Q5_K、Q6_K、Q8_K。9B Q4_K_M 的预填充是 Q4_K（up、
gate、attention）和 Q6_K（down）。0.8B MoE 里走批量 GEMM 的投影是 Q4_K、
Q8_0、Q5_0。专家 GEMV 仍是 GEMV。

`cuda_init` 时读 `ODINFER_CUDA_TC`：

| 值 | 行为 |
|----|------|
| 不设 | 算力 ≥ 7.0 且 `T ≥ 16` 时用 tensor core |
| `1` | 所有 `T > 1` 用 tensor core（`T == 1` 仍是 GEMV） |
| `0` | `T > 1` 用 MMQ，解码用 GEMV |

`T == 1` 永远是 GEMV。没有对应内核时退回 MMQ。

正确性（打印 `max_abs`、该元素的 tol、`atol`、`rtol`）和微基准：

```sh
bin/cuda_kernel_test
bin/cuda_kernel_test --bench
```

`--bench` 对 9B 的 w1/w3、wq、wo（Q4_K）和 w2（Q6_K），以及 0.8B 规模的
Q8_0、Q5_0，分别打印 tensor core 和 MMQ 的毫秒数和 TFLOPS（2·T·rows·cols）。

端到端预填充对比（同一条 prompt）。`prefill N tok` 里的 N 至少要 16，默认
策略才会走上 tensor core；`ODINFER_CUDA_TC=1` 则只要 N>1。

```sh
ODINFER_CUDA_TC=0 bin/cuda_smoke "$MODEL" 8 2048
bin/cuda_smoke "$MODEL" 8 2048
ODINFER_CUDA_TC=1 bin/cuda_smoke "$MODEL" 8 2048
```

这三次的解码 tok/s 应当一致。初始化日志会写 `tensor cores auto (T>=16)`、
`off` 或 `on for T>1`。

## 解码 GEMV

`T==1` 是访存 GEMV。标量内核（一行一个 warp、按字节加载）留在
`ODINFER_CUDA_GEMV=old`。带宽内核有两种形状。四行一个 warp、每行 8 个线程，
用于高矩阵（FFN up、27B 的 Q、logits）。这 8 个线程各发一次 `ld.global.v4`，
一行的量化数据是一次 128 字节的合并读，点积是 `dp4a.s32.s32`。Q6_K 的块是
210 字节，16 字节加载是两个对齐窗口的移位；权重分配多留 64 字节，sm_70 上
不会发出未对齐的 `v4`。Q6 按字节拆码，PTX 里没有 `vsub4`：NVRTC 会发出
`vsub4.s32.s32.s32`，sm_70 的 JIT 拒绝这条指令，整模块都加载失败。scale
留在寄存器里。

8192 行及以下用一行一个 warp（注意力输出、FFN down、9B 的 Q）。4096 行配
四行一个 warp 只有 1024 个 warp，V100S 上大约每个 SM 13 个，所以这些形状
只有 1134 GB/s 的大约 20%。一行一个 warp 是 4096 个 warp。总 warp 数是
`rows / rows_per_warp`，改每块的 warp 数并不改变它，所以没有加 split-K。
再发一次 kernel 填不满更多 SM，融合的 split 还要原子加。
`ODINFER_CUDA_GEMV_ROWS=1` 或 `=4` 强制 Q4_K / Q6_K 的形状。

16 MiB 以下的 Q8_0 和 Q5_0 留在标量内核。2048×2048 上新内核更慢（Q8_0
118.5 对 121.8 GB/s，Q5_0 49.2 对 69.7）。9B 和 27B 的热权重是 Q4_K 和
Q6_K，不走这条。`ODINFER_CUDA_GEMV=new` 让这些小形状也用新内核。

激活按 32 个一组量化成 q8（有符号 int8，scale 是 `amax/127`），每条向量一次。
Q4_K 和 Q5_K 另外记下每 16 个 float 的精确和，`dmin` 那一项不量化。单行
RMSNorm 在同一个内核里写出这份 q8；Q、K、V 和两路 FFN up 共用它。SwiGLU
和多行 norm 会使缓存失效。没有带宽内核的种类（F32、Q2_K、Q3_K、Q8_K、
Q4_0 这一族）仍走标量 GEMV。K 不是量化块的整数倍时也走标量。

块大小是 `ODINFER_CUDA_GEMV_WARPS` 个 warp（默认 8，范围 1..16）。四行时
每块 32 行；一行时每块的每个 warp 一行。V100S 有 80 个 SM、每个 SM 64 个
warp 槽。

稠密 `T==1` 时，注意力输出、FFN down 和线性层输出的 GEMV 直接加进残差，
这一步不再单独发 residual。预填充（`T>1`）和标量回退仍用单独的加。logits
的 GEMV 不累加。

| 变量 | 默认 | 作用 |
|------|------|------|
| `ODINFER_CUDA_GEMV` | auto | `old` 或 `0`：标量 GEMV。`new`：小 Q5_0 / Q8_0 也用带宽内核。不设：带宽内核，但 16 MiB 以下的 Q5_0 / Q8_0 用标量 |
| `ODINFER_CUDA_GEMV_ROWS` | auto | Q4_K / Q6_K 每个 warp 的行数。不设：`rows≤8192` 用 1，否则 4。`1` 或 `4` 强制 |
| `ODINFER_CUDA_GEMV_WARPS` | `8` | 带宽 GEMV 每块的 warp 数（1..16） |
| `ODINFER_CUDA_LAUNCHES` | 关 | `=1` 时每次 forward 打印内核启动次数 |
| `ODINFER_CUDA_PEAK_GBS` | `1134` | `--bench-gemv` 用来算峰值百分比 |
| `ODINFER_CUDA_GRAPH` | 开 | 稠密 `T==1` 是一张 CUDA graph。`=0` 每次都即时启动。MoE 和预填充仍即时。`ODINFER_CUDA_PROFILE=1` 会关掉 graph |
| `ODINFER_CUDA_PROFILE` | 关 | `=1` 打印这次 forward 每个内核的 GPU 时间，最慢的在前 |

`cuda_selftest` 用 CPU fp32 点积对四行内核做 Q4_K、Q6_K、Q8_0、Q5_0、Q5_K
的 64×512 和 6×256（行数不是 4 的倍数）。Q4_K 和 Q6_K 再跑一行内核，包括
`accumulate=1`（输出先写成 0.25，期望 `ref+0.25`）。容差是 `1 + 0.05*|ref|`。
和 CPU 的差来自 `x` 的 q8；半字节或 scale 对调会差到 `|ref|` 这一量级。

```sh
bin/cuda_kernel_test
bin/cuda_kernel_test --bench-gemv
ODINFER_CUDA_GEMV_WARPS=4 bin/cuda_kernel_test --bench-gemv
```

`--bench` 先打 tensor core 的 TFLOPS，再打同一张 GEMV 表。`--bench-gemv`
只打 GEMV：毫秒、GB/s、占峰值的百分比。四行内核是 `r4`，有一行内核的形状再打
`r1`，标量内核是 `old`。形状是 `T=1` 的 9B / 27B（w1、wq、wo、w2、logits）
以及 2048 的 Q8_0 / Q5_0 和 5120 的 Q5_K。
GB/s 只算这一次启动读的权重字节。`x` 的 q8 在计时循环外面，和缓存命中一样。
两行 logits 大约 1 GB，分配失败会打印并继续其余形状。设备不是 V100S 时用
`ODINFER_CUDA_PEAK_GBS` 改峰值。

端到端解码，标量内核对比默认。冒烟的 prompt 很短，这是短上下文的数（9B
上以前是 32.7 tok/s）。990 token 的预填充是另一次测量：那个长度上每个完整
注意力层的 KV 只有几 MB，权重大小仍然是整个模型。

```sh
ODINFER_CUDA_GEMV=old bin/cuda_smoke "$MODEL9B" 16 2048
bin/cuda_smoke "$MODEL9B" 16 2048
ODINFER_CUDA_GEMV=old bin/cuda_smoke "$MODEL27B" 16 2048
bin/cuda_smoke "$MODEL27B" 16 2048
ODINFER_CUDA_LAUNCHES=1 bin/cuda_smoke "$MODEL9B" 8 2048
```

初始化日志会写 GEMV 的行策略，以及 `graph on for dense T=1` 或 `gemv old`。

稠密解码（`T==1`，不是 MoE）捕获一次 CUDA graph，之后重放。捕获本身不跑
内核；第一个 token 付出捕获的代价，然后启动这张图。token 位置是设备符号
`g_decode_pos`，每次重放前写入，因为内核参数会在捕获时被写死。预填充传入
真实位置，并且保持即时启动，所以 graph 改不了 `T>1` 的结果。捕获或实例化
失败时，这个 token 改走即时路径，之后也不再捕获。`ODINFER_CUDA_GRAPH=0`
就是即时路径。`ODINFER_CUDA_PROFILE=1` 关掉 graph，并按从慢到快打印每个
内核的 GPU 时间（名字、次数、毫秒），以及墙钟和内核时间之和。采样仍在
CPU 上，在 logits 拷回之后。graph 和 event 符号是单独查找的；驱动里没有
它们时，graph 和 profiler 关掉，GEMV 照常加载。

## 检查脚本

`scripts/cuda_check.sh` 会编 CUDA 测试和冒烟二进制，把内核源编译成
`compute_70` 的 PTX（这一步不需要 GPU），机器上能看到 GPU 时再跑内核自测和
端到端生成。

```sh
./scripts/cuda_check.sh
```

模型按下面的顺序找，命中即停（`MODEL` 优先）：

| 路径 | 大小 | 覆盖 |
|------|------|------|
| `$MODEL` | 加载器能读的任意 Qwen3.5 GGUF | 你指定的文件 |
| `models/Qwen3.5-0.8B-MoE-A0.6B-Q4_K_M.gguf` | 524 MB | MoE，混合 Q4_K / Q8_0 / Q5_0 |
| `$HOME/.odinfer/models/Qwen3.5-0.8B-MoE-A0.6B-Q4_K_M.gguf` | 524 MB | 同上，安装器目录 |
| `models/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf` | Mac 基准，sha256 前缀 `a692de95` | 稠密 Q4_K_M（Q4_K + Q6_K + F32 norm） |

冒烟参数：`SMOKE_TOKENS` 默认 16，`SMOKE_CTX` 默认 2048。没有模型文件时内核
测试照跑，冒烟跳过，除非 `CUDA_CHECK_REQUIRE_SMOKE=1`。

脚本会打印 `sha256 <十六进制> <路径>`。Mac 上的 9B 基准文件以前缀
`a692de95` 开头。`mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-GGUF`
当前的 Hugging Face `main`（提交 `29319295d35e` 到 `a713868aa6cd`）是另一份
量化：sha256 前缀 `b4363b57`，5780092768 字节。不能把这个提交钉成基准，
因为 `a692de95` 已经不在该仓库现存历史里。heretic 9B 的哈希不是
`a692de95` 时，`cuda_check.sh` 拒绝冒烟，除非 `CUDA_CHECK_ALLOW_REQUANT=1`。
`scripts/install.sh --model qwen9b` 仍会拉 HF `main`，打印 sha256，并在
不是 `a692de95` 时给出警告。

脚本包住的命令：

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

`cuda_kernel_test` 把 GEMV/MMQ、RMSNorm、RoPE、SwiGLU、混合 V 的 KV 写入和
注意力、门控 delta 步，和 `packages/ggml` 与 `packages/qwen3_5/ops.odin` 的
CPU 参考对比。`cuda_smoke` 打印 prefill 和 decode 的 tok/s。采样仍用现有的
CPU sampler（冒烟工具温度是 0）。

## 已移植

GEMV（解码）和 MMQ（预填，最多 8 个 token 一块，任意 `T`，输出按 token
主序），fp32 累加。默认解码 GEMV 是 Q4_K、Q5_K、Q6_K、Q8_0、Q5_0 的带宽
内核；每种量化仍保留一行一个 warp 的标量内核：

F32、F16、Q8_0、Q4_0、Q4_1、Q5_0、Q5_1、Q2_K、Q3_K、Q4_K、Q5_K、Q6_K、Q8_K。

另外：embedding 用 CPU 反量化再上传，RMSNorm，按头的 Q RMSNorm，部分 RoPE，
解码和预填的注意力，f16 / 对称 Q8_0 / 40 字节混合 V 的 KV 写入和注意力，
注意力门，SwiGLU，残差，深度卷积 conv1d+SiLU，l2norm+缩放，逐 token 门控
delta，RMSNormGated，MoE softmax top-k、逐专家 GEMV，以及可选的共享专家。

默认的 9B Q4_K_M（Q4_K、Q6_K、F32 norm，embedding 常见 Q8_0 或 Q6_K）和
0.8B MoE（Q4_K、Q8_0、Q5_0、F32）都在这个集合里。初始化会扫每一层投影，
遇到名单外的量化种类就失败，并打出张量名。Embedding 可以是 `ggml` 在 CPU
上能反量化的任意种类，包括 IQ；输出投影不行。

## 未移植

IQ1_S、IQ1_M、IQ2_XXS、IQ2_XS、IQ2_S、IQ3_XXS、IQ3_S、IQ4_NL、IQ4_XS、Q8_1。

也不在这个后端里：Metal 按形状特化的 GEMV（`QGEMVSPEC`）、宽 N 自动调优和
tune 缓存、fast-math、MoE 分组 GEMM（`forward_gpu_batch_moe`）、
`engine_verify_batch`（投机校验）。CUDA 的单行 RMSNorm 会写下一次 GEMV 读的
q8 缓存。MoE 预填的注意力投影走 MMQ，专家走逐 token GEMV。稠密预填按
`CUDA_MAX_BATCH`（128，激活缓冲分配失败时更小）分块。解码用 GEMV，不用 MMQ。

权重在初始化时拷上设备一次（GGUF mmap 的 `cuMemcpyHtoD`，再加上主机上已经
反量化的 MoE router 和共享专家）。5.6 GB 的 Q4_K_M 9B 放得进 32 GB 的
V100S；KV 另算，布局和 Metal 的 `KV_Q8` / `KV_V` 相同，快照可以互换。

CUDA 里的 sigmoid 和 softplus 对齐 `ops.odin`（稳定 sigmoid，softplus 在
±20 截断）。Metal fast-math 的 `1/(1+exp(-x))` 可能差几个 ulp。内核测试对
的是 CPU 参考，不是 Metal 的 golden。GEMV 容差大约是
`2e-2 + 1e-3 * |ref|`，因为 warp 归约的求和顺序和 CPU 从左到右不一样。
