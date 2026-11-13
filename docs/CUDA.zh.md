# CUDA 后端

[English](CUDA.md)

Qwen3.5 引擎的 Linux x86_64 后端。目标是 **sm_70 及以上**（按 Tesla V100S、
计算能力 7.0、驱动 CUDA 13.0 来写）。累加是 fp32，没有 bf16。预填充 GEMM
（`T>1`）可以用 Volta tensor core（WMMA 16×16×16，fp16 乘、fp32 累加）。
解码（`T==1`）走带宽 GEMV：128 位加载、每个 warp 四行、`dp4a`，激活每条向量
只量化成 q8 一次。`ODINFER_CUDA_GEMV=old` 回到原来的一行一个 warp。

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
`ODINFER_CUDA_GEMV=old`。默认内核每个 warp 算四行，每行 8 个线程，每个线程
一次 `ld.global.v4`，一行的量化数据是一次 128 字节的合并读。点积是
`dp4a.s32.s32`。Q6_K 和 Q8_0 / Q5_0 的块不是 16 字节对齐的，加载前会看地址，
sm_70 上不会发出未对齐的 `v4`。

激活按 32 个一组量化成 q8（有符号 int8，scale 是 `amax/127`），每条向量一次。
Q4_K 和 Q5_K 另外记下每 16 个 float 的精确和，`dmin` 那一项不量化。单行
RMSNorm 在同一个内核里写出这份 q8；Q、K、V 和两路 FFN up 共用它。SwiGLU
和多行 norm 会使缓存失效。没有带宽内核的种类（F32、Q2_K、Q3_K、Q8_K、
Q4_0 这一族）仍走标量 GEMV。K 不是量化块的整数倍时也走标量。

块大小是 `ODINFER_CUDA_GEMV_WARPS` 个 warp（默认 8，范围 1..16）。8 个 warp
是 256 个线程、每块 32 行。V100S 有 80 个 SM、每个 SM 64 个 warp 槽，32
线程的块填不满。

| 变量 | 默认 | 作用 |
|------|------|------|
| `ODINFER_CUDA_GEMV` | new | `old` 或 `0` 用标量 GEMV |
| `ODINFER_CUDA_GEMV_WARPS` | `8` | 带宽 GEMV 每块的 warp 数（1..16） |
| `ODINFER_CUDA_LAUNCHES` | 关 | `=1` 时每次 forward 打印内核启动次数 |
| `ODINFER_CUDA_PEAK_GBS` | `1134` | `--bench-gemv` 用来算峰值百分比 |

`cuda_selftest` 用 CPU fp32 点积对 Q4_K、Q6_K、Q8_0、Q5_0、Q5_K 做 64×512
和 6×256（行数不是 4 的倍数）。容差是 `1 + 0.05*|ref|`。和 CPU 的差来自
`x` 的 q8；半字节或 scale 对调会差到 `|ref|` 这一量级。

```sh
bin/cuda_kernel_test
bin/cuda_kernel_test --bench-gemv
ODINFER_CUDA_GEMV_WARPS=4 bin/cuda_kernel_test --bench-gemv
```

`--bench` 先打 tensor core 的 TFLOPS，再打同一张 GEMV 表。`--bench-gemv`
只打 GEMV：毫秒、GB/s、占峰值的百分比，旧内核对比新内核，形状是 `T=1` 的
9B / 27B（w1、wq、wo、w2、logits）以及 2048 的 Q8_0 / Q5_0 和 5120 的 Q5_K。
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

初始化日志是 `gemv dp4a  warps=8` 或 `gemv old`。

没有用 CUDA graph。稠密 27B 一个 token 大约一千次启动，每次几微秒，一共
几毫秒。15.8 GB 的 Q4_K_M 测到 5.28 tok/s，大约 190 ms，几乎全是权重流，
大约是 1134 GB/s 的 7%。驱动绑定里没有 graph API。`ODINFER_CUDA_LAUNCHES=1`
是给 V100 上核对这个次数的。几千 token 的 KV 注意力搬不了这么多字节。

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
tune 缓存、fast-math、MoE 分组 GEMM（`forward_gpu_batch_moe`）、CUDA graph、
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
