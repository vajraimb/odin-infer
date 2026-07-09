# Task: Batched Prefill (simdgroup GEMM) — 根治 agent/tool 慢

**状态:** Stage 1b + MMQ 调优 ✅ — prefill 4.9×;有状态部分仍逐 token
**优先级:** 高 — 这是让 Ornith 接 agent/工具调用的关键瓶颈
**最后更新:** 本轮会话(MMQ tile 调优,prefill 49s → 23.4s)

## ✅ 本轮成果(Stage 1a kernel 正确性已解锁)

独立验证 harness:`qwen3_5/gemm_test.odin`(`odin run gemm_test.odin -file -o:speed`,
无需 GGUF)。全部 8 项正确性测试通过 + 性能达标:

```
[OK] Step1: 8x8 all-ones -> 8              max_abs=0
[OK] Step2: 16x16 all-ones -> 16 (K loop)  max_abs=0
[OK] Step3a: identity(A) x seq(B) -> seq   (A 未转置)
[OK] Step3b: row-distinct(A)               (A 未转置)
[OK] Step3c: col-distinct(B)               (B 未转置)
[OK] Step4: 4096x4096 x 4096x64 all-ones -> 4096   max_abs=0   ← 旧 bug 是 2048
[OK] random 32x32x32 vs CPU ref            max_abs=1.5e-6
[OK] random 64x64x256 vs CPU ref           max_abs=9.8e-6
[PERF] prefill projection shape  best=3.44 ms   624 GFLOPS   ← 验收 ≥300 GFLOPS
[PERF] square 4096x4096x4096     best=205 ms    671 GFLOPS
```

### 2048 bug 根因 = K loop 步长写成了 16(已确诊并复现)

用 `gemm_f16_f32_kstep16`(除 `k += 16` 外与正确 kernel 完全相同)在 4096³×64 全 1 上
跑,**C[0] 恰好 = 2048** —— 精确复现旧 bug。`k += 8` 给 4096。由此锁定:

| 候选假设 | 判定 | 依据 |
|---|---|---|
| **K loop 步长 = 16 而非 8** | ✅ **确诊根因** | 每次 MAC 覆盖 8 个 K 元素,`k+=16` 跳过一半切片 → 全 1 下恰好 K/2=2048。k16 变体实测复现。 |
| stride 传了字节数而非元素数 | ❌ 排除 | 全 1 输入对 stride 不敏感(跳行读到的还是 1),应仍得 4096,不会是 2048。 |
| origin 写成 (row,col) 而非 (col,row) | ❌ 排除 | 全 1 输入对转置不敏感,应仍得 4096(边界 tile 才会越界读垃圾)。 |
| accumulator 用对角初始化(`acc(0)`)而非清零 | ⚠️ 次要来源 | 解释「或 NaN」的那一半症状,非 2048 的主因。已改用 `make_filled_simdgroup_matrix<float,8>(0.0f)`。 |

### 正确的布局约定(写进 metal 集成时照抄)

经 llama.cpp `kernel_mul_mm`(legacy raw-simdgroup 分支)逐行对齐确认:

```c
simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);  // 必须显式清零
for (uint k = 0; k < K; k += 8) {                                       // ← 步长是 8 不是 16
    simdgroup_load(a_tile, A + (ulong)row0*K + k, K);   // stride = K 元素数, 非 K*sizeof
    simdgroup_load(b_tile, B + (ulong)k*N    + col0, N); // stride = N 元素数
    simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc); // acc += a*b (half*half->float OK)
}
simdgroup_store(acc, C + (ulong)row0*N + col0, N);
```

- `simdgroup_load/store` 的 stride 是**元素数**(不是字节);origin 烘焙进指针,用 3-arg 形式避开 (col,row) 之争。
- `simdgroup_multiply_accumulate(D, A, B, C)` = `D = A×B + C`,half×half→float 累加是支持的(llama.cpp 也这么用)。
- 一个 threadgroup = 一个 simdgroup(32 线程)算一个 8×8 输出 tile。624 GFLOPS 已达标,先不急着上多-simdgroup-per-tg 的 tiling。

## 为什么做这个

实测:agent 调用工具时,工具结果(几百~几千 token)要 prefill,而我们的引擎是**逐 token GEMV**(一次一个 token),prefill 速度 ≈ 生成速度(~11 tok/s)。一个 1500 token 的工具结果 = 几分钟。llama.cpp 用**矩阵×矩阵**(一次一批 token)所以 prefill 快 10–100×。

这是接 `pie`/agent 慢的根因,也是 prefix caching 在工具调用上失效的根源(qwen3_5 线性状态无法回滚,只能精确延伸才续算)。

## 已验证的事实(别重新发现)

1. **性能潜力是真的**:`gemm_f16_f32` kernel(裸 simdgroup 8×8 tile)在 4096×4096 × 4096×64 上跑到 **536 GFLOPS / 4ms**。对比逐 token GEMV,prefill 潜在 ~50× 提速。simdgroup 8×8 tile 这条路方向正确。
2. **kernel 编译通过、跑得快,但输出错**(全 1 输入应得 K=4096,实际 2048 或 NaN)—— 对 simdgroup matrix 的**布局约定**用错了。
3. **llama.cpp 现役代码**已不用裸 simdgroup API,改用更高层的 `mpp::tensor_ops::matmul2d`(tensor + cooperative_tensor)。但**旧的裸 simdgroup 模式仍可用且更简单**,建议从它入手。

## 关键技术坑(都已踩过,记下来)

- **类型**:`simdgroup_float8x8`(typedef,不是模板 `simdgroup<float,8,8>`),`simdgroup_half8x8`。
- **函数**:`simdgroup_load(mat, src, srcStride)`、`simdgroup_multiply(d, a, b)`(d=a*b,不累加)、`simdgroup_multiply_accumulate(d, a, b, c)`(d=a*b+c)、`simdgroup_store`。
- **`simdgroup_float8x8 acc(0)` 是对角线初始化**(只置对角,非对角是垃圾)→ 必须**首步用 `simdgroup_multiply`,后续才 `simdgroup_multiply_accumulate`**,或先从零 buffer load。
- **kernel 参数不能混 `uint2`/`uint`**("all scalar or all vector")→ 统一用 `uint3 tgpig [[threadgroup_position_in_grid]]` + `uint3 tidg [[thread_position_in_threadgroup]]`(或都 uint)。simdgroup index = `tidg.x / 32`。
- **Odin Metal 绑定**:`dev->newBufferWithBytes(bytes: []byte, options)`(切片自带长度,2 参数);`ob->contentsAsSlice([]f32)` 取回数据;`MTL.ResourceStorageModeShared`。

## ~~阻塞点:simdgroup_load 的布局约定~~ ✅ 已解决(见文首)

> 原始记录(保留供回溯):

我的用法:
```c
simdgroup_load(wt, W + r*K + k, K);   // 想取 W[r:r+8, k:k+8]
simdgroup_load(xt, X + k*N + c, N);   // 想取 X[k:k+8, c:c+8]
simdgroup_multiply(acc, wt, xt);       // 想 acc = W_tile @ X_tile
```
输出错(2048=一半 / NaN)。可能的原因(待逐个排查):
- `srcStride` 的语义(行间距?是否要对某个操作数 transpose?)
- half8x8 的 load 是否只填了一半元素(2048=一半 提示这点)
- 线程→元素的映射(每个 thread 持有哪 2 个元素)没对齐 multiply 的约定
- `simdgroup_load` 可能需要第 4 参数 `srcOffset`(`ulong2`)或 transpose 标志

**解决路径(推荐)**:找一个**已知正确的最小 simdgroup GEMM 示例**,逐行对齐 load/multiply/store 的约定,而不是猜。来源:
- llama.cpp master 之前的裸 simdgroup `kernel_mul_mm`(git 历史,在引入 `mpp::tensor_ops` 之前)— 直接抄其 load/stride 写法
- Apple 官方 "Metal simdgroup matrix" 示例代码
- ggml 早期 metal 实现

## 验收标准(Stage 1a 的"完成"定义)

1. `gemm_f16_f32` kernel:对随机 F16 输入,与 CPU 参考逐元素对比 **max_err < 1e-2**;全 1 输入时 out[m,n] **恰好 = K**。
2. 速度 ≥ 300 GFLOPS(4096 量级)。
3. 一个独立测试程序复现(验证 harness 见下)。

验证 harness 结构(本轮写过,清掉了,按此重建):
- Odin 程序:建 Metal device、编译 MSL 串、分配 W[M,K]/X[K,N]/out[M,N] 的 shared buffer(全 1 或随机)、dispatch、读回、对比 CPU 参考、测速。
- 起始尺寸:M=K=4096, N=64(模拟 64 token prefill 的一个投影)。

## 分阶段集成计划(每阶段独立交付)

- **Stage 1a — MLP 批量投影(gate/up/down)**
  - 先 F16 GEMM 验证正确,再加 **Q4_K 在线反量化**(把量化块在线解进 8×8 half tile —— 这是 Stage 1a 里最硬的部分,对照 llama.cpp 的 mmq dequant 写法)
  - T pad 到 8 的倍数(Odin 侧);T < 阈值(如 < 8)走旧 GEMV
  - 验证:单层 batch 输出 vs 逐 token GEMV 参考逐元素一致
- **Stage 1b — QKV/O 批量投影**(同一个 GEMM kernel);attention 仍逐 token
- **Stage 1c — 全注意力层批量 attention + causal mask**(此时全注意力层完整 batch 化)
- **Stage 1d — 线性注意力层只 batch 投影;delta 递推不动**(递推仍逐 token,占算力小)

出真实 prefill tok/s 数字 → 决定继续优化 Stage 1 还是跳 Stage 2(chunked delta rule)。

## Stage 2 预告(之后的硬骨头)

chunked gated delta rule:让线性注意力层也能批量 prefill + 状态可从前缀重建(解锁 qwen3_5 在任意 L 续算的 prefix caching,根治工具调用全量 reset)。算法见 transformers `torch_chunk_gated_delta_rule`。

## 接续入口(下次直接从这里开始)

**Stage 1a 已端到端接通并实测有提速**(本轮)。新增:
- `metal.odin`:`gemm_q4k_f32` + `gemm_q6k_f32` MMQ kernel、`cast_f32_f16`/`copy_f32`/
  `transpose_to_f16`/`transpose_f32` 辅助 kernel、`forward_gpu_batch`(T 个 token 一批,
  Phase A 逐 token attention + Phase B 批量 MLP)、`enc_mm`(按 weight kind 选 Q4_K/Q6_K)。
- `engine.odin`:`engine_forward_batch`(分块 ≤512 且 %8==0,尾部 <8 走 per-token)+ `engine_batch_max`。
- `main.odin`(CLI 仓 `odin-infeer`):`chat_q35` prefill 调 `engine_forward_batch` 处理前 N-1
  个 prompt token,最后一个 token 走原循环以自然吐出第一个生成 token。
- 验证 harness 移到 `gemm_harness/gemm_test.odin`(独立 package,不污染 qwen3_5 集合)。

### 实测(Ornith-1.0-9B Q4_K_M,Apple M3,1062-token prompt)

| 路径 | TTFT | ms/token | vs 逐token |
|---|---|---|---|
| 逐 token GEMV(旧) | 122.1 s | 115 | 1.0× |
| Stage 1a 批量 MLP | 86.3 s | 81 | 1.42× |
| Stage 1b + 批量 attention 投影 | 48.8 s | 46 | 2.5× |
| **Stage 1b + MMQ tile 调优** | **23.4 s** | **22** | **4.9×** |

**prefill 4.9× 提速**(超目标 25–35s),全程贪心解码输出**逐位一致**
(`2+2 → 4`、`capital of France → Paris`、1062-token 长文摘要完全相同)。
1500-token 工具结果 prefill 从 ~3 分钟降到 ~33 秒。

### MMQ tile 调优(本轮,零风险)

把 Q4_K/Q6_K MMQ 从 1-simdgroup/8×8-tile 升级为 **4 simdgroups/tg + BM=32×BN=8 tile
+ sa/sb threadgroup staging + 反量化摊薄**:
- 关键洞察:一个 8 元素 chunk 的 8 个 K 列必落在同一 Q4_K/Q6_K super-block 的同一 half
  → **共享同一个 scale**,反量化时每个 chunk 只算一次 scale(原来逐元素算,~32× 浪费)。
- 每个线程读所在行的 block 一次,反量化 8 个元素,写进 sa;4 个 simdgroup 共享 sa/sb。
- 数学完全相同 → **v3 与 v1 输出逐位相同**(harness 实测 v3v1_diff=0.00e+00)。
- kernel GFLOPS:Q4_K 335→928(2.8×,超过 F16 的 624,因 Q4_K 只读 1/4 字节);Q6_K 326→460(1.4×)。

### Stage 1b 怎么做的(本轮)

每层把**纯投影矩阵乘**全部批量成 MMQ,有状态的部分逐 token:
- **全注意力层**:批量 `wq/wk/wv`(pre)→ 逐 token qnorm/RoPE/store_kv/attention/gate → 批量 `wo`(post)。
- **线性注意力层**:批量 `in_qkv/in_z/in_b/in_a`(pre)→ 逐 token conv1d/l2norm/delta/gated-rmsnorm → 批量 `out`(post)。
- 新增 `enc_proj_fwd`(transpose→MMQ→transpose,token-major 进出)+ 一组 batch buffer。
- Q4_K_M 的 **`in_qkv`/`wv` 是 Q6_K**(其余投影 Q4_K),`enc_mm` 按 weight kind 分派,无需新 kernel。
- 新增启动 quant-summary 打印(防 Q6_K 类 bug:`metal: quant (linear) ... qkv=Q6_K ...`)。

### 剩余瓶颈(为什么不是 50×,以及下一步)

MMQ 调优已把投影层榨干(4.9×)。逐 token 只剩**有状态部分**:全注意力的 score/softmax/value
计算,线性注意力的 conv1d+delta 递推。按 ROI 排序的下一步:

1. **Stage 2 — chunked gated delta rule**(最大杠杆,第二优先)。把线性层的 O(T) 串行递推
   转成 chunk 内并行 GEMM + chunk 间少量状态传递。分步降险:先单独批量 conv1d(低风险 causal
   local op)→ 建 float64 CPU reference 对拍 → 最后才接 chunked scan。推导 WY 展开 + 处理数值
   稳定性,silent-error 风险最高。
2. **分解式批量 attention**(可选,排最后)。全注意力 score 只占 ~5%,且 Rigel 论文实测:Apple
   统一内存上分解式路径(simdgroup GEMM + device-memory S×S)反而比融合 flash 快数倍——融合
   softmax 会把计算赶离矩阵单元。如未来 profiler 证明 attn core 占比高再做,且走分解式而非全融合。

### 已知约束(别踩)
- `forward_gpu_batch` 要求 `T % 8 == 0` 且 `T ≤ MAX_BATCH_T(512)`;`engine_forward_batch` 自动分块。
- 调优后的 MMQ 要求 `M % 32`(本模型所有投影 out_dim 都满足:n_vh=32、kv_dim=1024、dim/hidden/conv_dim/qproj);新架构务必核验。
- 新架构务必确认每种投影权重的 quant type 都有对应 MMQ(看启动 quant-summary);当前覆盖 Q4_K/Q6_K。
- 改 kernel 后先跑 `gemm_harness/gemm_test.odin` 回归(F16 GEMM + Q4_K/Q6_K MMQ 正确性 + v1/v3 逐位对比)。

【关键避坑】K loop 步长写 `+= 8`(不是 16);accumulator 用 `make_filled_simdgroup_matrix<float,8>(0.0f)`。
`gemm_test.odin` 里的 k16 变体是回归防线。
