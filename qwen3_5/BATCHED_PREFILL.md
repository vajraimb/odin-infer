# Task: Batched Prefill (simdgroup GEMM) — 根治 agent/tool 慢

**状态:** 进行中 · Stage 1a kernel 性能已验证但正确性未通过(阻塞)
**优先级:** 高 — 这是让 Ornith 接 agent/工具调用的关键瓶颈
**最后更新:** 本轮会话

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

## 阻塞点:simdgroup_load 的布局约定

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

1. 重建验证 harness(见上),写 `gemm_f16_f32`,目标是 max_err<1e-2 + 全 1 得 K。
2. 找正确裸 simdgroup 示例对齐布局约定(最可能卡在这一步)。
3. 通过后 → 接 Q4_K 在线反量化 → Stage 1a 接进 `qwen3_5/metal.odin` 的投影层。
