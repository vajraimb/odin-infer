# Roofline / Profiling Notes (数据驱动的优化空间封顶)

记录对本引擎(Ornith-1.0-9B Q4_K_M,Apple M3 Air)prefill/decode 的 roofline 分析,
用于决定"继续优化 vs 收工"。结论:**prefill 已在 MMQ 算力墙上,decode 已在带宽墙上;
剩余空间封顶 ~2×,且只有"死磕 MMQ dequant"一条硬路。**

## 关键事实(实测 + 算术)

模型 **dense**(非 MoE,单 MLP/层),~9B 参数,Q4_K_M ~5.6GB。M3 Air 统一内存 ~100GB/s,
GPU f32 ~2.15 TFLOPS / fp16 ~4.3 TFLOPS。

### Prefill(批量,Stage 1b + tuned MMQ,当前 ~15-22 ms/tok)
- 每 chunk(T=512)投影 FLOP = **7.085e12**(dense,全参数激活)。
- 实测 MMQ 吞吐 **~900 GFLOPS**(isolated benchmark,引擎实测匹配)。
- 7.085e12 / 900e9 = **15.4 ms/tok** ← 精确匹配实测 15-20ms/tok。
- **= 投影 COMPUTE @ MMQ 算力墙**(dequant-ALU 受限)。不是带宽(带宽地板 0.11ms/tok
  需 127 TFLOPS,物理不可能),不是 dispatch(~1248 dispatch/chunk × 5µs = 0.012ms/tok,
  可忽略),不是逐 token 状态循环(消融实测线性层 ~2ms/tok、全注意力 ~1-4ms/tok)。
- **引擎 MMQ 已达 isolated benchmark 的 900 GFLOPS** → 没有"engine dispatch 形态效率损失",
  已在 kernel 物理上限。继续推只能重写 dequant kernel 本身,不是改 dispatch。

### Decode(逐 token 生成,~91 ms/tok)
- 每 token 读全 5.6GB 权重 → 5.6e9/100e9 = **56ms/tok 带宽地板**。
- 实测 ~91ms/tok → ~60% 带宽效率,**已接近带宽墙**,没什么可优化。

## 优化空间封顶

| 路径 | 收益 | 状态 |
|---|---|---|
| Stage 1b 批量投影 + tuned MMQ | prefill 122→23 ms/tok(**4.9×**) | ✅ 已交付 |
| Stage 2 chunked delta(线性层递推) | ~2 ms/tok | ❌ 数据否决(低 ROI),kernel 验证过但未接 |
| MMQ dequant/staging 往 1.5 TFLOPS 推 | prefill 15→~9-12 ms/tok(~1.5×) | 硬工程,唯一真杠杆,收益封顶 ~2× |
| tile sweep(真实 shape) | ~10-20%(900→1100) | 低成本,"若继续"的第一步 |
| fp16 累加路径 | 死路 | ❌ half 操作数已用;全 half 累加 K=4096 精度崩;dequant 是瓶颈非 MAC |
| 全注意力批量(Stage 1c) | ~1-4 ms/tok | 低 ROI(全注意力只 8 层) |
| Flash attention | ~5% 以下 | 跳过(Rigel:分解式优于融合) |

**prefill 现实极限 ~8-9 ms/tok**(MMQ 推到 ~1.8 TFLOPS),decode 已到墙。整个项目优化空间
已封顶:从当前 4.9× 出发,最多再 ~2×(prefill),且需重写 dequant kernel。

## Profiling 工具(env 控制,默认行为不变)
- `QFASTMATH=0`:CompileOptions 关 fast-math(precise 着色器,f32 漂移排查)。
- `QTIMING=1`:forward_gpu_batch 打印 `GPUStartTime/GPUEndTime` 精确 GPU ms。
- `QPROF_NOLIN=1` / `QPROF_NOFULL=1`:消融跳过线性层/全注意力逐 token 状态循环(输出垃圾,
  GPU 时间有效),用于归因。
- 启动打印 `maxThreadgroupMemoryLength`(M3=32KB,fused kernel 的硬预算)。

## 决策
4.9× prefill + bit-identical + 完整 profiling 工具链 + roofline 封顶 = 优化项目闭环完成。
是否继续推 MMQ 到 ~1.5 TFLOPS(再 ~2×、硬工程)是一个**有数据支撑的收工/继续决策**,
非直觉押注。
