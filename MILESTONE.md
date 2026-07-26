# Milestone 备忘(v1.0)— 待灌入 memecsy 记忆

> 写于 memecsy daemon 不可达期间。daemon 恢复后把本文按 subject 分条 `remember` 灌入。

## subject: odin-infer(项目)

- **是什么**:纯 Odin 写的 GGUF 推理引擎,Qwen3.5 / Ornith-1.0-9B,Apple Silicon Metal GPU 后端。
  混合注意力(8 full + 24 linear,gated delta net)。平台:Apple M3 Air / 16GB / Odin lang。
- **三仓库**(均已在 origin/main 打 tag `v1.0`):
  - `odin-infer`(库,GitHub vajraimb/odin-infer,HEAD 9328f37)— 引擎/MMQ/profiling。
  - `odin-infeer`(CLI,GitHub vajraimb/odin-infer-mac,HEAD 60f9a3d)— interactive chat;binary tracked。
  - `odin-infer-server`(HTTP,GitHub vajraimb/odin-infer-server,HEAD ffb6944)— Ollama 兼容;/api/chat、/api/generate、tool calling、prefix caching;binary gitignored(需 ./build.sh)。
- **模型文件不入 git**:`models/ornith-1.0-9b-Q4_K_M.gguf`(~5.6GB),dense(非 MoE)。

## subject: batched-prefill(v1.0 里程碑)

- **交付**:prefill **122 → 23 ms/tok(4.9×)**,贪心解码 **bit-identical**。
  端到端:1500-tok prefill + 500-tok 生成 ≈ 78s(原 ~6 分钟)。
- **怎么做的(分阶段,每个独立验证)**:
  - Stage 1a:MLP 投影(w1/w3/w2)批量 → Q4_K MMQ(1.42×)。
  - Stage 1b:全 attention 投影(wq/wk/wv/wo、in_qkv/in_z/in_b/in_a/out)也批量(2.5×)。
  - MMQ tile 调优:1-sg/tile → 4-sg/tg + sa/sb threadgroup staging + 反量化摊薄(1 scale/8 元素)→ 4.9×。Q4_K MMQ 335→928 GFLOPS(2.8×),bit-identical。
  - conv1d 批量(Stage 2 step1,causal 左 padding,bit-identical)。
- **新加 engine API**:`engine_forward_batch`(分块 ≤512 且 %8==0,尾部 <8 走 per-token)。

## subject: roofline(优化空间封顶)

- **prefill 15ms/tok = 投影 COMPUTE @ MMQ ~900 GFLOPS(dequant-ALU 墙)**。
  每 chunk(T=512)投影 FLOP = 7.085e12;÷900e9 = 15.4ms/tok(精确匹配实测)。
- **不是带宽墙**:带宽地板 0.11ms/tok 需 127 TFLOPS(GPU 仅 ~2.2TFLOPS f32)→ 物理不可能。
- **不是 dispatch**:`~1248 dispatch/chunk × 5µs = 0.012ms/tok`,可忽略。
- **不是逐 token 状态循环**:消融实测线性层 ~2ms/tok、全注意力 ~1-4ms/tok。
- **引擎 MMQ 已达 isolated benchmark 的 900 GFLOPS** → 无 dispatch 形态损耗,已在 kernel 物理上限。
- **decode ~91ms/tok 已在带宽墙**(每 token 读全 5.6GB → 56ms/tok 地板)。
- **fp16 路径死路**:half 操作数已用;全 half 累加 K=4096 精度崩;v1→v3(335→900)实证瓶颈是 dequant 不是 MAC。
- **现实封顶**:prefill 再 ~2×(15→~8-9ms/tok),只有"重写 MMQ dequant 往 1.5TFLOPS 推"一条硬路。tile sweep 是低成本的"若继续"第一步。
- **决策:收工于 v1.0**(端到端死磕极限仅 ~15% 改善,decode 已到墙,杠杆率极低)。

## subject: mmq(kernel 坑 + 事实)

- **2048 bug 根因 = K loop 步长写成 16 而非 8**(全 1 输入应得 K=4096,实际 2048)。`gemm_test.odin` 里 `gemm_f16_f32_kstep16` 变体复现 = 回归防线。
- **Q4_K_M 的 Q6_K 陷阱**:`ffn_down`(w2)、`attn_qkv`(in_qkv)、`wv`、`output` 是 **Q6_K**(其余 Q4_K)。一开始 MMQ 只写 Q4_K → 把 Q6_K 当 Q4_K 读 → 残差全 garbage → 输出 `!!!!!`。修:`gemm_q6k_f32` + `enc_mm` 按 weight kind 分派。
- **tuned MMQ kernel**:`gemm_q4k_f32`/`gemm_q6k_f32`,4 simdgroups/tg,BM=32×BN=8,sa/sb TG staging,1 scale/8 元素。要求 M%32、N%8、K%256。
- **Metal simdgroup 约定**:`make_filled_simdgroup_matrix<float,8>(0.0f)` 清零(非对角初始化);stride 是元素数非字节;`simdgroup_multiply_accumulate(D,A,B,C)` = D=A×B+C,half×half→float OK。

## subject: stage2(chunked delta — 低 ROI,挂起)

- **数学已验证**:affine 分解 `S_t = A_t S_{t-1} + B_t`(左乘),A_t=g_t(I−β_t k_t k_tᵀ),B_t=β_t k_t v_tᵀ。chunk 推导 = 下三角 δ 系统,**前向替换、不求逆、不形成 hkd×hkd**。
- **CPU 原型** extra_error_ratio ~3e-6(chunked vs tokenwise-f32);**GPU kernel** 对齐 CPU ~1e-7(C=16 含变 beta/g 全过)。harness 在 `delta_harness/`(delta_ref.odin + chunked_gpu.odin)。
- **engine 集成失败已回滚**:v1(S 留 global、per-v-col、per-chunk dispatch)FLOPs 比 per-token 还多、并行度差 → 不提速;且不 bit-identical。
- **C=16 悬案**:engine C=16 出 `!!!!!`,C≤8 正常;math 在 harness C=16 全过(含变 g)。`maxTotalThreadsPerThreadgroup=1024` 排除资源上限。**重启条件**:做 fused sequence-wise kernel 时第一件事用 `QFASTMATH=0` 跑 engine C=16 排查 fast-math;若否则查 host-side dispatch/buffer layout。
- **fused 版**(未来):sequence-wise,一个 threadgroup 走完一个 head 全部 chunk,S 驻留 threadgroup(V_TILE tiling 过 32KB),三角替换限 1 个 simdgroup 内串行。地基已就绪,设计空间被 v1 失败完全约束。

## subject: tooling(profiling,env 门控,默认关)

- `QFASTMATH=0`:CompileOptions 关 fast-math(precise 着色器)。
- `QTIMING=1`:forward_gpu_batch 打印 `GPUStartTime/GPUEndTime` 精确 GPU ms。
- `QPROF_NOLIN=1` / `QPROF_NOFULL=1`:消融跳过线性层/全注意力逐 token 循环(输出垃圾,GPU 时间有效)。
- 启动打印 `maxThreadgroupMemoryLength`(M3=32KB)与 quant summary(每个投影的 quant type)。
- harness:`gemm_harness/gemm_test.odin`(F16 GEMM + Q4_K/Q6_K MMQ + kstep16 诊断 + v1/v2/v3 对比,回归用)。

## subject: user(协作偏好)

- **数据驱动**:每个决策要数据/算术支撑,不接受直觉押注;喜欢 roofline/ablation 这类定性闭环。
- **三模型交叉验证**:常带 GPT-5.5/Claude Opus/Gemini 的分析来对照,要我接住并校正其中不准确处(如 simdgroup 竞态诊断不适用 v1 的 per-v-col 结构)。
- **要直白**:成绩和失败都要如实报(v1 不提速、C=16 撞鬼都直接说);不要 hype。
- **中文交流**;偏好简短、要点式、带表格/数字。
- **工程纪律**:bit-identical 作为硬验收(除数学上不可能的 chunked↔tokenwise,降级为语义等价);每阶段独立 commit + push;harness 作回归防线。
