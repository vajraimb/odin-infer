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
| tile sweep(真实 shape) | ~10-20%(900→1100) | ✅ 已交付并采纳进 engine:`gemm_harness/gemm_autotune.odin` + `metal.odin` 的 `gemm_q4k_f32_b64`(M%64 走 bm=64 bk=64,其余回退 v3)。Q4K isolated w1 874→1028(+18%);**engine 实测 prefill ~1.2×**(QTIMING 位置对齐 median 39.2→32.4 ms/tok)。Q6K b64 是 0.66× 回归已排除(w2 保持 v3 ~400 GFLOPS 上限);fm on/off 无差 |
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

## Continuation 续写缓存收益实测(2026-10-31,A/B 双二进制)

修复前(c2591c0,已含 b64 MMQ)vs 修复后(HEAD,continuation 路径),同一 5 轮中文对话,
贪心、num_predict=32、冷启动,server 端 prefill= 计时:

| 轮次 | 修复前 prefill | 修复后 prefill | 路径变化 |
|---|---|---|---|
| 1 | 5209 ms(含一次性快照保存) | 2314 ms | full prefill(同一路径,差值为噪声) |
| 2 | 2066 ms | 1122 ms | snapshot HIT → continuation hit |
| 3 | 3405 ms | 1587 ms | 同上 |
| 4 | 3808 ms | 1340 ms | 同上 |
| 5 | 5568 ms | 802 ms(**6.9×**) | 同上 |

- 修复前:每轮 prefill 随会话长度**线性增长**(整段历史从系统边界重算,O(n²) 累计);
  修复后:每轮只 prefill 新回合 ~15–25 token,**基本恒定**(O(n) 累计)。
- 轮 2–5 合计:14.8s → 4.9s(**3.0×**);会话越长收益越大,轮 10 外推 ~10s → ~1s。
- 计数器佐证:修复前 4 次 snapshot hit,修复后 4 次 continuation hit。
- 保真度:continuation 命中与冷全量 prefill 逐字节相同(test_prefix_bitidentical.py 多轮阶段)。

## 客户端契约:prompt 字节必须跨轮稳定(2026-10-31 晚,真实流量教训)

continuation 修复后实测 pie 真实会话仍 100% miss。服务端 QDUMP_DIVERGE=1 打点定位:
**pie 每轮把工具 schema 经 json.Value(map) 重序列化,Odin map 遍历序不稳定 → system 块
~600 token 处起字节全变**,快照与 continuation 双双失效,每轮全量重算 1100–1800 token
(76–218s,热节流加剧)。修复在 pie-odin 侧(Raw_Json 逐字节透传,commit 6e8cda1):
修复后同 prompt 二次请求 PREFIX CACHE HIT(restored 1119/1137,prefill 81s→3.2s,**25×**)。

**契约**:任何客户端只要改变 system 块/历史消息的**字节**(哪怕语义等价的 JSON 键序),
前缀缓存即失效。稳定序列化是客户端责任;服务端 `QDUMP_DIVERGE=1` 可打点定位漂移段。

## v1.0 Release Status(收工基线)
- **主线交付**:prefill 122→23 ms/tok(**4.9×**),贪心解码 **bit-identical**
  (`2+2=4`、`Paris`、1062-token 摘要逐字一致),三仓库(origin/main)同步。
- 端到端体感:1500-tok prefill + 500-tok 生成 ≈ 78s(原 ~6 分钟)。继续死磕 MMQ 极限
  仅换 ~15% 端到端(78→67s),且 decode 已在带宽墙 → **杠杆率极低,决定收工**。
- profiling 工具(`QFASTMATH`/`QTIMING`/`QPROF_*`)保留,**默认关闭**(env 门控)。
- **C=16 chunked-delta 悬案(已结案,2026-10-31)**:engine 在 C=16 出 `!!!!!`,C≤8 正常。
  用 `delta_harness/engine_chunked_gpu.odin`(engine 内核逐字 + engine 维度/布局 + CPU 逐行双胞)
  做了判决性实验:
  - **fast-math 排除**:良性输入下 C∈{8,16} × fast-math{on,off} 四组合全过
    (GPU vs CPU 双胞 max|dOut| ~5e-9),engine 内核本身无辜。
  - **真凶定位(机制级)**:对抗用例(cp 累积衰减下溢:单个 head 内两个 token 的
    g=exp(a_dec·softplus)≈1e-14 → cp[t] 下溢为字面 0 → `cp[t]/cp[i]` = 0/0 → NaN),
    CPU 双胞与 GPU 产生**完全一致的 NaN 模式**(各 19968 个)。这不是 fast-math,也不是
    dispatch,而是 chunked 公式本身的脆弱点——tokenwise 路径从不构造 cp 比率所以免疫;
    C 越大,一个 chunk 内含多个极小-g token 的概率越高,与"C≤8 正常、C=16 炸"的症状吻合。
  - **对 fused kernel 的硬性设计要求**:必须稳定化 cp 比率——clamp 累积衰减下界
    (cp ≥ 1e-20 量级)或改 log-space 比率(exp(cumsum 差)并 clamp),否则 fused 版会
    在真实激活上随机炸 NaN。
  chunked delta 整体只值 ~2ms/tok,非优先。
