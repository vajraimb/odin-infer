# Prefix State Snapshot 计划（Step 0–4）

> 来源：NNUE → odin-infer 迁移讨论（稀疏状态检查点 / Finny Tables 思想在 GDN 递归状态上的对应）。
> 原则：不加速重算，**消除重算**。这是唯一不在 roofline 封顶表上的杠杆（prefill 15 ms/tok 是 dequant-ALU 物理墙，snapshot 不碰 kernel）。
> **状态：Step 0–3 已闭环（2026-10-31），结案总结见 `WRAPUP_PREFIX_SNAPSHOT.md`。**

## 背景数字

- prefill 23 ms/tok（v1.0 实测）；1500-token 固定前缀 ≈ 34.5 s TTFT。
- GDN 递归状态 S：24 linear 层 × 32 v-heads × 128(k) × 128(v) × f32 ≈ **48 MB / 快照**。
- conv1d 状态：depthwise causal conv，kernel=4，每层仅 (kernel-1) × conv_dim，KB 级，**但必须进快照**，漏了就是 silent corruption。
- 快照三要素：**S + conv state + KV prefix**（KV 已有 `kv_persist.odin` 模式可抄）。
- 恢复语义：快照只读，恢复时**拷贝不共享**，避免跨请求污染。

## Step 0 — Hit-rate 埋点 + 快照可逆性验证（½–1 天）

- 0a. server 层统计最近 N 个请求的 prompt 前缀重合长度分布（纯日志，不改引擎）。
- 0b. harness 级验证：跑到 chunk 边界（%8==0）dump S + conv state → 重新载入 → 继续跑，与一口气跑完 **bit-identical**。

**闸门**：0b 不 bit-identical → 停，查状态完整性（首选怀疑 conv state / 归一化隐藏缓冲）。
0a hit rate < 50% → 整个计划止步，损失半天。

## Step 1 — Prefix state snapshot，内存版（2–3 天）

- `engine_forward_batch` chunk 边界导出 S + conv state + KV prefix 指针，按 prompt hash 索引。
- 命中时跳过固定前缀，只对增量后缀 prefill。单份常驻快照（48 MB），不做多 checkpoint。

**期望**：1500-token 前缀 prefill 34.5 s → ~0.05 s；端到端 78 s → ~48 s；TTFT ~35 s → 2–3 s。
fallback 路径 = 现状，无回归风险。

**闸门**：命中 vs 非命中输出 bit-identical；连续 100 请求无状态污染。

## Step 2 — 快照落盘（1 天，可选）✅ 已交付（随 Step 1 一同 ship），验收已加固（2026-10-31）

- 照 `kv_persist.odin` 模式写盘，server 重启预热即恢复。磁盘读 48 MB < 0.1 s。
- 不做多版本管理：前缀变了整体失效重存。
- ✅ 落盘实现早已就位：`prefix.oikv` + `prefix.tokens` + `session.oikv` + `session.tokens`，
  重启恢复（CRC + model fingerprint + seq_len 校验），首个请求即 PREFIX CACHE HIT。
- ✅ 验收加固（2026-10-31）：`serve_harness/test_kv_persist.py` 从"回复是干净中文"
  升级为 **bit-identical 判据**——冷、暖两轮发同一首轮 prompt（temperature=0 贪心），
  断言两次回复逐字节相同（f16↔f32 落盘往返无损的强判据）。实测 PASS，
  暖启动 prefill=1295ms。

## Step 3 — Autotune 基建（2 天，可与 1/2 并行）✅ 已交付（2026-10-31）

- ✅ `gemm_harness/gemm_autotune.odin`：Odin 端程序化生成 MSL（BM/BK/NT/TPR/RSTEP 编成常量），
  每候选先过正确性门（随机量化块 + `ggml.dequant_row` + CPU f64 参考，gate 256×64×512，tol 5e-3），
  再 warmup 3 + median-of-7 计时；结果追加缓存 `gemm_harness/autotune_cache.jsonl`，命中即跳过。
- 扫描网格：BM ∈ {16,32,64} × BK ∈ {16,32,64} × fastmath ∈ {on,off} = 18 配置 × (Q4K+Q6K) = 36 配置，全过门。
- **实测结果（M3 Air，median-of-7）**：
  - Q4K w1 4096×64×4096：v3 基线 874 → **bm=64 bk=64 = 1028 GFLOPS（+18%）**
  - Q4K hidden_x_T 12288×64×4096：933 → **1052 GFLOPS（+13%）**
  - Q6K w2 4096×64×12288：~400 GFLOPS 持平，bk=32 必需，bm 不敏感 → v3 已在平台上限
  - fast-math on/off 全部 <2% 噪声 → 无精度/性能权衡
- 附带战果：正确性门第一跑抓到泛化 bug（BK≠32 时 qs 索引应用组内列 `e_base&31` 而非 `col_base`；
  BK=32 时两者恰相等，v3 从未暴露）。
- ✅ C=16 悬案已先于本步结案（`delta_harness/engine_chunked_gpu.odin`，cp 比率下溢，见 ROOFLINE.md）。
- ✅ 已采纳进 engine（2026-10-31）：`metal.odin` 新增 `gemm_q4k_f32_b64`，M%64==0 走 bm=64 bk=64
  （in_a/in_b 的 M=n_vh=32 回退 v3）；**Q6K 不走 b64**（w2 实测 0.66× 回归，v3 保持）。
  验证：`test_prefix_bitidentical.py` PASS + `serve_harness/ab_b64_check.py`（新旧二进制 A/B，
  贪心回复逐字节相同，QTIMING 位置对齐 4 chunk median 39.2→32.4 ms/tok，**prefill ~1.2×**）。

**期望复盘**：预估 0–10%，实测 Q4K +13–18%，超预期；Q6K 验证了"已在上限"。

## Step 4 — Stage 2 fused kernel（1–2 周，最后做）

- sequence-wise fused kernel（一 threadgroup 一 head 全 chunk，S 驻 threadgroup，V_TILE tiling 过 32 KB）。
- **诚实预期**：端到端 prefill ~4–5%。Step 1 落地后相对价值进一步下降，按需重启。

## 总账

Step 0–2 约一周拿走几乎全部收益；Step 3–4 是长尾。Step 0 不过关则整体止损。

---

## 现状盘点（2026-08-06，开工时发现）

**Step 1 / Step 2 已 ship**，比计划乐观：

- `qwen3_5/engine.odin`：`engine_save_full_state` / `engine_restore_full_state`（内存快照，含 KV + conv + recurrent，O(1) memcpy 恢复）。
- `qwen3_5/kv_persist.odin`：`engine_save_kv` / `engine_load_kv` 落盘，conv_states + recurrent_states 已在格式内（Metal 路径含 f16↔f32 布局转换）。
- `serve_harness/serve.odin`：系统提示词边界（首个 `<|im_end|>` + "\n"）快照 → 命中时 restore + 只 prefill 后缀；`prefix.oikv` + `prefix.tokens` 落盘，重启后首个请求即 HIT。
- `serve_harness/test_prefix_cache.py`：快照错位 bug（suffix 在已推进的 recurrent state 上重放 → 24/32 层污染）的回归测试。
- `serve_harness/test_kv_persist.py`：重启恢复回归，已升级为 bit-identical 判据（冷/暖两轮贪心输出逐字节比对，2026-10-31）。

**Continuation 修复（2026-10-31，hit-rate 实测驱动）**：hit-rate 计数器上线后实测发现
`pc_continuation_hits` 恒为 0——多轮续写路径在 chat 场景是死代码。三个叠加原因，全部修复：
1. 模板不对称：生成 prompt 以 `<|im_start|>assistant\n<think>\n\n</think>\n` 结尾，但历史渲染
   assistant 回合时**不带空 think 桩** → token 流在 `assistant\n` 后必然分叉。修复：历史渲染补同款
   空 think 桩（全量重算与续算看到同一 token 流，保住 bit-identical）。
2. pos 语义差一：`g_state.pos` 是"最后转发 token 的下标"（= 已转发数 − 1），continuation 条件
   写成 `pos == common` 永假。改为 `pos == common - 1`。
3. 快照检查顺序截胡：快照匹配跑在 continuation 前面，多轮请求总是先命中系统边界快照（25 tokens）
   而非整段会话（64+ tokens）。修复：continuation 优先，仅当复用量 < 快照时才用快照。
验证：`test_prefix_bitidentical.py` 新增多轮阶段——continuation hit（64/81 复用，只 prefill 16 新
token）且与冷启动全量 prefill **逐字节相同**。

**剩余空白（本次开工范围）**：

- Step 0a：prefix cache 命中/未命中/续写命中**计数器**，暴露在 `/api/state`，供 hit rate 量化。 ← 本次做
- Step 0b：**bit-identical 验证**——同一请求连发两次（第一次 MISS 全量算 + 快照，第二次 HIT 走恢复快照），贪心解码下两次输出必须逐 token 相同。现有测试只查"不乱码"，不查保真度。 ← 本次做
- 多 checkpoint（512/1024 稀疏档位）：**不做**。固定 system prompt 场景单边界快照已是最优；多档位只对"变长共享前缀"有意义，等 hit-rate 数据说话。
