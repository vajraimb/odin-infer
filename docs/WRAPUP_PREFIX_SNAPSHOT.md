# Prefix Snapshot / MMQ 优化轮 · 结案总结（2026-10-31）

> 范围：NNUE → odin-infer 迁移讨论催生的 Step 0–4 路线图（见 `PREFIX_SNAPSHOT_PLAN.md`）。
> 结论：**Step 0–3 全部闭环交付，Step 4 有数据支撑地搁置**。本轮所有性能声明均有
> bit-identical 回归测试背书。

## 起点

v1.0 收工基线（见 `qwen3_5/ROOFLINE.md`）：prefill 122→23 ms/tok（4.9×），decode ~91 ms/tok
已在带宽墙（~60% 带宽效率）。roofline 判定：prefill 在 MMQ dequant-ALU 算力墙，常规 kernel
优化空间封顶 ~2×。**唯一不在封顶表上的杠杆是"消除重算"而非"加速重算"**——这正是本轮主题。

## 各步交付与实测

### Step 0 — 埋点 + 保真度基线（commit `2d07a4e`）

- `serve.odin`：prefix cache 命中/续写命中/未命中计数器，暴露在 `/api/state`，
  供真实流量 hit-rate 量化（决定多 checkpoint 是否值得做的数据闸门）。
- `serve.odin`：修 `options.temperature` 被静默忽略的问题；temperature=0 走贪心 argmax——
  一切 bit-identical 判据的前置能力。
- `test_prefix_bitidentical.py`：同一请求连发两次（MISS 全量算 + 快照 → HIT 恢复快照），
  贪心下两次输出必须**逐字节相同**。把"不崩溃/不乱码"的弱验收升级为保真度强验收。

### Step 1 — Prefix state snapshot，内存版（此前已 ship，本轮验证）

- 系统提示词边界快照：命中时跳过 ~1000–1500 固定 token 的 prefill，只对增量后缀重算。
  TTFT 从 ~30s 量级降到 ~1–2s。命中 vs 非命中输出 bit-identical。
- 回归测试 `test_prefix_cache.py`：覆盖快照错位 bug（suffix 在已推进的 recurrent state 上
  重放 → 24/32 层污染）。

### Step 2 — 快照落盘（此前随 Step 1 ship，本轮加固验收，commit `addcb53`）

- `prefix.oikv` + `prefix.tokens` + `session.oikv` + `session.tokens` 落盘，
  重启恢复（CRC + model fingerprint + seq_len 校验），重启后首个请求即 PREFIX CACHE HIT。
- 本轮加固：`test_kv_persist.py` 从"回复是干净中文"升级为 **bit-identical 判据**——
  冷、暖两轮发同一首轮 prompt（temperature=0），断言回复逐字节相同，
  证明 f16↔f32 落盘往返无损。实测 PASS，暖启动 prefill=1295ms。

### Step 3 — MMQ autotune（commits `8482129` + `bc3941c`）· 超预期

- `gemm_harness/gemm_autotune.odin`：程序化生成 MSL，BM/BK/NT/TPR/RSTEP 编成常量；
  每候选先过正确性门（随机量化块 + CPU f64 参考，tol 5e-3）再 median-of-7 计时；
  `autotune_cache.jsonl` 缓存命中即跳过。18 配置 × (Q4K+Q6K) 全过门。
- 实测（M3 Air）：Q4K w1 874→**1028 GFLOPS（+18%）**，hidden_x_T 933→**1052（+13%）**；
  Q6K ~400 GFLOPS 持平（已在平台上限，bm 不敏感）；fast-math on/off <2% 噪声。
- engine 采纳：`metal.odin` 新增 `gemm_q4k_f32_b64`，M%64==0 走 bm=64 bk=64（M=32 回退 v3）；
  **Q6K 不走 b64**（w2 上 0.66× 回归，用数据否决）。
- 验证：`serve_harness/ab_b64_check.py` 新旧二进制 A/B，贪心回复逐字节相同，
  QTIMING 位置对齐 median **39.2→32.4 ms/tok，prefill ~1.2×**。
- 附带战果：正确性门第一跑抓到泛化 bug（BK≠32 时 qs 索引须用 `e_base&31` 而非 `col_base`；
  BK=32 时恰相等，v3 从未暴露）——正确性门不只守门，还能捕鱼。

### C=16 悬案结案（commit `a984d90`，Step 3 前置）

- 症状：chunked delta 在 C=16 出 `!!!!!`，C≤8 正常。
- 判决性实验排除 fast-math（GPU vs CPU 双胞 max|dOut| ~5e-9）；真凶是 **cp 比率下溢**：
  单 head 内两个极小-g token → `cp[t]` 下溢为字面 0 → `cp[t]/cp[i]` = 0/0 → NaN。
  CPU 双胞复现完全一致 NaN 模式（各 19968 个）→ 公式本身的脆弱点，非实现 bug。
- 遗产：fused kernel 的硬性设计要求——cp 比率必须稳定化（clamp 下界或 log-space），
  否则真实激活上随机炸 NaN。

## 验证文化（本轮最重要的方法论沉淀）

所有性能声明三件套：**贪心 bit-identical（逐字节）+ QTIMING 位置对齐计时 + A/B 双二进制**。
M3 Air 上 wall-clock 噪声 ±20%，只用 `[timing] forward_gpu_batch` 行下结论。
"不乱码"不是验收，"逐字节相同"才是。

## 剩余项与决策

| 项 | 预期收益 | 状态 | 理由 |
|---|---|---|---|
| Step 4 Stage 2 fused kernel | 端到端 ~4–5% | 搁置，按需重启 | 相对价值低；cp 稳定化是前置硬要求 |
| MMQ dequant 推 ~1.5 TFLOPS | prefill 再 ~1.5×（封顶 ~2×） | 未启动 | 唯一真杠杆，硬工程；需重写 dequant kernel |
| 多 checkpoint 稀疏档位 | 取决于 hit-rate | 不做 | 固定 system prompt 场景单边界已最优；等 `/api/state` 计数器数据说话 |
| decode 优化 | — | 到墙 | ~91 ms/tok，~60% 带宽效率，物理地板 56ms/tok |

## 总账

本轮把"消除重算"路线（snapshot 内存版 → 落盘 → 保真度验收）走到底，同时用 autotune 基建
在"加速重算"路线上拿走了计划外但真实的 ~1.2× prefill。路线图 Step 0–3 闭环，Step 4 有数据
支撑地搁置。引擎当前状态：prefill ~32 ms/tok（固定前缀场景 TTFT ~1–2s），decode 带宽墙，
全部有回归测试守护。

## 结案后追加：Continuation 路径修复（2026-10-31 晚）

结案后实测 hit-rate 计数器发现 `pc_continuation_hits` 恒为 0——多轮续写是死代码。
三因叠加：历史渲染缺空 think 桩（token 流结构性分叉）+ continuation 条件 `pos == common`
差一永假 + 快照检查顺序截胡。全部修复后：多轮请求整段会话复用（demo 中 64/81 tokens，
只 prefill 16 个新 token），且与冷全量 prefill **逐字节相同**（`test_prefix_bitidentical.py`
新增多轮阶段固化）。这正是 Step 0a 埋点的设计目的：没有计数器，这条死路径会永远躺着。

## 结案后追加：连环 tool-call 修复（2026-10-31 深夜）

**症状**：pie 里问天气类问题，模型答完一轮工具调用后无限重调同一个工具，永不给正文。

**根因**：`openai_extract_messages` 只抓 role+content，把 assistant 历史消息里的
`tool_calls` 字段整个丢掉。渲染出来的历史里 assistant 从没调用过工具，但下一条却是
tool 结果——模型看到的因果链断裂，只能"再调一次试试"，死循环。

**修复**（serve.odin）：
- `Message_Pair` 增加 `tool_calls_raw`（raw JSON 数组，用 `json_balanced_end` 平衡括号
  扫描逐字节捕获，不重序列化）；
- assistant 历史渲染改用 JSON 形式 `<tool_call>\n{...}\n</tool_call>`（function 对象
  逐字节嵌入）；tool 角色结果渲染为 `<|im_start|>user\n工具执行结果: ...<|im_end|>\n`。

**为什么是 JSON 形式而不是原生 `<function=` 形式**：探针 4 变体贪心实测——历史里放
原生 `<function=` 形式 = 连环重调；`<tool_response>` 包装（无论 tool/user 角色）= 连环；
只有 **JSON 形式 + 纯 user 角色结果** = 正常收尾。模型对训练分布外的历史形态会退化为
"继续生成 tool_call"，这是行为事实，不是猜测。

**已知取舍**：跨工具轮次 continuation 不命中（历史里的 JSON 形式 ≠ 模型生成的原生
形式，token 流不同），走 snapshot 路径，可接受。

**回归**：`serve_harness/test_toolloop.py`——turn2 必须无 tool call、缓存命中、有正文。
`test_prefix_cache.py` 断言放宽为认 snapshot 或 continuation 两种命中（88b2c42 之后
turn 2 走 continuation 是正确行为，原断言过期）。
