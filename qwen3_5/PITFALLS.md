# Qwen3.5 移植踩坑记录

把 Qwen3.5（Ornith-1.0-9B，混合注意力架构）移植到这个 Odin 引擎过程中踩过的坑，
按类别记录。每条包含 **症状 → 根因 → 修复 → 通用教训**。给未来移植其它新架构
（或回头调 Qwen3.5）的人省时间。

---

## 一、架构 / 数学（CPU + Metal 共享，最致命）

### 1. RMSNorm 的 `(1+w)` 烘焙方向搞反
- **症状**：输出全是乱码（替换字符、随机 CJK）。
- **根因**：HF 里 `Qwen3.5RMSNorm` 是零初始化、应用 `(1+w)·x`。我假设 GGUF 存的是
  零初始化偏移，在 loader 里再加 1。但 llama.cpp 转换时**已经把 `+1` 烘焙进权重**
  （`conversion/qwen.py:302`：`data_torch = data_torch + 1`）——我又加一次就变 2 倍多。
- **修复**：dump 真实权重值确认（`output_norm` ≈ 2.2、`attn_norm` ≈ 1.1，是有效乘数
  不是 ≈0 的偏移），设 `RMSNORM_BAKE_PLUS_ONE = false`。注意 `ssm_norm`（RMSNormGated）
  不加 1（它本身 ones 初始化）。
- **教训**：归一化权重的"存储约定"每个转换器不同，**dump 实际数值**比读文档快。

### 2. `ssm_a` 存的是 `-exp(A_log)`，不是原始 `A_log`
- **症状**：仍是乱码。
- **根因**：GGUF 里 `blk.N.ssm_a`（注意**无 `.weight` 后缀**）存的是 `conversion/qwen.py:296`
  预算好的 `-torch.exp(A_log)`。我把它当原始 `A_log` 又套一层 `-exp()`，算成
  `-exp(-exp(A_log))`，遗忘门完全错乱。
- **修复**：`g_decay = exp(ssm_a · softplus(a + dt_bias))`，直接用 ssm_a，不再 exp。
- **教训**：GGUF 张量存的可能是**预算变换后的值**，名字（`ssm_a`）不一定反映它经历了
  什么运算。一定查转换代码。

### 3. V-head 的 tiled 重排（最大的坑，藏得最深）
- **症状**：输出**流利但语义错**——能说英文、能引用 prompt，但 2+2 都答不对，多轮后退化。
  CPU 和 Metal 都中招（共享数学）。
- **根因**：`conversion/qwen.py:353` `_LinearAttentionVReorderBase` 把 V-head 从 HF 的
  grouped 序 `[G0_v0, G0_v1, G1_v0, G1_v1, ...]` 重排成 **tiled 序**
  `[G0_v0, G1_v0, ..., G0_v1, G1_v1, ...]` 存进 GGUF。所以 v-head `vh` 对应 k-head
  **`vh % n_k_heads`**，我写成 `vh / 2`（grouped 假设）→ 24 个线性注意力层的 q/k/v 全错配。
- **修复**：`kh = vh % n_k_heads`（CPU `forward.odin` + Metal `delta_recurrent` kernel）。
- **教训**：调研的 agent **白纸黑字警告过这点**，我跳过没验证就宣布成功。**别人标的"风险
  点"必须查证，不能因为读着像次要说明就略过。** 这个 bug 单靠推理找不到——我的逻辑自洽，
  只是和存储约定对不上。

---

## 二、张量命名 / 元数据（靠 dump 头部发现）

### 4. 张量名全是猜错的
- **症状**：loader 报 "missing required tensor"。
- **根因**：我按 llama.cpp 老习惯猜 `gdn_in_qkv`/`gdn_conv` 等。实际命名（写个 GGUF 头部
  dump 工具读出来的）：
  | 我猜的 | 实际 |
  |---|---|
  | `gdn_in_qkv.weight` | `attn_qkv.weight` |
  | `gdn_in_z.weight` | `attn_gate.weight`（注意：是线性层的 z 门，不是全注意力的）|
  | `gdn_in_b.weight` | `ssm_beta.weight` |
  | `gdn_in_a.weight` | `ssm_alpha.weight` |
  | `gdn_conv.weight` | `ssm_conv1d.weight` |
  | `gdn_dt_bias.weight` | `ssm_dt.bias` |
  | `gdn_a_log.weight` | `ssm_a`（**无 .weight 后缀**）|
  | `gdn_norm.weight` | `ssm_norm.weight` |
  | `gdn_out.weight` | `ssm_out.weight` |
- 架构名是 `qwen35`（**无下划线**），元数据前缀 `qwen35.*`。MLP 的 norm 叫
  `post_attention_norm.weight`，不是 `ffn_norm.weight`。
- **教训**：永远先写个 ~50 行的 GGUF 头部 dump 工具，打印所有张量名 + metadata，**别猜**。

---

## 三、Tokenizer

### 5. HuggingFace Xet 存储拿不到文件
- **症状**：`curl -L` 下载 `tokenizer.json` 只得到 133 字节的 LFS 指针。
- **根因**：该模型用 Xet 存储，普通 HTTP range/redirect 取不到内容。
- **修复**：用 `hf download` CLI（自动处理 Xet）。

### 6. 嵌套 `map[string]map[string]int` 释放 → segfault
- **症状**：`free_tokenizer` 大量 "bad free" 级联成 segfault。
- **根因**：`delete(嵌套 map 的内层)` 在 Odin 里行为不可靠。
- **修复**：拍平成单个 `map[string]int`，复合键 `"<left>\n<right>"`。查表用栈 buffer
  构键零分配（Odin 的 string map 按内容哈希，栈字符串查表合法）。

### 7. 全局 `unicode_to_byte` map 并发竞争
- **症状**：`odin test`（8 线程）偶发 segfault，单测却过。
- **根因**：原 Qwen3 tokenizer 把 `unicode_to_byte` 放成包级全局，每个 `build_tokenizer`
  重新 `make`、`free_tokenizer` 里 `delete`。并行测试并发 make/delete 同一个全局 → use-after-free。
  （原 Qwen3 包有同样隐患，只是运气好没崩。）
- **修复**：移进 `Tokenizer` 结构体，每实例一份。

### 8. 24 万条 merges 的线性扫描不可用
- **症状**：encode 一次要几十秒。
- **根因**：`get_merge_rank` 是对 247k 条 merges 的 O(n) 线性扫描，BPE 合并循环让它变成
  O(tokens² · merges)。
- **修复**：复合键哈希表 O(1) 查表（见 #6）。

---

## 四、Metal（CPU 对的，Metal 错的——每个后端各有各的坏）

### 9. norm 权重用堆拷贝，Metal 的 mmap 偏移就废了
- **症状**：Metal 输出 `!!!!!`（argmax 恒为 0，x 变成 NaN/0）。
- **根因**：我为 `(1+w)` 烘焙用了 `tensor_as_f32_copy`（堆分配），但 Metal 的 `woff(ptr)`
  假设 `ptr` 在 mmap 里（`ptr - mmap_base`）。堆指针算出垃圾偏移 → rmsnorm 读到垃圾。
  （顺带：`free_transformer` 里 `delete` mmap 切片是双重释放隐患。）
- **修复**：norm 权重改用 `tensor_as_f32`（mmap 切片，无拷贝）；CPU 直接读 F32 字节，Metal
  用 woff，都对。移除所有 mmap 切片的 delete。
- **教训**：CPU 跑通**不代表** Metal 跑通。共享 buffer 假设（"这个指针在 mmap 里"）是
  Metal 特有的失败模式。

### 10. Metal buffer 不清零
- **症状**：Metal 输出 `!!!!!`。
- **根因**：`newBufferWithLength` 内容未定义。conv_state（初始应 0 历史）和
  recurrent_state（初始应 0 状态）在 pos 0 读到垃圾。CPU 的 `make([]f32)` 会清零，Metal 不会。
- **修复**：`metal_init` 里显式零填这两个 buffer。

### 11. shader 编译 typo 导致静默回退 CPU
- **症状**：速度只有 ~0.1 tok/s（CPU 速度），但没报错。
- **根因**：复制 GEMV kernel 时把 `thread_position_in_threadgroup` 损坏成
  `thread_position_in_threadthreadgroup_in_threadgroup`，shader 编译失败，
  `metal_init` 打印错误后 `return false`，引擎**静默回退 CPU**。我看到"有输出"没注意速度。
- **修复**：读 shader 编译错误信息，修 typo。
- **教训**：Metal 初始化失败会静默回退，**一定检查实际 tok/s 和 "Metal: enabled" 日志**。

### 12. `rmsnorm_gated` 把 `silu` 写成 `sigmoid`
- **症状**：Metal 输出流利但语义错（同 #3 症状）。
- **根因**：HF `RMSNormGated` 乘以 `silu(gate) = gate·sigmoid(gate)`，我 Metal kernel 写成
  `sigmoid(gate)`，少了 `gate·`。CPU 写对了（`silu_f32`），Metal 抄错。影响全部 24 个线性层。
- **修复**：`gz = z·(1/(1+exp(-z)))`。
- **教训**：CPU 和 Metal 的同一数学**要逐项对拍**，不能"逻辑一样所以肯定对"。

---

## 五、Odin 语言层面

### 13. Odin API 和预想不一致
- `math.abs_f32` / `math.log_f32(x)` **不存在** → 用泛型 `math.abs`，自然对数 `math.ln_f64`。
- 运行时字符串 `+` 受限（`arch + "."` 编译报错）→ 用 `fmt.tprintf` 或常量拼接。
- `delete(^T)`（`new` 出来的指针）**不能直接 free** → 把对象放栈上用 `&obj`，或用 `core:mem` 的 free。
- `for v in slice` 是值拷贝，不能赋值 → 用 `for &v`（注意 Odin 里仍不能 `v^=`，得用索引循环 `s[i]=0`）。
- map 值是 rvalue，`m[k][k2] = v` 不行 → 取到局部 handle 再写（map 是引用类型，共享底层表）。
- `for &v in slice { v^ = 0 }` 在某些上下文也报错，最稳是索引循环。

---

## 六、验证方法论（最该记住的元教训）

### 14. "输出像人话" 不是成功标准
- **症状**：我两次在输出还是错的时宣布"跑通了"。
- **根因**：**流利的错误输出比崩溃更难发现**。"The user is asking what to me" 语法没毛病，
  语义全错。我一直靠"看一眼像不像英文"判断，却没做最基本的事——**问它 2+2 等不等于 4**。
- **修复**：拿 `llama-cli`（升级到支持 qwen35 的版本）跑同一个 GGUF 当**地面真值**。它答对，
  我答错 → 锁定 bug 在我的引擎。然后读 `conversion/qwen.py` + `qwen35.cpp` 对齐存储约定。
- **教训**：移植新架构，**第一天就接上参考实现**做数值/行为对照，别等到"看起来不对"才接。
  事实性问答（算术、已知事实）是最便宜的回归测试。

### 15. Metal 的正确性独立于 CPU
- CPU 跑通只证明共享数学对。Metal 还要单独验：buffer 初始化、权重来源（mmap vs 堆）、
  kernel 里的每个激活函数。CPU vs Metal 的 logits 应该逐位比对（本引擎现在两边输出一致）。

---

## 快速自检清单（下次移植新架构）

- [ ] 写 GGUF 头部 dump 工具，拿真实张量名 + metadata，**别猜**
- [ ] 读 `conversion/*.py`：每个权重存进 GGUF 前**经历了什么变换**（+1？-exp？transpose？reorder？）
- [ ] 读参考实现的 forward，逐项对齐归一化约定、激活函数（silu vs sigmoid）、head 映射
- [ ] **接参考实现做行为对照**（llama-cli / transformers），用事实性问答验证
- [ ] CPU 跑通后，Metal 单独验证：buffer 清零、权重在 mmap、kernel 激活函数逐个对拍
- [ ] 关注实际 tok/s 和 "Metal: enabled"，别被静默回退 CPU 骗了
