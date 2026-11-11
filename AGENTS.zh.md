# AGENTS.zh.md — AI agent 速查（构建 / 测试 / 使用）

这是给 AI coding agent（Cursor / Claude Code 等）的项目速查。读完即可独立
构建、测试、运行和改动本项目。

[English version / 英文版](AGENTS.md)

## 一句话架构

Apple Silicon 上的 Qwen3.5 级推理引擎（纯 Odin，无 Python 运行时依赖）。
**产品定位：Qwen3.5 及以上**——服务端（主形态）只跑 `packages/qwen3_5` 引擎
（混合注意力 + gated delta），这是有意为之。`packages/infer`（Qwen3 密集）
仅作 CLI 的历史兼容路径，不要在产品文档中扩展。共享 `packages/ggml`
（GGUF 解析/量化）。服务端在 `server/serve_harness/serve.odin`，CLI 在
`cli/main.odin`。

## 发布

- 零编译二进制由 GitHub Actions 构建：`.github/workflows/release.yml`
  （macos-15 arm64 runner，Odin 锁定 dev-2026-07，`./build.sh`，推 `v*` tag
  自动 `gh release create` 发布 bin/odin-serve + bin/odin-cli）。
- 改发布流程后：push tag 触发；`gh run watch` 观察。

## 安装与运行（零编译——agent 先读这里）

无需工具链：安装脚本下载 release 二进制（可选加模型）并打印可直接运行的命令。

```sh
bash scripts/install.sh --model moe   # 二进制 + 524MB MoE 测试模型
export PATH="$PATH:$HOME/.odinfer/bin"
odin-serve "$HOME/.odinfer/models/Qwen3.5-0.8B-MoE-A0.6B-Q4_K_M.gguf" --mcp --metal
```

- `--model qwen9b` 改下载 5.6GB 的 heretic 模型。
- `--prefix` / `MODEL_DIR` 覆盖安装路径。
- 手动等价做法：curl 下载
  `https://github.com/vajraimb/odinfer/releases/latest/download/` 下的两个二进制。

下面的源码构建仅供开发用。

## 构建与测试（必跑顺序）

```sh
./build.sh        # 全量：bin/odin-serve + bin/odin-cli + 全部单测（约 1-2 分钟）
```

单测单独跑：

```sh
( cd packages/qwen3_5 && ./build.sh )        # 11 个单测（含 3 个 MoE 数学测试）
( cd packages/tokenizer && odin test . -define:ODIN_TEST_THREADS=1 )
# ⚠️ tokenizer 测试必须单线程：5 个并行 runner 各建 248k 词表映射，内存紧张时
#    会 OOM 段错误——这是环境性已知问题，不是代码 bug。
```

集成测试（会加载真实模型，每个 ~30-60s）：

```sh
python3 server/serve_harness/test_mcp.py    # MCP 全握手（需 Qwen3.5 GGUF）
python3 server/serve_harness/test_tune.py   # 首启调优 + 缓存复用
# 通用环境变量：MODEL=<路径> 覆盖默认模型
```

> 集成测试的二进制缓存在 `server/serve_harness/.test-serve.bin`；**只在
> serve.odin 或 packages/*.odin 更新时自动重建**（按最新 mtime 判断）。

## 运行

```sh
# HTTP 服务（开发模式）
MODEL=./models/<qwen3.5-gguf>.gguf ./serve.sh 9748 8192
# 端点：POST /v1/chat/completions、/api/chat、/v1/responses(SSE)、GET /api/state

# MCP 模式（stdout 是协议通道，禁止往 stdout 打日志）
bin/odin-serve <model.gguf> --mcp --metal --max-ctx 4096

# CLI（自动识别 Qwen3 / Qwen3.5）
bin/odin-cli <model.gguf> -g 1 -c 2048
```

## 模型文件

- `models/` 是 gitignored 软链目录（见 models/README.zh.md）。
- 默认基准模型：`models/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf`
  （→ ollama blob 软链，Qwen3.5，5.6GB）。
- MoE 测试模型：`models/Qwen3.5-0.8B-MoE-A0.6B-Q4_K_M.gguf`（524MB，
  bayliner1980）——tools/moe_smoke.odin 用它。
- 快速冒烟用 0.6B：任意 `Qwen3-0.6B-Q4_K_M.gguf`（Qwen3，历史兼容路径，
  加载 ~1s）——仅 CLI；serve 只认 Qwen3.5，产品定位即 3.5+。

## 关键约定（改代码前必读）

1. **诊断一律 stderr**：`fmt.eprint*`。`--mcp` 模式下 stdout 只允许协议消息；
   库层启动打印已迁到 stderr，不要回退。
2. **fmt.tprintf 的 `{`/`}` 是动词**：JSON 模板必须写 `{{`/`}}`。
3. **json_escape 自带外层引号**（输出 `"escaped"`），调用方不要再加引号。
4. **集合导入**：`import q35 "qwen3_5:qwen3_5"`，构建必须挂 6 个
   `-collection:<name>=packages`（ggml/infer/qwen3_5/qwen3_5_tokenizer/
   sampler/tokenizer），见 build.sh 的 COLLECTIONS。
5. **内核 A/B 开关**（env，默认全关，均为实测回归项）：`QGEMV3=1`
   （llama.cpp 行切块移植）、`QGEMV2R=1`（2 行/线程）、`QGEMVNORM=1`
   （norm 融合）。消融开关：写 `/tmp/qprof_flags`（noout/nomlp/nosmall/
   nonorm/noattn/nodelta），每 token 读一次，免重启。
6. **调优缓存**：`~/.odin-infer/<model>.tune`（`ODIN_TUNE_DIR` 覆盖，
   `ODIN_TUNE=0` 关闭）；改了内核性能特征后删除缓存强制重测。
7. **KV 持久化**：`KV_PERSIST=0` 关闭，`KV_PERSIST_DIR` 改目录；跨 max-ctx
   不兼容（seq_len mismatch 属正常日志）。
8. **不要提交二进制**：构建产物在 `bin/`（已 ignore）；历史上误提交过
   `tokenizer`/`serve` 两个编译产物，已清理并加了 `/tokenizer`、`/serve`
   到 .gitignore。

## MoE（qwen35moe）

- CPU + Metal 双路径已支持。Metal MoE：GPU router（moe_gemv_f32 +
  moe_softmax_topk，每层一次同步）、专家 GEMV 复用生产内核（张量基址+专家
  偏移）、moe_combine 加权求和；共享专家经 F32 上传走 moe_gemv_f32。
  MOE_CPU=1 强制走 CPU。MoE 批量 prefill（forward_gpu_batch_moe，分组
  GEMM）自 v0.3.0 起为默认 Metal 路径（三处叠加 bug 已修，与逐 token
  max_abs 8e-4），MOE_BATCH=0 退回逐 token 做 A/B。
- Metal IQ/k-quant GEMV 内核：gemv_q2_k/q3_k/q5_k/q8_k + 9 个 IQ 内核
  （生产形状）。codebook 表在 packages/qwen3_5/msl_tables.odin，由
  tools/gen_msl_tables.py 从 packages/ggml/tables.odin 生成（唯一事实源）；
  build.sh 自动重生成，CI 用 --check 防漂移。不要手改 msl_tables.odin。Qwen3.8-27B 可 GPU 解码
  （~0.4-1.3s/token）；ctx 8192 已验证（Q8_0 KV 0.27GB，总占用 7.7GB），
  ctx 32768 kv 1.14GB（总占用 8.56GB）。
- KV 缓存默认 Q8_0 量化（KV_Q8=0 退回旧 f16 Metal 路径）。对称量化
  d=amax/127，与 llama.cpp 逐位一致。布局 position-major
  [n_full][seq_len][kv_dim/32 块][34B]；CPU Run_State 保持 fp32、在快照边界
  转换（CPU 恢复保真 ~4e-2，Metal 快照逐字节精确）。.oikv 为 KV_VERSION=2
  （v1 文件明确拒绝）。离群值重的 V 缓存上 Q8 误差真实存在（27B logits
  ~1e-1），但 argmax/质量不受影响——llama.cpp 标准做法。
- dense 批量 prefill（enc_mm）仅支持 Q4_K/Q6_K；model_batch_mm_ok 把关，
  混合量化模型回退逐 token Metal。IQ 批量 prefill 已实现（见下一条）。
- dense 批量 prefill（enc_mm）现已支持 13 种量化——Q4_K/Q6_K/Q8_0/Q2_K/Q3_K
  + IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS（v3 形状 MMQ
  内核，dequant 与 GEMV 一致）。27B 的 Dynamic 混合量化 prefill 已批量化
  （~1.8x GPU，30-token prompt 227 vs 420 ms/tok）；批量 vs 逐 token prefill
  max_abs ~5.7e-3、解码步 ~6.6e-3（64 层 MLP 舍入残留，argmax 一致）。
  真实 token 的注意力列用 GEMV 覆盖以保持状态近精确。IQ 的 b64 变体为
  后续性能项。GGUF 检查工具：tools/inspect_iq_gguf.py + tools/gguf_meta.py。
- 前缀缓存/会话连续性：两条 HTTP 路径（OpenAI 与 Ollama）都跨请求复用引擎
  状态——延续（prompt 是上次 token 序列的精确前缀延伸时只 prefill 新尾）、
  系统边界快照（恢复后只 prefill 后缀）、否则全量 reset。OpenAI 路径在
  v0.3.x 周期补齐（MCP 的 odin.chat 自动继承）。计数器：/api/state 的
  pc_continuation_hits / pc_snapshot_hits。
- 专家张量布局（已对真实 unsloth 35B-A3B 头部验证）：`ffn_gate_inp`
  [dim, n_expert]，`ffn_up_exps`/`ffn_gate_exps` [dim, n_ff_exp, n_expert]，
  `ffn_down_exps` [n_ff_exp, dim, n_expert]——ne0 是收缩维，量化粒度 = 一行
  ne0。
- 路由数学与 llama.cpp build_moe_ffn 一致（softmax → top-k → 权重归一化，
  除数下限 6.1035e-5 → expert_weights_scale → 每专家 silu(gate)×up →
  sigmoid 门控共享专家）。
- 验证：`tools/make_moe_gguf.py` 写合成全 F32 MoE GGUF + Python 参考；
  `tools/moe_load_test.odin` 对拍 logits（当前逐位一致）。`./build.sh` 自动
  执行。
- 真实模型冒烟：`tools/moe_smoke.odin <model.gguf> [steps]`（CPU）。已在
  `bayliner1980/Qwen3.5-0.8B-MoE-A0.6B-GGUF` Q4_K_M（524MB，24 层，28 专家，
  top-14，248k 词表）验证——加载、预填充、解码连贯（CPU ~1.8s/token）。
  注意：我们的加载器接受任意层数（llama.cpp 的 qwen35moe 类型表只认
  40/48/60 层）。

## 已验证的坑（不要再踩）

- GEMV 内核形状：本机（M3）最优 = 256 线程/32 行/8 lanes；其余多行/多线程/
  融合变体实测回归（数据在 packages/qwen3_5/README.md 与 gemm_harness）。
- 内核计时必须**墙钟 + 正确性验证**：GPU 计时器（GPUEndTime）对短 dispatch
  会给出假数据（曾误判 22.5ms 输出层，实际 6.4ms）。
- Odin 语法小坑：for 循环里不能直接放数组字面量（先命名变量）；proc 参数取
  地址需先拷到局部变量；`os.to_stream(f)` 用于写文件流。
- MoE SwiGLU：silu 作用于 GATE 投影（silu(gate)×up），不是 up——对称测试
  数据会掩盖此错。
- GGUF 张量维度是 ggml ne 序（ne0 最快）= numpy shape 的反转；手写 GGUF
  写入器要反转 dims 且数据段 32 字节对齐。
- MTL setBuffer 签名是 (buffer, offset, index)——第二个参数当 index 写会静默
  重绑 index 0（曾导致 router 权重写进 logits 缓冲区）。
- 专家 GEMV 偏移是张量内相对偏移；m_weights 是整模型 mmap，必须加
  woff(张量) 基址。
- Q5_0 高比特：元素 j+16 用的是 qh 第 j+16 位（CPU 解量化是 >>(j+12) 后掩
  0x10）。
- 社区 MoE GGUF 逐层混合量化种类（0.8B-MoE 的 down_exps 在 Q8_0/Q5_0 间
  交替）——按张量 kind 分派，不要假设。
- MSL 保留字：变量名 'half' 会让整个 shader 库编译失败。过去 metal_init 失败
  会静默回退 CPU（所有模型）；现在 --metal / -g 1 下初始化失败直接报错退出
  （METAL_FALLBACK=1 恢复旧回退行为），build.sh 在 macOS 上自动跑
  tools/metal_smoke.odin：编译全部 kernel、逐个建 pipeline、核对 make_pso 名字。
- 已解决（v0.3.0）：MoE 分组 GEMM 批量 prefill 曾与逐 token 发散，根因是三处
  叠加 bug（M=16 ssm 张量 MMQ 零网格、top-k 跨 token 线程组竞争、MMQ 顺序
  放大误差），现已默认开启。

## 验证清单（改动后自检）

```sh
./build.sh                                          # 构建 + 单测全绿
python3 server/serve_harness/test_mcp.py            # MCP 握手全过
MODEL=./models/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf ./serve.sh &
curl -s http://127.0.0.1:9748/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"odinfer","messages":[{"role":"user","content":"What is 2+2? Answer in one word."}],"max_tokens":8}'
# 期望：content 为 "4"，stderr 有 mem: 行与（首次）tune: 行
