# AGENTS.md — AI agent 速查（构建 / 测试 / 使用）

这是给 AI coding agent（Cursor / Claude Code 等）的项目速查。读完即可独立
构建、测试、运行和改动本项目。

## 一句话架构

Apple Silicon 上的 Qwen3/Qwen3.5 推理引擎（纯 Odin，无 Python 运行时依赖）。
两个引擎：`packages/infer`（Qwen3 密集）、`packages/qwen3_5`（Qwen3.5 混合
注意力，gated delta）；共享 `packages/ggml`（GGUF 解析/量化）。服务端在
`server/serve_harness/serve.odin`（单文件包），CLI 在 `cli/main.odin`。

## 发布

- 零编译二进制由 GitHub Actions 构建：`.github/workflows/release.yml`
  （macos-14 arm64 runner，brew 装 Odin → ./build.sh → 推 tag `v*` 时
  `gh release create` 发布 bin/odin-serve + bin/odin-cli）
- 改发布流程后：push tag 触发；`gh run watch` 观察

## 构建与测试（必跑顺序）

```sh
./build.sh        # 全量：bin/odin-serve + bin/odin-cli + 全部单测（约 1-2 分钟）
```

单测单独跑：

```sh
( cd packages/qwen3_5 && ./build.sh )        # 8 个纯数学单测
( cd packages/tokenizer && odin test . -define:ODIN_TEST_THREADS=1 )
# ⚠️ tokenizer 测试必须单线程：5 个并行 runner 各建 248k 词表映射，
#    内存紧张时（<100MB free）会 OOM 段错误——这是环境性已知问题
```

集成测试（会加载真实模型，每个 ~30-60s）：

```sh
python3 server/serve_harness/test_mcp.py    # MCP 全握手（需 Qwen3.5 GGUF）
python3 server/serve_harness/test_tune.py   # 首启调优 + 缓存复用（同上）
# 通用环境变量：MODEL=<路径> 覆盖默认模型
```

> 集成测试的二进制缓存在 `server/serve_harness/.test-serve.bin`；**测试脚本
> 只在 serve.odin 或 packages/*.odin 更新时自动重建**（按最新 mtime 判断）。

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

- `models/` 是 gitignored 软链目录（见 models/README.md）
- 默认基准模型：`models/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf`
  （→ ollama blob 软链，Qwen3.5，5.6GB）
- 快速冒烟用 0.6B：`/Users/vajra/Claw/odin-infeer/Qwen3-0.6B-Q4_K_M.gguf`
  （Qwen3，走 infer 引擎，加载 ~1s）——仅 CLI 支持，serve 只认 Qwen3.5

## 关键约定（改代码前必读）

1. **诊断一律 stderr**：`fmt.eprint*`。`--mcp` 模式下 stdout 只允许协议消息；
   库层（packages/*）的所有启动打印已迁到 stderr，不要回退。
2. **fmt.tprintf 的 `{`/`}` 是动词**：JSON 模板必须写 `{{`/`}}`。
3. **json_escape 自带外层引号**（输出 `"escaped"`），调用方不要再加引号。
4. **集合导入**：`import q35 "qwen3_5:qwen3_5"`，构建必须挂
   `-collection:<name>=packages`（共 6 个：ggml/infer/qwen3_5/
   qwen3_5_tokenizer/sampler/tokenizer），见 build.sh 的 COLLECTIONS。
5. **内核 A/B 开关**（env，默认全关，均为实测回归项）：`QGEMV3=1`
   （llama.cpp 行切块移植）、`QGEMV2R=1`（2 行/线程）、`QGEMVNORM=1`
   （norm 融合）。消融开关：写 `/tmp/qprof_flags`（noout/nomlp/nosmall/
   nonorm/noattn/nodelta），每 token 读一次，免重启。
6. **调优缓存**：`~/.odin-infer/<model>.tune`（`ODIN_TUNE_DIR` 覆盖，
   `ODIN_TUNE=0` 关闭）；改了内核变体的性能特征后删除缓存强制重测。
7. **KV 持久化**：`KV_PERSIST=0` 关闭，`KV_PERSIST_DIR` 改目录；跨
   max-ctx 不兼容（seq_len mismatch 属正常日志）。
8. **不要提交二进制**：构建产物在 `bin/`（已 ignore）；历史上误提交过
   `tokenizer`/`serve` 两个编译产物，已清理并加了 `/tokenizer`、`/serve`
   到 .gitignore。

## 已验证的坑（不要再踩）

- GEMV 内核形状：本机（M3）最优 = 256 线程/32 行/8 lanes；所有多行/多线程/
  融合变体实测回归（数据在 packages/qwen3_5/README.md 与 gemm_harness）。
- 内核计时必须**墙钟 + 正确性验证**：GPU 计时器（GPUEndTime）对短 dispatch
  会给出假数据（曾误判 22.5ms 输出层，实际 6.4ms）。
- Odin 语法小坑：for 循环里不能直接放数组字面量（先命名变量）；取 proc
  参数的地址需先拷到局部变量；`os.to_stream(f)` 用于写文件流。

## 验证清单（改动后自检）

```sh
./build.sh                                          # 构建 + 单测全绿
python3 server/serve_harness/test_mcp.py            # MCP 握手全过
MODEL=./models/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf ./serve.sh &
curl -s http://127.0.0.1:9748/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"odin-infer","messages":[{"role":"user","content":"What is 2+2? Answer in one word."}],"max_tokens":8}'
# 期望：content 为 "4"，stderr 有 mem: 行与（首次）tune: 行
```
