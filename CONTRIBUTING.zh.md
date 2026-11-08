# Contributing.zh.md · 贡献指南

[English version / 英文版](CONTRIBUTING.md)

感谢考虑贡献！这是窄定位项目（Apple Silicon 上的 Qwen3.5 级模型——见
POSITIONING.zh.md），动手前请先确认你的想法符合定位。

## 基本规则

1. **正确性先于速度**。每个内核都有参考实现（CPU 或 Python）；数值改动必须
   保持对拍通过。没有对照验证的"更快"内核不允许合入。
2. **诊断走 stderr**——stdout 是 MCP 协议通道。
3. **二进制不进 git**（`bin/`、散落的 Mach-O 文件）。
4. 文档双语：面向用户的文档以完整英文+中文两版发布（`README.md` +
   `README.zh.md` 风格，互相链接）。

## 流程

```sh
./build.sh                      # 构建 + 全部单测 + MoE 参考对拍
python3 server/serve_harness/test_mcp.py    # MCP 握手（需 Qwen3.5 GGUF）
# 内核工作：tools/moe_smoke.odin --compare 做 CPU vs GPU 对拍
```

- 修了 bug → 加进 AGENTS.md 的坑清单，别让后人重踩。
- 首启调优 / 内存预算 / MoE 布局都在 AGENTS.md 里——改引擎代码前先读。

## PR 清单

- [ ] `./build.sh` 全绿
- [ ] 相关集成测试（test_mcp / test_tune / moe_smoke）跑过
- [ ] 文档已更新（用户可见改动需中英双版）
- [ ] 无个人机器路径、无二进制、无凭据

## License

参与贡献即表示你的作品以 MIT 许可证（见 LICENSE）发布。
