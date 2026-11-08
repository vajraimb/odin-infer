# models/ 模型目录

GGUF 文件被 gitignore；这里放软链或下载脚本。

[English version / 英文版](README.md)

- `ornith-1.0-9b-Q4_K_M.gguf` → 默认使用的 Qwen3.5-9B Ornith Q4_K_M
- `DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf` → heretic
  Q4_K_M，性能工作的基准模型
- `Qwen3.5-0.8B-MoE-A0.6B-Q4_K_M.gguf`（可选）→ bayliner1980 的小 qwen35moe
  测试模型，MoE 冒烟测试用

建议布局：真实 GGUF 放 `~/models`（或任意路径），软链到这里。
