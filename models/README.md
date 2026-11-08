# models/

GGUF files are gitignored; keep symlinks or download scripts here.

[中文版 / Chinese version](README.zh.md)

- `ornith-1.0-9b-Q4_K_M.gguf` → the Qwen3.5-9B Ornith Q4_K_M used by default.
- `DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf` → the heretic
  Q4_K_M, benchmark model of the performance work.
- `Qwen3.5-0.8B-MoE-A0.6B-Q4_K_M.gguf` (optional) → small qwen35moe test model
  from bayliner1980, used by the MoE smoke test.

Suggested layout: store real GGUFs in `~/models` (or anywhere) and symlink
them here.
