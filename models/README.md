# models/

GGUF files are gitignored; keep symlinks or download scripts here.

- `ornith-1.0-9b-Q4_K_M.gguf` -> the Qwen3.5-9B Ornith Q4_K_M used by default
- `DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf` -> the heretic
  Q4_K_M (the benchmark model of the performance work)

Suggested local layout: store real GGUFs in `~/models` (or anywhere) and
symlink them here.
