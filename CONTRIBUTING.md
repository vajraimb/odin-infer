# Contributing · 贡献指南

[中文版 / Chinese version](CONTRIBUTING.zh.md)

Thanks for considering a contribution! This is a narrow-scope project
(Qwen3.5-class models on Apple Silicon — see POSITIONING.md), so please check
that your idea fits the positioning before investing in it.

## Ground rules

1. **Correctness before speed.** Every kernel has a reference (CPU or Python);
   numeric changes must keep the comparisons green. Never ship a "faster"
   kernel without a bit/tolerance check against the reference.
2. **Diagnostics to stderr** — stdout is the MCP protocol channel.
3. **No binaries in git** (`bin/`, stray Mach-O files).
4. Docs are bilingual: user-facing docs ship as complete EN + ZH versions
   (`README.md` + `README.zh.md` style, cross-linked).

## Workflow

```sh
./build.sh                      # build + all unit tests + MoE reference check
python3 server/serve_harness/test_mcp.py    # MCP handshake (needs a Qwen3.5 GGUF)
# kernel work: compare CPU vs GPU with tools/moe_smoke.odin --compare
```

- Fix a bug → add it to the traps list in AGENTS.md so nobody re-learns it.
- First-run autotune / memory budget / MoE layouts are covered in AGENTS.md —
  read it before editing engine code.

## PR checklist

- [ ] `./build.sh` green
- [ ] Relevant integration test (test_mcp / test_tune / moe_smoke) run
- [ ] Docs updated (EN + ZH for user-facing changes)
- [ ] No personal machine paths, no binaries, no credentials

## License

By contributing you agree your work is licensed under the MIT License
(see LICENSE).
