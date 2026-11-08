# Security Policy · 安全策略

[中文版 / Chinese version](#中文)

## Supported versions

Only the latest release (and `main`) receive security fixes. The engine is a
local inference tool: it executes model files you feed it, so treat GGUF files
like any other executable input from the network.

## Reporting a vulnerability

Please report vulnerabilities privately — do not open a public issue:

- Email: `vajraimb` at GitHub (see the profile) or use GitHub's
  ["Report a vulnerability"](https://github.com/vajraimb/odin-infer/security/advisories/new)
  private advisory flow.

We aim to acknowledge within 48 hours and fix within 2 weeks.

## Out of scope

- Issues requiring physical access to the device
- DoS via oversized local model files (use `--max-ctx` / memory budget; see
  docs/MEMORY.md)

## 中文

仅最新 release 与 main 分支获得安全修复。本引擎在本地执行你提供的模型文件，
请把 GGUF 视作来自网络的任意输入。漏洞请**私下**报告（GitHub 私有安全通告
流程），不要开公开 issue；我们会在 48 小时内确认、2 周内修复。物理接触设备
与超大模型文件导致的本地 DoS 不在范围内。
