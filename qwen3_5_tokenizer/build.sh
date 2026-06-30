#!/usr/bin/env bash
# Build/test the Qwen3.5 BPE tokenizer package. vocab.txt/merges.txt are baked in
# at compile time via #load; regenerate them from an HF tokenizer.json with
# tools/export_vocab.py.
set -euo pipefail
cd "$(dirname "$0")"
odin test . -o:speed
echo "qwen3_5_tokenizer tests OK"
