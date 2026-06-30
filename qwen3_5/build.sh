#!/usr/bin/env bash
# Build/test the Qwen3.5 inference package (CPU path). End-to-end runs require a
# Qwen3.5 GGUF; the default target runs the pure-math unit tests.
set -euo pipefail
cd "$(dirname "$0")"
LIB="$(cd .. && pwd)"
odin test . \
  -collection:ggml="$LIB" \
  -collection:qwen3_5="$LIB" \
  -o:speed
echo "qwen3_5 tests OK"
