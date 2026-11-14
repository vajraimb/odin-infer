#!/usr/bin/env bash
# Build all odin-infer artifacts (server + CLI) and run the unit tests.
# Output: bin/odin-serve, bin/odin-cli
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p bin

COLLECTIONS=(-collection:ggml=packages -collection:infer=packages \
  -collection:qwen3_5=packages -collection:qwen3_5_tokenizer=packages \
  -collection:sampler=packages -collection:tokenizer=packages \
  -collection:cuda=packages -collection:vulkan=packages)

echo "== gen MSL tables =="
python3 tools/gen_msl_tables.py

echo "== build odin-serve =="
# Release flags: -o:speed -no-bounds-check -disable-assert for BOTH binaries.
# Deliberately NO -microarch:native: the CI runner compiles on newer Apple
# silicon and native emits ARMv8.6 matrix instructions (fmul.4s) that SIGILL
# on M1/M2. Baseline arm64 runs on all M1-M4.
odin build server/serve_harness/serve.odin -file -out:bin/odin-serve -o:speed -no-bounds-check -disable-assert "${COLLECTIONS[@]}"

echo "== build odin-cli =="
odin build cli -out:bin/odin-cli -o:speed -no-bounds-check -disable-assert "${COLLECTIONS[@]}"

echo "== unit tests: packages/qwen3_5 =="
( cd packages/qwen3_5 && ./build.sh )

echo "== unit tests: packages/tokenizer =="
( cd packages/tokenizer && odin test . -define:ODIN_TEST_THREADS=1 )

# Metal shader smoke test (no model needed): compile the MSL library, build a
# pipeline for every kernel, check every make_pso() name resolves. Catches the
# v0.2.0-class bug where the shader library failed to compile and every model
# silently ran on CPU. Skip with SKIP_METAL_SMOKE=1 (e.g. headless non-Metal hosts).
if [[ "$(uname -s)" == "Darwin" && "${SKIP_METAL_SMOKE:-0}" != "1" ]]; then
  echo "== metal_smoke: packages/qwen3_5 shaders =="
  odin run tools/metal_smoke.odin -file -out:/tmp/odinfer-metal-smoke "${COLLECTIONS[@]}"
fi

echo ""
# MoE (qwen35moe) end-to-end validation: synthetic GGUF + Python reference
if command -v python3 >/dev/null 2>&1; then
  python3 tools/make_moe_gguf.py >/dev/null
  if odin run tools/moe_load_test.odin -file "${COLLECTIONS[@]}" >/dev/null 2>&1; then
    echo "== moe_load_test: PASSED =="
  else
    echo "== moe_load_test: FAILED =="
    exit 1
  fi
fi

echo "== done: bin/odin-serve bin/odin-cli =="
