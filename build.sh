#!/usr/bin/env bash
# Build all odin-infer artifacts (server + CLI) and run the unit tests.
# Output: bin/odin-serve, bin/odin-cli
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p bin

COLLECTIONS=(-collection:ggml=packages -collection:infer=packages \
  -collection:qwen3_5=packages -collection:qwen3_5_tokenizer=packages \
  -collection:sampler=packages -collection:tokenizer=packages)

echo "== build odin-serve =="
odin build server/serve_harness/serve.odin -file -out:bin/odin-serve -o:speed "${COLLECTIONS[@]}"

echo "== build odin-cli =="
odin build cli -out:bin/odin-cli -o:speed -no-bounds-check -disable-assert -microarch:native "${COLLECTIONS[@]}"

echo "== unit tests: packages/qwen3_5 =="
( cd packages/qwen3_5 && ./build.sh )

echo "== unit tests: packages/tokenizer =="
( cd packages/tokenizer && odin test . )

echo ""
echo "== done: bin/odin-serve bin/odin-cli =="
