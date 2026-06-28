#!/usr/bin/env bash
# Run tests for all library packages.
set -euo pipefail
cd "$(dirname "$0")"
LIB="$(pwd)"

cd "$LIB/tokenizer"
odin test . -define:ODIN_TEST_THREADS=1
echo "All library tests passed."
