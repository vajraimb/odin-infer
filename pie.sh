#!/usr/bin/env bash
# Run pie-odin against the local odin-infer server.
# Prerequisite: ./serve.sh must be running.
#
# Usage:
#   ./pie.sh "your prompt"   # one-shot via stdin
#   ./pie.sh                 # interactive REPL
set -euo pipefail

PORT="${PORT:-9748}"
PIE_BIN="${PIE_BIN:-/Users/vajra/Claw/firstmate/projects/pie-odin/build/pie}"

export ODINFERSERVER_BASE_URL="http://127.0.0.1:${PORT}"

if [[ $# -ge 1 ]]; then
  # One-shot: feed prompt via stdin, then close
  echo "$1" | exec "$PIE_BIN" --model odinfer 2>/dev/null
else
  # Interactive
  exec "$PIE_BIN" --model odinfer
fi
