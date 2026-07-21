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

# Clear other provider keys so pie doesn't accidentally fall back to them
# for auxiliary tasks (title generation, etc.) and send "odinfer" as a
# model id to OpenAI/Anthropic/etc., which would 400.
unset OPENAI_API_KEY ANTHROPIC_API_KEY OPENROUTER_API_KEY GROQ_API_KEY \
      MISTRAL_API_KEY GEMINI_API_KEY GOOGLE_API_KEY OPENAI_BASE_URL 2>/dev/null || true

export ODINFERSERVER_BASE_URL="http://127.0.0.1:${PORT}"

if [[ $# -ge 1 ]]; then
  # One-shot: feed prompt via stdin, then close
  echo "$1" | exec "$PIE_BIN" --model odinfer 2>/dev/null
else
  # Interactive
  exec "$PIE_BIN" --model odinfer
fi
