#!/usr/bin/env bash
# Start the odin-infer HTTP server (dashboard + OpenAI-compatible API + Codex SSE).
# Usage: ./serve.sh [port] [max_ctx]
set -euo pipefail
cd "$(dirname "$0")"

PORT="${1:-9748}"
MAX_CTX="${2:-8192}"
MODEL="${MODEL:-/Users/vajra/Claw/odin-infeer/models/ornith-1.0-9b-Q4_K_M.gguf}"

# Kill any stale instance on this port
pkill -9 -f "serve_harness/serve.odin" 2>/dev/null || true
lsof -ti tcp:"$PORT" 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 1

echo "starting server on :$PORT (max_ctx=$MAX_CTX)"
echo "model: $MODEL"
echo "stop with Ctrl-C"
echo ""

exec odin run serve_harness/serve.odin -file -o:speed \
  -collection:ggml=. -collection:infer=. \
  -collection:qwen3_5=. -collection:qwen3_5_tokenizer=. \
  -collection:sampler=. -collection:tokenizer=. \
  -- \
  "$MODEL" \
  --metal --max-ctx "$MAX_CTX" --port "$PORT"
