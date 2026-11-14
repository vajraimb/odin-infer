#!/usr/bin/env bash
# Start the odin-infer HTTP server (dashboard + OpenAI-compatible API + Codex SSE).
# Usage: ./serve.sh [port] [max_ctx]
set -euo pipefail
cd "$(dirname "$0")"

PORT="${1:-9748}"
MAX_CTX="${2:-8192}"
MODEL="${MODEL:-./models/ornith-1.0-9b-Q4_K_M.gguf}"

# Kill any stale instance on this port
pkill -9 -f "server/serve_harness/serve.odin" 2>/dev/null || true
lsof -ti tcp:"$PORT" 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 1

echo "starting server on :$PORT (max_ctx=$MAX_CTX)"
echo "model: $MODEL"
echo "stop with Ctrl-C"
echo ""

EXTRA=()
GPU=(--metal)
if [[ "${ODINFER_BACKEND:-}" == "cuda" ]]; then
  EXTRA=(-define:ODINFER_BACKEND="cuda")
  GPU=(--cuda)
elif [[ "${ODINFER_BACKEND:-}" == "vulkan" ]]; then
  EXTRA=(-define:ODINFER_BACKEND="vulkan")
  GPU=(--vulkan)
elif [[ "$(uname -s)" != "Darwin" ]]; then
  GPU=()
fi

exec odin run server/serve_harness/serve.odin -file -o:speed \
  "${EXTRA[@]}" \
  -collection:ggml=packages -collection:infer=packages \
  -collection:qwen3_5=packages -collection:qwen3_5_tokenizer=packages \
  -collection:sampler=packages -collection:tokenizer=packages \
  -collection:cuda=packages -collection:vulkan=packages \
  -- \
  "$MODEL" \
  "${GPU[@]}" --max-ctx "$MAX_CTX" --port "$PORT"
