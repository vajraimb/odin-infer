#!/usr/bin/env bash
# Run Codex CLI against the local odin-infer server.
# Prerequisite: ./serve.sh must be running (wait for "serve: listening" line).
#
# Usage:
#   ./codex.sh "your prompt"            # one-shot
#   ./codex.sh                          # interactive
set -euo pipefail

PORT="${PORT:-9748}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local-only}"

exec codex ${1:+exec --skip-git-repo-check "$1"} \
  -c model_provider="odin" \
  -c model="odin-infer" \
  -c model_reasoning_effort="none" \
  -c 'model_providers.odin.name="odin-infer local"' \
  -c "model_providers.odin.base_url=\"http://127.0.0.1:${PORT}/v1\"" \
  -c 'model_providers.odin.wire_api="responses"' \
  -c 'model_providers.odin.env_key="OPENAI_API_KEY"'
