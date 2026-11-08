#!/usr/bin/env bash
# Odinfer zero-compile installer — run this to download the prebuilt binaries.
# Usage:
#   bash scripts/install.sh                 # binaries only
#   bash scripts/install.sh --model moe     # + the 0.8B MoE test model (524 MB)
#   bash scripts/install.sh --model qwen9b  # + Qwen3.5-9B heretic (5.6 GB)
set -euo pipefail

REPO="https://github.com/vajraimb/odinfer"
PREFIX="${PREFIX:-$HOME/.odinfer/bin}"
MODEL_DIR="${MODEL_DIR:-$HOME/.odinfer/models}"
MODEL_CHOICE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_CHOICE="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$PREFIX" "$MODEL_DIR"

echo "== downloading odin-serve + odin-cli =="
curl -fL -o "$PREFIX/odin-serve" "$REPO/releases/latest/download/odin-serve"
curl -fL -o "$PREFIX/odin-cli"   "$REPO/releases/latest/download/odin-cli"
chmod +x "$PREFIX/odin-serve" "$PREFIX/odin-cli"
xattr -cr "$PREFIX/odin-serve" "$PREFIX/odin-cli" 2>/dev/null || true

case "$MODEL_CHOICE" in
  moe)
    echo "== downloading Qwen3.5-0.8B-MoE (524 MB) =="
    curl -fL -o "$MODEL_DIR/Qwen3.5-0.8B-MoE-A0.6B-Q4_K_M.gguf" \
      "https://huggingface.co/bayliner1980/Qwen3.5-0.8B-MoE-A0.6B-GGUF/resolve/main/Qwen3.5-0.8B-MoE-A0.6B-Q4_K_M.gguf"
    MODEL="$MODEL_DIR/Qwen3.5-0.8B-MoE-A0.6B-Q4_K_M.gguf"
    ;;
  qwen9b)
    echo "== downloading Qwen3.5-9B heretic (5.6 GB) =="
    curl -fL -o "$MODEL_DIR/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf" \
      "https://huggingface.co/mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-GGUF/resolve/main/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf"
    MODEL="$MODEL_DIR/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf"
    ;;
  "")
    MODEL=""
    ;;
  *)
    echo "unknown model: $MODEL_CHOICE (use: moe | qwen9b)" >&2
    exit 2
    ;;
esac

echo
echo "== installed =="
echo "binaries: $PREFIX/odin-serve $PREFIX/odin-cli"
echo "add to PATH:  export PATH=\"\$PATH:$PREFIX\""
if [[ -n "$MODEL" ]]; then
  echo "model: $MODEL"
  echo "serve:        $PREFIX/odin-serve \"$MODEL\" --metal --port 9748"
  echo "MCP:          $PREFIX/odin-serve \"$MODEL\" --mcp --metal"
fi
