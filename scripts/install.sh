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
    echo "== downloading Qwen3.5-9B heretic =="
    echo "   Mac benchmark sha256 prefix is a692de95."
    echo "   HF main (commit a713868aa6cd) is a re-quant, sha256 prefix b4363b57, 5780092768 bytes."
    echo "   That blob is not pinned: it is not the cross-backend file."
    curl -fL -o "$MODEL_DIR/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf" \
      "https://huggingface.co/mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-GGUF/resolve/main/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf"
    MODEL="$MODEL_DIR/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf"
    if command -v sha256sum >/dev/null 2>&1; then
      SHA="$(sha256sum "$MODEL" | awk '{print $1}')"
    else
      SHA="$(shasum -a 256 "$MODEL" | awk '{print $1}')"
    fi
    echo "sha256 ${SHA}  ${MODEL}"
    case "$SHA" in
      a692de95*) echo "matches the Mac benchmark file" ;;
      *)
        echo "warning: this is not the Mac benchmark file (sha256 prefix a692de95)." >&2
        echo "warning: do not compare tok/s against the Metal numbers with this file." >&2
        ;;
    esac
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
