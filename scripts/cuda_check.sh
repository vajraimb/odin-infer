#!/usr/bin/env bash
# Build the CUDA backend and run what this machine can run.
# NVRTC syntax check (compute_70) does not need a GPU.
# Kernel self-test and the generation smoke need a CUDA device.
#
#   ./scripts/cuda_check.sh
#
# Model (first existing path; MODEL overrides). Sizes are the files this
# repo's docs name, not something the script downloads:
#   models/Qwen3.5-0.8B-MoE-A0.6B-Q4_K_M.gguf                          524 MB
#   $HOME/.odinfer/models/Qwen3.5-0.8B-MoE-A0.6B-Q4_K_M.gguf            524 MB
#   models/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf
#     Mac benchmark file: sha256 prefix a692de95.
#     HF main was re-quantized (sha256 prefix b4363b57, 5780092768 bytes,
#     commits 29319295d35e through a713868aa6cd). That file is not the
#     cross-backend benchmark. CUDA_CHECK_ALLOW_REQUANT=1 smokes it anyway.
#
# SMOKE_TOKENS (default 16), SMOKE_CTX (default 2048).
# CUDA_CHECK_REQUIRE_SMOKE=1 fails when no model file is present.
#
# compute_70 (sm_70, Tesla V100) cannot be compiled by CUDA 13 NVRTC.
# The script prefers NVRTC 12.x: $ODINFER_NVRTC_PATH if set, else pip's
# nvidia-cuda-nvrtc-cu12 libnvrtc.so.12. NVRTC 12.9's "architectures prior
# to sm_75 are deprecated" warning is expected and is not a failure.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v odin >/dev/null 2>&1; then
  echo "cuda_check: odin is not on PATH (this repo pins dev-2026-07)" >&2
  exit 1
fi

mkdir -p bin
COLLECTIONS=(
  -collection:ggml=packages
  -collection:infer=packages
  -collection:qwen3_5=packages
  -collection:qwen3_5_tokenizer=packages
  -collection:sampler=packages
  -collection:tokenizer=packages
  -collection:cuda=packages
)
DEFINE=(-define:ODINFER_BACKEND="cuda")

echo "== build cuda_kernel_test =="
odin build tools/cuda_kernel_test.odin -file -out:bin/cuda_kernel_test \
  "${DEFINE[@]}" "${COLLECTIONS[@]}"

echo "== build cuda_smoke =="
odin build tools/cuda_smoke.odin -file -out:bin/cuda_smoke \
  "${DEFINE[@]}" "${COLLECTIONS[@]}"

# Print a libnvrtc.so.12 path from pip site-packages, or nothing.
find_pip_nvrtc12() {
  python3 - <<'PY'
import pathlib, site
roots = []
for p in list(site.getsitepackages()) + [site.getusersitepackages()]:
    roots.append(pathlib.Path(p))
for root in roots:
    if not root.is_dir():
        continue
    hits = sorted(root.glob("**/libnvrtc.so.12"))
    if hits:
        print(hits[0])
        break
PY
}

# compute_70 needs NVRTC 12. CUDA 13's library cannot target it.
# ODINFER_NVRTC_PATH is exclusive inside the runtime, so set it to the
# 12.x soname before the syntax check.
prefer_nvrtc12() {
  if [[ -n "${ODINFER_NVRTC_PATH:-}" ]]; then
    echo "cuda_check: ODINFER_NVRTC_PATH=${ODINFER_NVRTC_PATH}"
    return 0
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    return 1
  fi
  local so
  so="$(find_pip_nvrtc12 || true)"
  if [[ -z "${so}" ]]; then
    echo "== installing pip nvidia-cuda-nvrtc-cu12 (NVRTC 12.x for compute_70) =="
    python3 -m pip install --user nvidia-cuda-nvrtc-cu12 >/dev/null
    so="$(find_pip_nvrtc12 || true)"
  fi
  if [[ -n "${so}" && -e "${so}" ]]; then
    ODINFER_NVRTC_PATH="${so}"
    NVRTC_DIR="$(dirname "${so}")"
    export ODINFER_NVRTC_PATH NVRTC_DIR
    echo "cuda_check: ODINFER_NVRTC_PATH=${ODINFER_NVRTC_PATH}"
    return 0
  fi
  return 1
}

echo "== NVRTC 12.x for compute_70 (CUDA 13 NVRTC cannot target sm_70) =="
if ! prefer_nvrtc12; then
  echo "cuda_check: no libnvrtc.so.12. Set ODINFER_NVRTC_PATH to NVRTC 12.x (pip nvidia-cuda-nvrtc-cu12)." >&2
  exit 1
fi

echo "== NVRTC syntax check compute_70 (no GPU required) =="
if ! bin/cuda_kernel_test --nvrtc-only compute_70; then
  echo "cuda_check: NVRTC 12.x compile of compute_70 failed (see ODINFER_NVRTC_PATH)." >&2
  echo "cuda_check: CUDA 13 NVRTC cannot target compute_70. A 13.x driver can JIT PTX from NVRTC 12.9." >&2
  exit 1
fi

has_gpu=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  has_gpu=1
fi

if [[ "${has_gpu}" == "1" ]]; then
  echo "== kernel self-test (CPU reference) =="
  bin/cuda_kernel_test
else
  echo "== no CUDA device visible; skipping cuda_selftest =="
fi

find_model() {
  if [[ -n "${MODEL:-}" && -f "${MODEL}" ]]; then
    echo "${MODEL}"
    return
  fi
  local c
  for c in \
    "models/Qwen3.5-0.8B-MoE-A0.6B-Q4_K_M.gguf" \
    "${HOME}/.odinfer/models/Qwen3.5-0.8B-MoE-A0.6B-Q4_K_M.gguf" \
    "models/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf"
  do
    if [[ -f "${c}" ]]; then
      echo "${c}"
      return
    fi
  done
}

# Mac benchmark 9B (cross-backend numbers). HF main no longer serves this blob.
MAC_9B_SHA_PREFIX="a692de95"
# Re-quant currently on mradermacher main (5780092768 bytes).
REQUANT_9B_SHA_PREFIX="b4363b57"
HERETIC_9B_NAME="DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf"

file_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

model="$(find_model || true)"
checksum_ok=1
if [[ -n "${model}" ]]; then
  sha="$(file_sha256 "${model}")"
  echo "cuda_check: sha256 ${sha}  ${model}"
  base="$(basename "${model}")"
  if [[ "${base}" == "${HERETIC_9B_NAME}" || "${sha}" == ${REQUANT_9B_SHA_PREFIX}* ]]; then
    if [[ "${sha}" == ${MAC_9B_SHA_PREFIX}* ]]; then
      echo "cuda_check: 9B file matches the Mac benchmark (sha256 prefix ${MAC_9B_SHA_PREFIX})"
    else
      echo "cuda_check: this 9B file is not the Mac benchmark (sha256 prefix ${MAC_9B_SHA_PREFIX})." >&2
      echo "cuda_check: HF main was re-quantized (sha256 prefix ${REQUANT_9B_SHA_PREFIX}, 5780092768 bytes, commits 29319295d35e..a713868aa6cd)." >&2
      echo "cuda_check: cross-backend tok/s need the ${MAC_9B_SHA_PREFIX} file. CUDA_CHECK_ALLOW_REQUANT=1 smokes this file anyway." >&2
      if [[ "${CUDA_CHECK_ALLOW_REQUANT:-0}" != "1" ]]; then
        checksum_ok=0
        if [[ "${CUDA_CHECK_REQUIRE_SMOKE:-0}" == "1" || "${has_gpu}" == "1" ]]; then
          exit 1
        fi
      fi
    fi
  fi
fi
if [[ -n "${model}" && "${checksum_ok}" == "1" && "${has_gpu}" == "1" ]]; then
  echo "== smoke ${model} =="
  bin/cuda_smoke "${model}" "${SMOKE_TOKENS:-16}" "${SMOKE_CTX:-2048}"
elif [[ -z "${model}" ]]; then
  echo "== no model file found; smoke skipped =="
  echo "   expected one of:"
  echo "   models/Qwen3.5-0.8B-MoE-A0.6B-Q4_K_M.gguf (524 MB)"
  echo "   ${HOME}/.odinfer/models/Qwen3.5-0.8B-MoE-A0.6B-Q4_K_M.gguf (524 MB)"
  echo "   models/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf (Mac benchmark sha256 prefix ${MAC_9B_SHA_PREFIX})"
  if [[ "${CUDA_CHECK_REQUIRE_SMOKE:-0}" == "1" ]]; then
    exit 1
  fi
elif [[ "${checksum_ok}" != "1" ]]; then
  echo "== smoke skipped; 9B sha256 is not the Mac benchmark prefix ${MAC_9B_SHA_PREFIX} =="
else
  echo "== model ${model} present but no GPU; smoke skipped =="
fi

echo "== cuda_check done =="
