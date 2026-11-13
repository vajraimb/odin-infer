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
#   models/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf    5.6 GB
#
# SMOKE_TOKENS (default 16), SMOKE_CTX (default 2048).
# CUDA_CHECK_REQUIRE_SMOKE=1 fails when no model file is present.
# NVRTC_DIR is honoured by the runtime. If NVRTC is missing, the script
# tries a user pip install of nvidia-cuda-nvrtc-cu12 and points NVRTC_DIR
# at the wheel's lib directory.
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

try_pip_nvrtc() {
  if ! command -v python3 >/dev/null 2>&1; then
    return 1
  fi
  echo "== NVRTC not found; trying pip nvidia-cuda-nvrtc-cu12 =="
  python3 -m pip install --user nvidia-cuda-nvrtc-cu12 >/dev/null
  local lib
  lib="$(python3 - <<'PY'
import pathlib, site
roots = []
for p in list(site.getsitepackages()) + [site.getusersitepackages()]:
    roots.append(pathlib.Path(p))
cands = []
for root in roots:
    if root.is_dir():
        cands += list(root.glob("**/libnvrtc.so.13"))
        cands += list(root.glob("**/libnvrtc.so.12"))
print(cands[0].parent if cands else "")
PY
)"
  if [[ -n "${lib}" && -d "${lib}" ]]; then
    export NVRTC_DIR="${lib}"
    echo "cuda_check: NVRTC_DIR=${NVRTC_DIR}"
    return 0
  fi
  return 1
}

echo "== NVRTC syntax check compute_70 (no GPU required) =="
if ! bin/cuda_kernel_test --nvrtc-only compute_70; then
  if try_pip_nvrtc; then
    bin/cuda_kernel_test --nvrtc-only compute_70
  else
    echo "cuda_check: NVRTC compile failed and no wheel was installed" >&2
    exit 1
  fi
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

model="$(find_model || true)"
if [[ -n "${model}" && "${has_gpu}" == "1" ]]; then
  echo "== smoke ${model} =="
  bin/cuda_smoke "${model}" "${SMOKE_TOKENS:-16}" "${SMOKE_CTX:-2048}"
elif [[ -z "${model}" ]]; then
  echo "== no model file found; smoke skipped =="
  echo "   expected one of:"
  echo "   models/Qwen3.5-0.8B-MoE-A0.6B-Q4_K_M.gguf (524 MB)"
  echo "   ${HOME}/.odinfer/models/Qwen3.5-0.8B-MoE-A0.6B-Q4_K_M.gguf (524 MB)"
  echo "   models/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf (5.6 GB)"
  if [[ "${CUDA_CHECK_REQUIRE_SMOKE:-0}" == "1" ]]; then
    exit 1
  fi
else
  echo "== model ${model} present but no GPU; smoke skipped =="
fi

echo "== cuda_check done =="
