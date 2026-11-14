#!/usr/bin/env bash
# Build the Vulkan backend and, when a device is available, check it.
#
#   ./scripts/vulkan_check.sh
#
# Odin dev-2026-07 must be on PATH (the same pin as
# .github/workflows/release.yml). SPIR-V is already in packages/qwen3_5/spirv;
# this script does not call glslangValidator.
#
# A machine with no DRM node and Mesa's lavapipe ICD is pointed at that ICD.
# Set VK_ICD_FILENAMES yourself to choose a device. On Windows the loader
# uses the driver's vulkan-1.dll; do not install the Vulkan SDK to build
# or to run.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v odin >/dev/null 2>&1; then
  echo "vulkan_check: odin is not on PATH (this repo pins dev-2026-07)" >&2
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
  -collection:vulkan=packages
)
DEFINE=(-define:ODINFER_BACKEND=vulkan)

echo "== build vulkan_kernel_test =="
odin build tools/vulkan_kernel_test.odin -file -out:bin/vulkan_kernel_test \
  "${DEFINE[@]}" "${COLLECTIONS[@]}"

echo "== build vulkan_smoke (not executed without a real GGUF) =="
odin build tools/vulkan_smoke.odin -file -out:bin/vulkan_smoke \
  "${DEFINE[@]}" "${COLLECTIONS[@]}"

echo "== build greedy_dump (cpu + vulkan) =="
odin build tools/greedy_dump.odin -file -out:bin/greedy_cpu "${COLLECTIONS[@]}"
odin build tools/greedy_dump.odin -file -out:bin/greedy_vk \
  "${DEFINE[@]}" "${COLLECTIONS[@]}"

if [[ -z "${VK_ICD_FILENAMES:-}${VK_DRIVER_FILES:-}" && ! -e /dev/dri && -f /usr/share/vulkan/icd.d/lvp_icd.json ]]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
  export VK_DRIVER_FILES="$VK_ICD_FILENAMES"
  echo "vulkan_check: no /dev/dri, using lavapipe ($VK_ICD_FILENAMES)"
fi

echo "== vulkan_kernel_test =="
bin/vulkan_kernel_test

TINY="${TINY_GGUF:-/tmp/odinfer_vulkan_tiny.gguf}"
echo "== synthetic hybrid GGUF =="
python3 tools/make_vulkan_gguf.py "$TINY"

echo "== greedy exact =="
ODINFER_LOGITS_OUT=/tmp/odinfer_logits_cpu.f32 bin/greedy_cpu "$TINY" 32 4 1 2 3 4 >/tmp/odinfer_greedy_cpu.txt
ODINFER_LOGITS_OUT=/tmp/odinfer_logits_vk.f32 bin/greedy_vk "$TINY" 32 4 1 2 3 4 >/tmp/odinfer_greedy_vk.txt
echo "  cpu: $(cat /tmp/odinfer_greedy_cpu.txt)"
echo "  vk:  $(cat /tmp/odinfer_greedy_vk.txt)"
if ! cmp -s /tmp/odinfer_greedy_cpu.txt /tmp/odinfer_greedy_vk.txt; then
  echo "vulkan_check: exact greedy token mismatch" >&2
  exit 1
fi
python3 - <<'PY'
import numpy as np
a = np.fromfile("/tmp/odinfer_logits_cpu.f32", dtype=np.float32)
b = np.fromfile("/tmp/odinfer_logits_vk.f32", dtype=np.float32)
d = np.max(np.abs(a - b))
print(f"vulkan_check: exact prefill logits max_abs={d:.8g} argmax={int(a.argmax())}/{int(b.argmax())}")
PY

echo "== greedy parallel =="
ODINFER_VULKAN_PARALLEL=1 ODINFER_LOGITS_OUT=/tmp/odinfer_logits_par.f32 \
  bin/greedy_vk "$TINY" 32 4 1 2 3 4 >/tmp/odinfer_greedy_par.txt
echo "  par: $(cat /tmp/odinfer_greedy_par.txt)"
if ! cmp -s /tmp/odinfer_greedy_cpu.txt /tmp/odinfer_greedy_par.txt; then
  echo "vulkan_check: parallel greedy token mismatch" >&2
  exit 1
fi
python3 - <<'PY'
import numpy as np
a = np.fromfile("/tmp/odinfer_logits_cpu.f32", dtype=np.float32)
b = np.fromfile("/tmp/odinfer_logits_par.f32", dtype=np.float32)
d = np.max(np.abs(a - b))
print(f"vulkan_check: parallel prefill logits max_abs={d:.8g} argmax={int(a.argmax())}/{int(b.argmax())}")
PY

if { [[ -d models ]] || mkdir -p models; } && python3 tools/make_moe_gguf.py >/dev/null; then
  echo "== moe greedy (F32 qwen35moe, both layers full attention) =="
  moe_cpu="$(bin/greedy_cpu models/moe_test.gguf 64 4 3 5 7)"
  moe_vk="$(bin/greedy_vk models/moe_test.gguf 64 4 3 5 7)"
  echo "  cpu: $moe_cpu"
  echo "  vk:  $moe_vk"
  if [[ "$moe_cpu" != "$moe_vk" ]]; then
    echo "vulkan_check: moe greedy token mismatch" >&2
    exit 1
  fi
fi

echo "vulkan_check: PASS"
