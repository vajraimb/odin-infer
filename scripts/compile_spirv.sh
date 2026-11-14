#!/usr/bin/env bash
# Regenerate embedded SPIR-V. A normal Windows build does not need this:
# packages/qwen3_5/spirv/*.spv is committed. Requires glslangValidator
# (Vulkan SDK or the glslang-tools package) on PATH.
set -euo pipefail
cd "$(dirname "$0")/.."
S=packages/qwen3_5/shaders
O=packages/qwen3_5/spirv
mkdir -p "$O"
if ! command -v glslangValidator >/dev/null 2>&1; then
	echo "compile_spirv: glslangValidator not on PATH" >&2
	exit 1
fi
# gemv/element shaders target Vulkan 1.2. The optional integer-dot and
# cooperative-matrix shaders need Vulkan 1.3 and a glslang that knows
# GL_EXT_integer_dot_product (tested with glslang 16.6).
V12=(glslangValidator -V --target-env vulkan1.2 -I"$S")
V13=(glslangValidator -V --target-env vulkan1.3 -I"$S")
compile() {
	echo "== $* =="
	"${V12[@]}" "$@"
}
compile -DLOCAL_X=1 -o "$O/gemv_exact.spv" "$S/gemv.comp"
compile -DLOCAL_X=64 -o "$O/gemv_par.spv" "$S/gemv.comp"
compile -DLOCAL_X=64 -DSUBGROUP=1 -o "$O/gemv_sub.spv" "$S/gemv.comp"
compile -o "$O/elem.spv" "$S/elem.comp"
compile -o "$O/rmsnorm.spv" "$S/rmsnorm.comp"
compile -o "$O/l2norm.spv" "$S/l2norm.comp"
compile -o "$O/rope.spv" "$S/rope.comp"
compile -o "$O/gated_rms.spv" "$S/gated_rms.comp"
compile -o "$O/attn.spv" "$S/attn.comp"
compile -o "$O/conv.spv" "$S/conv.comp"
compile -o "$O/delta.spv" "$S/delta.comp"
echo "== q8_dp4a =="
"${V13[@]}" -o "$O/q8_dp4a.spv" "$S/q8_dp4a.comp"
echo "== coopmat =="
"${V13[@]}" -o "$O/coopmat.spv" "$S/coopmat.comp"
echo "compile_spirv: wrote $O"
