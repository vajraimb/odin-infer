// Shared bindings for odinfer's Vulkan compute shaders.
//
// Design (on-the-fly GGUF dequant, subgroup reductions, optional
// integer dot product and cooperative matrix, host-visible memory for
// UMA iGPUs, one command buffer per forward) follows llama.cpp's
// ggml-vulkan backend, which is MIT licensed:
//   Copyright (c) 2023-2025 The ggml authors
//   https://github.com/ggml-org/llama.cpp
// The GLSL here is written for odinfer. It is not a copy of those shaders.
// Dequant formulas match packages/ggml/quant.odin (the engine's CPU path).

layout(set = 0, binding = 0) readonly buffer Weights { uint wdata[]; };
layout(set = 0, binding = 1) buffer Acts { float adata[]; };

layout(push_constant) uniform PC {
	uint u[24];
	float f[8];
} pc;

uint load_u8(uint off) {
	uint v = wdata[off >> 2u];
	return (v >> ((off & 3u) * 8u)) & 0xFFu;
}

uint load_u32(uint off) {
	return load_u8(off) | (load_u8(off + 1u) << 8u) | (load_u8(off + 2u) << 16u) | (load_u8(off + 3u) << 24u);
}

float load_f32(uint off) {
	return uintBitsToFloat(load_u32(off));
}

float load_f16(uint off) {
	uint h = load_u8(off) | (load_u8(off + 1u) << 8u);
	uint sign = (h >> 15u) & 1u;
	uint exp = (h >> 10u) & 0x1Fu;
	uint mant = h & 0x3FFu;
	uint outb;
	if (exp == 0u) {
		if (mant == 0u) {
			outb = sign << 31u;
		} else {
			exp = 127u - 15u + 1u;
			while ((mant & 0x400u) == 0u) {
				mant <<= 1u;
				exp -= 1u;
			}
			mant &= 0x3FFu;
			outb = (sign << 31u) | (exp << 23u) | (mant << 13u);
		}
	} else if (exp == 31u) {
		outb = (sign << 31u) | (0xFFu << 23u) | (mant << 13u);
	} else {
		outb = (sign << 31u) | ((exp + (127u - 15u)) << 23u) | (mant << 13u);
	}
	return uintBitsToFloat(outb);
}

float X(uint i) { return adata[i]; }
void Y(uint i, float v) { adata[i] = v; }

float sigmoid(float x) {
	if (x >= 0.0) {
		float z = exp(-x);
		return 1.0 / (1.0 + z);
	}
	float z = exp(x);
	return z / (1.0 + z);
}

float silu(float x) { return x * sigmoid(x); }

float softplus(float x) {
	if (x > 20.0) return x;
	if (x < -20.0) return 0.0;
	return log(1.0 + exp(x));
}

uvec2 scale_min_k4(uint j, uint scales_off) {
	uint sc;
	uint m;
	if (j < 4u) {
		sc = load_u8(scales_off + j) & 63u;
		m = load_u8(scales_off + j + 4u) & 63u;
	} else {
		sc = (load_u8(scales_off + j + 4u) & 0xFu) | ((load_u8(scales_off + j - 4u) >> 6u) << 4u);
		m = (load_u8(scales_off + j + 4u) >> 4u) | ((load_u8(scales_off + j) >> 6u) << 4u);
	}
	return uvec2(sc, m);
}
