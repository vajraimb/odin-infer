/* Elementwise ops for Qwen3.5 forward (CPU path). */

package qwen3_5

import "core:math"

sigmoid_f32 :: proc(x: f32) -> f32 {
	if x >= 0.0 {
		z := math.exp_f32(-x)
		return 1.0 / (1.0 + z)
	}
	z := math.exp_f32(x)
	return z / (1.0 + z)
}

silu_f32 :: proc(x: f32) -> f32 {
	return x * sigmoid_f32(x)
}

softplus_f32 :: proc(x: f32) -> f32 {
	if x > 20.0 do return x
	if x < -20.0 do return 0.0
	return f32(math.ln_f64(f64(1.0 + math.exp_f32(x))))
}

// l2norm matching ggml's ggml_l2_norm: scale = 1 / max(||x||, eps) (eps is a
// floor on the norm, not inside the sqrt).
l2norm_into :: proc(o, x: []f32, n: int, eps: f32) {
	ss: f32 = 0
	for i in 0 ..< n {
		ss += x[i] * x[i]
	}
	inv := 1.0 / math.max(math.sqrt_f32(ss), eps)
	for i in 0 ..< n {
		o[i] = x[i] * inv
	}
}

// Standard RMSNorm. weight must already encode the effective scale; the loader
// bakes (1+w) for Qwen3.5RMSNorm tensors so all call sites use this one form.
rmsnorm :: proc(o, x, weight: []f32, eps: f32) {
	size := len(x)
	ss: f32 = 0
	for j in 0 ..< size {
		ss += x[j] * x[j]
	}
	ss = ss / f32(size) + eps
	inv := 1.0 / math.sqrt_f32(ss)
	for j in 0 ..< size {
		o[j] = weight[j] * (inv * x[j])
	}
}

softmax :: proc(x: []f32) {
	if len(x) == 0 do return
	max_val := x[0]
	for i in 1 ..< len(x) {
		if x[i] > max_val {
			max_val = x[i]
		}
	}
	sum: f32 = 0
	for i in 0 ..< len(x) {
		x[i] = math.exp_f32(x[i] - max_val)
		sum += x[i]
	}
	for i in 0 ..< len(x) {
		x[i] /= sum
	}
}

// Partial-rotary MRoPE for text-only inputs. For text tokens the three MRoPE
// grids (T,H,W) share the same position id, so the interleaved layout in
// Qwen3_5TextRotaryEmbedding.apply_interleaved_mrope collapses to a plain 1D
// RoPE over the first rotary_dim elements of the head; the rest passes through.
// Pairs are (v[i], v[i + rotary_dim/2]) for i in [0, rotary_dim/2).
apply_mrope_text_head :: proc(v: []f32, pos: int, inv_freq: []f32, rotary_dim: int) {
	half := rotary_dim / 2
	pf := f32(pos)
	for i in 0 ..< half {
		angle := pf * inv_freq[i]
		c := math.cos_f32(angle)
		s := math.sin_f32(angle)
		x := v[i]
		y := v[i + half]
		v[i] = x * c - y * s
		v[i + half] = x * s + y * c
	}
}

// Depthwise causal conv1d single-token update.
//   state[c]  holds the previous (kernel-1) inputs for channel c (oldest first)
//   window  = state[c] ++ [new_in[c]]   (length = kernel)
//   out[c]  = silu(sum_k weight[c,k] * window[k])
//   state[c] := window[1:]              (slide)
// weight is laid out as [conv_dim, kernel] (row-major).
conv1d_step :: proc(
	out, new_in, state: []f32,
	weight: []f32,
	conv_dim, kernel: int,
) {
	km1 := kernel - 1
	for c in 0 ..< conv_dim {
		wrow := c * kernel
		sb := c * km1
		acc: f32 = weight[wrow + km1] * new_in[c]
		for k in 0 ..< km1 {
			acc += weight[wrow + k] * state[sb + k]
		}
		out[c] = silu_f32(acc)
		for k in 0 ..< km1 - 1 {
			state[sb + k] = state[sb + k + 1]
		}
		state[sb + km1 - 1] = new_in[c]
	}
}

// Gated delta-rule recurrent step for ONE value head (per-token decode path,
// matches torch_recurrent_gated_delta_rule).
//   state:  [head_k_dim * head_v_dim], row-major (k outer, v inner)
//   q_t:    [head_k_dim]  (already l2-normalized AND scaled by 1/sqrt(head_k_dim))
//   k_t:    [head_k_dim]  (already l2-normalized)
//   v_t:    [head_v_dim]
//   beta:   scalar in (0,1)
//   g_decay: scalar = exp(g), g = -exp(A_log) * softplus(a + dt_bias)  (so in (0,1])
//   delta_scratch: [head_v_dim] temporary
//   out:    [head_v_dim]
delta_recurrent_step :: proc(
	out, state, q_t, k_t, v_t, delta_scratch: []f32,
	beta, g_decay: f32,
	head_k_dim, head_v_dim: int,
) {
	for i in 0 ..< head_k_dim * head_v_dim {
		state[i] *= g_decay
	}

	delta := delta_scratch
	for j in 0 ..< head_v_dim {
		s: f32 = 0
		for i in 0 ..< head_k_dim {
			s += state[i * head_v_dim + j] * k_t[i]
		}
		delta[j] = (v_t[j] - s) * beta
	}

	for i in 0 ..< head_k_dim {
		ki := k_t[i]
		base := i * head_v_dim
		for j in 0 ..< head_v_dim {
			state[base + j] += ki * delta[j]
		}
	}

	for j in 0 ..< head_v_dim {
		s: f32 = 0
		for i in 0 ..< head_k_dim {
			s += state[i * head_v_dim + j] * q_t[i]
		}
		out[j] = s
	}
}

inv_freq_default :: proc(rope_theta: f32, rotary_dim: int, out: []f32) {
	rd := f32(rotary_dim)
	for i in 0 ..< rotary_dim / 2 {
		out[i] = 1.0 / math.pow_f32(rope_theta, f32(i) / (rd / 2.0))
	}
}
