/* Unit tests for the novel Qwen3.5 math (no GGUF required). */

package qwen3_5

import "core:fmt"
import "core:math"
import "core:testing"

approx :: proc(t: ^testing.T, a, b, tol: f32, msg: string) {
	testing.expect(t, math.abs(a - b) <= tol, fmt.tprintf("%s: |%f - %f| > %f", msg, a, b, tol))
}

@(test)
delta_recurrent_first_write :: proc(t: ^testing.T) {
	// head_k_dim=2, head_v_dim=2; state starts at zero; beta=1, g_decay=1.
	// With q_t = k_t = [1,0] and v_t = [1,1], the first step must reproduce v.
	state := make([]f32, 4)
	defer delete(state)
	q := []f32{1, 0}
	k := []f32{1, 0}
	v := []f32{1, 1}
	out := make([]f32, 2)
	defer delete(out)
	delta := make([]f32, 2)
	defer delete(delta)

	delta_recurrent_step(out, state, q, k, v, delta, 1.0, 1.0, 2, 2)

	approx(t, out[0], 1.0, 1e-5, "out[0]")
	approx(t, out[1], 1.0, 1e-5, "out[1]")
	approx(t, state[0], 1.0, 1e-5, "state[0,0]")
	approx(t, state[1], 1.0, 1e-5, "state[0,1]")
	approx(t, state[2], 0.0, 1e-5, "state[1,0]")
	approx(t, state[3], 0.0, 1e-5, "state[1,1]")
}

@(test)
delta_recurrent_partial_beta :: proc(t: ^testing.T) {
	// beta=0.5 should halve the first write.
	state := make([]f32, 4)
	defer delete(state)
	q := []f32{1, 0}
	k := []f32{1, 0}
	v := []f32{1, 1}
	out := make([]f32, 2)
	defer delete(out)
	delta := make([]f32, 2)
	defer delete(delta)

	delta_recurrent_step(out, state, q, k, v, delta, 0.5, 1.0, 2, 2)

	approx(t, out[0], 0.5, 1e-5, "out[0]")
	approx(t, out[1], 0.5, 1e-5, "out[1]")
}

@(test)
delta_recurrent_decay :: proc(t: ^testing.T) {
	// Pre-fill state, g_decay=0.5; with v matching the post-decay memory the
	// delta-rule writes nothing (delta=0), so state just halves and out reads it.
	// state[0,0]=2 -> 1 after decay; kv_mem=1; v=1 -> delta=0; out reads state=1.
	state := []f32{2.0, 0.0, 0.0, 0.0}
	q := []f32{1, 0}
	k := []f32{1, 0}
	v := []f32{1.0, 0.0}
	out := make([]f32, 2)
	defer delete(out)
	delta := make([]f32, 2)
	defer delete(delta)

	delta_recurrent_step(out, state, q, k, v, delta, 1.0, 0.5, 2, 2)

	approx(t, state[0], 1.0, 1e-5, "state[0,0] after decay (no correct)")
	approx(t, out[0], 1.0, 1e-5, "out[0] reads decayed memory")
}

@(test)
conv1d_single_token :: proc(t: ^testing.T) {
	// conv_dim=1, kernel=2; weight=[1,1]; state=[2]; new=[0].
	// out = silu(1*2 + 1*0) = silu(2) = 2*sigmoid(2) ~= 1.7616; state slides to [0].
	conv_dim := 1
	kernel := 2
	w := []f32{1.0, 1.0}
	state := make([]f32, conv_dim * (kernel - 1))
	state[0] = 2.0
	defer delete(state)
	new_in := []f32{0.0}
	out := make([]f32, conv_dim)
	defer delete(out)

	conv1d_step(out, new_in, state, w, conv_dim, kernel)

	expected := silu_f32(2.0)
	approx(t, out[0], expected, 1e-5, "conv out")
	approx(t, state[0], 0.0, 1e-5, "state slid to new_in")
}

@(test)
mrope_text_zero_pos_is_identity :: proc(t: ^testing.T) {
	// At pos=0 every angle is 0 (cos=1, sin=0), so the vector is unchanged.
	rotary_dim := 4
	inv_freq := make([]f32, rotary_dim / 2)
	defer delete(inv_freq)
	inv_freq_default(10000.0, rotary_dim, inv_freq)
	v := []f32{1.5, -2.0, 3.0, 4.0}
	apply_mrope_text_head(v, 0, inv_freq, rotary_dim)
	approx(t, v[0], 1.5, 1e-5, "v[0]")
	approx(t, v[1], -2.0, 1e-5, "v[1]")
	approx(t, v[2], 3.0, 1e-5, "v[2]")
	approx(t, v[3], 4.0, 1e-5, "v[3]")
}

@(test)
mrope_text_partial_passthrough :: proc(t: ^testing.T) {
	// rotary_dim=2 on a head_dim=4 vector: only the first 2 elements rotate,
	// the rest must pass through untouched.
	rotary_dim := 2
	inv_freq := make([]f32, rotary_dim / 2)
	defer delete(inv_freq)
	inv_freq_default(10000.0, rotary_dim, inv_freq)
	v := []f32{1.0, 0.0, 5.0, 6.0}
	apply_mrope_text_head(v, 7, inv_freq, rotary_dim)
	// v[2], v[3] are beyond rotary_dim -> unchanged.
	approx(t, v[2], 5.0, 1e-5, "v[2] passes through")
	approx(t, v[3], 6.0, 1e-5, "v[3] passes through")
}

@(test)
rmsnorm_known :: proc(t: ^testing.T) {
	// x = [3,4]; rms = sqrt((9+16)/2) = sqrt(12.5); normed * weight.
	// With weight all 1.0 and eps tiny: out = x / sqrt(12.5).
	x := []f32{3.0, 4.0}
	w := []f32{1.0, 1.0}
	o := make([]f32, 2)
	defer delete(o)
	rmsnorm(o, x, w, 1e-8)
	inv := 1.0 / math.sqrt_f32(12.5)
	approx(t, o[0], 3.0 * inv, 1e-5, "o[0]")
	approx(t, o[1], 4.0 * inv, 1e-5, "o[1]")
}

@(test)
l2norm_unit :: proc(t: ^testing.T) {
	x := []f32{3.0, 4.0}
	o := make([]f32, 2)
	defer delete(o)
	l2norm_into(o, x, 2, 1e-6)
	// |x| = 5, so o = [0.6, 0.8].
	approx(t, o[0], 0.6, 1e-5, "o[0]")
	approx(t, o[1], 0.8, 1e-5, "o[1]")
}

// ---------- Q8_0 KV cache block math (kv_q8.odin) ----------

@(test)
q8_block_unit_scale_roundtrip :: proc(t: ^testing.T) {
	// amax = 127 -> d = 1.0 exactly, so q = round(v) and integer values
	// round-trip exactly.
	src := make([]f32, QK8_0)
	defer delete(src)
	for j in 0 ..< QK8_0 {
		src[j] = f32(j) - 16.0 // all integers
	}
	src[7] = 127.0
	src[21] = -127.0
	blk := make([]u8, Q8_0_BLOCK_BYTES)
	defer delete(blk)
	out := make([]f32, QK8_0)
	defer delete(out)
	quantize_q8_block(src, blk)
	dequantize_q8_block(blk, out)
	for j in 0 ..< QK8_0 {
		approx(t, out[j], src[j], 1e-6, fmt.tprintf("q8 exact roundtrip [%d]", j))
	}
	// .5 ties round half away from zero (matches MSL round()/llama.cpp roundf)
	src2 := make([]f32, QK8_0)
	defer delete(src2)
	src2[0] = 9.5
	src2[1] = -9.5
	src2[2] = 127.0
	quantize_q8_block(src2, blk)
	dequantize_q8_block(blk, out)
	approx(t, out[0], 10.0, 1e-6, "q8 +9.5 rounds to +10")
	approx(t, out[1], -10.0, 1e-6, "q8 -9.5 rounds to -10")
}

@(test)
q8_block_zero_block :: proc(t: ^testing.T) {
	// All-zero block: d = 0, q = 0, dequant stays zero (matches llama.cpp
	// id = d ? 1/d : 0 and the MSL store_kv_q8 kernel).
	src := make([]f32, QK8_0)
	defer delete(src)
	blk := make([]u8, Q8_0_BLOCK_BYTES)
	defer delete(blk)
	out := make([]f32, QK8_0)
	defer delete(out)
	quantize_q8_block(src, blk)
	dequantize_q8_block(blk, out)
	for j in 0 ..< QK8_0 {
		approx(t, out[j], 0.0, 1e-12, fmt.tprintf("q8 zero [%d]", j))
	}
}

@(test)
q8_block_quantization_error_bound :: proc(t: ^testing.T) {
	// Error per element is < d/2 (round-to-nearest), and d = amax/127.
	src := make([]f32, QK8_0)
	defer delete(src)
	amax: f32 = 0
	for j in 0 ..< QK8_0 {
		src[j] = math.sin(f32(j) * 0.7) * 3.7 + f32(j) * 0.01
		a := math.abs(src[j])
		if a > amax { amax = a }
	}
	blk := make([]u8, Q8_0_BLOCK_BYTES)
	defer delete(blk)
	out := make([]f32, QK8_0)
	defer delete(out)
	quantize_q8_block(src, blk)
	dequantize_q8_block(blk, out)
	// |q - v/d| <= 0.5 and |d' - d| <= d*2^-11 (f16), so |out - src| <=
	// d*(0.5 + 127*2^-11) < d*0.5621.
	bound := (amax / 127.0) * 0.5621
	for j in 0 ..< QK8_0 {
		testing.expect(t, math.abs(out[j] - src[j]) <= bound, fmt.tprintf("q8 err [%d]: |%f-%f| > %f", j, out[j], src[j], bound))
	}
}

@(test)
q8_v_mix_outlier_exact :: proc(t: ^testing.T) {
	// One spike sets a symmetric Q8 scale and coarsens the other 31 values.
	// The mixed block keeps that spike as f16 and quantizes the rest against
	// their own (much smaller) amax.
	src := make([]f32, QK8_0)
	defer delete(src)
	for j in 0 ..< QK8_0 {
		src[j] = math.sin(f32(j) * 0.4) * 0.8
	}
	src[17] = 85.0
	blk := make([]u8, Q8_V_MIX_BLOCK_BYTES)
	defer delete(blk)
	out := make([]f32, QK8_0)
	defer delete(out)
	quantize_q8_v_mix_block(src, blk)
	testing.expect(t, int(blk[2]) == 17, "outlier index is the spike")
	dequantize_q8_v_mix_block(blk, out)
	approx(t, out[17], 85.0, 1e-3, "spike survives as f16")
	// Residual amax is < 1, so the Q8 error on the other elements is tiny
	// compared with d=85/127 of a symmetric block.
	for j in 0 ..< QK8_0 {
		if j == 17 { continue }
		testing.expect(t, math.abs(out[j] - src[j]) < 0.01, fmt.tprintf("mix residual [%d]", j))
	}
}

@(test)
q8_v_mix_tie_takes_lowest_index :: proc(t: ^testing.T) {
	src := make([]f32, QK8_0)
	defer delete(src)
	src[3] = 4.0
	src[9] = -4.0
	blk := make([]u8, Q8_V_MIX_BLOCK_BYTES)
	defer delete(blk)
	quantize_q8_v_mix_block(src, blk)
	testing.expect(t, int(blk[2]) == 3, "tie keeps the lowest index")
	out := make([]f32, QK8_0)
	defer delete(out)
	dequantize_q8_v_mix_block(blk, out)
	approx(t, out[3], 4.0, 1e-4, "tied outlier exact")
	approx(t, out[9], -4.0, 1e-3, "second peak quantized against itself as amax")
}

@(test)
q8_kv_byte_math :: proc(t: ^testing.T) {
	testing.expect(t, kv_pos_bytes(1024) == (1024 / 32) * 34, "kv_pos_bytes(1024)")
	testing.expect(t, kv_pos_bytes(512) == (512 / 32) * 34, "kv_pos_bytes(512)")
	testing.expect(t, kv_layer_bytes_q8(4096, 1024) == 4096 * 32 * 34, "kv_layer_bytes_q8")
	testing.expect(t, kv_total_bytes_q8(16, 32768, 1024) == 16 * 32768 * 32 * 34, "kv_total_bytes_q8(27B@32k)")
	// 27B fp32 KV @ 32k = 2*16*32768*1024*4 = 4.295 GB; Q8 = 1.141 GB -> 3.76x smaller.
	testing.expect(t, f64(2 * 16 * 32768 * 1024 * 4) / f64(kv_total_bytes_q8(16, 32768, 1024)) > 3.7, "q8 vs fp32 ratio")
}

@(test)
q8_kv_row_roundtrip :: proc(t: ^testing.T) {
	// Full-row (kv_dim=512) quantize/dequantize roundtrip through the row helpers.
	row := make([]f32, 512)
	defer delete(row)
	for j in 0 ..< len(row) {
		row[j] = math.cos(f32(j) * 0.13) * 5.0
	}
	blk := make([]u8, kv_pos_bytes(512))
	defer delete(blk)
	out := make([]f32, 512)
	defer delete(out)
	quantize_kv_row_q8(row, blk)
	dequantize_kv_row_q8(blk, out)
	amax: f32 = 0
	for j in 0 ..< len(row) {
		a := math.abs(row[j])
		if a > amax { amax = a }
	}
	bound := (amax / 127.0) * 0.5621
	for j in 0 ..< len(row) {
		testing.expect(t, math.abs(out[j] - row[j]) <= bound, fmt.tprintf("q8 row err [%d]", j))
	}
}
