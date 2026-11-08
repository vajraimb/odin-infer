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
