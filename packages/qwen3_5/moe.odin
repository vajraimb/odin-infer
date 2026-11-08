/* Qwen3.5-MoE CPU forward (qwen35moe): routed experts with softmax top-k
   gating + gated shared expert. Mirrors llama.cpp build_moe_ffn for
   LLM_ARCH_QWEN35MOE (LLM_FFN_SILU, norm_w=true, SOFTMAX gating, no bias,
   expert_weights_scale applied to normalized router weights).

   Expert tensor layout (verified against the real unsloth 35B-A3B GGUF):
     ffn_gate_inp.weight   [dim, n_expert]           (ne0 = dim)
     ffn_up_exps.weight    [dim, n_ff_exp, n_expert] (ne0 = dim, contraction)
     ffn_gate_exps.weight  [dim, n_ff_exp, n_expert]
     ffn_down_exps.weight  [n_ff_exp, dim, n_expert] (ne0 = n_ff_exp)
     ffn_*_shexp.weight    2D, same orientation as the dense MLP
     ffn_gate_inp_shexp.weight  [dim]

   Routing: logits = x @ gate_inp^T -> softmax -> top-k -> weights renormalized
   (sum clamped to >= 6.1035e-5, the F16 min) -> weights *= expert_w_scale.
*/

package qwen3_5

import ggml "ggml:ggml"

import "core:fmt"
import "core:math"
import "core:testing"

MOE_SUM_CLAMP :: f32(6.103515625e-5) // F16 min normal, llama.cpp's divisor clamp
MOE_MAX_EXPERTS :: 256

// GEMV over one expert of a 3D [cols, rows, n_expert] tensor. Each expert's
// rows are quantized independently (row length = cols = ne0).
moe_expert_gemv :: proc(out, x, scratch: []f32, w: ^Tensor, expert, rows, cols, n_expert: int) {
	rb := ggml.row_byte_size(w.kind, cols)
	base := expert * rows * rb
	for r in 0 ..< rows {
		ggml.dequant_row(w.kind, w.data[base + r * rb:], cols, scratch)
		sum: f32 = 0.0
		for j in 0 ..< cols {
			sum += scratch[j] * x[j]
		}
		out[r] = sum
	}
}

// MoE FFN on a normalized input vector; writes the result into out (not
// accumulated). Buffers are allocated per call (CPU correctness path; the
// Metal MoE path will manage its own scratch).
moe_ffn :: proc(x, out: []f32, mw: ^Moe_Weights, p: Config) {
	n_exp := p.n_expert
	used := min(p.n_expert_used, n_exp)

	logits := make([]f32, n_exp)
	defer delete(logits)
	weights := make([]f32, used)
	defer delete(weights)
	h := make([]f32, p.n_ff_exp)
	defer delete(h)
	g := make([]f32, p.n_ff_exp)
	defer delete(g)
	scratch := make([]f32, max(p.dim, p.n_ff_exp))
	defer delete(scratch)

	// router: logits[j] = dot(x, gate_inp[j])
	gw := mw.gate_inp
	for j in 0 ..< n_exp {
		row := gw[j * p.dim:(j + 1) * p.dim]
		sum: f32 = 0.0
		for i in 0 ..< p.dim {
			sum += row[i] * x[i]
		}
		logits[j] = sum
	}
	softmax(logits)

	// top-k selection (masking argmax loop; n_expert <= 256)
	sel: [MOE_MAX_EXPERTS]int
	picked: [MOE_MAX_EXPERTS]bool
	for k in 0 ..< used {
		best := -1
		best_v: f32 = -1e30
		for j in 0 ..< n_exp {
			if !picked[j] && logits[j] > best_v {
				best_v = logits[j]
				best = j
			}
		}
		sel[k] = best
		picked[best] = true
	}
	// normalized weights (clamped divisor) + expert_weights_scale
	sum: f32 = 0.0
	for k in 0 ..< used {
		weights[k] = logits[sel[k]]
		sum += weights[k]
	}
	if sum < MOE_SUM_CLAMP {
		sum = MOE_SUM_CLAMP
	}
	for k in 0 ..< used {
		weights[k] /= sum
	}
	if p.expert_w_scale != 1.0 && p.expert_w_scale != 0.0 {
		for k in 0 ..< used {
			weights[k] *= p.expert_w_scale
		}
	}

	for i in 0 ..< p.dim {
		out[i] = 0.0
	}

	for k in 0 ..< used {
		e := sel[k]
		wgt := weights[k]
		moe_expert_gemv(h, x, scratch, &mw.up_exps, e, p.n_ff_exp, p.dim, n_exp)
		moe_expert_gemv(g, x, scratch, &mw.gate_exps, e, p.n_ff_exp, p.dim, n_exp)
		for f in 0 ..< p.n_ff_exp {
			h[f] = silu_f32(g[f]) * h[f] // SwiGLU: silu(gate) * up (llama convention)
		}
		// down: out += wgt * (h @ down_e^T); down_e is [dim rows, n_ff_exp cols]
		rb_d := ggml.row_byte_size(mw.down_exps.kind, p.n_ff_exp)
		base_d := e * p.dim * rb_d
		for r in 0 ..< p.dim {
			ggml.dequant_row(mw.down_exps.kind, mw.down_exps.data[base_d + r * rb_d:], p.n_ff_exp, scratch)
			s: f32 = 0.0
			for f in 0 ..< p.n_ff_exp {
				s += scratch[f] * h[f]
			}
			out[r] += wgt * s
		}
	}

	// gated shared expert (optional; probed at load time)
	if mw.shexp_up != nil {
		sh := h[:p.n_ff_shexp]
		sg := g[:p.n_ff_shexp]
		for f in 0 ..< p.n_ff_shexp {
			row_u := mw.shexp_up[f * p.dim:(f + 1) * p.dim]
			row_g := mw.shexp_gate[f * p.dim:(f + 1) * p.dim]
			su: f32 = 0.0
			sgg: f32 = 0.0
			for i in 0 ..< p.dim {
				su += row_u[i] * x[i]
				sgg += row_g[i] * x[i]
			}
			sh[f] = silu_f32(sgg) * su
		}
		gsum: f32 = 0.0
		for i in 0 ..< p.dim {
			gsum += mw.shexp_gate_inp[i] * x[i]
		}
		gv := sigmoid_f32(gsum)
		for r in 0 ..< p.dim {
			row_d := mw.shexp_down[r * p.n_ff_shexp:(r + 1) * p.n_ff_shexp]
			s: f32 = 0.0
			for f in 0 ..< p.n_ff_shexp {
				s += row_d[f] * sh[f]
			}
			out[r] += gv * s
		}
	}
}

moe_ffn_layer :: proc(t: ^Transformer, l: int) {
	p := &t.config
	w := &t.weights
	s := &t.state
	lw := &w.layers[l]

	rmsnorm(s.xb, s.x, lw.ffn_norm, p.rms_eps)
	moe_ffn(s.xb, s.xb2, &lw.moe, p^)
	for i in 0 ..< p.dim {
		s.x[i] += s.xb2[i]
	}
}

// ---- unit tests (pure math, no GGUF) ----

@(private = "file")
moe_test_config :: proc() -> Config {
	c: Config
	c.dim = 8
	c.hidden_dim = 16
	c.n_expert = 6
	c.n_expert_used = 2
	c.n_ff_exp = 4
	c.n_ff_shexp = 4
	c.expert_w_scale = 1.0
	c.rms_eps = 1e-6
	return c
}

@(private = "file")
f32_bytes :: proc(a: []f32) -> []u8 {
	return ([^]u8)(raw_data(a))[:len(a) * size_of(f32)]
}

@(test)
moe_router_test :: proc(t: ^testing.T) {
	p := moe_test_config()
	// router weights chosen so expert 2 and expert 5 dominate
	gw := []f32{
		// expert rows [n_expert][dim]
		0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // 0
		0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // 1
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // 2 (strong)
		0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, // 3
		0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, // 4
		0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, // 5 (second)
	}
	x := []f32{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}

	mw: Moe_Weights
	mw.gate_inp = gw
	// experts: zero everything except chosen ones contribute
	up := make([]f32, p.n_ff_exp * p.dim * p.n_expert)
	gate := make([]f32, p.n_ff_exp * p.dim * p.n_expert)
	down := make([]f32, p.dim * p.n_ff_exp * p.n_expert)
	// expert 2: up row 0 = 1s -> h[0] = silu(1)*1; down rows = 1s
	for i in 0 ..< p.dim {
		up[(2 * p.n_ff_exp + 0) * p.dim + i] = 1.0
		gate[(2 * p.n_ff_exp + 0) * p.dim + i] = 1.0
		down[(2 * p.dim + i) * p.n_ff_exp + 0] = 1.0
	}
	mw.up_exps = Tensor{.F32, f32_bytes(up)}
	mw.gate_exps = Tensor{.F32, f32_bytes(gate)}
	mw.down_exps = Tensor{.F32, f32_bytes(down)}

	out := make([]f32, p.dim)
	moe_ffn(x, out, &mw, p)

	// expected: logits = dot(x, gw[j]) = {0.8, 1.6, 8, 2.4, 3.2, 4}
	// softmax top-2 = experts 2 (8) and 5 (4); norm: w2 = e8/(e8+e4), w5 = e4/(e8+e4)
	// out[i] = w2 * 1 * 1 * 1 + w5 * 0
	ex := math.exp_f32(8.0)
	exx := math.exp_f32(4.0)
	w2 := ex / (ex + exx)
	// expert 2: up row0 dot = 8, gate row0 dot = 8 -> h[0] = silu(8)*8
	want := w2 * silu_f32(8.0) * 8.0
	for i in 0 ..< p.dim {
		testing.expectf(t, math.abs(out[i] - want) < 1e-5, "expert output %d: got %f want %f", i, out[i], want)
	}
}

@(test)
moe_shared_expert_test :: proc(t: ^testing.T) {
	p := moe_test_config()
	x := []f32{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}

	mw: Moe_Weights
	mw.gate_inp = make([]f32, p.n_expert * p.dim) // all-zero router
	up := make([]f32, p.n_ff_exp * p.dim * p.n_expert)
	gate := make([]f32, p.n_ff_exp * p.dim * p.n_expert)
	down := make([]f32, p.dim * p.n_ff_exp * p.n_expert)
	mw.up_exps = Tensor{.F32, f32_bytes(up)}
	mw.gate_exps = Tensor{.F32, f32_bytes(gate)}
	mw.down_exps = Tensor{.F32, f32_bytes(down)}

	// shared expert: up row 0 = 1, gate row 0 = 1, down[i][0] = 1, gate_inp = 0 -> sigmoid(0)=0.5
	mw.shexp_up = make([]f32, p.n_ff_shexp * p.dim)
	mw.shexp_gate = make([]f32, p.n_ff_shexp * p.dim)
	mw.shexp_down = make([]f32, p.dim * p.n_ff_shexp)
	mw.shexp_gate_inp = make([]f32, p.dim)
	for i in 0 ..< p.dim {
		mw.shexp_up[i] = 1.0
		mw.shexp_gate[i] = 1.0
		mw.shexp_down[i * p.n_ff_shexp] = 1.0
	}
	out := make([]f32, p.dim)
	moe_ffn(x, out, &mw, p)

	// routed experts contribute 0 (zero weights); shared: up dot = 8, gate dot = 8
	// -> h[0] = silu(8)*8; gate = sigmoid(0) = 0.5; out[i] = 0.5 * silu(8)*8
	want := 0.5 * silu_f32(8.0) * 8.0
	for i in 0 ..< p.dim {
		testing.expectf(t, math.abs(out[i] - want) < 1e-5, "shared expert %d: got %f want %f", i, out[i], want)
	}
}

@(test)
moe_expert_gemv_quant_test :: proc(t: ^testing.T) {
	// expert GEMV over F32 rows vs naive reference
	p := moe_test_config()
	x := []f32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
	rows, cols, n_exp := 4, 8, 6
	w := make([]f32, rows * cols * n_exp)
	for i in 0 ..< len(w) {
		w[i] = f32((i * 7) % 13) - 6.0
	}
	tens := Tensor{.F32, f32_bytes(w)}
	scratch := make([]f32, cols)
	out := make([]f32, rows)
	moe_expert_gemv(out, x, scratch, &tens, 3, rows, cols, n_exp)
	for r in 0 ..< rows {
		want: f32 = 0.0
		for j in 0 ..< cols {
			want += w[(3 * rows + r) * cols + j] * x[j]
		}
		testing.expectf(t, math.abs(out[r] - want) < 1e-4, "row %d: got %f want %f", r, out[r], want)
	}
}
