/* Vulkan forward for Qwen3.5. Compiled only with -define:ODINFER_BACKEND="vulkan".
   SPIR-V is embedded from packages/qwen3_5/spirv. A Windows build does not
   invoke a shader compiler; the driver supplies vulkan-1.dll at run time.

   GEMV defaults to one thread per row so the dot matches the CPU association.
   ODINFER_VULKAN_PARALLEL=1 selects the 64-wide kernel (subgroup add when the
   device subgroup size divides 64). ODINFER_VULKAN_EXACT=1 forces the exact
   kernel even if PARALLEL is set; EXACT=0 forces the parallel kernel.
*/

package qwen3_5

import "core:fmt"
import "core:math"
import "core:mem"
import "core:os"
import ggml "ggml:ggml"
import vkr "vulkan:vulkan"
import vk "vendor:vulkan"

// Referenced so a CPU/Metal/CUDA build (GPU_VULKAN false, body stripped) still
// uses these imports. Never called.
@(private)
_vulkan_import_keep :: proc() {
	_ = vkr.device_info
	_ = vk.API_VERSION_1_2
}

when GPU_VULKAN {

@(private)
GEMV_EXACT_SPV :: #load("spirv/gemv_exact.spv", string)
@(private)
GEMV_PAR_SPV :: #load("spirv/gemv_par.spv", string)
@(private)
GEMV_SUB_SPV :: #load("spirv/gemv_sub.spv", string)
@(private)
ELEM_SPV :: #load("spirv/elem.spv", string)
@(private)
RMS_SPV :: #load("spirv/rmsnorm.spv", string)
@(private)
L2_SPV :: #load("spirv/l2norm.spv", string)
@(private)
ROPE_SPV :: #load("spirv/rope.spv", string)
@(private)
GATED_SPV :: #load("spirv/gated_rms.spv", string)
@(private)
ATTN_SPV :: #load("spirv/attn.spv", string)
@(private)
CONV_SPV :: #load("spirv/conv.spv", string)
@(private)
DELTA_SPV :: #load("spirv/delta.spv", string)
@(private)
Q8_SPV :: #load("spirv/q8_dp4a.spv", string)
@(private)
COOP_SPV :: #load("spirv/coopmat.spv", string)

@(private)
Push :: struct {
	u: [24]u32,
	f: [8]f32,
}

@(private)
Wpart :: struct {
	chunk:     int,
	off:       int,
	row0:      int,
	rows:      int,
	cols:      int,
	row_bytes: int,
	kind:      u32,
}

@(private)
Wref :: struct {
	parts: [dynamic]Wpart,
}

@(private)
Blob :: struct {
	src:  []u8,
	keep: []f32,
}

@(private)
Layer_GPU :: struct {
	attn_norm: int,
	ffn_norm:  int,
	q_norm:    int,
	k_norm:    int,
	a_decay:   int,
	dt_bias:   int,
	lin_norm:  int,
	wq, wk, wv, wo: Wref,
	in_qkv, in_z, in_b, in_a, conv, lin_out: Wref,
	w1, w2, w3: Wref,
	gate_inp, up_exps, gate_exps, down_exps: Wref,
	shexp_up, shexp_gate, shexp_down, shexp_gin: Wref,
	has_shexp: bool,
}

@(private)
g_on: bool
@(private)
g_exact: bool
@(private)
g_info: vkr.Info
@(private)
g_act: vkr.Buffer
@(private)
g_chunks: [dynamic]vkr.Buffer
@(private)
g_blobs: [dynamic]Blob
@(private)
g_chunk_used: [dynamic]int
@(private)
g_upload_ok: bool
@(private)
g_layers: []Layer_GPU
@(private)
g_logits: []f32
@(private)
g_tr: ^Transformer

@(private)
g_gemv, g_gemv_exact, g_gemv_par, g_gemv_sub: vk.Pipeline
@(private)
g_elem, g_rms, g_l2, g_rope, g_gated, g_attn, g_convp, g_deltap: vk.Pipeline
@(private)
g_q8, g_coop: vk.Pipeline

@(private)
g_max_wg: int
@(private)
g_at: int
@(private)
g_x, g_xb, g_xb2, g_hb, g_hb2: int
@(private)
g_qproj, g_q, g_xb3, g_att, g_ylog: int
@(private)
g_qkv_raw, g_qkv_out, g_z, g_b, g_a, g_qlin, g_klin, g_linout: int
@(private)
g_kvk, g_kvv, g_convst, g_recst: int
@(private)
g_out_norm, g_inv, g_router, g_scalar, g_scratch: int
@(private)
g_Tcap: int

@(private)
spv_bytes :: proc(s: string) -> []u8 {
	return ([^]u8)(raw_data(s))[:len(s)]
}

@(private)
fslice :: proc(off, n: int) -> []f32 {
	if n <= 0 || g_act.ptr == nil {
		return nil
	}
	p := cast([^]f32)(rawptr(uintptr(g_act.ptr) + uintptr(off * 4)))
	return p[:n]
}

vulkan_batch_max :: proc() -> int {
	return 8
}

vulkan_init :: proc(t: ^Transformer, _path: string) -> bool {
	if g_on {
		return true
	}
	info, okb := vkr.boot()
	if !okb {
		return false
	}
	g_info = info
	g_max_wg = int(info.max_wg)
	if g_max_wg <= 0 {
		g_max_wg = 65535
	}
	if !make_pipes() {
		vulkan_destroy()
		return false
	}
	g_tr = t
	g_upload_ok = true
	if !upload_model(t) {
		vulkan_destroy()
		return false
	}
	mode := "exact"
	if !g_exact {
		if g_gemv == g_gemv_sub {
			mode = "subgroup"
		} else {
			mode = "parallel"
		}
	}
	fmt.eprintfln("vulkan: gemv=%s intdot_pipe=%v coopmat_pipe=%v", mode, g_q8 != 0, g_coop != 0)
	g_on = true
	return true
}

vulkan_destroy :: proc() {
	for i in 0 ..< len(g_layers) {
		free_wref(&g_layers[i].wq)
		free_wref(&g_layers[i].wk)
		free_wref(&g_layers[i].wv)
		free_wref(&g_layers[i].wo)
		free_wref(&g_layers[i].in_qkv)
		free_wref(&g_layers[i].in_z)
		free_wref(&g_layers[i].in_b)
		free_wref(&g_layers[i].in_a)
		free_wref(&g_layers[i].conv)
		free_wref(&g_layers[i].lin_out)
		free_wref(&g_layers[i].w1)
		free_wref(&g_layers[i].w2)
		free_wref(&g_layers[i].w3)
		free_wref(&g_layers[i].gate_inp)
		free_wref(&g_layers[i].up_exps)
		free_wref(&g_layers[i].gate_exps)
		free_wref(&g_layers[i].down_exps)
		free_wref(&g_layers[i].shexp_up)
		free_wref(&g_layers[i].shexp_gate)
		free_wref(&g_layers[i].shexp_down)
		free_wref(&g_layers[i].shexp_gin)
	}
	if g_layers != nil {
		delete(g_layers)
	}
	for b in g_blobs {
		if b.keep != nil {
			delete(b.keep)
		}
	}
	delete(g_blobs)
	delete(g_chunk_used)
	delete(g_chunks)
	if g_logits != nil {
		delete(g_logits)
	}
	vkr.shutdown()
	g_on = false
	g_act = {}
	g_tr = nil
	g_gemv = 0
	g_gemv_exact = 0
	g_gemv_par = 0
	g_gemv_sub = 0
	g_q8 = 0
	g_coop = 0
}

vulkan_reset_state :: proc() {
	if !g_on {
		return
	}
	vkr.idle()
	c := &g_tr.config
	kv_dim := c.n_kv_heads * c.head_dim
	zero_f(g_kvk, c.n_full * c.seq_len * kv_dim)
	zero_f(g_kvv, c.n_full * c.seq_len * kv_dim)
	km1 := max(c.lin_conv_kernel - 1, 0)
	conv_dim := c.lin_n_k_heads * c.lin_head_k_dim * 2 + c.lin_n_v_heads * c.lin_head_v_dim
	zero_f(g_convst, c.n_linear * conv_dim * km1)
	zero_f(g_recst, c.n_linear * c.lin_n_v_heads * c.lin_head_k_dim * c.lin_head_v_dim)
}

vulkan_forward :: proc(t: ^Transformer, token, pos: int) -> []f32 {
	tok := [1]int{token}
	return vulkan_forward_chunk(t, tok[:], pos)
}

vulkan_forward_tokens :: proc(t: ^Transformer, tokens: []int, pos0: int) -> []f32 {
	logits: []f32
	i := 0
	for i < len(tokens) {
		n := min(vulkan_batch_max(), len(tokens) - i)
		logits = vulkan_forward_chunk(t, tokens[i:][:n], pos0 + i)
		i += n
	}
	return logits
}

@(private)
vulkan_forward_chunk :: proc(t: ^Transformer, tokens: []int, pos0: int) -> []f32 {
	T := len(tokens)
	if T <= 0 {
		return nil
	}
	c := &t.config
	if pos0 < 0 || pos0 + T > c.seq_len {
		fmt.eprintfln("vulkan: pos %d..%d outside seq_len %d", pos0, pos0 + T, c.seq_len)
		return nil
	}
	dim := c.dim
	for ti in 0 ..< T {
		tok := tokens[ti]
		if tok < 0 || tok >= c.vocab_size {
			fmt.eprintfln("vulkan: token %d outside vocab %d", tok, c.vocab_size)
			return nil
		}
		get_embedding_row(&t.weights.token_embedding, tok, dim, fslice(g_x + ti * dim, dim))
	}
	vkr.flush_range(g_act, g_x * 4, T * dim * 4)
	if !vkr.begin() {
		return nil
	}
	for l in 0 ..< c.n_layers {
		if c.layer_types[l] == .Full_Attention {
			if !forward_full(t, l, pos0, T) {
				return nil
			}
		} else {
			if !forward_linear(t, l, pos0, T) {
				return nil
			}
		}
	}
	last := (T - 1) * dim
	rms(g_x + last, g_x + last, g_out_norm, 1, dim, dim, dim, c.rms_eps)
	gemv_rows(g_tr_out(t), g_x + last, g_ylog, 0, c.vocab_size, 1, dim, c.vocab_size)
	if !vkr.submit_wait() {
		return nil
	}
	vkr.invalidate_range(g_act, g_ylog * 4, c.vocab_size * 4)
	copy(g_logits, fslice(g_ylog, c.vocab_size))
	return g_logits
}

@(private)
g_output: Wref

@(private)
g_tr_out :: proc(t: ^Transformer) -> Wref {
	_ = t
	return g_output
}

@(private)
forward_full :: proc(t: ^Transformer, l, pos0, T: int) -> bool {
	c := &t.config
	lw := &g_layers[l]
	dim := c.dim
	hd := c.head_dim
	nh := c.n_heads
	nkv := c.n_kv_heads
	kv_dim := nkv * hd
	seq := c.seq_len
	eps := c.rms_eps
	slot := t.full_slot[l]
	kbase := g_kvk + slot * seq * kv_dim
	vbase := g_kvv + slot * seq * kv_dim
	rms(g_x, g_xb, lw.attn_norm, T, dim, dim, dim, eps)
	gemv_rows(lw.wq, g_xb, g_qproj, 0, nh * hd * 2, T, dim, nh * hd * 2)
	gemv_rows(lw.wk, g_xb, kbase + pos0 * kv_dim, 0, kv_dim, T, dim, kv_dim)
	gemv_rows(lw.wv, g_xb, vbase + pos0 * kv_dim, 0, kv_dim, T, dim, kv_dim)
	rms(g_qproj, g_q, lw.q_norm, T * nh, hd, hd * 2, hd, eps)
	rms(kbase + pos0 * kv_dim, kbase + pos0 * kv_dim, lw.k_norm, T * nkv, hd, hd, hd, eps)
	rope(g_q, g_inv, T * nh, hd, c.rotary_dim, pos0, nh)
	rope(kbase + pos0 * kv_dim, g_inv, T * nkv, hd, c.rotary_dim, pos0, nkv)
	scale := f32(1.0) / math.sqrt_f32(f32(hd))
	for ti in 0 ..< T {
		p := Push{}
		p.u[0] = u32(g_q + ti * nh * hd)
		p.u[1] = u32(kbase)
		p.u[2] = u32(vbase)
		p.u[3] = u32(g_xb3 + ti * nh * hd)
		p.u[4] = u32(g_att)
		p.u[5] = u32(hd)
		p.u[6] = u32(nh)
		p.u[7] = u32(nkv)
		p.u[8] = u32(kv_dim)
		p.u[9] = u32(pos0 + ti)
		p.u[10] = u32(seq)
		p.f[0] = scale
		if nh > g_max_wg {
			fmt.eprintln("vulkan: n_heads exceeds maxComputeWorkGroupCount")
			return false
		}
		vkr.barrier()
		vkr.dispatch(g_attn, u32(nh), 1, 1, &p, size_of(p), 0)
	}
	gp := Push{}
	gp.u[0] = 3
	gp.u[1] = u32(g_xb3)
	gp.u[2] = u32(g_qproj)
	gp.u[3] = u32(T)
	gp.u[4] = u32(nh)
	gp.u[5] = u32(hd)
	elem(gp, T * nh * hd)
	gemv_rows(lw.wo, g_xb3, g_xb2, 0, dim, T, nh * hd, dim)
	axpy(g_x, g_xb2, T * dim, 1)
	return forward_ffn(t, l, T)
}

@(private)
forward_linear :: proc(t: ^Transformer, l, pos0, T: int) -> bool {
	_ = pos0
	c := &t.config
	lw := &g_layers[l]
	dim := c.dim
	eps := c.rms_eps
	hk := c.lin_head_k_dim
	hv := c.lin_head_v_dim
	nkh := c.lin_n_k_heads
	nvh := c.lin_n_v_heads
	key_dim := nkh * hk
	value_dim := nvh * hv
	conv_dim := key_dim * 2 + value_dim
	kernel := c.lin_conv_kernel
	km1 := kernel - 1
	slot := t.lin_slot[l]
	conv_state := g_convst + slot * conv_dim * max(km1, 0)
	rec := g_recst + slot * nvh * hk * hv
	rms(g_x, g_xb, lw.attn_norm, T, dim, dim, dim, eps)
	gemv_rows(lw.in_qkv, g_xb, g_qkv_raw, 0, conv_dim, T, dim, conv_dim)
	gemv_rows(lw.in_z, g_xb, g_z, 0, value_dim, T, dim, value_dim)
	gemv_rows(lw.in_b, g_xb, g_b, 0, nvh, T, dim, nvh)
	gemv_rows(lw.in_a, g_xb, g_a, 0, nvh, T, dim, nvh)
	qscale := f32(1.0) / math.sqrt_f32(f32(max(hk, 1)))
	for ti in 0 ..< T {
		conv_dispatch(lw.conv, g_qkv_raw + ti * conv_dim, g_qkv_out + ti * conv_dim, conv_state, conv_dim, kernel)
		l2(g_qkv_out + ti * conv_dim, g_qlin + ti * key_dim, nkh, hk, hk, hk, qscale, 1e-6)
		l2(g_qkv_out + ti * conv_dim + key_dim, g_klin + ti * key_dim, nkh, hk, hk, hk, 1, 1e-6)
		if nvh > g_max_wg {
			fmt.eprintln("vulkan: value heads exceed maxComputeWorkGroupCount")
			return false
		}
		p := Push{}
		p.u[0] = u32(g_qlin + ti * key_dim)
		p.u[1] = u32(g_klin + ti * key_dim)
		p.u[2] = u32(g_qkv_out + ti * conv_dim + key_dim * 2)
		p.u[3] = u32(rec)
		p.u[4] = u32(g_linout + ti * value_dim)
		p.u[5] = u32(g_b + ti * nvh)
		p.u[6] = u32(g_a + ti * nvh)
		p.u[7] = u32(lw.a_decay)
		p.u[8] = u32(lw.dt_bias)
		p.u[9] = u32(hk)
		p.u[10] = u32(hv)
		p.u[11] = u32(max(nkh, 1))
		vkr.barrier()
		vkr.dispatch(g_deltap, u32(max(nvh, 1)), 1, 1, &p, size_of(p), 0)
		gated(g_linout + ti * value_dim, g_z + ti * value_dim, lw.lin_norm, nvh, hv, eps)
	}
	gemv_rows(lw.lin_out, g_linout, g_xb2, 0, dim, T, value_dim, dim)
	axpy(g_x, g_xb2, T * dim, 1)
	return forward_ffn(t, l, T)
}

@(private)
forward_ffn :: proc(t: ^Transformer, l, T: int) -> bool {
	c := &t.config
	lw := &g_layers[l]
	dim := c.dim
	eps := c.rms_eps
	if c.is_moe {
		return forward_moe(t, l, T)
	}
	hid := c.hidden_dim
	rms(g_x, g_xb, lw.ffn_norm, T, dim, dim, dim, eps)
	gemv_rows(lw.w1, g_xb, g_hb, 0, hid, T, dim, hid)
	gemv_rows(lw.w3, g_xb, g_hb2, 0, hid, T, dim, hid)
	swiglu(g_hb, g_hb2, T * hid)
	gemv_rows(lw.w2, g_hb2, g_xb2, 0, dim, T, hid, dim)
	axpy(g_x, g_xb2, T * dim, 1)
	return true
}

@(private)
forward_moe :: proc(t: ^Transformer, l, T: int) -> bool {
	c := &t.config
	lw := &g_layers[l]
	dim := c.dim
	n_exp := c.n_expert
	used := min(c.n_expert_used, n_exp)
	ff := c.n_ff_exp
	eps := c.rms_eps
	rms(g_x, g_xb, lw.ffn_norm, T, dim, dim, dim, eps)
	gemv_rows(lw.gate_inp, g_xb, g_router, 0, n_exp, T, dim, n_exp)
	if !vkr.submit_wait() {
		return false
	}
	vkr.invalidate_range(g_act, g_router * 4, T * n_exp * 4)
	host := make([]f32, T * n_exp)
	defer delete(host)
	copy(host, fslice(g_router, T * n_exp))
	if !vkr.begin() {
		return false
	}
	for ti in 0 ..< T {
		logits := host[ti * n_exp:][:n_exp]
		softmax(logits)
		sel: [MOE_MAX_EXPERTS]int
		picked: [MOE_MAX_EXPERTS]bool
		weights: [MOE_MAX_EXPERTS]f32
		for k in 0 ..< used {
			best := 0
			best_v := f32(-1e30)
			for j in 0 ..< n_exp {
				if !picked[j] && logits[j] > best_v {
					best_v = logits[j]
					best = j
				}
			}
			sel[k] = best
			picked[best] = true
		}
		sum: f32 = 0
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
		if c.expert_w_scale != 1 && c.expert_w_scale != 0 {
			for k in 0 ..< used {
				weights[k] *= c.expert_w_scale
			}
		}
		fill(g_xb2 + ti * dim, dim, 0)
		xb := g_xb + ti * dim
		for k in 0 ..< used {
			e := sel[k]
			gemv_rows(lw.up_exps, xb, g_hb, e * ff, ff, 1, dim, ff)
			gemv_rows(lw.gate_exps, xb, g_hb2, e * ff, ff, 1, dim, ff)
			swiglu(g_hb2, g_hb, ff)
			gemv_rows(lw.down_exps, g_hb, g_scratch, e * dim, dim, 1, ff, dim)
			axpy(g_xb2 + ti * dim, g_scratch, dim, weights[k])
		}
		if lw.has_shexp {
			sff := c.n_ff_shexp
			gemv_rows(lw.shexp_up, xb, g_hb, 0, sff, 1, dim, sff)
			gemv_rows(lw.shexp_gate, xb, g_hb2, 0, sff, 1, dim, sff)
			swiglu(g_hb2, g_hb, sff)
			gemv_rows(lw.shexp_down, g_hb, g_scratch, 0, dim, 1, sff, dim)
			gemv_rows(lw.shexp_gin, xb, g_scalar, 0, 1, 1, dim, 1)
			axpy_sigmoid(g_xb2 + ti * dim, g_scratch, dim, g_scalar)
		}
		axpy(g_x + ti * dim, g_xb2 + ti * dim, dim, 1)
	}
	return true
}

@(private)
zero_f :: proc(off, n: int) {
	if n <= 0 {
		return
	}
	mem.zero(raw_data(fslice(off, n)), n * 4)
	vkr.flush_range(g_act, off * 4, n * 4)
}

@(private)
bind_state :: proc(t: ^Transformer) {
	c := &t.config
	kv_dim := c.n_kv_heads * c.head_dim
	if c.n_full > 0 {
		n := c.n_full * c.seq_len * kv_dim
		t.state.key_cache = fslice(g_kvk, n)
		t.state.value_cache = fslice(g_kvv, n)
	}
	if c.n_linear > 0 {
		km1 := max(c.lin_conv_kernel - 1, 0)
		conv_dim := c.lin_n_k_heads * c.lin_head_k_dim * 2 + c.lin_n_v_heads * c.lin_head_v_dim
		t.state.conv_states = fslice(g_convst, c.n_linear * conv_dim * km1)
		t.state.recurrent_states = fslice(g_recst, c.n_linear * c.lin_n_v_heads * c.lin_head_k_dim * c.lin_head_v_dim)
	}
}

@(private)
unbind_state :: proc(t: ^Transformer) {
	t.state.key_cache = nil
	t.state.value_cache = nil
	t.state.conv_states = nil
	t.state.recurrent_states = nil
}

vulkan_save_kv :: proc(t: ^Transformer, path: string, n_valid_pos: int) -> bool {
	vkr.idle()
	vkr.invalidate_range(g_act, 0, g_act.size)
	bind_state(t)
	ok := cpu_save_kv(t, path, n_valid_pos)
	unbind_state(t)
	return ok
}

vulkan_load_kv :: proc(t: ^Transformer, path: string) -> (int, bool) {
	vkr.idle()
	bind_state(t)
	n, ok := cpu_load_kv(t, path)
	unbind_state(t)
	vkr.flush_range(g_act, 0, g_act.size)
	return n, ok
}

vulkan_save_full_state :: proc(t: ^Transformer) -> (kv_k, kv_v, conv, rec: []u8) {
	vkr.idle()
	vkr.invalidate_range(g_act, 0, g_act.size)
	c := &t.config
	if c.n_full == 0 {
		km1 := max(c.lin_conv_kernel - 1, 0)
		conv_dim := c.lin_n_k_heads * c.lin_head_k_dim * 2 + c.lin_n_v_heads * c.lin_head_v_dim
		cn := c.n_linear * conv_dim * km1
		rn := c.n_linear * c.lin_n_v_heads * c.lin_head_k_dim * c.lin_head_v_dim
		if cn > 0 {
			conv = make([]u8, cn * 4)
			mem.copy(raw_data(conv), raw_data(fslice(g_convst, cn)), cn * 4)
		}
		if rn > 0 {
			rec = make([]u8, rn * 4)
			mem.copy(raw_data(rec), raw_data(fslice(g_recst, rn)), rn * 4)
		}
		return
	}
	bind_state(t)
	kv_k, kv_v, conv, rec = cpu_save_full_state(t)
	unbind_state(t)
	return
}

vulkan_restore_full_state :: proc(t: ^Transformer, kv_k, kv_v, conv, rec: []u8) {
	vkr.idle()
	c := &t.config
	if c.n_full == 0 {
		if len(conv) > 0 {
			mem.copy(raw_data(fslice(g_convst, len(conv) / 4)), raw_data(conv), len(conv))
		}
		if len(rec) > 0 {
			mem.copy(raw_data(fslice(g_recst, len(rec) / 4)), raw_data(rec), len(rec))
		}
		vkr.flush_range(g_act, 0, g_act.size)
		return
	}
	bind_state(t)
	cpu_restore_full_state(t, kv_k, kv_v, conv, rec)
	unbind_state(t)
	vkr.flush_range(g_act, 0, g_act.size)
}

vulkan_save_spec_state :: proc() -> (conv, rec: []u8) {
	if g_tr == nil {
		return
	}
	vkr.idle()
	vkr.invalidate_range(g_act, 0, g_act.size)
	c := &g_tr.config
	km1 := max(c.lin_conv_kernel - 1, 0)
	conv_dim := c.lin_n_k_heads * c.lin_head_k_dim * 2 + c.lin_n_v_heads * c.lin_head_v_dim
	cn := c.n_linear * conv_dim * km1
	rn := c.n_linear * c.lin_n_v_heads * c.lin_head_k_dim * c.lin_head_v_dim
	if cn > 0 {
		conv = make([]u8, cn * 4)
		mem.copy(raw_data(conv), raw_data(fslice(g_convst, cn)), cn * 4)
	}
	if rn > 0 {
		rec = make([]u8, rn * 4)
		mem.copy(raw_data(rec), raw_data(fslice(g_recst, rn)), rn * 4)
	}
	return
}

vulkan_restore_spec_state :: proc(conv, rec: []u8) {
	if len(conv) > 0 {
		mem.copy(raw_data(fslice(g_convst, len(conv) / 4)), raw_data(conv), len(conv))
	}
	if len(rec) > 0 {
		mem.copy(raw_data(fslice(g_recst, len(rec) / 4)), raw_data(rec), len(rec))
	}
	vkr.flush_range(g_act, 0, g_act.size)
}

@(private)
copy_wref :: proc(w: ^Wref, bi: ^int) {
	for p in w.parts {
		if bi^ >= len(g_blobs) {
			g_upload_ok = false
			return
		}
		n := p.rows * p.row_bytes
		dst := cast([^]byte)(rawptr(uintptr(g_chunks[p.chunk].ptr) + uintptr(p.off)))
		mem.copy(dst, raw_data(g_blobs[bi^].src), n)
		bi^ += 1
	}
}

@(private)
free_wref :: proc(w: ^Wref) {
	delete(w.parts)
	w^ = {}
}

@(private)
make_pipes :: proc() -> bool {
	ex := os.get_env("ODINFER_VULKAN_EXACT", context.temp_allocator)
	par := os.get_env("ODINFER_VULKAN_PARALLEL", context.temp_allocator)
	g_exact = true
	if ex != "" {
		g_exact = ex != "0"
	} else if par == "1" {
		g_exact = false
	}
	okp: bool
	g_gemv_exact, okp = vkr.make_pipeline(spv_bytes(GEMV_EXACT_SPV), 0, false)
	if !okp {
		fmt.eprintln("vulkan: exact GEMV pipeline failed")
		return false
	}
	g_gemv_par, okp = vkr.make_pipeline(spv_bytes(GEMV_PAR_SPV), 0, false)
	if !okp {
		fmt.eprintln("vulkan: parallel GEMV pipeline failed")
		return false
	}
	g_gemv_sub, okp = vkr.make_pipeline(spv_bytes(GEMV_SUB_SPV), 0, false)
	if !okp {
		g_gemv_sub = 0
	}
	g_elem, okp = vkr.make_pipeline(spv_bytes(ELEM_SPV), 0, false)
	g_rms, okp = pipe_ok(okp, spv_bytes(RMS_SPV), &g_rms)
	if !g_elem_ok() || g_rms == 0 {
		fmt.eprintln("vulkan: element or rmsnorm pipeline failed")
		return false
	}
	_ = okp
	if !pipe(&g_l2, L2_SPV) || !pipe(&g_rope, ROPE_SPV) || !pipe(&g_gated, GATED_SPV) || !pipe(&g_attn, ATTN_SPV) || !pipe(&g_convp, CONV_SPV) || !pipe(&g_deltap, DELTA_SPV) {
		fmt.eprintln("vulkan: a compute pipeline failed to compile")
		return false
	}
	g_gemv = g_gemv_exact
	if !g_exact {
		if g_info.subgroup_arith && g_gemv_sub != 0 {
			g_gemv = g_gemv_sub
		} else {
			g_gemv = g_gemv_par
		}
	}
	if g_info.intdot {
		g_q8, _ = vkr.make_pipeline(spv_bytes(Q8_SPV), 0, false)
	}
	if g_info.coopmat && g_info.subgroup > 0 {
		g_coop, _ = vkr.make_pipeline(spv_bytes(COOP_SPV), g_info.subgroup, true)
	}
	return true
}

@(private)
g_elem_ok :: proc() -> bool {
	return g_elem != 0 && g_rms != 0
}

@(private)
pipe_ok :: proc(prev: bool, code: []u8, dst: ^vk.Pipeline) -> (vk.Pipeline, bool) {
	p, okp := vkr.make_pipeline(code, 0, false)
	if okp {
		dst^ = p
	}
	_ = prev
	return p, okp
}

@(private)
pipe :: proc(dst: ^vk.Pipeline, code: string) -> bool {
	p, okp := vkr.make_pipeline(spv_bytes(code), 0, false)
	if !okp {
		return false
	}
	dst^ = p
	return true
}

@(private)
elem :: proc(p: Push, n: int) {
	if n <= 0 {
		return
	}
	max_e := g_max_wg * 64
	if p.u[0] >= 3 || n <= max_e {
		groups := (n + 63) / 64
		if groups > g_max_wg {
			fmt.eprintln("vulkan: element dispatch exceeds maxComputeWorkGroupCount")
			return
		}
		q := p
		vkr.barrier()
		vkr.dispatch(g_elem, u32(groups), 1, 1, &q, size_of(q), 0)
		return
	}
	done := 0
	for done < n {
		m := min(max_e, n - done)
		q := p
		op := q.u[0]
		if op == 0 {
			q.u[1] += u32(done)
			q.u[2] = u32(m)
		} else if op == 1 || op == 2 {
			q.u[1] += u32(done)
			q.u[2] += u32(done)
			q.u[3] = u32(m)
		}
		vkr.barrier()
		vkr.dispatch(g_elem, u32((m + 63) / 64), 1, 1, &q, size_of(q), 0)
		done += m
	}
}

@(private)
fill :: proc(off, n: int, v: f32) {
	p := Push{}
	p.u[0] = 0
	p.u[1] = u32(off)
	p.u[2] = u32(n)
	p.f[0] = v
	elem(p, n)
}

@(private)
axpy :: proc(y, x, n: int, a: f32) {
	p := Push{}
	p.u[0] = 1
	p.u[1] = u32(y)
	p.u[2] = u32(x)
	p.u[3] = u32(n)
	p.f[0] = a
	elem(p, n)
}

@(private)
axpy_sigmoid :: proc(y, x, n, scalar: int) {
	p := Push{}
	p.u[0] = 1
	p.u[1] = u32(y)
	p.u[2] = u32(x)
	p.u[3] = u32(n)
	p.u[4] = 1
	p.u[5] = u32(scalar)
	elem(p, n)
}

@(private)
swiglu :: proc(gate, up, n: int) {
	p := Push{}
	p.u[0] = 2
	p.u[1] = u32(gate)
	p.u[2] = u32(up)
	p.u[3] = u32(n)
	elem(p, n)
}

@(private)
gemv_rows :: proc(w: Wref, x, y, row0, nrows, T, x_stride, y_stride: int) {
	if nrows <= 0 || T <= 0 || len(w.parts) == 0 {
		return
	}
	Tc := T
	if Tc > 8 {
		Tc = 8
	}
	for i in 0 ..< len(w.parts) {
		p := w.parts[i]
		a := max(row0, p.row0)
		b := min(row0 + nrows, p.row0 + p.rows)
		if a >= b {
			continue
		}
		left := b - a
		done := 0
		for left > 0 {
			n := min(left, g_max_wg)
			q := Push{}
			abs := a + done
			q.u[0] = u32(p.off + (abs - p.row0) * p.row_bytes)
			q.u[1] = u32(x)
			q.u[2] = u32(y + (abs - row0))
			q.u[3] = u32(n)
			q.u[4] = u32(p.cols)
			q.u[5] = u32(Tc)
			q.u[6] = p.kind
			q.u[8] = u32(x_stride)
			q.u[9] = u32(y_stride)
			q.u[11] = u32(p.row_bytes)
			vkr.barrier()
			vkr.dispatch(g_gemv, u32(n), 1, 1, &q, size_of(q), p.chunk)
			done += n
			left -= n
		}
	}
}

@(private)
rms :: proc(inb, outb, w: int, rows, size, in_stride, out_stride: int, eps: f32) {
	if rows <= 0 {
		return
	}
	max_rows := g_max_wg * 64
	r0 := 0
	for r0 < rows {
		m := min(max_rows, rows - r0)
		q := Push{}
		q.u[0] = u32(inb + r0 * in_stride)
		q.u[1] = u32(outb + r0 * out_stride)
		q.u[2] = u32(w)
		q.u[3] = u32(m)
		q.u[4] = u32(size)
		q.u[5] = u32(in_stride)
		q.u[6] = u32(out_stride)
		q.f[0] = eps
		vkr.barrier()
		vkr.dispatch(g_rms, u32((m + 63) / 64), 1, 1, &q, size_of(q), 0)
		r0 += m
	}
}

@(private)
l2 :: proc(inb, outb: int, rows, size, in_stride, out_stride: int, scale, eps: f32) {
	if rows <= 0 {
		return
	}
	max_rows := g_max_wg * 64
	r0 := 0
	for r0 < rows {
		m := min(max_rows, rows - r0)
		q := Push{}
		q.u[0] = u32(inb + r0 * in_stride)
		q.u[1] = u32(outb + r0 * out_stride)
		q.u[2] = u32(m)
		q.u[3] = u32(size)
		q.u[4] = u32(in_stride)
		q.u[5] = u32(out_stride)
		q.f[0] = scale
		q.f[1] = eps
		vkr.barrier()
		vkr.dispatch(g_l2, u32((m + 63) / 64), 1, 1, &q, size_of(q), 0)
		r0 += m
	}
}

@(private)
rope :: proc(base, inv: int, rows, stride, rotary, pos0, heads_per_token: int) {
	if rows <= 0 || rotary <= 0 {
		return
	}
	max_rows := g_max_wg * 64
	r0 := 0
	for r0 < rows {
		m := min(max_rows, rows - r0)
		q := Push{}
		q.u[0] = u32(base + r0 * stride)
		q.u[1] = u32(inv)
		q.u[2] = u32(m)
		q.u[3] = u32(stride)
		q.u[4] = u32(rotary)
		// pos = pos0 + (r0+local)/heads_per_token. The shader adds local/heads
		// to pos0, so fold the starting row into pos0.
		q.u[5] = u32(pos0 + r0 / max(heads_per_token, 1))
		q.u[6] = u32(max(heads_per_token, 1))
		// Rows that are not aligned to a token need the shader's row index to
		// include r0. Dispatch from a base pointer already advanced by r0, and
		// pass pos0 adjusted by whole tokens only when r0 is token-aligned.
		vkr.barrier()
		vkr.dispatch(g_rope, u32((m + 63) / 64), 1, 1, &q, size_of(q), 0)
		r0 += m
	}
}

@(private)
gated :: proc(x, z, w: int, rows, head_v: int, eps: f32) {
	if rows <= 0 {
		return
	}
	q := Push{}
	q.u[0] = u32(x)
	q.u[1] = u32(z)
	q.u[2] = u32(w)
	q.u[3] = u32(rows)
	q.u[4] = u32(head_v)
	q.f[0] = eps
	groups := (rows + 63) / 64
	if groups > g_max_wg {
		groups = g_max_wg
	}
	vkr.barrier()
	vkr.dispatch(g_gated, u32(groups), 1, 1, &q, size_of(q), 0)
}

@(private)
conv_dispatch :: proc(w: Wref, inb, outb, state, conv_dim, kernel: int) {
	if len(w.parts) == 0 || conv_dim <= 0 {
		return
	}
	p := w.parts[0]
	q := Push{}
	q.u[0] = u32(p.off)
	q.u[1] = u32(inb)
	q.u[2] = u32(outb)
	q.u[3] = u32(state)
	q.u[4] = u32(conv_dim)
	q.u[5] = u32(kernel)
	groups := (conv_dim + 63) / 64
	if groups < 1 {
		groups = 1
	}
	vkr.barrier()
	vkr.dispatch(g_convp, u32(groups), 1, 1, &q, size_of(q), p.chunk)
}

@(private)
gpu_kind :: proc(k: ggml.GGML_Type, cols: int) -> (u32, bool) {
	#partial switch k {
	case .F32, .F16:
		return u32(k), true
	case .Q4_0, .Q4_1, .Q5_0, .Q5_1, .Q8_0:
		return u32(k), cols % 32 == 0
	case .Q2_K, .Q3_K, .Q4_K, .Q5_K, .Q6_K, .Q8_K:
		return u32(k), cols % 256 == 0
	case:
		return 0, false
	}
}

@(private)
upload_tensor :: proc(tensor: Tensor, rows, cols: int) -> Wref {
	kind, okk := gpu_kind(tensor.kind, cols)
	rb := ggml.row_byte_size(tensor.kind, cols)
	if okk && len(tensor.data) >= rows * rb {
		src := tensor.data[:rows * rb]
		return place_parts(src, nil, kind, rows, cols, rb)
	}
	f32s := make([]f32, rows * cols)
	if rb > 0 && len(tensor.data) >= rows * rb {
		for r in 0 ..< rows {
			ggml.dequant_row(tensor.kind, tensor.data[r * rb:], cols, f32s[r * cols:][:cols])
		}
	}
	src := ([^]u8)(raw_data(f32s))[:rows * cols * 4]
	return place_parts(src, f32s, 0, rows, cols, cols * 4)
}

@(private)
upload_f32 :: proc(data: []f32, rows, cols: int) -> Wref {
	if len(data) < rows * cols {
		fmt.eprintfln("vulkan: f32 weight shorter than %d", rows * cols)
		g_upload_ok = false
		w: Wref
		return w
	}
	src := ([^]u8)(raw_data(data))[:rows * cols * 4]
	return place_parts(src, nil, 0, rows, cols, cols * 4)
}

@(private)
place :: proc(nbytes: int) -> (chunk, off: int, ok: bool) {
	maxr := int(g_info.max_storage)
	if nbytes <= 0 || nbytes > maxr {
		return 0, 0, false
	}
	if len(g_chunk_used) == 0 {
		append(&g_chunk_used, 0)
	}
	c := len(g_chunk_used) - 1
	o := (g_chunk_used[c] + 255) & ~int(255)
	if o + nbytes > maxr {
		append(&g_chunk_used, 0)
		c = len(g_chunk_used) - 1
		o = 0
	}
	g_chunk_used[c] = o + nbytes
	return c, o, true
}

@(private)
place_parts :: proc(data: []u8, owned: []f32, kind: u32, rows, cols, row_bytes: int) -> Wref {
	w: Wref
	if rows <= 0 || row_bytes <= 0 {
		if owned != nil {
			delete(owned)
		}
		return w
	}
	maxr := int(g_info.max_storage)
	if row_bytes > maxr {
		fmt.eprintln("vulkan: one weight row exceeds maxStorageBufferRange")
		g_upload_ok = false
		if owned != nil {
			delete(owned)
		}
		return w
	}
	max_rows := maxr / row_bytes
	row := 0
	for row < rows {
		n := min(rows - row, max_rows)
		nbytes := n * row_bytes
		chunk, off, okp := place(nbytes)
		if !okp {
			g_upload_ok = false
			if owned != nil {
				delete(owned)
			}
			return w
		}
		b: Blob
		b.src = data[row * row_bytes:][:nbytes]
		if row == 0 {
			b.keep = owned
		}
		append(&g_blobs, b)
		append(&w.parts, Wpart{chunk = chunk, off = off, row0 = row, rows = n, cols = cols, row_bytes = row_bytes, kind = kind})
		row += n
	}
	return w
}

@(private)
alloc_f :: proc(n: int) -> int {
	nn := n
	if nn < 0 {
		nn = 0
	}
	off := g_at
	g_at += nn
	return off
}

@(private)
put_f32 :: proc(off: int, src: []f32) {
	if len(src) == 0 {
		return
	}
	mem.copy(raw_data(fslice(off, len(src))), raw_data(src), len(src) * 4)
}

@(private)
upload_model :: proc(t: ^Transformer) -> bool {
	c := &t.config
	w := &t.weights
	if c.is_moe && (c.n_expert > MOE_MAX_EXPERTS || c.n_expert <= 0) {
		fmt.eprintfln("vulkan: n_expert %d is outside 1..%d", c.n_expert, MOE_MAX_EXPERTS)
		return false
	}
	g_output = upload_tensor(w.output, c.vocab_size, c.dim)
	g_layers = make([]Layer_GPU, c.n_layers)
	key_dim := c.lin_n_k_heads * c.lin_head_k_dim
	value_dim := c.lin_n_v_heads * c.lin_head_v_dim
	conv_dim := key_dim * 2 + value_dim
	for l in 0 ..< c.n_layers {
		lw := &w.layers[l]
		g := &g_layers[l]
		switch lw.layer_type {
		case .Full_Attention:
			g.wq = upload_tensor(lw.full.wq, c.n_heads * c.head_dim * 2, c.dim)
			g.wk = upload_tensor(lw.full.wk, c.n_kv_heads * c.head_dim, c.dim)
			g.wv = upload_tensor(lw.full.wv, c.n_kv_heads * c.head_dim, c.dim)
			g.wo = upload_tensor(lw.full.wo, c.dim, c.n_heads * c.head_dim)
		case .Linear_Attention:
			la := &lw.linear
			g.in_qkv = upload_tensor(la.in_qkv, conv_dim, c.dim)
			g.in_z = upload_tensor(la.in_z, value_dim, c.dim)
			g.in_b = upload_tensor(la.in_b, c.lin_n_v_heads, c.dim)
			g.in_a = upload_tensor(la.in_a, c.lin_n_v_heads, c.dim)
			g.conv = upload_tensor(la.conv, conv_dim, c.lin_conv_kernel)
			g.lin_out = upload_tensor(la.out, c.dim, value_dim)
		}
		if c.is_moe {
			mw := &lw.moe
			g.gate_inp = upload_f32(mw.gate_inp, c.n_expert, c.dim)
			g.up_exps = upload_tensor(mw.up_exps, c.n_expert * c.n_ff_exp, c.dim)
			g.gate_exps = upload_tensor(mw.gate_exps, c.n_expert * c.n_ff_exp, c.dim)
			g.down_exps = upload_tensor(mw.down_exps, c.n_expert * c.dim, c.n_ff_exp)
			if mw.shexp_up != nil {
				g.has_shexp = true
				g.shexp_up = upload_f32(mw.shexp_up, c.n_ff_shexp, c.dim)
				g.shexp_gate = upload_f32(mw.shexp_gate, c.n_ff_shexp, c.dim)
				g.shexp_down = upload_f32(mw.shexp_down, c.dim, c.n_ff_shexp)
				g.shexp_gin = upload_f32(mw.shexp_gate_inp, 1, c.dim)
			}
		} else {
			g.w1 = upload_tensor(lw.w1, c.hidden_dim, c.dim)
			g.w3 = upload_tensor(lw.w3, c.hidden_dim, c.dim)
			g.w2 = upload_tensor(lw.w2, c.dim, c.hidden_dim)
		}
	}
	if !g_upload_ok {
		return false
	}

	T := vulkan_batch_max()
	g_Tcap = T
	dim := c.dim
	ff := c.hidden_dim
	if c.is_moe {
		ff = max(c.n_ff_exp, c.n_ff_shexp)
	}
	if ff < 1 {
		ff = 1
	}
	nh := c.n_heads
	hd := c.head_dim
	kv_dim := c.n_kv_heads * hd
	g_x = alloc_f(T * dim)
	g_xb = alloc_f(T * dim)
	g_xb2 = alloc_f(T * dim)
	g_hb = alloc_f(T * ff)
	g_hb2 = alloc_f(T * ff)
	g_qproj = alloc_f(T * nh * hd * 2)
	g_q = alloc_f(T * nh * hd)
	g_xb3 = alloc_f(T * nh * hd)
	g_att = alloc_f(max(nh, 1) * c.seq_len)
	g_ylog = alloc_f(c.vocab_size)
	g_qkv_raw = alloc_f(T * max(conv_dim, 1))
	g_qkv_out = alloc_f(T * max(conv_dim, 1))
	g_z = alloc_f(T * max(value_dim, 1))
	g_b = alloc_f(T * max(c.lin_n_v_heads, 1))
	g_a = alloc_f(T * max(c.lin_n_v_heads, 1))
	g_qlin = alloc_f(T * max(key_dim, 1))
	g_klin = alloc_f(T * max(key_dim, 1))
	g_linout = alloc_f(T * max(value_dim, 1))
	g_kvk = alloc_f(c.n_full * c.seq_len * kv_dim)
	g_kvv = alloc_f(c.n_full * c.seq_len * kv_dim)
	km1 := max(c.lin_conv_kernel - 1, 0)
	g_convst = alloc_f(c.n_linear * conv_dim * km1)
	g_recst = alloc_f(c.n_linear * c.lin_n_v_heads * c.lin_head_k_dim * c.lin_head_v_dim)
	g_out_norm = alloc_f(dim)
	g_inv = alloc_f(max(c.rotary_dim / 2, 1))
	nexp := max(c.n_expert, 1)
	g_router = alloc_f(T * nexp)
	g_scalar = alloc_f(4)
	g_scratch = alloc_f(max(dim, ff))
	for l in 0 ..< c.n_layers {
		lw := &w.layers[l]
		g := &g_layers[l]
		g.attn_norm = alloc_f(dim)
		g.ffn_norm = alloc_f(dim)
		if lw.layer_type == .Full_Attention {
			g.q_norm = alloc_f(hd)
			g.k_norm = alloc_f(hd)
		} else {
			g.a_decay = alloc_f(max(c.lin_n_v_heads, 1))
			g.dt_bias = alloc_f(max(c.lin_n_v_heads, 1))
			g.lin_norm = alloc_f(max(c.lin_head_v_dim, 1))
		}
	}
	bytes := g_at * 4
	if bytes < 4096 {
		bytes = 4096
	}
	act, oka := vkr.make_buffer(bytes)
	if !oka {
		return false
	}
	g_act = act
	for i in 0 ..< len(g_chunk_used) {
		buf, okb := vkr.make_buffer(max(g_chunk_used[i], 256))
		if !okb {
			return false
		}
		append(&g_chunks, buf)
	}
	// Blobs and parts were appended together, one blob per part.
	bi := 0
	copy_wref(&g_output, &bi)
	for l in 0 ..< c.n_layers {
		g := g_layers[l]
		copy_wref(&g.wq, &bi); copy_wref(&g.wk, &bi); copy_wref(&g.wv, &bi); copy_wref(&g.wo, &bi)
		copy_wref(&g.in_qkv, &bi); copy_wref(&g.in_z, &bi); copy_wref(&g.in_b, &bi); copy_wref(&g.in_a, &bi); copy_wref(&g.conv, &bi); copy_wref(&g.lin_out, &bi)
		// Upload order is gate, up, down (w1, w3, w2). The blob list follows that.
		copy_wref(&g.w1, &bi); copy_wref(&g.w3, &bi); copy_wref(&g.w2, &bi)
		copy_wref(&g.gate_inp, &bi); copy_wref(&g.up_exps, &bi); copy_wref(&g.gate_exps, &bi); copy_wref(&g.down_exps, &bi)
		copy_wref(&g.shexp_up, &bi); copy_wref(&g.shexp_gate, &bi); copy_wref(&g.shexp_down, &bi); copy_wref(&g.shexp_gin, &bi)
	}
	if bi != len(g_blobs) || !g_upload_ok {
		fmt.eprintfln("vulkan: weight copy mismatch (%d blobs, walked %d)", len(g_blobs), bi)
		return false
	}
	for ch in g_chunks {
		vkr.flush_range(ch, 0, ch.size)
	}
	if !vkr.bind_storage(g_chunks[:], g_act) {
		return false
	}
	put_f32(g_out_norm, w.output_norm)
	put_f32(g_inv, t.inv_freq)
	for l in 0 ..< c.n_layers {
		lw := &w.layers[l]
		g := &g_layers[l]
		put_f32(g.attn_norm, lw.attn_norm)
		put_f32(g.ffn_norm, lw.ffn_norm)
		if lw.layer_type == .Full_Attention {
			put_f32(g.q_norm, lw.full.q_norm)
			put_f32(g.k_norm, lw.full.k_norm)
		} else {
			put_f32(g.a_decay, lw.linear.a_decay)
			put_f32(g.dt_bias, lw.linear.dt_bias)
			put_f32(g.lin_norm, lw.linear.norm_w)
		}
	}
	vkr.flush_range(g_act, 0, g_act.size)
	for b in g_blobs {
		if b.keep != nil {
			delete(b.keep)
		}
	}
	// Drop the source alias so destroy does not free twice. The keep slice
	// was the only allocation; src aliases it or the mmap.
	for i in 0 ..< len(g_blobs) {
		g_blobs[i].keep = nil
		g_blobs[i].src = nil
	}
	g_logits = make([]f32, c.vocab_size)
	return true
}

} // GPU_VULKAN
