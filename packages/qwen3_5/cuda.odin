/* CUDA forward for Qwen3.5. Compiled only with -define:ODINFER_BACKEND="cuda".
   The driver and NVRTC are dlopen'd; nothing here links against libcuda.
   Decode (T==1) uses a bandwidth GEMV (dp4a, 4 rows/warp); ODINFER_CUDA_GEMV=old
   restores one warp per row. Prefill uses WMMA when it pays. KV layout matches the
   Metal/CPU snapshot bytes (Q8_0 K, mixed or Q8 V, or legacy f16). */

package qwen3_5

import cu "cuda:cuda"
import ggml "ggml:ggml"

import "core:c"
import "core:fmt"
import "core:math"
import "core:os"
import "core:strconv"
import "core:time"

@(private)
CUDA_KERNEL_SRC :: #load("cuda_kernels.cu", string)

// Referenced so a CPU/Metal build (GPU_CUDA false, body stripped) still uses
// these imports. Never called.
@(private = "file")
_cuda_import_keep :: proc() {
	_ = cu.BACKEND_TAG
	_ = ggml.QK_K
	_ = fmt.tprintf("")
	_ = math.abs(f32(0))
	_ = os.O_RDONLY
	_, _ = strconv.parse_int("0")
	_ = time.now()
	_ = c.int(0)
	_ = len(CUDA_KERNEL_SRC)
}

when GPU_CUDA {

CUDA_MAX_BATCH :: 128
// WMMA tile is 16. Below this, the fp16 quant round-trip does not pay for
// itself against the 8-token MMQ. ODINFER_CUDA_TC=1 uses tensor cores for
// every T>1; =0 stays on MMQ.
TC_MIN_T :: 16

@(private)
Store_Fn :: proc(k, v, kc, vc: cu.Device_Ptr, pos: int) -> bool
@(private)
Attn_Fn :: proc(q, kc, vc, xb3: cu.Device_Ptr, pos: int) -> bool

@(private)
g_ready: bool
@(private)
g_dev: cu.Device
@(private)
g_ctx_held: bool
@(private)
g_mod: cu.Module
@(private)
g_mmap_base: uintptr
@(private)
d_mmap: cu.Device_Ptr
@(private)
live: [dynamic]cu.Device_Ptr
@(private)
g_max_t: int
@(private)
g_timing: bool
@(private)
g_dim, g_hidden, g_vocab, g_seq, g_heads, g_kv_heads, g_head: int
@(private)
g_kv_dim, g_rot: int
@(private)
g_k_pos, g_v_pos, g_k_layer, g_v_layer: int
@(private)
g_k_bytes, g_v_bytes, g_conv_bytes, g_rec_bytes: int
@(private)
do_store: Store_Fn
@(private)
do_attn: Attn_Fn

@(private)
d_x, d_xb, d_xb2, d_xb3: cu.Device_Ptr
@(private)
d_qproj, d_q, d_k, d_v: cu.Device_Ptr
@(private)
d_hb, d_hb2: cu.Device_Ptr
@(private)
d_qkv, d_qkv2, d_z, d_b, d_a, d_qlin, d_klin, d_lout: cu.Device_Ptr
@(private)
d_logits, d_scores, d_partial, d_moe_logits, d_moe_wt, d_moe_sel: cu.Device_Ptr
@(private)
d_kc, d_vc, d_conv, d_rec: cu.Device_Ptr

@(private)
h_logits, h_embed, h_wt: []f32
@(private)
h_sel: []i32

@(private)
Moe_GPU :: struct {
	gate, up, gate_w, down, ginp: cu.Device_Ptr,
	has_shexp:                     bool,
}
@(private)
moe_gpu: []Moe_GPU

@(private)
gemv_fn: [40]cu.Function
@(private)
mmq_fn: [40]cu.Function
@(private)
tc_fn: [40]cu.Function
@(private)
g_sm_major: int
@(private)
g_tc_mode: int // -1 auto, 0 off, 1 on for every T>1
@(private)
g_gemv_new: bool
@(private)
g_gemv_warps: int
@(private)
g_launches: int
@(private)
g_count_launches: bool
@(private)
d_xq, d_xd, d_xs: cu.Device_Ptr
@(private)
g_q8_cap, g_q8_n: int
@(private)
g_q8_src: cu.Device_Ptr
@(private)
g_q8_valid: bool
@(private)
k_quant, k_rmsnorm_q8: cu.Function
@(private)
k_gemv2_q4, k_gemv2_q6, k_gemv2_q5k, k_gemv2_q8, k_gemv2_q50: cu.Function
@(private)
k_rmsnorm, k_qnorm, k_rope, k_gate, k_swiglu, k_residual, k_axpy: cu.Function
@(private)
k_store_f16, k_attn_f16, k_store_q8, k_attn_q8, k_store_q8v, k_attn_q8v: cu.Function
@(private)
k_conv, k_l2, k_delta, k_rmsgate, k_moe_topk: cu.Function

cuda_batch_max :: proc() -> int {
	if g_max_t > 0 {
		return g_max_t
	}
	return CUDA_MAX_BATCH
}

@(private)
ck :: proc(err: cu.Error, what: string) -> bool {
	if err == cu.OK {
		return true
	}
	fmt.eprintf("cuda: %s: %s\n", what, cu.error_string(err))
	return false
}

@(private)
klaunch :: proc(fn: cu.Function, gx, gy, gz, bx, by, bz: u32, args: []rawptr, what: string) -> bool {
	if g_count_launches {
		g_launches += 1
	}
	return ck(cu.launch(fn, gx, gy, gz, bx, by, bz, 0, args), what)
}

@(private)
must :: proc(ok: bool) {
	if !ok {
		os.exit(1)
	}
}

@(private)
fp :: proc(base: cu.Device_Ptr, nfloat: int) -> cu.Device_Ptr {
	return cu.ptr_add(base, nfloat * 4)
}

@(private)
wptr :: proc(p: rawptr) -> cu.Device_Ptr {
	return cu.ptr_add(d_mmap, int(uintptr(p) - g_mmap_base))
}

@(private)
dev_f32 :: proc(s: []f32) -> cu.Device_Ptr {
	return wptr(raw_data(s))
}

@(private)
cuda_kind_tag :: proc(k: ggml.GGML_Type) -> string {
	#partial switch k {
	case .F32:
		return "f32"
	case .F16:
		return "f16"
	case .Q4_0:
		return "q4_0"
	case .Q4_1:
		return "q4_1"
	case .Q5_0:
		return "q5_0"
	case .Q5_1:
		return "q5_1"
	case .Q8_0:
		return "q8_0"
	case .Q2_K:
		return "q2_k"
	case .Q3_K:
		return "q3_k"
	case .Q4_K:
		return "q4_k"
	case .Q5_K:
		return "q5_k"
	case .Q6_K:
		return "q6_k"
	case .Q8_K:
		return "q8_k"
	}
	return ""
}

@(private)
kind_ok :: proc(k: ggml.GGML_Type) -> bool {
	return cuda_kind_tag(k) != ""
}

@(private)
check_mat :: proc(name: string, k: ggml.GGML_Type, cols: int) -> bool {
	if !kind_ok(k) {
		fmt.eprintf("cuda: %s kind %v is not ported\n", name, k)
		fmt.eprintln("cuda: ported GEMV/MMQ: F32 F16 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q2_K Q3_K Q4_K Q5_K Q6_K Q8_K")
		fmt.eprintln("cuda: not ported: IQ1_S IQ1_M IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S IQ4_NL IQ4_XS Q8_1")
		return false
	}
	elems, _ := ggml.ggml_block_info(k)
	if elems <= 0 || cols % elems != 0 {
		fmt.eprintf("cuda: %s cols %d not divisible by block %d (%v)\n", name, cols, elems, k)
		return false
	}
	return true
}

@(private)
check_tensor :: proc(name: string, t: Tensor, cols: int) -> bool {
	if len(t.data) == 0 {
		return true
	}
	return check_mat(name, t.kind, cols)
}

@(private)
grid_n :: proc(n, block: int) -> u32 {
	if n <= 0 {
		return 0
	}
	return u32((n + block - 1) / block)
}

@(private)
get_fn :: proc(name: string) -> cu.Function {
	fn, err := cu.module_function(g_mod, name)
	if err != cu.OK || fn == nil {
		fmt.eprintf("cuda: kernel %s missing (%s)\n", name, cu.error_string(err))
		return nil
	}
	return fn
}

@(private)
alloc_bytes :: proc(n: int) -> (cu.Device_Ptr, bool) {
	if n <= 0 {
		return 0, true
	}
	p, err := cu.alloc(n)
	if err != cu.OK || p == 0 {
		fmt.eprintf("cuda: alloc %d bytes: %s\n", n, cu.error_string(err))
		return 0, false
	}
	append(&live, p)
	return p, true
}

@(private)
free_live :: proc() {
	for p in live {
		cu.free(p)
	}
	clear(&live)
}

@(private)
q8_drop :: proc() {
	g_q8_valid = false
}

@(private)
gemv2_for :: proc(kind: ggml.GGML_Type) -> cu.Function {
	if kind == .Q4_K do return k_gemv2_q4
	if kind == .Q6_K do return k_gemv2_q6
	if kind == .Q5_K do return k_gemv2_q5k
	if kind == .Q8_0 do return k_gemv2_q8
	if kind == .Q5_0 do return k_gemv2_q50
	return nil
}

@(private)
gemv2_aligned :: proc(kind: ggml.GGML_Type, n: int) -> bool {
	if n <= 0 || n > g_q8_cap || d_xq == 0 {
		return false
	}
	if kind == .Q4_K || kind == .Q6_K || kind == .Q5_K {
		return n % 256 == 0
	}
	if kind == .Q8_0 || kind == .Q5_0 {
		return n % 32 == 0
	}
	return false
}

@(private)
q8_ensure :: proc(x: cu.Device_Ptr, n: int) -> bool {
	if g_q8_valid && g_q8_src == x && g_q8_n == n {
		return true
	}
	if k_quant == nil || n > g_q8_cap || n % 32 != 0 {
		return false
	}
	xx := x
	qs, sc, sm := d_xq, d_xd, d_xs
	nn := i32(n)
	args := [?]rawptr{&xx, &qs, &sc, &sm, &nn}
	nb := n / 32
	if !klaunch(k_quant, u32((nb + 127) / 128), 1, 1, 128, 1, 1, args[:], "quant_q8") {
		return false
	}
	g_q8_src = x
	g_q8_n = n
	g_q8_valid = true
	return true
}

@(private)
launch_gemv_old :: proc(kind: ggml.GGML_Type, w, x, y: cu.Device_Ptr, n, d: int) -> bool {
	fn := gemv_fn[int(kind)]
	if fn == nil {
		fmt.eprintf("cuda: no GEMV for %v\n", kind)
		return false
	}
	wb, xx, yy := w, x, y
	nn, dd := i32(n), i32(d)
	args := [?]rawptr{&wb, &xx, &yy, &nn, &dd}
	return klaunch(fn, u32(d), 1, 1, 32, 1, 1, args[:], "gemv")
}

@(private)
launch_gemv2 :: proc(fn: cu.Function, kind: ggml.GGML_Type, w, y: cu.Device_Ptr, n, d: int) -> bool {
	warps := g_gemv_warps
	if warps < 1 {
		warps = 1
	}
	if warps > 16 {
		warps = 16
	}
	wb, yy := w, y
	qs, sc, sm := d_xq, d_xd, d_xs
	nn, dd := i32(n), i32(d)
	rows_per := warps * 4
	gx := u32((d + rows_per - 1) / rows_per)
	bx := u32(warps * 32)
	if kind == .Q4_K || kind == .Q5_K {
		args := [?]rawptr{&wb, &qs, &sc, &sm, &yy, &nn, &dd}
		return klaunch(fn, gx, 1, 1, bx, 1, 1, args[:], "gemv2")
	}
	args := [?]rawptr{&wb, &qs, &sc, &yy, &nn, &dd}
	return klaunch(fn, gx, 1, 1, bx, 1, 1, args[:], "gemv2")
}

launch_gemv :: proc(kind: ggml.GGML_Type, w, x, y: cu.Device_Ptr, n, d: int) -> bool {
	if d <= 0 || n <= 0 {
		return true
	}
	if g_gemv_new {
		fn := gemv2_for(kind)
		if fn != nil && gemv2_aligned(kind, n) && q8_ensure(x, n) {
			return launch_gemv2(fn, kind, w, y, n, d)
		}
	}
	return launch_gemv_old(kind, w, x, y, n, d)
}

@(private)
launch_mmq :: proc(kind: ggml.GGML_Type, w, x, y: cu.Device_Ptr, n, d, tcount: int) -> bool {
	if d <= 0 || n <= 0 || tcount <= 0 {
		return true
	}
	fn := mmq_fn[int(kind)]
	if fn == nil {
		fmt.eprintf("cuda: no MMQ for %v\n", kind)
		return false
	}
	wb, xx, yy := w, x, y
	nn, dd, tt := i32(n), i32(d), i32(tcount)
	args := [?]rawptr{&wb, &xx, &yy, &nn, &dd, &tt}
	return klaunch(fn, u32(d), 1, 1, 32, 1, 1, args[:], "mmq")
}

@(private)
use_tensor_cores :: proc(tcount: int) -> bool {
	if g_sm_major < 7 || tcount <= 1 || g_tc_mode == 0 {
		return false
	}
	if g_tc_mode == 1 {
		return true
	}
	return tcount >= TC_MIN_T
}

@(private)
launch_tc :: proc(kind: ggml.GGML_Type, w, x, y: cu.Device_Ptr, n, d, tcount: int) -> bool {
	fn := tc_fn[int(kind)]
	if fn == nil {
		return false
	}
	wb, xx, yy := w, x, y
	nn, dd, tt := i32(n), i32(d), i32(tcount)
	args := [?]rawptr{&wb, &xx, &yy, &nn, &dd, &tt}
	gx := u32((d + 63) / 64)
	gy := u32((tcount + 63) / 64)
	return klaunch(fn, gx, gy, 1, 512, 1, 1, args[:], "tc")
}

@(private)
launch_mm :: proc(t: Tensor, x, y: cu.Device_Ptr, n, d, tcount: int) -> bool {
	w := wptr(raw_data(t.data))
	if tcount <= 1 {
		return launch_gemv(t.kind, w, x, y, n, d)
	}
	if use_tensor_cores(tcount) && tc_fn[int(t.kind)] != nil {
		return launch_tc(t.kind, w, x, y, n, d, tcount)
	}
	return launch_mmq(t.kind, w, x, y, n, d, tcount)
}

@(private)
launch_rmsnorm :: proc(inp, outp, weight: cu.Device_Ptr, rows, size: int, eps: f32) -> bool {
	if rows <= 0 || size <= 0 {
		return true
	}
	// Decode (one row): RMSNorm and the q8 activation quant share one pass.
	// The following GEMVs of this vector then skip the quant kernel.
	if g_gemv_new && rows == 1 && size % 32 == 0 && size <= g_q8_cap && k_rmsnorm_q8 != nil && d_xq != 0 {
		a, b, c := inp, outp, weight
		qs, sc, sm := d_xq, d_xd, d_xs
		sz := i32(size)
		ep := eps
		args := [?]rawptr{&a, &b, &c, &qs, &sc, &sm, &sz, &ep}
		if !klaunch(k_rmsnorm_q8, 1, 1, 1, 128, 1, 1, args[:], "rmsnorm_q8") {
			return false
		}
		g_q8_src = outp
		g_q8_n = size
		g_q8_valid = true
		return true
	}
	q8_drop()
	a, b, c := inp, outp, weight
	sz := i32(size)
	ep := eps
	args := [?]rawptr{&a, &b, &c, &sz, &ep}
	return klaunch(k_rmsnorm, u32(rows), 1, 1, 128, 1, 1, args[:], "rmsnorm")
}

@(private)
launch_qnorm :: proc(qproj, q, qn: cu.Device_Ptr, nheads, head: int, eps: f32) -> bool {
	a, b, c := qproj, q, qn
	n := i32(head)
	ep := eps
	args := [?]rawptr{&a, &b, &c, &n, &ep}
	return klaunch(k_qnorm, u32(nheads), 1, 1, 128, 1, 1, args[:], "qnorm")
}

@(private)
launch_rope :: proc(vec: cu.Device_Ptr, head_dim, pos: int, theta: f32, rotary, nheads: int) -> bool {
	half := rotary / 2
	total := nheads * half
	if total <= 0 {
		return true
	}
	v := vec
	hd, ps, rd, nh := i32(head_dim), i32(pos), i32(rotary), i32(nheads)
	th := theta
	args := [?]rawptr{&v, &hd, &ps, &th, &rd, &nh}
	return klaunch(k_rope, grid_n(total, 128), 1, 1, 128, 1, 1, args[:], "rope")
}

@(private)
launch_gate :: proc(xb3, qproj: cu.Device_Ptr, head_dim, nheads: int) -> bool {
	a, b := xb3, qproj
	hd, nh := i32(head_dim), i32(nheads)
	args := [?]rawptr{&a, &b, &hd, &nh}
	n := nheads * head_dim
	return klaunch(k_gate, grid_n(n, 256), 1, 1, 256, 1, 1, args[:], "attn_gate")
}

@(private)
launch_swiglu :: proc(hb, hb2: cu.Device_Ptr, n: int) -> bool {
	q8_drop()
	a, b := hb, hb2
	nn := i32(n)
	args := [?]rawptr{&a, &b, &nn}
	return klaunch(k_swiglu, grid_n(n, 256), 1, 1, 256, 1, 1, args[:], "swiglu")
}

@(private)
launch_residual :: proc(x, y: cu.Device_Ptr, n: int) -> bool {
	a, b := x, y
	nn := i32(n)
	args := [?]rawptr{&a, &b, &nn}
	return klaunch(k_residual, grid_n(n, 256), 1, 1, 256, 1, 1, args[:], "residual")
}

@(private)
launch_axpy :: proc(y, x: cu.Device_Ptr, a: f32, n: int) -> bool {
	yy, xx := y, x
	aa := a
	nn := i32(n)
	args := [?]rawptr{&yy, &xx, &aa, &nn}
	return klaunch(k_axpy, grid_n(n, 256), 1, 1, 256, 1, 1, args[:], "axpy")
}

@(private)
store_f16_call :: proc(k, v, kc, vc: cu.Device_Ptr, pos: int) -> bool {
	a, b, c, d := k, v, kc, vc
	hd, sl, ps, nk := i32(g_head), i32(g_seq), i32(pos), i32(g_kv_heads)
	args := [?]rawptr{&a, &b, &c, &d, &hd, &sl, &ps, &nk}
	n := g_kv_heads * g_head
	return klaunch(k_store_f16, grid_n(n, 128), 1, 1, 128, 1, 1, args[:], "store_f16")
}

@(private)
attn_f16_call :: proc(q, kc, vc, xb3: cu.Device_Ptr, pos: int) -> bool {
	a, b, c, d := q, kc, vc, xb3
	sc := d_scores
	hd, sl, mul, ps := i32(g_head), i32(g_seq), i32(g_heads / g_kv_heads), i32(pos)
	args := [?]rawptr{&a, &b, &c, &d, &sc, &hd, &sl, &mul, &ps}
	return klaunch(k_attn_f16, u32(g_heads), 1, 1, 128, 1, 1, args[:], "attn_f16")
}

@(private)
store_q8_call :: proc(k, v, kc, vc: cu.Device_Ptr, pos: int) -> bool {
	a, b, c, d := k, v, kc, vc
	nb, pb, ps := i32(g_kv_dim / 32), i32(g_k_pos), i32(pos)
	args := [?]rawptr{&a, &b, &c, &d, &nb, &pb, &ps}
	return klaunch(k_store_q8, grid_n(g_kv_dim / 32, 64), 1, 1, 64, 1, 1, args[:], "store_q8")
}

@(private)
attn_q8_call :: proc(q, kc, vc, xb3: cu.Device_Ptr, pos: int) -> bool {
	a, b, c, d := q, kc, vc, xb3
	sc := d_scores
	hd, pb, mul, ps, sl := i32(g_head), i32(g_k_pos), i32(g_heads / g_kv_heads), i32(pos), i32(g_seq)
	args := [?]rawptr{&a, &b, &c, &d, &sc, &hd, &pb, &mul, &ps, &sl}
	return klaunch(k_attn_q8, u32(g_heads), 1, 1, 128, 1, 1, args[:], "attn_q8")
}

@(private)
store_q8v_call :: proc(k, v, kc, vc: cu.Device_Ptr, pos: int) -> bool {
	a, b, c, d := k, v, kc, vc
	nb, kb, vb, ps := i32(g_kv_dim / 32), i32(g_k_pos), i32(g_v_pos), i32(pos)
	args := [?]rawptr{&a, &b, &c, &d, &nb, &kb, &vb, &ps}
	return klaunch(k_store_q8v, grid_n(g_kv_dim / 32, 64), 1, 1, 64, 1, 1, args[:], "store_q8v")
}

@(private)
attn_q8v_call :: proc(q, kc, vc, xb3: cu.Device_Ptr, pos: int) -> bool {
	a, b, c, d := q, kc, vc, xb3
	sc := d_scores
	hd := i32(g_head)
	kb, vb := i32(g_k_pos), i32(g_v_pos)
	mul, ps, sl := i32(g_heads / g_kv_heads), i32(pos), i32(g_seq)
	args := [?]rawptr{&a, &b, &c, &d, &sc, &hd, &kb, &vb, &mul, &ps, &sl}
	return klaunch(k_attn_q8v, u32(g_heads), 1, 1, 128, 1, 1, args[:], "attn_q8v")
}

@(private)
launch_conv :: proc(outp, inp, state, weight: cu.Device_Ptr, conv_dim, kernel: int) -> bool {
	a, b, c, d := outp, inp, state, weight
	cd, k := i32(conv_dim), i32(kernel)
	args := [?]rawptr{&a, &b, &c, &d, &cd, &k}
	return klaunch(k_conv, grid_n(conv_dim, 128), 1, 1, 128, 1, 1, args[:], "conv")
}

@(private)
launch_l2 :: proc(q_out, k_out, q_in, k_in: cu.Device_Ptr, n_kh, head_k: int, eps, scale: f32) -> bool {
	a, b, c, d := q_out, k_out, q_in, k_in
	hk := i32(head_k)
	ep, sc := eps, scale
	args := [?]rawptr{&a, &b, &c, &d, &hk, &ep, &sc}
	return klaunch(k_l2, u32(n_kh), 1, 1, 128, 1, 1, args[:], "l2norm")
}

@(private)
launch_delta :: proc(outp, state, q, k, v, b, a, decay, dt: cu.Device_Ptr, hkd, hvd, nkh, nvh: int) -> bool {
	o, st, qq, kk, vv, bb, aa, dec, dtb := outp, state, q, k, v, b, a, decay, dt
	ik, iv, nk := i32(hkd), i32(hvd), i32(nkh)
	args := [?]rawptr{&o, &st, &qq, &kk, &vv, &bb, &aa, &dec, &dtb, &ik, &iv, &nk}
	return klaunch(k_delta, u32(nvh), 1, 1, 128, 1, 1, args[:], "delta")
}

@(private)
launch_gated :: proc(outp, inp, weight, gate: cu.Device_Ptr, n_vh, hvd: int, eps: f32) -> bool {
	a, b, c, d := outp, inp, weight, gate
	sz := i32(hvd)
	ep := eps
	args := [?]rawptr{&a, &b, &c, &d, &sz, &ep}
	return klaunch(k_rmsgate, u32(n_vh), 1, 1, 128, 1, 1, args[:], "rmsnorm_gated")
}

@(private)
launch_topk :: proc(n, k: int, clamp_v, scale: f32) -> bool {
	logits, sel, wt := d_moe_logits, d_moe_sel, d_moe_wt
	nn, kk := i32(n), i32(k)
	cl, sc := clamp_v, scale
	args := [?]rawptr{&logits, &sel, &wt, &nn, &kk, &cl, &sc}
	return klaunch(k_moe_topk, 1, 1, 1, 1, 1, 1, args[:], "moe_topk")
}

@(private)
bind_kernels :: proc() -> bool {
	tags := [?]string{
		"f32", "f16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0",
		"q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_k",
	}
	kinds := [?]ggml.GGML_Type{
		.F32, .F16, .Q4_0, .Q4_1, .Q5_0, .Q5_1, .Q8_0,
		.Q2_K, .Q3_K, .Q4_K, .Q5_K, .Q6_K, .Q8_K,
	}
	for tag, i in tags {
		gemv_fn[int(kinds[i])] = get_fn(fmt.tprintf("gemv_%s", tag))
		mmq_fn[int(kinds[i])] = get_fn(fmt.tprintf("mmq_%s", tag))
		tc_fn[int(kinds[i])] = get_fn(fmt.tprintf("tc_%s", tag))
		if gemv_fn[int(kinds[i])] == nil || mmq_fn[int(kinds[i])] == nil || tc_fn[int(kinds[i])] == nil {
			return false
		}
	}
	k_rmsnorm = get_fn("rmsnorm_k")
	k_qnorm = get_fn("qnorm_extract")
	k_rope = get_fn("rope_partial")
	k_gate = get_fn("attn_gate")
	k_swiglu = get_fn("swiglu")
	k_residual = get_fn("residual")
	k_axpy = get_fn("axpy_k")
	k_store_f16 = get_fn("store_kv_f16")
	k_attn_f16 = get_fn("attention_f16")
	k_store_q8 = get_fn("store_kv_q8")
	k_attn_q8 = get_fn("attention_q8")
	k_store_q8v = get_fn("store_kv_q8v")
	k_attn_q8v = get_fn("attention_q8v")
	k_conv = get_fn("conv1d_step")
	k_l2 = get_fn("l2norm_scale")
	k_delta = get_fn("delta_recurrent")
	k_rmsgate = get_fn("rmsnorm_gated")
	k_moe_topk = get_fn("moe_softmax_topk")
	k_quant = get_fn("quant_q8")
	k_rmsnorm_q8 = get_fn("rmsnorm_quant_q8")
	k_gemv2_q4 = get_fn("gemv2_q4_k")
	k_gemv2_q6 = get_fn("gemv2_q6_k")
	k_gemv2_q5k = get_fn("gemv2_q5_k")
	k_gemv2_q8 = get_fn("gemv2_q8_0")
	k_gemv2_q50 = get_fn("gemv2_q5_0")
	handles := [?]cu.Function{
		k_rmsnorm, k_qnorm, k_rope, k_gate, k_swiglu, k_residual, k_axpy,
		k_store_f16, k_attn_f16, k_store_q8, k_attn_q8, k_store_q8v, k_attn_q8v,
		k_conv, k_l2, k_delta, k_rmsgate, k_moe_topk,
		k_quant, k_rmsnorm_q8, k_gemv2_q4, k_gemv2_q6, k_gemv2_q5k, k_gemv2_q8, k_gemv2_q50,
	}
	for h in handles {
		if h == nil {
			return false
		}
	}
	return true
}

@(private)
scan_model :: proc(t: ^Transformer) -> bool {
	p := &t.config
	w := &t.weights
	if !ggml.is_supported_quant(w.token_embedding.kind) {
		fmt.eprintf("cuda: embedding kind %v is not supported\n", w.token_embedding.kind)
		return false
	}
	if !check_tensor("output", w.output, p.dim) {
		return false
	}
	qdim := p.n_heads * p.head_dim * 2
	kvdim := p.n_kv_heads * p.head_dim
	attdim := p.n_heads * p.head_dim
	keydim := p.lin_n_k_heads * p.lin_head_k_dim
	valdim := p.lin_n_v_heads * p.lin_head_v_dim
	convdim := keydim * 2 + valdim
	for l in 0 ..< p.n_layers {
		lw := &w.layers[l]
		name := fmt.tprintf("layer %d", l)
		if p.is_moe {
			mw := &lw.moe
			if !check_tensor(fmt.tprintf("%s up_exps", name), mw.up_exps, p.dim) do return false
			if !check_tensor(fmt.tprintf("%s gate_exps", name), mw.gate_exps, p.dim) do return false
			if !check_tensor(fmt.tprintf("%s down_exps", name), mw.down_exps, p.n_ff_exp) do return false
		} else {
			if !check_tensor(fmt.tprintf("%s w1", name), lw.w1, p.dim) do return false
			if !check_tensor(fmt.tprintf("%s w3", name), lw.w3, p.dim) do return false
			if !check_tensor(fmt.tprintf("%s w2", name), lw.w2, p.hidden_dim) do return false
		}
		switch lw.layer_type {
		case .Full_Attention:
			fa := &lw.full
			if !check_tensor(fmt.tprintf("%s wq", name), fa.wq, p.dim) do return false
			if !check_tensor(fmt.tprintf("%s wk", name), fa.wk, p.dim) do return false
			if !check_tensor(fmt.tprintf("%s wv", name), fa.wv, p.dim) do return false
			if !check_tensor(fmt.tprintf("%s wo", name), fa.wo, attdim) do return false
			_ = qdim
			_ = kvdim
		case .Linear_Attention:
			la := &lw.linear
			if !check_tensor(fmt.tprintf("%s qkv", name), la.in_qkv, p.dim) do return false
			if !check_tensor(fmt.tprintf("%s z", name), la.in_z, p.dim) do return false
			if !check_tensor(fmt.tprintf("%s b", name), la.in_b, p.dim) do return false
			if !check_tensor(fmt.tprintf("%s a", name), la.in_a, p.dim) do return false
			if !check_tensor(fmt.tprintf("%s out", name), la.out, valdim) do return false
			if la.conv.kind != .F32 {
				fmt.eprintf("cuda: %s conv kind %v, want F32\n", name, la.conv.kind)
				return false
			}
			_ = convdim
		}
	}
	if p.n_expert > MOE_MAX_EXPERTS {
		fmt.eprintf("cuda: n_expert %d > %d\n", p.n_expert, MOE_MAX_EXPERTS)
		return false
	}
	return true
}

@(private)
alloc_acts :: proc(t: ^Transformer, T: int) -> bool {
	p := &t.config
	dim := p.dim
	hidden := p.hidden_dim
	qdim := max(p.n_heads * p.head_dim * 2, 1)
	kvdim := max(p.n_kv_heads * p.head_dim, 1)
	attdim := max(p.n_heads * p.head_dim, 1)
	keydim := p.lin_n_k_heads * p.lin_head_k_dim
	valdim := p.lin_n_v_heads * p.lin_head_v_dim
	convdim := keydim * 2 + valdim
	nff := max(hidden, p.n_ff_exp, p.n_ff_shexp)
	nvh := max(p.lin_n_v_heads, 1)
	nexp := max(p.n_expert, 1)
	ok: bool
	d_x, ok = alloc_bytes(T * dim * 4); if !ok do return false
	d_xb, ok = alloc_bytes(T * dim * 4); if !ok do return false
	d_xb2, ok = alloc_bytes(T * dim * 4); if !ok do return false
	d_xb3, ok = alloc_bytes(T * attdim * 4); if !ok do return false
	d_qproj, ok = alloc_bytes(T * qdim * 4); if !ok do return false
	d_q, ok = alloc_bytes(T * attdim * 4); if !ok do return false
	d_k, ok = alloc_bytes(T * kvdim * 4); if !ok do return false
	d_v, ok = alloc_bytes(T * kvdim * 4); if !ok do return false
	d_hb, ok = alloc_bytes(T * max(nff, valdim, 1) * 4); if !ok do return false
	d_hb2, ok = alloc_bytes(T * max(nff, 1) * 4); if !ok do return false
	if convdim > 0 {
		d_qkv, ok = alloc_bytes(T * convdim * 4); if !ok do return false
		d_qkv2, ok = alloc_bytes(T * convdim * 4); if !ok do return false
	}
	if valdim > 0 {
		d_z, ok = alloc_bytes(T * valdim * 4); if !ok do return false
		d_lout, ok = alloc_bytes(T * valdim * 4); if !ok do return false
	}
	d_b, ok = alloc_bytes(T * nvh * 4); if !ok do return false
	d_a, ok = alloc_bytes(T * nvh * 4); if !ok do return false
	if keydim > 0 {
		d_qlin, ok = alloc_bytes(T * keydim * 4); if !ok do return false
		d_klin, ok = alloc_bytes(T * keydim * 4); if !ok do return false
	}
	d_logits, ok = alloc_bytes(p.vocab_size * 4); if !ok do return false
	if p.n_heads > 0 {
		d_scores, ok = alloc_bytes(p.n_heads * p.seq_len * 4); if !ok do return false
	}
	d_partial, ok = alloc_bytes(dim * 4); if !ok do return false
	max_k := dim
	max_k = max(max_k, hidden)
	max_k = max(max_k, qdim)
	max_k = max(max_k, attdim)
	max_k = max(max_k, convdim)
	max_k = max(max_k, valdim)
	max_k = max(max_k, nff)
	if max_k < 32 {
		max_k = 32
	}
	if max_k % 32 != 0 {
		max_k = ((max_k + 31) / 32) * 32
	}
	g_q8_cap = max_k
	d_xq, ok = alloc_bytes(max_k); if !ok do return false
	d_xd, ok = alloc_bytes((max_k / 32) * 4); if !ok do return false
	d_xs, ok = alloc_bytes((max_k / 32) * 8); if !ok do return false
	if p.is_moe {
		d_moe_logits, ok = alloc_bytes(nexp * 4); if !ok do return false
		d_moe_wt, ok = alloc_bytes(nexp * 4); if !ok do return false
		d_moe_sel, ok = alloc_bytes(nexp * size_of(i32)); if !ok do return false
	}
	return true
}

@(private)
upload_moe :: proc(t: ^Transformer) -> bool {
	if !t.config.is_moe {
		return true
	}
	p := &t.config
	moe_gpu = make([]Moe_GPU, p.n_layers)
	for l in 0 ..< p.n_layers {
		mw := &t.weights.layers[l].moe
		g := &moe_gpu[l]
		nbytes := p.n_expert * p.dim * 4
		ok: bool
		g.gate, ok = alloc_bytes(nbytes)
		if !ok do return false
		if !ck(cu.upload(g.gate, raw_data(mw.gate_inp), nbytes), "upload gate_inp") {
			return false
		}
		if mw.shexp_up != nil {
			g.has_shexp = true
			up_n := p.n_ff_shexp * p.dim * 4
			down_n := p.dim * p.n_ff_shexp * 4
			g.up, ok = alloc_bytes(up_n); if !ok do return false
			g.gate_w, ok = alloc_bytes(up_n); if !ok do return false
			g.down, ok = alloc_bytes(down_n); if !ok do return false
			g.ginp, ok = alloc_bytes(p.dim * 4); if !ok do return false
			if !ck(cu.upload(g.up, raw_data(mw.shexp_up), up_n), "shexp_up") do return false
			if !ck(cu.upload(g.gate_w, raw_data(mw.shexp_gate), up_n), "shexp_gate") do return false
			if !ck(cu.upload(g.down, raw_data(mw.shexp_down), down_n), "shexp_down") do return false
			if !ck(cu.upload(g.ginp, raw_data(mw.shexp_gate_inp), p.dim * 4), "shexp_ginp") do return false
		}
	}
	return true
}

cuda_destroy :: proc() {
	g_ready = false
	free_live()
	if d_mmap != 0 {
		cu.free(d_mmap)
		d_mmap = 0
	}
	if g_mod != nil {
		cu.drv.ModuleUnload(g_mod)
		g_mod = nil
	}
	if g_ctx_held {
		cu.drv.DevicePrimaryCtxRelease_v2(g_dev)
		g_ctx_held = false
	}
	d_xq, d_xd, d_xs = 0, 0, 0
	g_q8_valid = false
	g_q8_cap = 0
	delete(h_logits); h_logits = nil
	delete(h_embed); h_embed = nil
	delete(h_wt); h_wt = nil
	delete(h_sel); h_sel = nil
	delete(moe_gpu); moe_gpu = nil
	g_max_t = 0
	do_store = nil
	do_attn = nil
}

cuda_init :: proc(t: ^Transformer, model_path: string) -> bool {
	cuda_destroy()
	if cu.load_driver() != cu.OK {
		fmt.eprintln("cuda: libcuda.so.1 not found")
		return false
	}
	if !ck(cu.drv.Init(0), "cuInit") {
		return false
	}
	ord := 0
	if s := os.get_env("CUDA_DEVICE", context.temp_allocator); s != "" {
		if v, ok := strconv.parse_int(s); ok {
			ord = int(v)
		}
	}
	if !ck(cu.drv.DeviceGet(&g_dev, c.int(ord)), "cuDeviceGet") {
		return false
	}
	namebuf: [128]u8
	cu.drv.DeviceGetName(raw_data(namebuf[:]), 128, g_dev)
	major, minor: c.int
	cu.drv.DeviceGetAttribute(&major, cu.ATTR_COMPUTE_CAPABILITY_MAJOR, g_dev)
	cu.drv.DeviceGetAttribute(&minor, cu.ATTR_COMPUTE_CAPABILITY_MINOR, g_dev)
	if int(major) < 7 {
		fmt.eprintf("cuda: sm_%d%d is below sm_70\n", major, minor)
		return false
	}
	g_sm_major = int(major)
	tc_env := os.get_env("ODINFER_CUDA_TC", context.temp_allocator)
	if tc_env == "0" {
		g_tc_mode = 0
	} else if tc_env == "1" {
		g_tc_mode = 1
	} else {
		g_tc_mode = -1
	}
	gemv_env := os.get_env("ODINFER_CUDA_GEMV", context.temp_allocator)
	g_gemv_new = gemv_env != "old" && gemv_env != "0"
	g_gemv_warps = 8
	if s := os.get_env("ODINFER_CUDA_GEMV_WARPS", context.temp_allocator); s != "" {
		if v, ok := strconv.parse_int(s); ok && v >= 1 && v <= 16 {
			g_gemv_warps = int(v)
		}
	}
	g_count_launches = os.get_env("ODINFER_CUDA_LAUNCHES", context.temp_allocator) == "1"
	g_launches = 0
	g_q8_valid = false
	mem_bytes: c.size_t
	cu.drv.DeviceTotalMem_v2(&mem_bytes, g_dev)
	ctx: cu.Context
	if !ck(cu.drv.DevicePrimaryCtxRetain(&ctx, g_dev), "primary ctx") {
		return false
	}
	g_ctx_held = true
	if !ck(cu.drv.CtxSetCurrent(ctx), "ctx set") {
		cuda_destroy()
		return false
	}
	arch := fmt.tprintf("compute_%d%d", major, minor)
	opt := fmt.tprintf("--gpu-architecture=%s", arch)
	ptx, log, err := cu.compile_ptx(CUDA_KERNEL_SRC, "odinfer_cuda.cu", []string{opt})
	if len(log) > 0 {
		fmt.eprintfln("cuda: nvrtc log:\n%s", log)
	}
	if err != cu.OK {
		fmt.eprintf("cuda: NVRTC compile for %s failed: %s\n", arch, cu.error_string(err))
		pre75 := int(major) < 7 || (int(major) == 7 && int(minor) < 5)
		if pre75 {
			fmt.eprintln("cuda: sm_70 needs NVRTC 12.x. Set ODINFER_NVRTC_PATH to libnvrtc.so.12 from pip nvidia-cuda-nvrtc-cu12. CUDA 13 NVRTC cannot target compute_70.")
		} else {
			fmt.eprintln("cuda: set NVRTC_DIR or ODINFER_NVRTC_PATH to the directory of libnvrtc.so.13 (versioned soname, not the bare .so)")
		}
		cuda_destroy()
		return false
	}
	mod, jit, merr := cu.module_load_ptx(ptx)
	if len(jit) > 0 {
		fmt.eprintfln("cuda: jit: %s", jit)
	}
	if merr != cu.OK {
		fmt.eprintf("cuda: cuModuleLoadDataEx failed: %s\n", cu.error_string(merr))
		cuda_destroy()
		return false
	}
	g_mod = mod
	if !bind_kernels() {
		cuda_destroy()
		return false
	}
	if !scan_model(t) {
		cuda_destroy()
		return false
	}
	p := &t.config
	if g_kv_q8 && (p.n_kv_heads * p.head_dim) % 32 != 0 {
		fmt.eprintln("cuda: KV_Q8 requires kv_dim % 32 == 0")
		cuda_destroy()
		return false
	}
	mmap := t.gguf.mmap
	dp, aerr := cu.alloc(len(mmap))
	if aerr != cu.OK {
		fmt.eprintf("cuda: weight alloc %d bytes: %s\n", len(mmap), cu.error_string(aerr))
		cuda_destroy()
		return false
	}
	d_mmap = dp
	g_mmap_base = uintptr(raw_data(mmap))
	if !ck(cu.upload(d_mmap, raw_data(mmap), len(mmap)), "upload weights") {
		cuda_destroy()
		return false
	}

	g_dim = p.dim
	g_hidden = p.hidden_dim
	g_vocab = p.vocab_size
	g_seq = p.seq_len
	g_heads = p.n_heads
	g_kv_heads = p.n_kv_heads
	g_head = p.head_dim
	g_kv_dim = p.n_kv_heads * p.head_dim
	g_rot = p.rotary_dim
	if g_kv_q8 {
		g_k_pos = kv_pos_bytes(g_kv_dim)
		g_k_layer = kv_layer_bytes_q8(p.seq_len, g_kv_dim)
		g_k_bytes = kv_total_bytes_q8(p.n_full, p.seq_len, g_kv_dim)
		if g_kv_v_mix {
			g_v_pos = kv_v_mix_pos_bytes(g_kv_dim)
			g_v_layer = kv_v_mix_layer_bytes(p.seq_len, g_kv_dim)
			g_v_bytes = kv_v_mix_total_bytes(p.n_full, p.seq_len, g_kv_dim)
			do_store = store_q8v_call
			do_attn = attn_q8v_call
		} else {
			g_v_pos = g_k_pos
			g_v_layer = g_k_layer
			g_v_bytes = g_k_bytes
			do_store = store_q8_call
			do_attn = attn_q8_call
		}
	} else {
		g_k_layer = p.seq_len * g_kv_dim * 2
		g_v_layer = g_k_layer
		g_k_bytes = p.n_full * g_k_layer
		g_v_bytes = g_k_bytes
		do_store = store_f16_call
		do_attn = attn_f16_call
	}
	conv_f := 0
	rec_f := 0
	if p.n_linear > 0 {
		cd := compute_conv_dim(p)
		conv_f = p.n_linear * cd * max(p.lin_conv_kernel - 1, 0)
		rec_f = p.n_linear * p.lin_n_v_heads * p.lin_head_k_dim * p.lin_head_v_dim
	}
	g_conv_bytes = conv_f * 4
	g_rec_bytes = rec_f * 4

	chosen := 0
	batches := [?]int{CUDA_MAX_BATCH, 32, 8, 1}
	for batch in batches {
		free_live()
		if alloc_acts(t, batch) {
			chosen = batch
			break
		}
		fmt.eprintf("cuda: batch %d does not fit, trying smaller\n", batch)
	}
	if chosen == 0 {
		fmt.eprintln("cuda: activation alloc failed")
		cuda_destroy()
		return false
	}
	g_max_t = chosen
	ok: bool
	d_kc, ok = alloc_bytes(g_k_bytes); if !ok { cuda_destroy(); return false }
	d_vc, ok = alloc_bytes(g_v_bytes); if !ok { cuda_destroy(); return false }
	d_conv, ok = alloc_bytes(g_conv_bytes); if !ok { cuda_destroy(); return false }
	d_rec, ok = alloc_bytes(g_rec_bytes); if !ok { cuda_destroy(); return false }
	cu.memset(d_kc, 0, g_k_bytes)
	cu.memset(d_vc, 0, g_v_bytes)
	cu.memset(d_conv, 0, g_conv_bytes)
	cu.memset(d_rec, 0, g_rec_bytes)
	if !upload_moe(t) {
		cuda_destroy()
		return false
	}
	h_logits = make([]f32, p.vocab_size)
	h_embed = make([]f32, chosen * p.dim)
	if p.is_moe {
		h_sel = make([]i32, p.n_expert)
		h_wt = make([]f32, p.n_expert)
	}
	g_timing = os.get_env("QTIMING", context.temp_allocator) == "1"
	g_ready = true
	nlen := 0
	for nlen < len(namebuf) && namebuf[nlen] != 0 {
		nlen += 1
	}
	fmt.eprintfln(
		"cuda: %s sm_%d%d  %.1f GiB  %s  nvrtc %s  kv %s  batch %d",
		string(namebuf[:nlen]), major, minor,
		f64(mem_bytes) / (1024 * 1024 * 1024), arch, cu.nv_loaded_from,
		kv_resident_tag(), chosen,
	)
	fmt.eprintfln("cuda: weights %d MiB from %s", len(mmap) / (1024 * 1024), model_path)
	tc_note := "auto (T>=16)"
	if g_tc_mode == 0 {
		tc_note = "off (ODINFER_CUDA_TC=0)"
	} else if g_tc_mode == 1 {
		tc_note = "on for T>1 (ODINFER_CUDA_TC=1)"
	}
	fmt.eprintfln("cuda: tensor cores %s  WMMA 16x16x16 fp16, fp32 acc", tc_note)
	if g_gemv_new {
		fmt.eprintfln("cuda: gemv dp4a  warps={}  4 rows/warp  q8 act  (ODINFER_CUDA_GEMV=old for the scalar kernel)", g_gemv_warps)
	} else {
		fmt.eprintln("cuda: gemv old (ODINFER_CUDA_GEMV=old)")
	}
	return true
}

cuda_reset_state :: proc() {
	if !g_ready {
		return
	}
	cu.memset(d_kc, 0, g_k_bytes)
	cu.memset(d_vc, 0, g_v_bytes)
	cu.memset(d_conv, 0, g_conv_bytes)
	cu.memset(d_rec, 0, g_rec_bytes)
	cu.sync()
}

@(private)
cuda_full :: proc(tr: ^Transformer, l, pos0, T: int) {
	p := &tr.config
	lw := &tr.weights.layers[l]
	fa := &lw.full
	dim := p.dim
	qdim := p.n_heads * p.head_dim * 2
	kvdim := p.n_kv_heads * p.head_dim
	attdim := p.n_heads * p.head_dim
	eps := p.rms_eps
	must(launch_mm(fa.wq, d_xb, d_qproj, dim, qdim, T))
	must(launch_mm(fa.wk, d_xb, d_k, dim, kvdim, T))
	must(launch_mm(fa.wv, d_xb, d_v, dim, kvdim, T))
	must(launch_qnorm(d_qproj, d_q, dev_f32(fa.q_norm), T * p.n_heads, p.head_dim, eps))
	must(launch_rmsnorm(d_k, d_k, dev_f32(fa.k_norm), T * p.n_kv_heads, p.head_dim, eps))
	slot := tr.full_slot[l]
	kc := cu.ptr_add(d_kc, slot * g_k_layer)
	vc := cu.ptr_add(d_vc, slot * g_v_layer)
	for ti in 0 ..< T {
		qrow := fp(d_q, ti * attdim)
		krow := fp(d_k, ti * kvdim)
		vrow := fp(d_v, ti * kvdim)
		qproj := fp(d_qproj, ti * qdim)
		xb3 := fp(d_xb3, ti * attdim)
		pos := pos0 + ti
		must(launch_rope(qrow, p.head_dim, pos, p.rope_theta, p.rotary_dim, p.n_heads))
		must(launch_rope(krow, p.head_dim, pos, p.rope_theta, p.rotary_dim, p.n_kv_heads))
		must(do_store(krow, vrow, kc, vc, pos))
		must(do_attn(qrow, kc, vc, xb3, pos))
		must(launch_gate(xb3, qproj, p.head_dim, p.n_heads))
	}
	must(launch_mm(fa.wo, d_xb3, d_xb2, attdim, dim, T))
	must(launch_residual(d_x, d_xb2, T * dim))
}

@(private)
cuda_linear :: proc(tr: ^Transformer, l, pos0, T: int) {
	p := &tr.config
	lw := &tr.weights.layers[l]
	la := &lw.linear
	dim := p.dim
	eps := p.rms_eps
	key := p.lin_n_k_heads * p.lin_head_k_dim
	val := p.lin_n_v_heads * p.lin_head_v_dim
	convdim := key * 2 + val
	_ = pos0
	must(launch_mm(la.in_qkv, d_xb, d_qkv, dim, convdim, T))
	must(launch_mm(la.in_z, d_xb, d_z, dim, val, T))
	must(launch_mm(la.in_b, d_xb, d_b, dim, p.lin_n_v_heads, T))
	must(launch_mm(la.in_a, d_xb, d_a, dim, p.lin_n_v_heads, T))
	slot := tr.lin_slot[l]
	km1 := p.lin_conv_kernel - 1
	cstate := cu.ptr_add(d_conv, slot * convdim * km1 * 4)
	rstate := cu.ptr_add(d_rec, slot * p.lin_n_v_heads * p.lin_head_k_dim * p.lin_head_v_dim * 4)
	wconv := wptr(raw_data(la.conv.data))
	scale := 1.0 / math.sqrt_f32(f32(p.lin_head_k_dim))
	for ti in 0 ..< T {
		qin := fp(d_qkv, ti * convdim)
		qout := fp(d_qkv2, ti * convdim)
		must(launch_conv(qout, qin, cstate, wconv, convdim, p.lin_conv_kernel))
		must(launch_l2(fp(d_qlin, ti * key), fp(d_klin, ti * key), qout, fp(qout, key), p.lin_n_k_heads, p.lin_head_k_dim, eps, scale))
		must(launch_delta(
			fp(d_lout, ti * val), rstate,
			fp(d_qlin, ti * key), fp(d_klin, ti * key), fp(qout, key * 2),
			fp(d_b, ti * p.lin_n_v_heads), fp(d_a, ti * p.lin_n_v_heads),
			dev_f32(la.a_decay), dev_f32(la.dt_bias),
			p.lin_head_k_dim, p.lin_head_v_dim, p.lin_n_k_heads, p.lin_n_v_heads,
		))
		lout := fp(d_lout, ti * val)
		must(launch_gated(lout, lout, dev_f32(la.norm_w), fp(d_z, ti * val), p.lin_n_v_heads, p.lin_head_v_dim, eps))
	}
	must(launch_mm(la.out, d_lout, d_xb2, val, dim, T))
	must(launch_residual(d_x, d_xb2, T * dim))
}

@(private)
cuda_mlp :: proc(tr: ^Transformer, l, T: int) {
	p := &tr.config
	lw := &tr.weights.layers[l]
	dim := p.dim
	hid := p.hidden_dim
	must(launch_rmsnorm(d_x, d_xb, dev_f32(lw.ffn_norm), T, dim, p.rms_eps))
	must(launch_mm(lw.w1, d_xb, d_hb, dim, hid, T))
	must(launch_mm(lw.w3, d_xb, d_hb2, dim, hid, T))
	// swiglu(gate, up): result lands in the gate buffer. w1 is gate, w3 is up.
	must(launch_swiglu(d_hb, d_hb2, T * hid))
	must(launch_mm(lw.w2, d_hb, d_xb2, hid, dim, T))
	must(launch_residual(d_x, d_xb2, T * dim))
}

@(private)
cuda_moe_token :: proc(tr: ^Transformer, l, ti: int) {
	p := &tr.config
	lw := &tr.weights.layers[l]
	mw := &lw.moe
	dim := p.dim
	xrow := fp(d_x, ti * dim)
	// Norm this token into d_xb[0:dim]. Earlier tokens in the chunk are already
	// residual-updated; d_xb is scratch from here on.
	must(launch_rmsnorm(xrow, d_xb, dev_f32(lw.ffn_norm), 1, dim, p.rms_eps))
	nexp := p.n_expert
	used := min(p.n_expert_used, nexp)
	must(launch_gemv(.F32, moe_gpu[l].gate, d_xb, d_moe_logits, dim, nexp))
	must(launch_topk(nexp, used, MOE_SUM_CLAMP, p.expert_w_scale))
	must(ck(cu.download(raw_data(h_sel), d_moe_sel, used * size_of(i32)), "moe sel"))
	must(ck(cu.download(raw_data(h_wt), d_moe_wt, used * 4), "moe wt"))
	must(ck(cu.memset(d_xb2, 0, dim * 4), "moe zero"))
	for ki in 0 ..< used {
		e := int(h_sel[ki])
		if e < 0 || e >= nexp {
			fmt.eprintf("cuda: moe expert %d out of range\n", e)
			os.exit(1)
		}
		wgt := h_wt[ki]
		rb_up := ggml.row_byte_size(mw.up_exps.kind, dim)
		rb_g := ggml.row_byte_size(mw.gate_exps.kind, dim)
		rb_d := ggml.row_byte_size(mw.down_exps.kind, p.n_ff_exp)
		up := cu.ptr_add(wptr(raw_data(mw.up_exps.data)), e * p.n_ff_exp * rb_up)
		gate := cu.ptr_add(wptr(raw_data(mw.gate_exps.data)), e * p.n_ff_exp * rb_g)
		down := cu.ptr_add(wptr(raw_data(mw.down_exps.data)), e * dim * rb_d)
		must(launch_gemv(mw.up_exps.kind, up, d_xb, d_hb, dim, p.n_ff_exp))
		must(launch_gemv(mw.gate_exps.kind, gate, d_xb, d_hb2, dim, p.n_ff_exp))
		// result in the gate buffer: silu(gate) * up
		must(launch_swiglu(d_hb2, d_hb, p.n_ff_exp))
		must(launch_gemv(mw.down_exps.kind, down, d_hb2, d_partial, p.n_ff_exp, dim))
		must(launch_axpy(d_xb2, d_partial, wgt, dim))
	}
	if moe_gpu[l].has_shexp {
		nf := p.n_ff_shexp
		must(launch_gemv(.F32, moe_gpu[l].up, d_xb, d_hb, dim, nf))
		must(launch_gemv(.F32, moe_gpu[l].gate_w, d_xb, d_hb2, dim, nf))
		must(launch_swiglu(d_hb2, d_hb, nf))
		must(launch_gemv(.F32, moe_gpu[l].down, d_hb2, d_partial, nf, dim))
		must(launch_gemv(.F32, moe_gpu[l].ginp, d_xb, d_moe_logits, dim, 1))
		gsum: f32
		must(ck(cu.download(&gsum, d_moe_logits, 4), "shexp gate"))
		must(launch_axpy(d_xb2, d_partial, sigmoid_f32(gsum), dim))
	}
	must(launch_residual(xrow, d_xb2, dim))
}

@(private)
cuda_forward_chunk :: proc(tr: ^Transformer, tokens: []int, pos0: int) -> []f32 {
	p := &tr.config
	w := &tr.weights
	T := len(tokens)
	if T <= 0 {
		fmt.eprintln("cuda: empty chunk")
		os.exit(1)
	}
	if pos0 < 0 || pos0 + T > p.seq_len {
		fmt.eprintf("cuda: pos %d..+%d out of range (seq_len=%d)\n", pos0, T, p.seq_len)
		os.exit(1)
	}
	t0: time.Tick
	if g_timing {
		t0 = time.tick_now()
	}
	if g_count_launches {
		g_launches = 0
	}
	for ti in 0 ..< T {
		tok := tokens[ti]
		if tok < 0 || tok >= p.vocab_size {
			fmt.eprintf("cuda: token %d out of range (vocab=%d)\n", tok, p.vocab_size)
			os.exit(1)
		}
		get_embedding_row(&w.token_embedding, tok, p.dim, h_embed[ti * p.dim:][:p.dim])
	}
	must(ck(cu.upload(d_x, raw_data(h_embed), T * p.dim * 4), "embed"))
	for l in 0 ..< p.n_layers {
		lw := &w.layers[l]
		must(launch_rmsnorm(d_x, d_xb, dev_f32(lw.attn_norm), T, p.dim, p.rms_eps))
		switch lw.layer_type {
		case .Full_Attention:
			cuda_full(tr, l, pos0, T)
		case .Linear_Attention:
			cuda_linear(tr, l, pos0, T)
		}
		if p.is_moe {
			for ti in 0 ..< T {
				cuda_moe_token(tr, l, ti)
			}
		} else {
			cuda_mlp(tr, l, T)
		}
	}
	last := (T - 1) * p.dim
	must(launch_rmsnorm(fp(d_x, last), d_xb, dev_f32(w.output_norm), 1, p.dim, p.rms_eps))
	must(launch_gemv(w.output.kind, wptr(raw_data(w.output.data)), d_xb, d_logits, p.dim, p.vocab_size))
	must(ck(cu.download(raw_data(h_logits), d_logits, p.vocab_size * 4), "logits"))
	if g_timing {
		ms := time.duration_milliseconds(time.tick_diff(t0, time.tick_now()))
		fmt.eprintfln("[timing] cuda_forward T={} pos={} wall={:.2f} ms ({:.2f} ms/tok)", T, pos0, ms, ms / f64(T))
	}
	if g_count_launches {
		fmt.eprintfln("cuda: launches {} this forward (T={} pos={})", g_launches, T, pos0)
	}
	return h_logits[:p.vocab_size]
}

cuda_forward :: proc(tr: ^Transformer, token, pos: int) -> []f32 {
	tok := [1]int{token}
	return cuda_forward_chunk(tr, tok[:], pos)
}

cuda_forward_tokens :: proc(tr: ^Transformer, tokens: []int, pos0: int) -> []f32 {
	logits: []f32
	i := 0
	for i < len(tokens) {
		n := min(g_max_t, len(tokens) - i)
		logits = cuda_forward_chunk(tr, tokens[i:][:n], pos0 + i)
		i += n
	}
	return logits
}

} // GPU_CUDA
