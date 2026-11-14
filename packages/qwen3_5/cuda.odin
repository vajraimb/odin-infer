/* CUDA forward for Qwen3.5. Compiled only with -define:ODINFER_BACKEND="cuda".
   The driver and NVRTC are dlopen'd; nothing here links against libcuda.
   Decode (T==1) uses a bandwidth GEMV, four rows per warp.
   ODINFER_CUDA_GEMV_ROWS=1 selects one row per warp. ODINFER_CUDA_GEMV=old
   restores the scalar kernel.
   Dense decode can be one CUDA graph (ODINFER_CUDA_GRAPH=1 enables it).
   Prefill stays eager and uses WMMA when it pays. KV layout matches the
   Metal/CPU snapshot bytes (Q8_0 K, mixed or Q8 V, or legacy f16). */

package qwen3_5

import cu "cuda:cuda"
import ggml "ggml:ggml"

import "core:c"
import "core:fmt"
import "core:math"
import "core:os"
import "core:strconv"
import "core:strings"
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
	_, _ = strings.replace("", "", "", 1)
	_ = time.now()
	_ = c.int(0)
	_ = len(CUDA_KERNEL_SRC)
}

when GPU_CUDA {

CUDA_MAX_BATCH :: 128
// Q5_0 / Q8_0 below this stay on the old GEMV. The new kernel lost at 2048x2048.
GEMV_SMALL_BYTES :: 16 * 1024 * 1024
PROF_MAX :: 8192
// WMMA tile is 16. Below this, the fp16 quant round-trip does not pay for
// itself against the 8-token MMQ. ODINFER_CUDA_TC=1 uses tensor cores for
// every T>1; =0 stays on MMQ.
TC_MIN_T :: 16

@(private)
Store_Fn :: proc(k, v, kc, vc: cu.Device_Ptr, pos: int) -> bool
@(private)
Attn_Fn :: proc(q, kc, vc, xb3: cu.Device_Ptr, pos: int) -> bool
@(private)
Attn_Batch_Fn :: proc(q, kc, vc, xb3: cu.Device_Ptr, pos0, T: int) -> bool

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
do_attn_batch: Attn_Batch_Fn

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
g_gemv_force: int // 0 old, 1 auto, 2 always the new kernel
@(private)
g_gemv_rows: int // 0 or 4: four rows/warp. 1: one row/warp.
@(private)
g_gemv_split: bool // short Q4_K / Q6_K GEMV splits K across grid.y
@(private)
g_q6_stage: int // -1 auto, 0 direct loads, 1 staged loads
@(private)
g_tc2_wide: bool // ODINFER_CUDA_TC2_WIDE=1: T>64 uses the 128-token tile
@(private)
g_tc2_block: bool // ODINFER_CUDA_TC2_BLOCK=1: T>64 uses the 64x64 block-scale tile
@(private)
g_tc2_reuse: bool // ODINFER_CUDA_TC2_REUSE=1: T>64 reuses one scalar weight tile across 128 tokens
@(private)
g_attn_batch: bool // prefill attention is one launch per chunk
@(private)
g_fuse_norm: bool // ODINFER_CUDA_FUSE_NORM=1: side-buffer residual_norm (slower)
@(private)
g_q4_nc: bool // ODINFER_CUDA_Q4_NC=1: ld.global.nc Q4 loads (slower on V100)
@(private)
g_q4_ilp: bool // ODINFER_CUDA_Q4_ILP=1: 8 rows/warp, two super-blocks
@(private)
g_skip_ffn, g_skip_attn: bool
@(private)
g_score_batch: int // how many query tokens d_scores can hold
@(private)
g_pos_host: i32 // stable source for the async decode-pos copy
@(private)
g_tc2: bool // Q4_K / Q6_K prefill uses the block-dequant WMMA kernel
@(private)
g_gemv_warps: int
@(private)
g_launches: int
@(private)
g_count_launches: bool
@(private)
g_profile: bool
@(private)
g_profile_full: bool // ODINFER_CUDA_PROFILE=1 prints every kernel
@(private)
g_phase: bool // ODINFER_CUDA_PHASE=1 prints linear/attn/kv/norm/delta/other
@(private)
g_phase_ms: [6]f64
@(private)
g_phase_n: [6]int
@(private)
g_use_graph: bool
@(private)
g_capturing: bool
@(private)
g_graph_ready: bool
@(private)
g_stream: rawptr
@(private)
g_exec: rawptr
@(private)
g_graph_launches: int
@(private)
g_pos_sym: cu.Device_Ptr
@(private)
g_pos_arg: cu.Device_Ptr // passed into kernels when the module symbol is not exported
@(private)
g_pos_buf: cu.Device_Ptr // fallback device int, not part of the weight mmap
@(private)
g_launch_stream: rawptr
@(private)
d_xq, d_xd, d_xs: cu.Device_Ptr
@(private)
g_q8_cap, g_q8_n: int
@(private)
g_q8_src: cu.Device_Ptr
@(private)
g_q8_valid: bool
@(private)
k_quant, k_rmsnorm_q8, k_res_norm: cu.Function
@(private)
k_gemv2_q4, k_gemv2_q4nc, k_gemv2_q4x, k_gemv2_q6, k_gemv2_q6s, k_gemv2_q5k, k_gemv2_q8, k_gemv2_q50: cu.Function
@(private)
k_gemv2r1_q4, k_gemv2r1_q6: cu.Function
@(private)
k_rmsnorm, k_qnorm, k_rope, k_gate, k_gate_q, k_swiglu, k_swiglu_q, k_residual, k_axpy: cu.Function
@(private)
k_store_f16, k_attn_f16, k_store_q8, k_attn_q8, k_store_q8v, k_attn_q8v: cu.Function
@(private)
k_conv, k_l2, k_delta, k_delta_chunk, k_rmsgate, k_rmsgate_q, k_moe_topk: cu.Function
@(private)
k_tc2_q4, k_tc2_q6, k_tc2s_q4, k_tc2s_q6, k_tc2m_q4, k_tc2m_q6, k_tc2r_q4, k_tc2r_q6: cu.Function

Prof_Bin :: struct {
	name:  string,
	count: int,
	ms:    f64,
}
@(private)
g_prof_ev: [PROF_MAX]rawptr
@(private)
g_prof_name: [PROF_MAX]string
@(private)
g_prof_n: int
@(private)
g_prof_ready: bool
@(private)
g_prof_nbins: int
@(private)
g_prof_launches: int
@(private)
g_prof_gpu: f64
@(private)
g_bins: [96]Prof_Bin

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
	ok := ck(cu.launch(fn, gx, gy, gz, bx, by, bz, 0, args, g_launch_stream), what)
	if ok && g_profile && g_prof_ready && !g_capturing {
		if g_prof_n >= PROF_MAX {
			profile_flush()
		}
		if g_prof_n < PROF_MAX {
			if cu.gapi.event_record(g_prof_ev[g_prof_n], g_launch_stream) == cu.OK {
				g_prof_name[g_prof_n] = what
				g_prof_n += 1
			}
		}
	}
	return ok
}

@(private)
decode_pos :: proc(pos: int) -> int {
	if g_capturing {
		return -1
	}
	return pos
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
	if kind == .Q4_K {
		// The wider kernel and the .nc loads are off until a V100 bench
		// says they beat plain gemv2_q4_k. ROWS=1 still wins over both.
		if g_q4_ilp && k_gemv2_q4x != nil {
			return k_gemv2_q4x
		}
		if g_q4_nc && k_gemv2_q4nc != nil {
			return k_gemv2_q4nc
		}
		return k_gemv2_q4
	}
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
gemv_what :: proc(kind: ggml.GGML_Type, r1: bool) -> string {
	if r1 {
		if kind == .Q4_K {
			return "gemv2r1_q4"
		}
		if kind == .Q6_K {
			return "gemv2r1_q6"
		}
		return "gemv2r1"
	}
	if kind == .Q4_K {
		return "gemv2_q4"
	}
	if kind == .Q6_K {
		return "gemv2_q6"
	}
	if kind == .Q5_K {
		return "gemv2_q5k"
	}
	if kind == .Q8_0 {
		return "gemv2_q8"
	}
	if kind == .Q5_0 {
		return "gemv2_q5_0"
	}
	return "gemv2"
}

// nil means "use the old GEMV". r1 is the one-row-per-warp kernel.
@(private)
gemv2_pick :: proc(kind: ggml.GGML_Type, n, d: int) -> (fn: cu.Function, r1: bool) {
	if !g_gemv_new || !gemv2_aligned(kind, n) {
		return nil, false
	}
	if g_gemv_force != 2 && (kind == .Q8_0 || kind == .Q5_0) {
		elems, bsz := ggml.ggml_block_info(kind)
		if elems > 0 && d * (n / elems) * bsz < GEMV_SMALL_BYTES {
			return nil, false
		}
	}
	base := gemv2_for(kind)
	if base == nil {
		return nil, false
	}
	// One row per warp lost on every V100 shape (9B default 44 tok/s, ROWS=1
	// 33, ROWS=4 62). Four rows is the default. ROWS=1 still selects it.
	want_r1 := g_gemv_rows == 1 && (kind == .Q4_K || kind == .Q6_K)
	if want_r1 {
		if kind == .Q4_K && k_gemv2r1_q4 != nil {
			return k_gemv2r1_q4, true
		}
		if kind == .Q6_K && k_gemv2r1_q6 != nil {
			return k_gemv2r1_q6, true
		}
	}
	return base, false
}

@(private)
launch_gemv2 :: proc(fn: cu.Function, kind: ggml.GGML_Type, w, y: cu.Device_Ptr, n, d: int, r1: bool, accumulate: int) -> bool {
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
	acc := i32(accumulate)
	rows_per := warps * 4
	if r1 {
		rows_per = warps
	} else if fn == k_gemv2_q4x {
		rows_per = warps * 8
	}
	splits := 1
	// Short Q4/Q6 rows (wq, wo, w2) do not fill the SMs. Split K so more
	// blocks are in flight. Tall rows stay one split. Capture includes the
	// split only when the zeroing memset can be recorded on g_stream.
	if g_gemv_split && !r1 && (kind == .Q4_K || kind == .Q6_K) && d > 0 && d <= 8192 {
		nb := n / 256
		if nb >= 4 {
			splits = 4
		} else if nb >= 2 {
			splits = 2
		}
	}
	if splits > 1 && g_capturing && (g_stream == nil || cu.gapi.memset_async == nil) {
		splits = 1
	}
	if splits > 1 && accumulate == 0 {
		if g_capturing {
			if cu.gapi.memset_async(y, 0, c.size_t(d * 4), g_stream) != cu.OK {
				fmt.eprintln("cuda: gemv split memset failed")
				return false
			}
		} else if cu.memset(y, 0, d * 4) != cu.OK {
			fmt.eprintln("cuda: gemv split memset failed")
			return false
		}
	}
	gx := u32((d + rows_per - 1) / rows_per)
	bx := u32(warps * 32)
	what := gemv_what(kind, r1)
	if fn == k_gemv2_q4x {
		what = "gemv2_q4x"
	} else if fn == k_gemv2_q4nc {
		what = "gemv2_q4nc"
	}
	fn_use := fn
	// 9B q6 is 33 launches: 32 of them are w2 (4096 x 12288). Staging those
	// was slower (3.95 -> 4.19 ms). 27B w2 is 5120 x 17408 and the logits
	// are 248320 rows; staging those was faster (7.99 -> 7.04 ms).
	if kind == .Q6_K && !r1 && k_gemv2_q6s != nil && q6_stage_on(n, d) {
		fn_use = k_gemv2_q6s
		what = "gemv2_q6s"
	}
	gy := u32(splits)
	if kind == .Q4_K || kind == .Q5_K {
		args := [?]rawptr{&wb, &qs, &sc, &sm, &yy, &nn, &dd, &acc}
		return klaunch(fn_use, gx, gy, 1, bx, 1, 1, args[:], what)
	}
	args := [?]rawptr{&wb, &qs, &sc, &yy, &nn, &dd, &acc}
	return klaunch(fn_use, gx, gy, 1, bx, 1, 1, args[:], what)
}

// Long K or a tall output matrix. The 9B down projection is neither.
@(private)
q6_stage_on :: proc(n, d: int) -> bool {
	if g_q6_stage == 0 {
		return false
	}
	if g_q6_stage == 1 {
		return true
	}
	return n >= 16384 || d > 8192
}

launch_gemv :: proc(kind: ggml.GGML_Type, w, x, y: cu.Device_Ptr, n, d: int) -> bool {
	if d <= 0 || n <= 0 {
		return true
	}
	fn, r1 := gemv2_pick(kind, n, d)
	if fn != nil && q8_ensure(x, n) {
		return launch_gemv2(fn, kind, w, y, n, d, r1, 0)
	}
	return launch_gemv_old(kind, w, x, y, n, d)
}

// Writes y += W*x with the bandwidth GEMV and skips the residual kernel.
// False leaves y untouched so the caller can run the two-kernel path.
@(private)
try_accum_gemv :: proc(t: Tensor, x, y: cu.Device_Ptr, n, d: int) -> bool {
	fn, r1 := gemv2_pick(t.kind, n, d)
	if fn == nil || !q8_ensure(x, n) {
		return false
	}
	w := wptr(raw_data(t.data))
	return launch_gemv2(fn, t.kind, w, y, n, d, r1, 1)
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
	wb, xx, yy := w, x, y
	nn, dd, tt := i32(n), i32(d), i32(tcount)
	args := [?]rawptr{&wb, &xx, &yy, &nn, &dd, &tt}
	if g_tc2 && n > 0 && n % 256 == 0 && (kind == .Q4_K || kind == .Q6_K) {
		// T<=64 stays on the 32-token tile: at T=19 the 128-token tile
		// still MMAs a 64-token N and was 323 ms vs 176 ms. T>64 measured
		// slower than the scalar tile (9B 581: 124.5 vs 138.9 tok/s, 2000:
		// 71.9 vs 76.7; 27B 581: 50.9 vs 53.1), so the scalar tile is the
		// default there. ODINFER_CUDA_TC2_WIDE=1 opts into tc2w.
		// ODINFER_CUDA_TC2_BLOCK=1 is the rejected scale-broadcast tile.
		// ODINFER_CUDA_TC2_REUSE=1 dequants the scalar tile once per 128 tokens.
		use_small := tcount <= 64
		use_reuse := g_tc2_reuse && tcount > 64
		use_wide := g_tc2_wide && tcount > 64 && !g_tc2_block && !g_tc2_reuse
		use_block := g_tc2_block && tcount > 64 && !g_tc2_reuse
		if use_reuse {
			fn := k_tc2r_q4
			if kind == .Q6_K {
				fn = k_tc2r_q6
			}
			if fn != nil {
				gx := u32((d + 63) / 64)
				gy := u32((tcount + 127) / 128)
				return klaunch(fn, gx, gy, 1, 512, 1, 1, args[:], "tc2r")
			}
		}
		if use_block {
			fn := k_tc2m_q4
			if kind == .Q6_K {
				fn = k_tc2m_q6
			}
			if fn != nil {
				gx := u32((d + 63) / 64)
				gy := u32((tcount + 63) / 64)
				return klaunch(fn, gx, gy, 1, 512, 1, 1, args[:], "tc2m")
			}
		}
		if use_small || use_wide {
			fn := k_tc2s_q4
			if use_wide {
				fn = k_tc2_q4
			}
			if kind == .Q6_K {
				fn = k_tc2s_q6
				if use_wide {
					fn = k_tc2_q6
				}
			}
			if fn != nil {
				gx := u32((d + 31) / 32)
				gy := u32((tcount + 31) / 32)
				name := "tc2"
				if use_wide {
					gy = u32((tcount + 127) / 128)
					name = "tc2w"
				}
				return klaunch(fn, gx, gy, 1, 256, 1, 1, args[:], name)
			}
		}
	}
	fn := tc_fn[int(kind)]
	if fn == nil {
		return false
	}
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
pos_slot :: proc() -> cu.Device_Ptr {
	if g_capturing {
		return g_pos_arg
	}
	return 0
}

@(private)
launch_rope :: proc(vec: cu.Device_Ptr, head_dim, pos: int, theta: f32, rotary, nheads: int) -> bool {
	half := rotary / 2
	total := nheads * half
	if total <= 0 {
		return true
	}
	v := vec
	hd, ps, rd, nh := i32(head_dim), i32(decode_pos(pos)), i32(rotary), i32(nheads)
	th := theta
	slot := pos_slot()
	args := [?]rawptr{&v, &hd, &ps, &th, &rd, &nh, &slot}
	return klaunch(k_rope, grid_n(total, 128), 1, 1, 128, 1, 1, args[:], "rope")
}

@(private)
launch_gate :: proc(xb3, qproj: cu.Device_Ptr, head_dim, nheads: int, quant: bool) -> bool {
	n := nheads * head_dim
	a, b := xb3, qproj
	hd, nh := i32(head_dim), i32(nheads)
	if quant && g_gemv_new && n % 32 == 0 && n <= g_q8_cap && k_gate_q != nil && d_xq != 0 {
		qs, sc, sm := d_xq, d_xd, d_xs
		args := [?]rawptr{&a, &b, &qs, &sc, &sm, &hd, &nh}
		if !klaunch(k_gate_q, grid_n(n, 256), 1, 1, 256, 1, 1, args[:], "attn_gate") {
			return false
		}
		g_q8_src = xb3
		g_q8_n = n
		g_q8_valid = true
		return true
	}
	q8_drop()
	args := [?]rawptr{&a, &b, &hd, &nh}
	return klaunch(k_gate, grid_n(n, 256), 1, 1, 256, 1, 1, args[:], "attn_gate")
}

@(private)
launch_swiglu :: proc(hb, hb2: cu.Device_Ptr, n: int, quant: bool) -> bool {
	a, b := hb, hb2
	nn := i32(n)
	if quant && g_gemv_new && n % 32 == 0 && n <= g_q8_cap && k_swiglu_q != nil && d_xq != 0 {
		qs, sc, sm := d_xq, d_xd, d_xs
		args := [?]rawptr{&a, &b, &qs, &sc, &sm, &nn}
		if !klaunch(k_swiglu_q, grid_n(n, 256), 1, 1, 256, 1, 1, args[:], "swiglu") {
			return false
		}
		g_q8_src = hb
		g_q8_n = n
		g_q8_valid = true
		return true
	}
	q8_drop()
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

// x += y, RMSNorm into out, quantize out. Same loops as residual then
// rmsnorm_quant_q8. The following stage skips its rmsnorm launch.
@(private)
launch_residual_norm :: proc(x, y, outp, weight: cu.Device_Ptr, n: int, eps: f32) -> bool {
	if k_res_norm == nil || weight == 0 || n <= 0 || n % 32 != 0 || n > g_q8_cap || d_xq == 0 {
		return false
	}
	a, b, c, d := x, y, outp, weight
	qs, sc, sm := d_xq, d_xd, d_xs
	nn := i32(n)
	ep := eps
	args := [?]rawptr{&a, &b, &c, &d, &qs, &sc, &sm, &nn, &ep}
	if !klaunch(k_res_norm, 1, 1, 1, 128, 1, 1, args[:], "residual_norm") {
		return false
	}
	g_q8_src = outp
	g_q8_n = n
	g_q8_valid = true
	return true
}

// Writes W*act into d_xb2, adds it onto d_x, and RMSNorms that sum into
// d_xb. applied means the residual already includes the projection, so the
// caller must not add it again. normed means d_xb is the next stage's
// input and the next launch_rmsnorm should be skipped.
@(private)
try_fuse_proj :: proc(t: Tensor, act: cu.Device_Ptr, n, d: int, norm_w: cu.Device_Ptr, eps: f32) -> (applied, normed: bool) {
	if !g_fuse_norm || !g_gemv_new || k_res_norm == nil || norm_w == 0 {
		return false, false
	}
	if d <= 0 || d % 32 != 0 || d > g_q8_cap {
		return false, false
	}
	w := wptr(raw_data(t.data))
	if !launch_gemv(t.kind, w, act, d_xb2, n, d) {
		return false, false
	}
	if launch_residual_norm(d_x, d_xb2, d_xb, norm_w, d, eps) {
		return true, true
	}
	if !launch_residual(d_x, d_xb2, d) {
		fmt.eprintln("cuda: residual add after a fused projection failed")
		os.exit(1)
	}
	return true, false
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
	hd, sl, ps, nk := i32(g_head), i32(g_seq), i32(decode_pos(pos)), i32(g_kv_heads)
	slot := pos_slot()
	args := [?]rawptr{&a, &b, &c, &d, &hd, &sl, &ps, &nk, &slot}
	n := g_kv_heads * g_head
	return klaunch(k_store_f16, grid_n(n, 128), 1, 1, 128, 1, 1, args[:], "store_f16")
}

@(private)
attn_f16_call :: proc(q, kc, vc, xb3: cu.Device_Ptr, pos: int) -> bool {
	a, b, c, d := q, kc, vc, xb3
	sc := d_scores
	hd, sl, mul, ps := i32(g_head), i32(g_seq), i32(g_heads / g_kv_heads), i32(decode_pos(pos))
	slot := pos_slot()
	stride := i32(0)
	args := [?]rawptr{&a, &b, &c, &d, &sc, &hd, &sl, &mul, &ps, &slot, &stride}
	return klaunch(k_attn_f16, u32(g_heads), 1, 1, 128, 1, 1, args[:], "attn_f16")
}

@(private)
attn_f16_batch :: proc(q, kc, vc, xb3: cu.Device_Ptr, pos0, T: int) -> bool {
	a, b, c, d := q, kc, vc, xb3
	sc := d_scores
	hd, sl, mul, ps := i32(g_head), i32(g_seq), i32(g_heads / g_kv_heads), i32(pos0)
	slot: cu.Device_Ptr
	stride := i32(g_heads * g_head)
	args := [?]rawptr{&a, &b, &c, &d, &sc, &hd, &sl, &mul, &ps, &slot, &stride}
	return klaunch(k_attn_f16, u32(g_heads), u32(T), 1, 128, 1, 1, args[:], "attn_f16")
}

@(private)
store_q8_call :: proc(k, v, kc, vc: cu.Device_Ptr, pos: int) -> bool {
	a, b, c, d := k, v, kc, vc
	nb, pb, ps := i32(g_kv_dim / 32), i32(g_k_pos), i32(decode_pos(pos))
	slot := pos_slot()
	args := [?]rawptr{&a, &b, &c, &d, &nb, &pb, &ps, &slot}
	return klaunch(k_store_q8, grid_n(g_kv_dim / 32, 64), 1, 1, 64, 1, 1, args[:], "store_q8")
}

@(private)
attn_q8_call :: proc(q, kc, vc, xb3: cu.Device_Ptr, pos: int) -> bool {
	a, b, c, d := q, kc, vc, xb3
	sc := d_scores
	hd, pb, mul, ps, sl := i32(g_head), i32(g_k_pos), i32(g_heads / g_kv_heads), i32(decode_pos(pos)), i32(g_seq)
	slot := pos_slot()
	stride := i32(0)
	args := [?]rawptr{&a, &b, &c, &d, &sc, &hd, &pb, &mul, &ps, &sl, &slot, &stride}
	return klaunch(k_attn_q8, u32(g_heads), 1, 1, 128, 1, 1, args[:], "attn_q8")
}

@(private)
attn_q8_batch :: proc(q, kc, vc, xb3: cu.Device_Ptr, pos0, T: int) -> bool {
	a, b, c, d := q, kc, vc, xb3
	sc := d_scores
	hd, pb, mul, ps, sl := i32(g_head), i32(g_k_pos), i32(g_heads / g_kv_heads), i32(pos0), i32(g_seq)
	slot: cu.Device_Ptr
	stride := i32(g_heads * g_head)
	args := [?]rawptr{&a, &b, &c, &d, &sc, &hd, &pb, &mul, &ps, &sl, &slot, &stride}
	return klaunch(k_attn_q8, u32(g_heads), u32(T), 1, 128, 1, 1, args[:], "attn_q8")
}

@(private)
store_q8v_call :: proc(k, v, kc, vc: cu.Device_Ptr, pos: int) -> bool {
	a, b, c, d := k, v, kc, vc
	nb, kb, vb, ps := i32(g_kv_dim / 32), i32(g_k_pos), i32(g_v_pos), i32(decode_pos(pos))
	slot := pos_slot()
	args := [?]rawptr{&a, &b, &c, &d, &nb, &kb, &vb, &ps, &slot}
	return klaunch(k_store_q8v, grid_n(g_kv_dim / 32, 64), 1, 1, 64, 1, 1, args[:], "store_q8v")
}

@(private)
attn_q8v_call :: proc(q, kc, vc, xb3: cu.Device_Ptr, pos: int) -> bool {
	a, b, c, d := q, kc, vc, xb3
	sc := d_scores
	hd := i32(g_head)
	kb, vb := i32(g_k_pos), i32(g_v_pos)
	mul, ps, sl := i32(g_heads / g_kv_heads), i32(decode_pos(pos)), i32(g_seq)
	slot := pos_slot()
	stride := i32(0)
	args := [?]rawptr{&a, &b, &c, &d, &sc, &hd, &kb, &vb, &mul, &ps, &sl, &slot, &stride}
	return klaunch(k_attn_q8v, u32(g_heads), 1, 1, 128, 1, 1, args[:], "attn_q8v")
}

@(private)
attn_q8v_batch :: proc(q, kc, vc, xb3: cu.Device_Ptr, pos0, T: int) -> bool {
	a, b, c, d := q, kc, vc, xb3
	sc := d_scores
	hd := i32(g_head)
	kb, vb := i32(g_k_pos), i32(g_v_pos)
	mul, ps, sl := i32(g_heads / g_kv_heads), i32(pos0), i32(g_seq)
	slot: cu.Device_Ptr
	stride := i32(g_heads * g_head)
	args := [?]rawptr{&a, &b, &c, &d, &sc, &hd, &kb, &vb, &mul, &ps, &sl, &slot, &stride}
	return klaunch(k_attn_q8v, u32(g_heads), u32(T), 1, 128, 1, 1, args[:], "attn_q8v")
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

// q_stride is the token stride of q and k. v_stride is the token stride of v.
@(private)
launch_delta_chunk :: proc(outp, state, q, k, v, b, a, decay, dt: cu.Device_Ptr, hkd, hvd, nkh, nvh, T, q_stride, v_stride: int) -> bool {
	o, st, qq, kk, vv, bb, aa, dec, dtb := outp, state, q, k, v, b, a, decay, dt
	ik, iv, nk, nv := i32(hkd), i32(hvd), i32(nkh), i32(nvh)
	tt, qs, vs := i32(T), i32(q_stride), i32(v_stride)
	args := [?]rawptr{&o, &st, &qq, &kk, &vv, &bb, &aa, &dec, &dtb, &ik, &iv, &nk, &nv, &tt, &qs, &vs}
	return klaunch(k_delta_chunk, u32(nvh), 1, 1, 128, 1, 1, args[:], "delta_chunk")
}

@(private)
launch_gated :: proc(outp, inp, weight, gate: cu.Device_Ptr, n_vh, hvd: int, eps: f32, quant: bool) -> bool {
	a, b, c, d := outp, inp, weight, gate
	sz := i32(hvd)
	ep := eps
	n := n_vh * hvd
	if quant && g_gemv_new && hvd % 32 == 0 && n <= g_q8_cap && k_rmsgate_q != nil && d_xq != 0 {
		qs, sc, sm := d_xq, d_xd, d_xs
		args := [?]rawptr{&a, &b, &c, &d, &qs, &sc, &sm, &sz, &ep}
		if !klaunch(k_rmsgate_q, u32(n_vh), 1, 1, 128, 1, 1, args[:], "rmsnorm_gated") {
			return false
		}
		g_q8_src = outp
		g_q8_n = n
		g_q8_valid = true
		return true
	}
	q8_drop()
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
	k_gate_q = get_fn("attn_gate_quant")
	k_swiglu = get_fn("swiglu")
	k_swiglu_q = get_fn("swiglu_quant")
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
	k_delta_chunk = get_fn("delta_recurrent_chunk")
	k_tc2_q4 = get_fn("tc2_q4_k")
	k_tc2_q6 = get_fn("tc2_q6_k")
	k_tc2s_q4 = get_fn("tc2s_q4_k")
	k_tc2s_q6 = get_fn("tc2s_q6_k")
	k_tc2m_q4 = get_fn("tc2m_q4_k")
	k_tc2m_q6 = get_fn("tc2m_q6_k")
	k_tc2r_q4 = get_fn("tc2r_q4_k")
	k_tc2r_q6 = get_fn("tc2r_q6_k")
	k_rmsgate = get_fn("rmsnorm_gated")
	k_rmsgate_q = get_fn("rmsnorm_gated_quant")
	k_moe_topk = get_fn("moe_softmax_topk")
	k_quant = get_fn("quant_q8")
	k_rmsnorm_q8 = get_fn("rmsnorm_quant_q8")
	k_res_norm = get_fn("residual_rmsnorm_q8")
	k_gemv2_q4 = get_fn("gemv2_q4_k")
	k_gemv2_q4nc = get_fn("gemv2_q4nc_k")
	k_gemv2_q4x = get_fn("gemv2_q4x_k")
	k_gemv2_q6 = get_fn("gemv2_q6_k")
	k_gemv2_q6s = get_fn("gemv2_q6s_k")
	k_gemv2_q5k = get_fn("gemv2_q5_k")
	k_gemv2_q8 = get_fn("gemv2_q8_0")
	k_gemv2_q50 = get_fn("gemv2_q5_0")
	k_gemv2r1_q4 = get_fn("gemv2r1_q4_k")
	k_gemv2r1_q6 = get_fn("gemv2r1_q6_k")
	handles := [?]cu.Function{
		k_rmsnorm, k_qnorm, k_rope, k_gate, k_swiglu, k_residual, k_axpy,
		k_store_f16, k_attn_f16, k_store_q8, k_attn_q8, k_store_q8v, k_attn_q8v,
		k_conv, k_l2, k_delta, k_delta_chunk, k_rmsgate, k_rmsgate_q, k_moe_topk,
		k_gate_q, k_swiglu_q,
		k_tc2_q4, k_tc2_q6, k_tc2s_q4, k_tc2s_q6, k_tc2m_q4, k_tc2m_q6, k_tc2r_q4, k_tc2r_q6,
		k_quant, k_rmsnorm_q8, k_res_norm, k_gemv2_q4, k_gemv2_q4nc, k_gemv2_q4x, k_gemv2_q6, k_gemv2_q6s, k_gemv2_q5k, k_gemv2_q8, k_gemv2_q50,
		k_gemv2r1_q4, k_gemv2r1_q6,
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
	g_score_batch = 1
	if p.n_heads > 0 && p.seq_len > 0 {
		one := p.n_heads * p.seq_len * 4
		if one <= 0 {
			fmt.eprintln("cuda: attention score buffer overflowed")
			return false
		}
		batch := 1
		// One score row per (token, head). Cap the buffer at 256 MiB and
		// batch the prefill attention in chunks of that many tokens.
		if g_attn_batch && one > 0 && one <= 256 * 1024 * 1024 {
			fit := (256 * 1024 * 1024) / one
			if fit > T {
				fit = T
			}
			if fit < 1 {
				fit = 1
			}
			batch = fit
		}
		d_scores, ok = alloc_bytes(batch * one); if !ok do return false
		g_score_batch = batch
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
	// Stream, graph exec, and events belong to the primary context. Releasing
	// the context first and destroying them after is the exit 139: the driver
	// faults in stream teardown even when the graph was never captured.
	if g_ctx_held {
		cu.drv.CtxSynchronize()
	}
	if cu.gapi.ready && g_exec != nil {
		cu.gapi.graph_exec_destroy(g_exec)
	}
	g_exec = nil
	g_graph_launches = 0
	g_graph_ready = false
	g_capturing = false
	g_launch_stream = nil
	g_pos_sym = 0
	g_pos_arg = 0
	if g_pos_buf != 0 {
		cu.free(g_pos_buf)
		g_pos_buf = 0
	}
	if cu.gapi.ready && g_stream != nil {
		cu.gapi.stream_destroy(g_stream)
	}
	g_stream = nil
	if cu.gapi.ready {
		for i in 0 ..< PROF_MAX {
			if g_prof_ev[i] != nil {
				cu.gapi.event_destroy(g_prof_ev[i])
				g_prof_ev[i] = nil
			}
		}
	}
	g_prof_ready = false
	g_prof_n = 0
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
	do_attn_batch = nil
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
	g_gemv_force = 1
	g_gemv_new = true
	if gemv_env == "old" || gemv_env == "0" {
		g_gemv_new = false
		g_gemv_force = 0
	} else if gemv_env == "new" {
		g_gemv_force = 2
	}
	g_gemv_warps = 8
	if s := os.get_env("ODINFER_CUDA_GEMV_WARPS", context.temp_allocator); s != "" {
		if v, ok := strconv.parse_int(s); ok && v >= 1 && v <= 16 {
			g_gemv_warps = int(v)
		}
	}
	g_gemv_rows = 0
	if s := os.get_env("ODINFER_CUDA_GEMV_ROWS", context.temp_allocator); s != "" {
		if v, ok := strconv.parse_int(s); ok && (v == 1 || v == 4) {
			g_gemv_rows = int(v)
		}
	}
	g_gemv_split = os.get_env("ODINFER_CUDA_GEMV_SPLIT", context.temp_allocator) != "0"
	g_q6_stage = -1
	if s := os.get_env("ODINFER_CUDA_Q6_STAGE", context.temp_allocator); s == "0" {
		g_q6_stage = 0
	} else if s == "1" {
		g_q6_stage = 1
	}
	g_tc2 = os.get_env("ODINFER_CUDA_TC2", context.temp_allocator) != "0"
	// Wide tile lost to the scalar tile at 581 and 2000 tokens. Opt in.
	g_tc2_wide = os.get_env("ODINFER_CUDA_TC2_WIDE", context.temp_allocator) == "1"
	// V100S: about −26% vs the scalar tile, and the text diverged. Leave off.
	g_tc2_block = os.get_env("ODINFER_CUDA_TC2_BLOCK", context.temp_allocator) == "1"
	// V100S: bit-exact with the scalar tile, but slower (9B 581 tokens 198.6
	// vs 208.3, 2000 tokens 192.4 vs 202.7; 27B 581 tokens 66.9 vs 70.8). Leave off.
	g_tc2_reuse = os.get_env("ODINFER_CUDA_TC2_REUSE", context.temp_allocator) == "1"
	g_attn_batch = os.get_env("ODINFER_CUDA_ATTN_BATCH", context.temp_allocator) != "0"
	// Side-buffer residual_norm was slower than accum GEMV + rmsnorm_q8
	// (9B 1.99 vs 1.41 ms, 27B 4.96 vs 3.53). Opt in.
	g_fuse_norm = os.get_env("ODINFER_CUDA_FUSE_NORM", context.temp_allocator) == "1"
	g_q4_nc = os.get_env("ODINFER_CUDA_Q4_NC", context.temp_allocator) == "1"
	g_q4_ilp = os.get_env("ODINFER_CUDA_Q4_ILP", context.temp_allocator) == "1"
	g_count_launches = os.get_env("ODINFER_CUDA_LAUNCHES", context.temp_allocator) == "1"
	g_profile_full = os.get_env("ODINFER_CUDA_PROFILE", context.temp_allocator) == "1"
	// Same event records as PROFILE. One summary line, no per-kernel dump
	// unless PROFILE is also set. Event sync means this run's tok/s is not
	// the production number.
	g_phase = os.get_env("ODINFER_CUDA_PHASE", context.temp_allocator) == "1"
	g_profile = g_profile_full || g_phase
	// Replay was slower than eager (9B 57.6 vs 63.0 tok/s): MemcpyHtoD_v2
	// synced the device every token, and capture dropped split-K. =1 opts in.
	g_use_graph = os.get_env("ODINFER_CUDA_GRAPH", context.temp_allocator) == "1"
	if g_profile {
		g_use_graph = false
	}
	if g_phase {
		fmt.eprintln("cuda: phase on (ODINFER_CUDA_PHASE=1). Event sync is on, so tok/s from this process is not the production number. Buckets: linear attn kv norm delta other")
	}
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
	// NVRTC emits `.global` without `.visible`. The kernels can still load it,
	// but cuModuleGetGlobal only finds an exported symbol, so the graph never
	// attached ("g_decode_pos missing").
	needle := ".global .align 4 .u32 g_decode_pos;"
	ptx_load := ptx
	if !strings.contains(ptx, needle) {
		fmt.eprintln("cuda: PTX has no g_decode_pos declaration; graph will be off")
	} else {
		ptx_load, _ = strings.replace(ptx, needle, ".visible .global .align 4 .u32 g_decode_pos;", 1)
	}
	// Direct cuModuleLoadData is the default. cuLink did not export
	// g_decode_pos on driver 580 (still CUresult 500) and the linked cubin
	// is a different JIT of gemv2_q4, whose source did not change while its
	// time went from 6.0 ms to 6.94 ms. ODINFER_CUDA_LINK=1 opts into cuLink.
	mod: cu.Module
	if os.get_env("ODINFER_CUDA_LINK", context.temp_allocator) == "1" {
		linked, lerr := cu.module_load_linked(ptx_load)
		if lerr == cu.OK && linked != nil {
			mod = linked
			fmt.eprintln("cuda: module linked (g_decode_pos stays exported)")
		} else if lerr != cu.OK && lerr != cu.ERR_SYMBOL {
			fmt.eprintf("cuda: cuLink failed: %s (%d); loading PTX directly\n", cu.error_string(lerr), int(lerr))
		}
	}
	if mod == nil {
		loaded, jit, merr := cu.module_load_ptx(ptx_load)
		if len(jit) > 0 {
			fmt.eprintfln("cuda: jit: %s", jit)
		}
		if merr != cu.OK {
			fmt.eprintf("cuda: cuModuleLoadData failed: %s (%d)\n", cu.error_string(merr), int(merr))
			cuda_destroy()
			return false
		}
		mod = loaded
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
	// 64 bytes past the last block so a misaligned Q6 vector load stays in range.
	dp, aerr := cu.alloc(len(mmap) + 64)
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
			do_attn_batch = attn_q8v_batch
		} else {
			g_v_pos = g_k_pos
			g_v_layer = g_k_layer
			g_v_bytes = g_k_bytes
			do_store = store_q8_call
			do_attn = attn_q8_call
			do_attn_batch = attn_q8_batch
		}
	} else {
		g_k_layer = p.seq_len * g_kv_dim * 2
		g_v_layer = g_k_layer
		g_k_bytes = p.n_full * g_k_layer
		g_v_bytes = g_k_bytes
		do_store = store_f16_call
		do_attn = attn_f16_call
		do_attn_batch = attn_f16_batch
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
	if g_tc2 {
		wide_note := "T<=64 is the 32-token tile; larger T uses the scalar tile (ODINFER_CUDA_TC2_REUSE=1 reuses that tile across 128 tokens)"
		if g_tc2_reuse {
			wide_note = "T<=64 is the 32-token tile, T>64 reuses one scalar weight tile across 128 tokens (ODINFER_CUDA_TC2_REUSE=1)"
		} else if g_tc2_block {
			wide_note = "T<=64 is the 32-token tile, T>64 is the 64x64 block-scale tile (ODINFER_CUDA_TC2_BLOCK=1)"
		} else if g_tc2_wide {
			wide_note = "T<=64 is the 32-token tile, T>64 is the 128-token tile"
		}
		fmt.eprintfln("cuda: tc2 on for Q4_K/Q6_K (%s; ODINFER_CUDA_TC2=0 uses the scalar dequant tile)", wide_note)
	}
	if g_fuse_norm {
		fmt.eprintln("cuda: decode fuses the residual add into the next RMSNorm (ODINFER_CUDA_FUSE_NORM=1). Slower than the accum GEMV on the V100S")
	}
	if g_q4_ilp {
		fmt.eprintln("cuda: Q4 GEMV is 8 rows/warp, two super-blocks (ODINFER_CUDA_Q4_ILP=1). Default is 4 rows and plain ld.global")
	} else if g_q4_nc {
		fmt.eprintln("cuda: Q4 GEMV uses ld.global.nc (ODINFER_CUDA_Q4_NC=1). Slower than plain ld.global on the V100S")
	}
	if g_gemv_split {
		fmt.eprintln("cuda: gemv split-K on for Q4_K/Q6_K with rows<=8192 (ODINFER_CUDA_GEMV_SPLIT=0 disables it)")
	}
	if g_gemv_new {
		rows_note := "4 rows/warp"
		if g_gemv_rows == 1 {
			rows_note = "1 row/warp"
		}
		force_note := ""
		if g_gemv_force == 2 {
			force_note = "  small q5_0/q8_0 stay on this kernel"
		}
		fmt.eprintfln("cuda: gemv dp4a  warps={}  {}  q8 act{}  (ODINFER_CUDA_GEMV=old for the scalar kernel)", g_gemv_warps, rows_note, force_note)
	} else {
		fmt.eprintln("cuda: gemv old (ODINFER_CUDA_GEMV=old)")
	}
	if g_use_graph || g_profile {
		if !cu.load_graph_api() {
			fmt.eprintln("cuda: driver has no graph/event symbols; graph and profile off")
			g_use_graph = false
			g_profile = false
		}
	}
	if g_use_graph {
		st: rawptr
		if cu.gapi.stream_create(&st, 0) != cu.OK {
			fmt.eprintln("cuda: stream create failed; graph off")
			g_use_graph = false
		} else {
			g_stream = st
			nbytes: c.size_t
			dptr: cu.Device_Ptr
			sym := [?]u8{'g', 'd', 'e', 'c', 'o', 'd', 'e', '_', 'p', 'o', 's', 0}
			gerr := cu.gapi.module_global(&dptr, &nbytes, g_mod, cstring(&sym[0]))
			if gerr == cu.OK && dptr != 0 {
				g_pos_sym = dptr
				g_pos_arg = 0
				fmt.eprintfln("cuda: g_decode_pos %d bytes", int(nbytes))
			} else {
				// The name is g_decode_pos. Driver 580 still returns 500 after
				// a direct PTX load. A device int captured as a kernel pointer
				// is the same graph slot: the host writes it before each replay.
				buf, aerr := cu.alloc(4)
				if aerr != cu.OK || buf == 0 {
					fmt.eprintf("cuda: g_decode_pos missing: %s (%d); graph off\n", cu.error_string(gerr), int(gerr))
					cu.gapi.stream_destroy(st)
					g_stream = nil
					g_use_graph = false
				} else {
					g_pos_buf = buf
					g_pos_sym = buf
					g_pos_arg = buf
					fmt.eprintf("cuda: g_decode_pos via device buffer (%s %d); graph on\n", cu.error_string(gerr), int(gerr))
				}
			}
		}
	}
	if g_use_graph {
		fmt.eprintln("cuda: graph on for dense T=1 (ODINFER_CUDA_GRAPH=1; MoE stays eager)")
	} else if g_profile {
		fmt.eprintln("cuda: profile on (graph off while profiling)")
	} else {
		fmt.eprintln("cuda: graph off (ODINFER_CUDA_GRAPH=1 enables it; replay was slower than eager)")
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
	// Store every token before attention. Query t only reads KV through
	// position t, so writing the rest of the chunk first does not change
	// the scores. The per-token launch was one block per head and left
	// most SMs idle; the chunk launch is still O(N^2) flops.
	for ti in 0 ..< T {
		qrow := fp(d_q, ti * attdim)
		krow := fp(d_k, ti * kvdim)
		vrow := fp(d_v, ti * kvdim)
		pos := pos0 + ti
		must(launch_rope(qrow, p.head_dim, pos, p.rope_theta, p.rotary_dim, p.n_heads))
		must(launch_rope(krow, p.head_dim, pos, p.rope_theta, p.rotary_dim, p.n_kv_heads))
		must(do_store(krow, vrow, kc, vc, pos))
	}
	batch := T > 1 && g_attn_batch && g_score_batch > 1 && do_attn_batch != nil
	if batch {
		for base := 0; base < T; base += g_score_batch {
			ntok := min(g_score_batch, T - base)
			must(do_attn_batch(fp(d_q, base * attdim), kc, vc, fp(d_xb3, base * attdim), pos0 + base, ntok))
		}
	}
	for ti in 0 ..< T {
		qproj := fp(d_qproj, ti * qdim)
		xb3 := fp(d_xb3, ti * attdim)
		if !batch {
			must(do_attn(fp(d_q, ti * attdim), kc, vc, xb3, pos0 + ti))
		}
		must(launch_gate(xb3, qproj, p.head_dim, p.n_heads, T == 1))
	}
	applied := false
	normed := false
	if T == 1 && !p.is_moe && g_fuse_norm {
		applied, normed = try_fuse_proj(fa.wo, d_xb3, attdim, dim, dev_f32(lw.ffn_norm), eps)
	}
	if !applied {
		if !(T == 1 && try_accum_gemv(fa.wo, d_xb3, d_x, attdim, dim)) {
			must(launch_mm(fa.wo, d_xb3, d_xb2, attdim, dim, T))
			must(launch_residual(d_x, d_xb2, T * dim))
		}
	}
	g_skip_ffn = normed
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
	}
	if T > 1 && k_delta_chunk != nil {
		must(launch_delta_chunk(
			d_lout, rstate, d_qlin, d_klin, fp(d_qkv2, key * 2),
			d_b, d_a, dev_f32(la.a_decay), dev_f32(la.dt_bias),
			p.lin_head_k_dim, p.lin_head_v_dim, p.lin_n_k_heads, p.lin_n_v_heads, T, key, convdim,
		))
	} else {
		for ti in 0 ..< T {
			must(launch_delta(
				fp(d_lout, ti * val), rstate,
				fp(d_qlin, ti * key), fp(d_klin, ti * key), fp(fp(d_qkv2, ti * convdim), key * 2),
				fp(d_b, ti * p.lin_n_v_heads), fp(d_a, ti * p.lin_n_v_heads),
				dev_f32(la.a_decay), dev_f32(la.dt_bias),
				p.lin_head_k_dim, p.lin_head_v_dim, p.lin_n_k_heads, p.lin_n_v_heads,
			))
		}
	}
	for ti in 0 ..< T {
		lout := fp(d_lout, ti * val)
		must(launch_gated(lout, lout, dev_f32(la.norm_w), fp(d_z, ti * val), p.lin_n_v_heads, p.lin_head_v_dim, eps, T == 1))
	}
	applied := false
	normed := false
	if T == 1 && !p.is_moe && g_fuse_norm {
		applied, normed = try_fuse_proj(la.out, d_lout, val, dim, dev_f32(lw.ffn_norm), eps)
	}
	if !applied {
		if !(T == 1 && try_accum_gemv(la.out, d_lout, d_x, val, dim)) {
			must(launch_mm(la.out, d_lout, d_xb2, val, dim, T))
			must(launch_residual(d_x, d_xb2, T * dim))
		}
	}
	g_skip_ffn = normed
}

@(private)
cuda_mlp :: proc(tr: ^Transformer, l, T: int, next_norm: cu.Device_Ptr) {
	p := &tr.config
	lw := &tr.weights.layers[l]
	dim := p.dim
	hid := p.hidden_dim
	if g_skip_ffn {
		g_skip_ffn = false
	} else {
		must(launch_rmsnorm(d_x, d_xb, dev_f32(lw.ffn_norm), T, dim, p.rms_eps))
	}
	must(launch_mm(lw.w1, d_xb, d_hb, dim, hid, T))
	must(launch_mm(lw.w3, d_xb, d_hb2, dim, hid, T))
	// swiglu(gate, up): result lands in the gate buffer. w1 is gate, w3 is up.
	must(launch_swiglu(d_hb, d_hb2, T * hid, T == 1))
	applied := false
	normed := false
	if T == 1 && g_fuse_norm {
		applied, normed = try_fuse_proj(lw.w2, d_hb, hid, dim, next_norm, p.rms_eps)
	}
	if !applied {
		if !(T == 1 && try_accum_gemv(lw.w2, d_hb, d_x, hid, dim)) {
			must(launch_mm(lw.w2, d_hb, d_xb2, hid, dim, T))
			must(launch_residual(d_x, d_xb2, T * dim))
		}
	}
	g_skip_attn = normed
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
		must(launch_swiglu(d_hb2, d_hb, p.n_ff_exp, false))
		must(launch_gemv(mw.down_exps.kind, down, d_hb2, d_partial, p.n_ff_exp, dim))
		must(launch_axpy(d_xb2, d_partial, wgt, dim))
	}
	if moe_gpu[l].has_shexp {
		nf := p.n_ff_shexp
		must(launch_gemv(.F32, moe_gpu[l].up, d_xb, d_hb, dim, nf))
		must(launch_gemv(.F32, moe_gpu[l].gate_w, d_xb, d_hb2, dim, nf))
		must(launch_swiglu(d_hb2, d_hb, nf, false))
		must(launch_gemv(.F32, moe_gpu[l].down, d_hb2, d_partial, nf, dim))
		must(launch_gemv(.F32, moe_gpu[l].ginp, d_xb, d_moe_logits, dim, 1))
		gsum: f32
		must(ck(cu.download(&gsum, d_moe_logits, 4), "shexp gate"))
		must(launch_axpy(d_xb2, d_partial, sigmoid_f32(gsum), dim))
	}
	must(launch_residual(xrow, d_xb2, dim))
}

@(private)
cuda_forward_body :: proc(tr: ^Transformer, pos0, T: int) {
	p := &tr.config
	w := &tr.weights
	g_q8_valid = false
	g_skip_ffn = false
	g_skip_attn = false
	for l in 0 ..< p.n_layers {
		lw := &w.layers[l]
		if g_skip_attn {
			g_skip_attn = false
		} else {
			must(launch_rmsnorm(d_x, d_xb, dev_f32(lw.attn_norm), T, p.dim, p.rms_eps))
		}
		switch lw.layer_type {
		case .Full_Attention:
			cuda_full(tr, l, pos0, T)
		case .Linear_Attention:
			cuda_linear(tr, l, pos0, T)
		}
		if p.is_moe {
			g_skip_ffn = false
			for ti in 0 ..< T {
				cuda_moe_token(tr, l, ti)
			}
		} else {
			next := dev_f32(w.output_norm)
			if l + 1 < p.n_layers {
				next = dev_f32(w.layers[l + 1].attn_norm)
			}
			cuda_mlp(tr, l, T, next)
		}
	}
	last := (T - 1) * p.dim
	if g_skip_attn {
		// The last down projection already wrote the output norm into d_xb.
		g_skip_attn = false
	} else {
		must(launch_rmsnorm(fp(d_x, last), d_xb, dev_f32(w.output_norm), 1, p.dim, p.rms_eps))
	}
	must(launch_gemv(w.output.kind, wptr(raw_data(w.output.data)), d_xb, d_logits, p.dim, p.vocab_size))
}

@(private)
profile_begin :: proc() {
	if !g_profile || !cu.gapi.ready {
		return
	}
	if !g_prof_ready {
		for i in 0 ..< PROF_MAX {
			ev: rawptr
			if cu.gapi.event_create(&ev, 0) != cu.OK {
				fmt.eprintln("cuda: event create failed; profile off")
				g_profile = false
				return
			}
			g_prof_ev[i] = ev
		}
		g_prof_ready = true
	}
	g_prof_n = 0
	g_prof_nbins = 0
	g_prof_launches = 0
	g_prof_gpu = 0
	for i in 0 ..< len(g_bins) {
		g_bins[i] = {}
	}
	for i in 0 ..< len(g_phase_ms) {
		g_phase_ms[i] = 0
		g_phase_n[i] = 0
	}
	if cu.gapi.event_record(g_prof_ev[0], nil) == cu.OK {
		g_prof_name[0] = ""
		g_prof_n = 1
	}
}

// Fold the recorded window into name totals and keep one fresh baseline.
// A 2000-token prefill is ~200k launches; the 8192-event ring cannot hold it.
@(private)
profile_flush :: proc() {
	if g_prof_n < 2 || cu.gapi.event_elapsed == nil {
		return
	}
	synced := cu.gapi.event_sync(g_prof_ev[g_prof_n - 1]) == cu.OK
	if synced {
		for i in 1 ..< g_prof_n {
			ms: f32
			if cu.gapi.event_elapsed(&ms, g_prof_ev[i - 1], g_prof_ev[i]) != cu.OK {
				continue
			}
			profile_note(g_prof_name[i], f64(ms))
		}
	}
	last := g_prof_n - 1
	g_prof_ev[0], g_prof_ev[last] = g_prof_ev[last], g_prof_ev[0]
	if cu.gapi.event_record(g_prof_ev[0], g_launch_stream) == cu.OK {
		g_prof_name[0] = ""
	}
	g_prof_n = 1
}

@(private)
profile_note :: proc(name: string, ms: f64) {
	g_prof_gpu += ms
	g_prof_launches += 1
	slot := -1
	for b in 0 ..< g_prof_nbins {
		if g_bins[b].name == name {
			slot = b
			break
		}
	}
	if slot < 0 {
		if g_prof_nbins >= len(g_bins) {
			return
		}
		slot = g_prof_nbins
		g_bins[slot].name = name
		g_prof_nbins += 1
	}
	g_bins[slot].count += 1
	g_bins[slot].ms += ms
	if g_phase {
		b := phase_bucket(name)
		g_phase_ms[b] += ms
		g_phase_n[b] += 1
	}
}

// 0 linear (tc/gemv/mmq), 1 attn, 2 kv store, 3 norm, 4 linear-attn (delta), 5 other.
@(private)
phase_bucket :: proc(name: string) -> int {
	if strings.has_prefix(name, "attn") {
		return 1
	}
	if strings.has_prefix(name, "store") {
		return 2
	}
	if strings.has_prefix(name, "delta") || name == "conv" || name == "l2norm" || name == "rmsnorm_gated" {
		return 4
	}
	if strings.has_prefix(name, "rmsnorm") || name == "residual_norm" || name == "qnorm" {
		return 3
	}
	if strings.has_prefix(name, "gemv") || strings.has_prefix(name, "tc") || name == "mmq" {
		return 0
	}
	return 5
}

@(private)
profile_end :: proc(T, pos0: int, t0: time.Tick) {
	if g_prof_launches == 0 && g_prof_n < 2 {
		return
	}
	profile_flush()
	wall := time.duration_milliseconds(time.tick_diff(t0, time.tick_now()))
	if g_phase {
		fmt.eprintfln("cuda: phase T={} pos={} wall {:.2f} linear {:.2f} attn {:.2f} kv {:.2f} norm {:.2f} delta {:.2f} other {:.2f} ms ({} launches)",
			T, pos0, wall, g_phase_ms[0], g_phase_ms[1], g_phase_ms[2], g_phase_ms[3], g_phase_ms[4], g_phase_ms[5], g_prof_launches)
	}
	if !g_profile_full {
		return
	}
	nbin := g_prof_nbins
	order := make([]int, nbin, context.temp_allocator)
	for i in 0 ..< nbin {
		order[i] = i
	}
	for i in 0 ..< nbin {
		best := i
		for j in i + 1 ..< nbin {
			if g_bins[order[j]].ms > g_bins[order[best]].ms {
				best = j
			}
		}
		order[i], order[best] = order[best], order[i]
	}
	fmt.eprintfln("cuda: profile T={} pos={}  wall {:.2f} ms  kernels {:.2f} ms  ({} launches)", T, pos0, wall, g_prof_gpu, g_prof_launches)
	for i in order {
		b := g_bins[i]
		fmt.eprintfln("  {}  count {}  {:.3f} ms", b.name, b.count, b.ms)
	}
}

@(private)
write_decode_pos :: proc(pos: int) -> bool {
	// MemcpyHtoD_v2 runs on the legacy null stream and syncs the device.
	// That sync, once per token, is why graph replay was slower than eager.
	// After the graph exists, the copy is ordered on g_stream ahead of the
	// replay. The first token still uses the sync copy so capture starts on
	// an idle stream. g_pos_host stays live until the logits download.
	g_pos_host = i32(pos)
	if g_graph_ready && g_stream != nil && cu.gapi.htod_async != nil {
		return ck(cu.gapi.htod_async(g_pos_sym, &g_pos_host, 4, g_stream), "decode pos")
	}
	return ck(cu.drv.MemcpyHtoD_v2(g_pos_sym, &g_pos_host, 4), "decode pos")
}

// Capture does not run the kernels. On success the caller replays the graph.
// On failure the stream is left idle and the caller runs the body for real.
@(private)
capture_decode_graph :: proc(tr: ^Transformer, pos0: int) -> bool {
	g_q8_valid = false
	g_capturing = true
	g_launch_stream = g_stream
	if cu.gapi.stream_begin(g_stream, 0) != cu.OK {
		fmt.eprintln("cuda: graph capture failed to start")
		g_capturing = false
		g_launch_stream = nil
		return false
	}
	cuda_forward_body(tr, pos0, 1)
	g_capturing = false
	graph: rawptr
	if cu.gapi.stream_end(g_stream, &graph) != cu.OK {
		fmt.eprintln("cuda: graph capture failed to end")
		g_launch_stream = nil
		g_q8_valid = false
		return false
	}
	logb := make([]u8, 1024, context.temp_allocator)
	exec: rawptr
	errnode: rawptr
	ierr := cu.gapi.graph_instantiate(&exec, graph, &errnode, raw_data(logb), c.size_t(len(logb)))
	cu.gapi.graph_destroy(graph)
	if ierr != cu.OK {
		n := 0
		for n < len(logb) && logb[n] != 0 {
			n += 1
		}
		if n > 0 {
			fmt.eprintfln("cuda: graph instantiate: %s", string(logb[:n]))
		} else {
			fmt.eprintfln("cuda: graph instantiate failed ({})", ierr)
		}
		g_launch_stream = nil
		g_q8_valid = false
		return false
	}
	g_exec = exec
	g_graph_ready = true
	g_graph_launches = g_launches
	g_launch_stream = nil
	return true
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
	use_graph := g_use_graph && T == 1 && !p.is_moe && g_stream != nil && g_pos_sym != 0 && cu.gapi.ready
	replayed := false
	if use_graph {
		if !write_decode_pos(pos0) {
			os.exit(1)
		}
		if !g_graph_ready {
			if !capture_decode_graph(tr, pos0) {
				g_use_graph = false
				fmt.eprintln("cuda: graph disabled; this token runs eager")
				g_q8_valid = false
				cuda_forward_body(tr, pos0, T)
			} else if cu.gapi.graph_launch(g_exec, g_stream) != cu.OK {
				fmt.eprintln("cuda: graph launch failed")
				os.exit(1)
			} else {
				replayed = true
			}
		} else if cu.gapi.graph_launch(g_exec, g_stream) != cu.OK {
			fmt.eprintln("cuda: graph launch failed")
			os.exit(1)
		} else {
			replayed = true
		}
	} else {
		tprof: time.Tick
		if g_profile {
			tprof = time.tick_now()
			profile_begin()
		}
		cuda_forward_body(tr, pos0, T)
		if g_profile {
			profile_end(T, pos0, tprof)
		}
	}
	if replayed {
		must(ck(cu.sync(), "graph sync"))
	}
	must(ck(cu.download(raw_data(h_logits), d_logits, p.vocab_size * 4), "logits"))
	if g_timing {
		ms := time.duration_milliseconds(time.tick_diff(t0, time.tick_now()))
		fmt.eprintfln("[timing] cuda_forward T={} pos={} wall={:.2f} ms ({:.2f} ms/tok)", T, pos0, ms, ms / f64(T))
	}
	if g_count_launches {
		if replayed {
			fmt.eprintfln("cuda: graph replay, {} launches captured (T={} pos={})", g_graph_launches, T, pos0)
		} else {
			fmt.eprintfln("cuda: launches {} this forward (T={} pos={})", g_launches, T, pos0)
		}
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
