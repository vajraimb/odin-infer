#+build darwin

/* MMQ autotune harness (Step 3): sweeps BM x BK x fastmath for the Q4_K / Q6_K
   simdgroup GEMM kernels on the engine's real prefill shapes.

   Kernel MSL is generated programmatically per config: BM/BK/NT/TPR/RSTEP are
   baked in as compile-time constants (no runtime params, no placeholder text
   substitution in the kernel body -- constants header + fixed body).

   Per config: compile -> correctness gate (small shape, CPU f64 ref via
   ggml.dequant_row) -> warmup 3 + median-of-7 timing per real shape.
   Results append to gemm_harness/autotune_cache.jsonl; cached keys are skipped.

   Run:  odin run gemm_harness/gemm_autotune.odin -file -o:speed
*/

package gemm_autotune

import "core:fmt"
import "core:math"
import "core:os"
import "core:strconv"
import "core:strings"
import "core:time"
import ggml "ggml:ggml"
import NS "core:sys/darwin/Foundation"
import MTL "vendor:darwin/Metal"

CACHE_PATH :: "gemm_harness/autotune_cache.jsonl"
GATE_M, GATE_N, GATE_K :: 256, 64, 512
GATE_TOL :: 5e-3
ITERS :: 7 // median-of-7

// ---------- Metal state ----------
@(private = "file")
g_dev: ^MTL.Device
@(private = "file")
g_queue: ^MTL.CommandQueue

GemmDims :: struct {
	M, N, K: u32,
}

// ---------- MSL templates ----------
// Constants header is generated per config; bodies reference BM/BK/NT/TPR/RSTEP.

MSL_HEAD :: `#include <metal_stdlib>
using namespace metal;
struct GemmDims { uint M; uint N; uint K; };
inline void k4_get_scale_min(int j, device const uchar * q, thread uchar & d, thread uchar & m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
`

// Generalized v3: threadgroup = NSG simdgroups (NT = NSG*32 threads) computes a
// BM x 8 output tile; sa[BM x BK] dequant staging, sb[BK x 8] activation tile.
// Dequant mapping: TPR = BK/8 threads per row, each owns 8 cols sharing one
// Q4_K scale; rows stride by RSTEP = NT/TPR (requires NT % TPR == 0).
MSL_Q4K_BODY :: `kernel void gemm_q4k_tune(
    device const uchar * A [[buffer(0)]], device const half * B [[buffer(1)]],
    device float * C [[buffer(2)]], constant GemmDims & dims [[buffer(3)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiitg [[thread_index_in_threadgroup]]) {
    const uint M = dims.M, N = dims.N, K = dims.K;
    const uint r0 = tgpig.y * BM, c0 = tgpig.x * 8;
    const ulong row_bytes = (ulong)(K / 256) * 144;
    threadgroup half sa[BM * BK];
    threadgroup half sb[BK * 8];
    simdgroup_half8x8 a_tile, b_tile;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
    const uint sg_row = sgitg * 8;
    for (uint k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint row = tiitg / TPR; row < BM; row += RSTEP) {
            const uint col_base = (tiitg % TPR) * 8;
            uint e_base = k + col_base;
            device const uchar * blk = A + (ulong)(r0 + row) * row_bytes + (uint)(e_base / 256) * 144;
            int sup = int(e_base & 255) >> 6;
            int hf = (int(e_base & 255) >> 5) & 1;
            int is = sup * 2 + hf;
            float d = (float)(*(device const half *)(blk));
            float dmin = (float)(*(device const half *)(blk + 2));
            device const uchar * scales = blk + 4;
            device const uchar * qs = blk + 16;
            uchar sc, m; k4_get_scale_min(is, scales, sc, m);
            float dl = d * (float)sc, ml = dmin * (float)m;
            int qoff = sup * 32, shift = 4 * hf;
            int l32 = int(e_base & 31); // within-32-group column (== col_base only when k%32==0)
            for (uint j = 0; j < 8; j++) {
                uchar nib = (qs[qoff + l32 + int(j)] >> shift) & 0xF;
                sa[row * BK + col_base + j] = (half)(dl * (float)nib - ml);
            }
        }
        for (uint t = tiitg; t < BK * 8; t += NT) {
            uint kr = t >> 3, nc = t & 7;
            sb[t] = B[(ulong)(k + kr) * N + c0 + nc];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ik = 0; ik < BK / 8; ik++) {
            simdgroup_load(a_tile, sa + sg_row * BK + ik * 8, BK);
            simdgroup_load(b_tile, sb + ik * 64, 8);
            simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc, C + (ulong)(r0 + sg_row) * N + c0, N);
}
`

MSL_Q6K_BODY :: `kernel void gemm_q6k_tune(
    device const uchar * A [[buffer(0)]], device const half * B [[buffer(1)]],
    device float * C [[buffer(2)]], constant GemmDims & dims [[buffer(3)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiitg [[thread_index_in_threadgroup]]) {
    const uint M = dims.M, N = dims.N, K = dims.K;
    const uint r0 = tgpig.y * BM, c0 = tgpig.x * 8;
    const ulong row_bytes = (ulong)(K / 256) * 210;
    threadgroup half sa[BM * BK];
    threadgroup half sb[BK * 8];
    simdgroup_half8x8 a_tile, b_tile;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
    const uint sg_row = sgitg * 8;
    for (uint k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint row = tiitg / TPR; row < BM; row += RSTEP) {
            const uint col_base = (tiitg % TPR) * 8;
            uint e_base = k + col_base;
            device const uchar * blk = A + (ulong)(r0 + row) * row_bytes + (uint)(e_base / 256) * 210;
            int ew = int(e_base & 255);
            int hf = ew >> 7, sub = (ew >> 5) & 3, is = (ew >> 4) & 1;
            float d = (float)(*(device const half *)(blk + 208));
            device const uchar * ql = blk;
            device const uchar * qh = blk + 128;
            device const char  * sc = (device const char *)(blk + 192);
            float dl = d * (float)sc[hf * 8 + is + 2 * sub];
            int qloff = hf * 64, qhoff = hf * 32;
            int l32 = int(e_base & 31); // within-32-group column (== col_base only when k%32==0)
            for (uint j = 0; j < 8; j++) {
                int l = l32 + int(j);
                int q;
                if (sub == 0)      q = ((ql[qloff + l]      & 0xF) | (((qh[qhoff + l] >> 0) & 3) << 4)) - 32;
                else if (sub == 1) q = ((ql[qloff + l + 32] & 0xF) | (((qh[qhoff + l] >> 2) & 3) << 4)) - 32;
                else if (sub == 2) q = ((ql[qloff + l] >> 4)       | (((qh[qhoff + l] >> 4) & 3) << 4)) - 32;
                else               q = ((ql[qloff + l + 32] >> 4)  | (((qh[qhoff + l] >> 6) & 3) << 4)) - 32;
                sa[row * BK + col_base + j] = (half)(dl * (float)q);
            }
        }
        for (uint t = tiitg; t < BK * 8; t += NT) {
            uint kr = t >> 3, nc = t & 7;
            sb[t] = B[(ulong)(k + kr) * N + c0 + nc];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ik = 0; ik < BK / 8; ik++) {
            simdgroup_load(a_tile, sa + sg_row * BK + ik * 8, BK);
            simdgroup_load(b_tile, sb + ik * 64, 8);
            simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc, C + (ulong)(r0 + sg_row) * N + c0, N);
}
`

@(private = "file")
make_src :: proc(q6: bool, bm, bk: int) -> string {
	nsg := bm / 8
	nt := nsg * 32
	tpr := bk / 8
	rstep := nt / tpr
	header := fmt.tprintf(
		"constant uint BM = %d;\nconstant uint BK = %d;\nconstant uint NT = %d;\nconstant uint TPR = %d;\nconstant uint RSTEP = %d;\n",
		bm, bk, nt, tpr, rstep,
	)
	body := q6 ? MSL_Q6K_BODY : MSL_Q4K_BODY
	s, err := strings.concatenate({MSL_HEAD, header, body})
	if err != nil {
		fmt.eprintln("concat failed")
	}
	return s
}

// ---------- helpers ----------
@(private = "file")
rng_f :: proc(seed: ^u32) -> f32 {
	x := seed^
	x = x ~ (x << 13)
	x = x ~ (x >> 17)
	x = x ~ (x << 5)
	seed^ = x
	return f32((x >> 8) & 0xFFFFFF) / f32(0x1000000)
}

@(private = "file")
write_f16_le :: proc(buf: []u8, off: int, v: f32) {
	h := f16(v)
	u := transmute(u16)h
	buf[off] = u8(u & 0xFF)
	buf[off + 1] = u8((u >> 8) & 0xFF)
}

@(private = "file")
copy_bytes :: proc(buf: ^MTL.Buffer, src: []u8) {
	dst := buf->contentsAsSlice([]u8)
	copy(dst, src)
}

@(private = "file")
copy_f16 :: proc(buf: ^MTL.Buffer, src: []f16) {
	dst := buf->contentsAsSlice([]u8)
	sb := ([^]u8)(raw_data(src))[: len(src) * 2]
	for i in 0 ..< len(sb) {
		dst[i] = sb[i]
	}
}

@(private = "file")
bytes_of :: proc(p: rawptr, n: int) -> []u8 {
	return ([^]u8)(p)[:n]
}

@(private = "file")
new_buf :: proc(n_bytes: int) -> ^MTL.Buffer {
	return g_dev->newBufferWithLength(NS.UInteger(n_bytes), MTL.ResourceStorageModeShared)
}

// ---------- compile one config ----------
@(private = "file")
compile_config :: proc(q6: bool, bm, bk: int, fastmath: bool) -> (^MTL.ComputePipelineState, bool) {
	src := make_src(q6, bm, bk)
	src_ns := NS.String.alloc()->initWithOdinString(src)
	defer src_ns->release()
	opts := MTL.CompileOptions.alloc()->init()
	defer opts->release()
	opts->setFastMathEnabled(fastmath)
	lib, err := g_dev->newLibraryWithSource(src_ns, opts)
	if err != nil {
		fmt.eprintfln("  [compile_error] q6={} bm={} bk={} fm={}: {}", q6, bm, bk, fastmath, err->localizedDescription()->odinString())
		return nil, false
	}
	defer lib->release()
	fn_name := q6 ? "gemm_q6k_tune" : "gemm_q4k_tune"
	fn_ := lib->newFunctionWithName(NS.String.alloc()->initWithOdinString(fn_name))
	if fn_ == nil {
		fmt.eprintfln("  [compile_error] kernel '%s' not found", fn_name)
		return nil, false
	}
	defer fn_->release()
	pso, perr := g_dev->newComputePipelineStateWithFunction(fn_)
	if perr != nil {
		fmt.eprintfln("  [compile_error] pipeline: %s", perr->localizedDescription()->odinString())
		return nil, false
	}
	return pso, true
}

// ---------- dispatch ----------
@(private = "file")
run_tune :: proc(pso: ^MTL.ComputePipelineState, bufA, bufB, bufC: ^MTL.Buffer, M, N, K, bm: int) {
	nt := (bm / 8) * 32
	cmd := g_queue->commandBuffer()
	enc := cmd->computeCommandEncoder()
	enc->setComputePipelineState(pso)
	enc->setBuffer(bufA, 0, 0)
	enc->setBuffer(bufB, 0, 1)
	enc->setBuffer(bufC, 0, 2)
	dims := GemmDims{u32(M), u32(N), u32(K)}
	enc->setBytes(bytes_of(&dims, size_of(GemmDims)), 3)
	enc->dispatchThreadgroups(MTL.Size{NS.Integer(N / 8), NS.Integer(M / bm), 1}, MTL.Size{NS.Integer(nt), 1, 1})
	enc->endEncoding()
	cmd->commit()
	cmd->waitUntilCompleted()
}

// ---------- test data ----------
@(private = "file")
gen_quant :: proc(q6: bool, M, K: int, seed_in: u32) -> []u8 {
	nb := K / 256
	bs := q6 ? 210 : 144
	a := make([]u8, M * nb * bs)
	seed := seed_in
	for b in 0 ..< M * nb {
		base := b * bs
		if q6 {
			write_f16_le(a, base + 208, rng_f(&seed)*0.18 + 0.02)
			for i in 0 ..< 128 { a[base + i] = u8(rng_f(&seed) * 256.0) }
			for i in 0 ..< 64 { a[base + 128 + i] = u8(rng_f(&seed) * 256.0) }
			for i in 0 ..< 16 { a[base + 192 + i] = u8(rng_f(&seed) * 256.0) }
		} else {
			write_f16_le(a, base, rng_f(&seed)*0.18 + 0.02)
			write_f16_le(a, base + 2, rng_f(&seed)*0.05)
			for s in 0 ..< 12 { a[base + 4 + s] = u8(rng_f(&seed) * 256.0) }
			for q in 0 ..< 128 { a[base + 16 + q] = u8(rng_f(&seed) * 256.0) }
		}
	}
	return a
}

@(private = "file")
gen_b :: proc(K, N: int, seed_in: u32) -> []f16 {
	b := make([]f16, K * N)
	seed := seed_in
	for i in 0 ..< K * N {
		b[i] = f16(rng_f(&seed)*2.0 - 1.0)
	}
	return b
}

// ---------- correctness gate (small shape, CPU f64 ref) ----------
@(private = "file")
gate :: proc(pso: ^MTL.ComputePipelineState, q6: bool, bm: int) -> (f64, bool) {
	M, N, K := GATE_M, GATE_N, GATE_K
	nb := K / 256
	bs := q6 ? 210 : 144
	a_bytes := gen_quant(q6, M, K, 0xBEEF)
	b_h := gen_b(K, N, 0xF00D)
	defer delete(a_bytes)
	defer delete(b_h)

	a_ref := make([]f32, M * K)
	defer delete(a_ref)
	ggml.dequant_row(q6 ? .Q6_K : .Q4_K, a_bytes, M * K, a_ref)
	for i in 0 ..< M * K {
		a_ref[i] = f32(f16(a_ref[i])) // match MMQ half staging
	}
	b_ref := make([]f32, K * N)
	defer delete(b_ref)
	for i in 0 ..< K * N {
		b_ref[i] = f32(b_h[i])
	}
	c_ref := make([]f64, M * N)
	defer delete(c_ref)
	for m in 0 ..< M {
		for n in 0 ..< N {
			s: f64 = 0
			for k in 0 ..< K {
				s += f64(a_ref[m*K+k]) * f64(b_ref[k*N+n])
			}
			c_ref[m*N+n] = s
		}
	}

	bufA := new_buf(M * nb * bs)
	bufB := new_buf(K * N * 2)
	bufC := new_buf(M * N * 4)
	defer bufA->release()
	defer bufB->release()
	defer bufC->release()
	copy_bytes(bufA, a_bytes)
	copy_f16(bufB, b_h)
	run_tune(pso, bufA, bufB, bufC, M, N, K, bm)
	c_gpu := bufC->contentsAsSlice([]f32)[: M * N]

	max_abs, scale: f64 = 0, 0
	for i in 0 ..< M * N {
		d := math.abs(f64(c_gpu[i]) - c_ref[i])
		if d > max_abs {
			max_abs = d
		}
		a := math.abs(c_ref[i])
		if a > scale {
			scale = a
		}
	}
	scaled := scale > 1e-6 ? max_abs / scale : max_abs
	return scaled, scaled <= GATE_TOL
}

// ---------- median-of-7 timing ----------
@(private = "file")
median7 :: proc(durs: ^[ITERS]time.Duration) -> time.Duration {
	for i in 1 ..< ITERS {
		v := durs[i]
		j := i
		for j > 0 && durs[j - 1] > v {
			durs[j] = durs[j - 1]
			j -= 1
		}
		durs[j] = v
	}
	return durs[ITERS / 2]
}

// ---------- result cache ----------
@(private = "file")
cache_key :: proc(q6: bool, bm, bk: int, fm: bool, M, N, K: int) -> string {
	return fmt.tprintf(`"q":"%s","bm":%d,"bk":%d,"fm":%d,"M":%d,"N":%d,"K":%d`, q6 ? "q6k" : "q4k", bm, bk, fm ? 1 : 0, M, N, K)
}

@(private = "file")
g_cache_text: string

@(private = "file")
cache_lookup :: proc(key: string) -> (f64, bool) {
	if len(g_cache_text) == 0 {
		return 0, false
	}
	lines, _ := strings.split(g_cache_text, "\n")
	defer delete(lines)
	for line in lines {
		if strings.contains(line, key) {
			idx := strings.index(line, `"gflops":`)
			if idx < 0 {
				continue
			}
			s := line[idx + len(`"gflops":`):]
			end := 0
			for end < len(s) {
				c := s[end]
				is_num := (c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '-' || c == '+'
				if !is_num {
					break
				}
				end += 1
			}
			v, ok := strconv.parse_f64(s[:end])
			return v, ok
		}
	}
	return 0, false
}

@(private = "file")
cache_append :: proc(key: string, gflops, ms: f64) {
	line := fmt.tprintf("{%s,\"gflops\":%.2f,\"ms\":%.4f}\n", key, gflops, ms)
	fd, err := os.open(CACHE_PATH, os.O_WRONLY + os.O_CREATE + os.O_APPEND)
	if err != nil {
		fmt.eprintfln("cache: open failed: {}", err)
		return
	}
	defer os.close(fd)
	os.write_string(fd, line)
}

// ---------- results ----------
Result :: struct {
	q6:             bool,
	bm, bk:         int,
	fm:             bool,
	m, n, k:        int,
	gflops, ms:     f64,
	cached:         bool,
}

@(private = "file")
g_results: [dynamic]Result

// ---------- per-config sweep ----------
@(private = "file")
PerfShape :: struct {
	m, n, k: int,
	name:    string,
}

@(private = "file")
sweep_config :: proc(q6: bool, bm, bk: int, fm: bool, shapes: []PerfShape) {
	nt := (bm / 8) * 32
	tpr := bk / 8
	if nt % tpr != 0 {
		fmt.printfln("  [skip] q6={} bm={} bk={}: nt%tpr!=0", q6, bm, bk)
		return
	}
	tag := fmt.tprintf("%s bm=%d bk=%d fm=%s", q6 ? "q6k" : "q4k", bm, bk, fm ? "on" : "off")

	// cache fast path: all shapes cached -> record and skip compile entirely
	all_cached := true
	for s in shapes {
		if _, ok := cache_lookup(cache_key(q6, bm, bk, fm, s.m, s.n, s.k)); !ok {
			all_cached = false
			break
		}
	}
	if all_cached {
		for s in shapes {
			g, _ := cache_lookup(cache_key(q6, bm, bk, fm, s.m, s.n, s.k))
			append(&g_results, Result{q6, bm, bk, fm, s.m, s.n, s.k, g, 0, true})
		}
		fmt.printfln("  [cached] %s", tag)
		return
	}

	pso, ok := compile_config(q6, bm, bk, fm)
	if !ok {
		return
	}
	defer pso->release()

	scaled, gate_ok := gate(pso, q6, bm)
	if !gate_ok {
		fmt.printfln("  [FAIL] %s gate scaled=%.2e (tol %.0e) -- not timed", tag, scaled, GATE_TOL)
		return
	}
	fmt.printf("  [OK] %s gate scaled=%.2e |", tag, scaled)

	for s in shapes {
		key := cache_key(q6, bm, bk, fm, s.m, s.n, s.k)
		if g, hit := cache_lookup(key); hit {
			append(&g_results, Result{q6, bm, bk, fm, s.m, s.n, s.k, g, 0, true})
			fmt.printf("  %s=%.0fG(cached)", s.name, g)
			continue
		}
		nb := s.k / 256
		bs := q6 ? 210 : 144
		a_bytes := gen_quant(q6, s.m, s.k, 0xABCDE)
		b_h := gen_b(s.k, s.n, 0x12345)
		bufA := new_buf(s.m * nb * bs)
		bufB := new_buf(s.k * s.n * 2)
		bufC := new_buf(s.m * s.n * 4)
		copy_bytes(bufA, a_bytes)
		copy_f16(bufB, b_h)

		for _ in 0 ..< 3 {
			run_tune(pso, bufA, bufB, bufC, s.m, s.n, s.k, bm)
		}
		durs: [ITERS]time.Duration
		for i in 0 ..< ITERS {
			t0 := time.tick_now()
			run_tune(pso, bufA, bufB, bufC, s.m, s.n, s.k, bm)
			durs[i] = time.tick_since(t0)
		}
		med := median7(&durs)
		secs := f64(i64(med)) / 1e9
		gflops := f64(2) * f64(s.m) * f64(s.n) * f64(s.k) / secs / 1e9
		cache_append(key, gflops, secs * 1e3)
		append(&g_results, Result{q6, bm, bk, fm, s.m, s.n, s.k, gflops, secs * 1e3, false})
		fmt.printf("  %s=%.0fG", s.name, gflops)

		bufA->release()
		bufB->release()
		bufC->release()
		delete(a_bytes)
		delete(b_h)
	}
	fmt.println()
}

// ---------- report ----------
@(private = "file")
report :: proc(q6: bool, s: PerfShape) {
	// collect indices for this shape, insertion sort by gflops desc
	idxs := make([dynamic]int)
	defer delete(idxs)
	base_g := 0.0
	for r, i in g_results {
		if r.q6 == q6 && r.m == s.m && r.n == s.n && r.k == s.k {
			append(&idxs, i)
			if r.bm == 32 && r.bk == 32 && r.fm {
				base_g = r.gflops
			}
		}
	}
	for i in 1 ..< len(idxs) {
		v := idxs[i]
		j := i
		for j > 0 && g_results[idxs[j - 1]].gflops < g_results[v].gflops {
			idxs[j] = idxs[j - 1]
			j -= 1
		}
		idxs[j] = v
	}
	fmt.printfln("\n== %s %s (M={} N={} K={}) == baseline v3 (bm=32 bk=32 fm=on) = {:.0f} GFLOPS", q6 ? "q6k" : "q4k", s.name, s.m, s.n, s.k, base_g)
	n_show := min(len(idxs), 8)
	for i in 0 ..< n_show {
		r := g_results[idxs[i]]
		speedup := base_g > 0 ? r.gflops / base_g : 0
		marker := r.bm == 32 && r.bk == 32 && r.fm ? " <- baseline" : ""
		fmt.printfln("   #{} bm={} bk={} fm={}  {:.0f} GFLOPS ({:.2f}x){}{}", i + 1, r.bm, r.bk, r.fm ? "on " : "off", r.gflops, speedup, marker, r.cached ? " [cached]" : "")
	}
}

main :: proc() {
	g_dev = MTL.CreateSystemDefaultDevice()
	if g_dev == nil {
		fmt.eprintln("metal: no device")
		os.exit(1)
	}
	fmt.printf("metal: %s\n", g_dev->name()->odinString())
	g_queue = g_dev->newCommandQueue()

	if data, ok := os.read_entire_file_from_path(CACHE_PATH, context.allocator); ok == nil {
		g_cache_text = string(data)
		fmt.printfln("cache: loaded %s ({} bytes)", CACHE_PATH, len(data))
	}

	q4k_shapes := []PerfShape{
		{4096, 64, 4096, "w1_n64"},
		{12288, 64, 4096, "hidden_x_T_n64"},
		{4096, 512, 4096, "w1_n512"},
		{12288, 512, 4096, "hidden_x_T_n512"},
	}
	q6k_shapes := []PerfShape{
		{4096, 64, 12288, "w2_n64"},
		{4096, 512, 12288, "w2_n512"},
	}
	bms := [?]int{16, 32, 64}
	bks := [?]int{16, 32, 64}
	fms := [?]bool{true, false}

	fmt.println("\n=== Q4_K sweep (18 configs x 2 shapes) ===")
	for bm in bms {
		for bk in bks {
			for fm in fms {
				sweep_config(false, bm, bk, fm, q4k_shapes)
			}
		}
	}
	fmt.println("\n=== Q6_K sweep (18 configs x 1 shape) ===")
	for bm in bms {
		for bk in bks {
			for fm in fms {
				sweep_config(true, bm, bk, fm, q6k_shapes)
			}
		}
	}

	for s in q4k_shapes {
		report(false, s)
	}
	for s in q6k_shapes {
		report(true, s)
	}
	fmt.printfln("\nresults appended to %s", CACHE_PATH)
}
