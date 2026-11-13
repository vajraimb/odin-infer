#+build darwin

/* Sweep MMQ output-column tile (BN). The shipped kernels fix BN=8, so each
   dequant of A is reused across only 8 columns. Q6_K sits near 400 GFLOPS
   while Q4_K is near 1000; this sweep asks whether a wider N tile amortizes
   the Q6 dequant on the real prefill shapes.

   Wall-clock around command-buffer wait (not GPUEndTime). Correctness gate
   is ggml.dequant_row rounded to f16, then an f64 dot.

   odin run tools/gemm_harness/qn_sweep.odin -file -o:speed \
     -collection:ggml=packages
*/

package qn_sweep

import "core:fmt"
import "core:math"
import "core:os"
import "core:strings"
import "core:time"
import ggml "ggml:ggml"
import NS "core:sys/darwin/Foundation"
import MTL "vendor:darwin/Metal"

GATE_TOL :: 5e-3
ITERS :: 5

Config :: struct {
	q6:      bool,
	bm, bn, bk: int,
}

Shape :: struct {
	m, n, k: int,
	name:    string,
}

g_dev: ^MTL.Device
g_queue: ^MTL.CommandQueue

MSL_HEAD :: `#include <metal_stdlib>
using namespace metal;
struct GemmDims { uint M; uint N; uint K; };
inline void k4_get_scale_min(int j, device const uchar * q, thread uchar & d, thread uchar & m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
`

// Numbers are @@TOKENS@@ so the body can keep single braces.
MSL_Q4 :: `kernel void @@NAME@@(
    device const uchar * A [[buffer(0)]], device const half * B [[buffer(1)]],
    device float * C [[buffer(2)]], constant GemmDims & dims [[buffer(3)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiitg [[thread_index_in_threadgroup]]) {
    const uint M = dims.M, N = dims.N, K = dims.K;
    const uint r0 = tgpig.y * @@BM@@, c0 = tgpig.x * @@BN@@;
    const ulong row_bytes = (ulong)(K / 256) * 144;
    threadgroup half sa[@@BM@@ * @@BK@@];
    threadgroup half sb[@@BK@@ * @@BN@@];
    simdgroup_half8x8 a_tile, b_tile;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
    const uint sg_row = (sgitg / @@NSGN@@) * 8;
    const uint sg_col = (sgitg % @@NSGN@@) * 8;
    for (uint k = 0; k < K; k += @@BK@@) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint row = tiitg / @@TPR@@; row < @@BM@@; row += @@RSTEP@@) {
            const uint col_base = (tiitg % @@TPR@@) * 8;
            uint e_base = k + col_base;
            device const uchar * blk = A + (ulong)(r0 + row) * row_bytes + (uint)(e_base / 256) * 144;
            int ew = int(e_base & 255), sup = ew >> 6, hf = (ew >> 5) & 1, is = sup * 2 + hf;
            float d = (float)(*(device const half *)(blk));
            float dmin = (float)(*(device const half *)(blk + 2));
            device const uchar * scales = blk + 4;
            device const uchar * qs = blk + 16;
            uchar sc, m; k4_get_scale_min(is, scales, sc, m);
            float dl = d * (float)sc, ml = dmin * (float)m;
            int qoff = sup * 32, shift = 4 * hf;
            int l32 = ew & 31;
            for (uint j = 0; j < 8; j++) {
                uchar nib = (qs[qoff + l32 + int(j)] >> shift) & 0xF;
                sa[row * @@BK@@ + col_base + j] = (half)(dl * (float)nib - ml);
            }
        }
        for (uint t = tiitg; t < @@BK@@ * @@BN@@; t += @@NT@@) {
            uint kr = t / @@BN@@, nc = t % @@BN@@;
            sb[t] = B[(ulong)(k + kr) * N + c0 + nc];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ik = 0; ik < @@BK@@ / 8; ik++) {
            simdgroup_load(a_tile, sa + sg_row * @@BK@@ + ik * 8, @@BK@@);
            simdgroup_load(b_tile, sb + ik * 8 * @@BN@@ + sg_col, @@BN@@);
            simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (r0 + sg_row < M) simdgroup_store(acc, C + (ulong)(r0 + sg_row) * N + c0 + sg_col, N);
}
`

// sub is uniform across the 8-element chunk; the switch sits outside the j loop.
MSL_Q6 :: `kernel void @@NAME@@(
    device const uchar * A [[buffer(0)]], device const half * B [[buffer(1)]],
    device float * C [[buffer(2)]], constant GemmDims & dims [[buffer(3)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiitg [[thread_index_in_threadgroup]]) {
    const uint M = dims.M, N = dims.N, K = dims.K;
    const uint r0 = tgpig.y * @@BM@@, c0 = tgpig.x * @@BN@@;
    const ulong row_bytes = (ulong)(K / 256) * 210;
    threadgroup half sa[@@BM@@ * @@BK@@];
    threadgroup half sb[@@BK@@ * @@BN@@];
    simdgroup_half8x8 a_tile, b_tile;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
    const uint sg_row = (sgitg / @@NSGN@@) * 8;
    const uint sg_col = (sgitg % @@NSGN@@) * 8;
    for (uint k = 0; k < K; k += @@BK@@) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint row = tiitg / @@TPR@@; row < @@BM@@; row += @@RSTEP@@) {
            const uint col_base = (tiitg % @@TPR@@) * 8;
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
            int l32 = ew & 31;
            for (uint j = 0; j < 8; j++) {
                int l = l32 + int(j);
                int q;
                if (sub == 0)      q = ((ql[qloff + l]      & 0xF) | (((qh[qhoff + l] >> 0) & 3) << 4)) - 32;
                else if (sub == 1) q = ((ql[qloff + l + 32] & 0xF) | (((qh[qhoff + l] >> 2) & 3) << 4)) - 32;
                else if (sub == 2) q = ((ql[qloff + l] >> 4)       | (((qh[qhoff + l] >> 4) & 3) << 4)) - 32;
                else               q = ((ql[qloff + l + 32] >> 4)  | (((qh[qhoff + l] >> 6) & 3) << 4)) - 32;
                sa[row * @@BK@@ + col_base + j] = (half)(dl * (float)q);
            }
        }
        for (uint t = tiitg; t < @@BK@@ * @@BN@@; t += @@NT@@) {
            uint kr = t / @@BN@@, nc = t % @@BN@@;
            sb[t] = B[(ulong)(k + kr) * N + c0 + nc];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ik = 0; ik < @@BK@@ / 8; ik++) {
            simdgroup_load(a_tile, sa + sg_row * @@BK@@ + ik * 8, @@BK@@);
            simdgroup_load(b_tile, sb + ik * 8 * @@BN@@ + sg_col, @@BN@@);
            simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (r0 + sg_row < M) simdgroup_store(acc, C + (ulong)(r0 + sg_row) * N + c0 + sg_col, N);
}
`

sub :: proc(s, a, b: string) -> string {
	out, _ := strings.replace_all(s, a, b)
	return out
}

kernel_src :: proc(c: Config) -> (string, string, int) {
	nsg_n := c.bn / 8
	nsg := (c.bm / 8) * nsg_n
	nt := nsg * 32
	tpr := c.bk / 8
	rstep := nt / tpr
	name := fmt.tprintf("gemm_{}_{}_{}_{}_{}", c.q6 ? "q6" : "q4", c.bm, c.bn, c.bk, nt)
	body := c.q6 ? MSL_Q6 : MSL_Q4
	body = sub(body, "@@NAME@@", name)
	body = sub(body, "@@BM@@", fmt.tprintf("{}", c.bm))
	body = sub(body, "@@BN@@", fmt.tprintf("{}", c.bn))
	body = sub(body, "@@BK@@", fmt.tprintf("{}", c.bk))
	body = sub(body, "@@NT@@", fmt.tprintf("{}", nt))
	body = sub(body, "@@TPR@@", fmt.tprintf("{}", tpr))
	body = sub(body, "@@RSTEP@@", fmt.tprintf("{}", rstep))
	body = sub(body, "@@NSGN@@", fmt.tprintf("{}", nsg_n))
	return name, body, nt
}

rng_f :: proc(seed: ^u32) -> f32 {
	x := seed^
	x = x ~ (x << 13)
	x = x ~ (x >> 17)
	x = x ~ (x << 5)
	seed^ = x
	return f32((x >> 8) & 0xFFFFFF) / f32(0x1000000)
}

write_f16_le :: proc(buf: []u8, off: int, v: f32) {
	h := f16(v)
	u := transmute(u16)h
	buf[off] = u8(u & 0xFF)
	buf[off + 1] = u8((u >> 8) & 0xFF)
}

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

gen_b :: proc(K, N: int, seed_in: u32) -> []f16 {
	b := make([]f16, K * N)
	seed := seed_in
	for i in 0 ..< K * N {
		b[i] = f16(rng_f(&seed)*2.0 - 1.0)
	}
	return b
}

GemmDims :: struct { M, N, K: u32 }

run_kernel :: proc(pso: ^MTL.ComputePipelineState, bufA, bufB, bufC: ^MTL.Buffer, M, N, K, bm, bn, nt: int) {
	cmd := g_queue->commandBuffer()
	enc := cmd->computeCommandEncoder()
	enc->setComputePipelineState(pso)
	enc->setBuffer(bufA, 0, 0)
	enc->setBuffer(bufB, 0, 1)
	enc->setBuffer(bufC, 0, 2)
	dims := GemmDims{u32(M), u32(N), u32(K)}
	enc->setBytes(([^]u8)(&dims)[:size_of(GemmDims)], 3)
	enc->dispatchThreadgroups(MTL.Size{NS.Integer(N / bn), NS.Integer(M / bm), 1}, MTL.Size{NS.Integer(nt), 1, 1})
	enc->endEncoding()
	cmd->commit()
	cmd->waitUntilCompleted()
}

gate :: proc(pso: ^MTL.ComputePipelineState, c: Config, nt: int) -> (f64, bool) {
	M, N, K := 192, 64, 256
	nb := K / 256
	bs := c.q6 ? 210 : 144
	a_bytes := gen_quant(c.q6, M, K, 0xBEEF)
	b_h := gen_b(K, N, 0xF00D)
	defer delete(a_bytes)
	defer delete(b_h)
	a_ref := make([]f32, M * K)
	defer delete(a_ref)
	ggml.dequant_row(c.q6 ? .Q6_K : .Q4_K, a_bytes, M * K, a_ref)
	for i in 0 ..< M * K {
		a_ref[i] = f32(f16(a_ref[i]))
	}
	c_ref := make([]f64, M * N)
	defer delete(c_ref)
	for m in 0 ..< M {
		for n in 0 ..< N {
			s: f64 = 0
			for k in 0 ..< K {
				s += f64(a_ref[m*K + k]) * f64(b_h[k*N + n])
			}
			c_ref[m*N + n] = s
		}
	}
	bufA := g_dev->newBufferWithLength(NS.UInteger(M * nb * bs), MTL.ResourceStorageModeShared)
	bufB := g_dev->newBufferWithLength(NS.UInteger(K * N * 2), MTL.ResourceStorageModeShared)
	bufC := g_dev->newBufferWithLength(NS.UInteger(M * N * 4), MTL.ResourceStorageModeShared)
	defer bufA->release()
	defer bufB->release()
	defer bufC->release()
	copy(bufA->contentsAsSlice([]u8), a_bytes)
	dstb := bufB->contentsAsSlice([]u8)
	sb := ([^]u8)(raw_data(b_h))[:len(b_h) * 2]
	copy(dstb, sb)
	run_kernel(pso, bufA, bufB, bufC, M, N, K, c.bm, c.bn, nt)
	c_gpu := bufC->contentsAsSlice([]f32)[:M * N]
	max_abs, scale: f64 = 0, 0
	for i in 0 ..< M * N {
		d := math.abs(f64(c_gpu[i]) - c_ref[i])
		if d > max_abs { max_abs = d }
		a := math.abs(c_ref[i])
		if a > scale { scale = a }
	}
	scaled := scale > 1e-6 ? max_abs / scale : max_abs
	return scaled, scaled <= GATE_TOL
}

time_shape :: proc(pso: ^MTL.ComputePipelineState, c: Config, nt: int, s: Shape, a_bytes: []u8, b_h: []f16) -> (f64, f64) {
	nb := s.k / 256
	bs := c.q6 ? 210 : 144
	bufA := g_dev->newBufferWithLength(NS.UInteger(s.m * nb * bs), MTL.ResourceStorageModeShared)
	bufB := g_dev->newBufferWithLength(NS.UInteger(s.k * s.n * 2), MTL.ResourceStorageModeShared)
	bufC := g_dev->newBufferWithLength(NS.UInteger(s.m * s.n * 4), MTL.ResourceStorageModeShared)
	defer bufA->release()
	defer bufB->release()
	defer bufC->release()
	copy(bufA->contentsAsSlice([]u8), a_bytes)
	dstb := bufB->contentsAsSlice([]u8)
	sb := ([^]u8)(raw_data(b_h))[:len(b_h) * 2]
	copy(dstb, sb)
	for _ in 0 ..< 2 {
		run_kernel(pso, bufA, bufB, bufC, s.m, s.n, s.k, c.bm, c.bn, nt)
	}
	durs: [ITERS]time.Duration
	for i in 0 ..< ITERS {
		t0 := time.tick_now()
		run_kernel(pso, bufA, bufB, bufC, s.m, s.n, s.k, c.bm, c.bn, nt)
		durs[i] = time.tick_since(t0)
	}
	for i in 1 ..< ITERS {
		v := durs[i]
		j := i
		for j > 0 && durs[j - 1] > v {
			durs[j] = durs[j - 1]
			j -= 1
		}
		durs[j] = v
	}
	secs := f64(i64(durs[ITERS / 2])) / 1e9
	gflops := f64(2) * f64(s.m) * f64(s.n) * f64(s.k) / secs / 1e9
	return gflops, secs * 1e3
}

main :: proc() {
	g_dev = MTL.CreateSystemDefaultDevice()
	if g_dev == nil {
		fmt.eprintln("metal: no device")
		os.exit(1)
	}
	g_queue = g_dev->newCommandQueue()
	fmt.eprintfln("metal: {}", g_dev->name()->odinString())

	configs := []Config{
		{false, 32, 32, 64},
		{false, 32, 32, 128},
		{false, 16, 32, 64},
		{false, 32, 64, 64},
		{true, 16, 64, 64},
		{true, 16, 64, 32},
		{true, 8, 64, 64},
		{true, 32, 64, 64},
		{true, 16, 64, 128},
	}

	parts := make([dynamic]string)
	append(&parts, MSL_HEAD)
	names := make([]string, len(configs))
	nts := make([]int, len(configs))
	for c, i in configs {
		name, body, nt := kernel_src(c)
		names[i] = name
		nts[i] = nt
		append(&parts, body)
	}
	src, cerr := strings.concatenate(parts[:])
	if cerr != nil {
		fmt.eprintln("concat failed")
		os.exit(1)
	}
	src_ns := NS.String.alloc()->initWithOdinString(src)
	defer src_ns->release()
	opts := MTL.CompileOptions.alloc()->init()
	defer opts->release()
	opts->setFastMathEnabled(true)
	fmt.eprintln("compiling...")
	lib, err := g_dev->newLibraryWithSource(src_ns, opts)
	if err != nil {
		fmt.eprintfln("compile: {}", err->localizedDescription()->odinString())
		os.exit(1)
	}
	defer lib->release()
	fmt.eprintln("compiled")

	shapes_q4 := []Shape{
		{12288, 512, 4096, "w1-512"},
		{12288, 64, 4096, "w1-64"},
		{4096, 512, 12288, "down-512"},
		{8192, 512, 4096, "qkv-512"},
	}
	shapes_q6 := []Shape{
		{4096, 512, 12288, "w2-512"},
		{4096, 64, 12288, "w2-64"},
		{8192, 512, 4096, "qkv-512"},
		{1024, 512, 4096, "small-512"},
	}

	Bufs :: struct {
		a: []u8,
		b: []f16,
	}
	q4b := make([]Bufs, len(shapes_q4))
	q6b := make([]Bufs, len(shapes_q6))
	for s, i in shapes_q4 {
		q4b[i] = {gen_quant(false, s.m, s.k, 0xABCDE), gen_b(s.k, s.n, 0x12345)}
	}
	for s, i in shapes_q6 {
		q6b[i] = {gen_quant(true, s.m, s.k, 0xABCDE), gen_b(s.k, s.n, 0x12345)}
	}

	for c, i in configs {
		fun := lib->newFunctionWithName(NS.String.alloc()->initWithOdinString(names[i]))
		if fun == nil {
			fmt.eprintfln("FAIL {} missing", names[i])
			continue
		}
		pso, perr := g_dev->newComputePipelineStateWithFunction(fun)
		fun->release()
		if perr != nil {
			fmt.eprintfln("FAIL {} pipeline: {}", names[i], perr->localizedDescription()->odinString())
			continue
		}
		rel, ok := gate(pso, c, nts[i])
		tag := c.q6 ? "q6" : "q4"
		if !ok {
			fmt.eprintfln("FAIL {} bm={} bn={} bk={} nt={} rel={:.3e}", tag, c.bm, c.bn, c.bk, nts[i], rel)
			pso->release()
			continue
		}
		fmt.eprintfln("OK   {} bm={} bn={} bk={} nt={} rel={:.3e}", tag, c.bm, c.bn, c.bk, nts[i], rel)
		shapes := c.q6 ? shapes_q6 : shapes_q4
		bufs := c.q6 ? q6b : q4b
		for s, si in shapes {
			if s.m % c.bm != 0 || s.n % c.bn != 0 || s.k % c.bk != 0 {
				fmt.eprintfln("     {}  skip (alignment)", s.name)
				continue
			}
			gflops, ms := time_shape(pso, c, nts[i], s, bufs[si].a, bufs[si].b)
			fmt.eprintfln("     {}  {:.0f} GFLOPS  {:.2f} ms", s.name, gflops, ms)
		}
		pso->release()
	}
}
