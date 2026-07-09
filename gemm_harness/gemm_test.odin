#+build darwin

/* Standalone simdgroup-GEMM correctness/perf harness for BATCHED_PREFILL Stage 1a.
   Builds a Metal device, compiles gemm_f16_f32 (raw simdgroup 8x8 tile), runs a
   ladder of correctness tests (all-ones, identity, row/col-distinct, random vs
   CPU reference) and a timing pass. No GGUF / no engine needed.

   Run:  odin run gemm_test.odin -file -o:speed
*/

package gemm_test

import "core:fmt"
import "core:math"
import "core:os"
import "core:time"
import ggml "ggml:ggml"
import NS "core:sys/darwin/Foundation"
import MTL "vendor:darwin/Metal"

MSL_SRC :: `
#include <metal_stdlib>
using namespace metal;

struct GemmDims { uint M; uint N; uint K; };

// C[M,N] = A[M,K] @ B[K,N]. All row-major. M, N, K must be multiples of 8.
// One threadgroup = one simdgroup (32 threads) computes one 8x8 output tile.
//
// Layout convention (aligned to llama.cpp kernel_mul_mm):
//   - simdgroup_load/store stride is in ELEMENTS (the source row length).
//   - Origin is baked into the pointer (3-arg form: ptr already points at the
//     tile's top-left element; origin defaults to (0,0)).
//   - Accumulator is explicitly zero-filled via make_filled_simdgroup_matrix
//     (NOT default-constructed, which only inits the diagonal).
//   - K loop advances by 8 per iteration (one 8-wide K slice per MAC).
kernel void gemm_f16_f32(
    device const half  * A [[buffer(0)]],
    device const half  * B [[buffer(1)]],
    device       float * C [[buffer(2)]],
    constant GemmDims & dims [[buffer(3)]],
    uint2 tgpig [[threadgroup_position_in_grid]])
{
    const uint M = dims.M, N = dims.N, K = dims.K;
    const uint row0 = tgpig.y * 8;   // output row-tile origin (M dimension)
    const uint col0 = tgpig.x * 8;   // output col-tile origin (N dimension)

    simdgroup_half8x8  a_tile;
    simdgroup_half8x8  b_tile;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);

    device const half * a_base = A + (ulong)row0 * K;   // &A[row0][0]
    for (uint k = 0; k < K; k += 8) {
        // a_tile[i][j] = A[row0+i][k+j]   (8x8: M_tile rows x K_tile cols)
        simdgroup_load(a_tile, a_base + k, K);
        // b_tile[i][j] = B[k+i][col0+j]   (8x8: K_tile rows x N_tile cols)
        simdgroup_load(b_tile, B + (ulong)k * N + col0, N);
        // acc += a_tile * b_tile   (M_tile x K_tile) @ (K_tile x N_tile) = (M_tile x N_tile)
        simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
    }

    // C[row0+i][col0+j] = acc[i][j]
    simdgroup_store(acc, C + (ulong)row0 * N + col0, N);
}

// ===================== Q4_K MMQ (dequant fused into GEMM) =====================
// Matches the engine's dequant_q4_k (ggml/quant.odin) exactly, so the kernel's
// view of a Q4_K block is identical to the CPU reference (ggml.dequant_row).
constant uint QK_K = 256;
inline void k4_get_scale_min(int j, device const uchar * q, thread uchar & d, thread uchar & m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
// dequantize one element at index e (0..255) within a 144-byte Q4_K block
inline float k4_dequant_elem(device const uchar * blk, int e) {
    float d   = (float)(*(device const half *)(blk));
    float dmin = (float)(*(device const half *)(blk + 2));
    device const uchar * scales = blk + 4;
    device const uchar * qs = blk + 16;
    int sup = e >> 6;     // e / 64
    int ws  = e & 63;     // e % 64
    int hf = ws >> 5;     // ws / 32
    int l   = ws & 31;    // ws % 32
    int is  = sup * 2 + hf;
    int qoff = sup * 32;
    uchar sc, m;
    k4_get_scale_min(is, scales, sc, m);
    float dl = d * (float)sc;
    float ml = dmin * (float)m;
    uchar nib = (qs[qoff + l] >> (4 * hf)) & 0xF;
    return dl * (float)nib - ml;
}

// C[M,N] = A[M,K] @ B[K,N]. A is Q4_K (row r at A + r*(K/256)*144). B is F16, C F32.
// 1 simdgroup / threadgroup, 8x8 output tile. Each K-iter (k+=8): 32 threads
// cooperatively dequant the 8x8 A tile into threadgroup memory, then MAC.
kernel void gemm_q4k_f32(
    device const uchar * A [[buffer(0)]],
    device const half  * B [[buffer(1)]],
    device       float * C [[buffer(2)]],
    constant GemmDims & dims [[buffer(3)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    const uint M = dims.M, N = dims.N, K = dims.K;
    const uint row0 = tgpig.y * 8;
    const uint col0 = tgpig.x * 8;
    const ulong row_bytes = (ulong)(K / 256) * 144;

    threadgroup half sa[64]; // 8x8 A tile, row-major stride 8
    simdgroup_half8x8  a_tile, b_tile;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (uint k = 0; k < K; k += 8) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // 64 elements, 32 threads -> 2 per thread
        for (uint t = tid; t < 64; t += 32) {
            uint i = t >> 3;   // row 0..7
            uint j = t & 7;    // col 0..7
            uint e = k + j;    // global K index within row
            device const uchar * blk = A + (ulong)(row0 + i) * row_bytes + (uint)(e / 256) * 144;
            sa[t] = (half)k4_dequant_elem(blk, e & 255);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(a_tile, sa, 8);
        simdgroup_load(b_tile, B + (ulong)k * N + col0, N);
        simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
    }
    simdgroup_store(acc, C + (ulong)row0 * N + col0, N);
}

// ===================== Q6_K MMQ (ffn_down / output / qkv are Q6_K) =====================
inline float k6_dequant_elem(device const uchar * blk, int e) {
    float d = (float)(*(device const half *)(blk + 208));
    device const uchar * ql = blk;
    device const uchar * qh = blk + 128;
    device const char  * sc = (device const char *)(blk + 192);
    int hf = e >> 7, within = e & 127, l = within & 31, sub = within >> 5, is = l >> 4;
    int qloff = hf * 64, qhoff = hf * 32, scoff = hf * 8;
    int q, scidx;
    if (sub == 0)      { q = ((ql[qloff + l] & 0xF)      | (((qh[qhoff + l] >> 0) & 3) << 4)) - 32; scidx = scoff + is + 0; }
    else if (sub == 1) { q = ((ql[qloff + l + 32] & 0xF) | (((qh[qhoff + l] >> 2) & 3) << 4)) - 32; scidx = scoff + is + 2; }
    else if (sub == 2) { q = ((ql[qloff + l] >> 4)       | (((qh[qhoff + l] >> 4) & 3) << 4)) - 32; scidx = scoff + is + 4; }
    else               { q = ((ql[qloff + l + 32] >> 4)  | (((qh[qhoff + l] >> 6) & 3) << 4)) - 32; scidx = scoff + is + 6; }
    return d * (float)sc[scidx] * (float)q;
}
kernel void gemm_q6k_f32(
    device const uchar * A [[buffer(0)]], device const half * B [[buffer(1)]],
    device float * C [[buffer(2)]], constant GemmDims & dims [[buffer(3)]],
    uint2 tgpig [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint M = dims.M, N = dims.N, K = dims.K;
    const uint row0 = tgpig.y * 8, col0 = tgpig.x * 8;
    const ulong row_bytes = (ulong)(K / 256) * 210;
    threadgroup half sa[64];
    simdgroup_half8x8 a_tile, b_tile;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint k = 0; k < K; k += 8) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t = tid; t < 64; t += 32) {
            uint i = t >> 3, j = t & 7, e = k + j;
            device const uchar * blk = A + (ulong)(row0 + i) * row_bytes + (uint)(e / 256) * 210;
            sa[t] = (half)k6_dequant_elem(blk, e & 255);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(a_tile, sa, 8);
        simdgroup_load(b_tile, B + (ulong)k * N + col0, N);
        simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
    }
    simdgroup_store(acc, C + (ulong)row0 * N + col0, N);
}
// 8-wide K slice). Used only to PROVE this is the old 2048 bug.
kernel void gemm_f16_f32_kstep16(
    device const half  * A [[buffer(0)]],
    device const half  * B [[buffer(1)]],
    device       float * C [[buffer(2)]],
    constant GemmDims & dims [[buffer(3)]],
    uint2 tgpig [[threadgroup_position_in_grid]])
{
    const uint M = dims.M, N = dims.N, K = dims.K;
    const uint row0 = tgpig.y * 8, col0 = tgpig.x * 8;
    simdgroup_half8x8 a_tile, b_tile;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
    device const half * a_base = A + (ulong)row0 * K;
    for (uint k = 0; k < K; k += 16) {                 // <-- bug: 16 instead of 8
        simdgroup_load(a_tile, a_base + k, K);
        simdgroup_load(b_tile, B + (ulong)k * N + col0, N);
        simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
    }
    simdgroup_store(acc, C + (ulong)row0 * N + col0, N);
}
`

// ---------- Metal state ----------
@(private = "file")
g_dev: ^MTL.Device
@(private = "file")
g_queue: ^MTL.CommandQueue
@(private = "file")
g_pso: ^MTL.ComputePipelineState
@(private = "file")
g_pso_k16: ^MTL.ComputePipelineState  // diagnostic: K-step 16
@(private = "file")
g_pso_q4k: ^MTL.ComputePipelineState  // Q4_K MMQ
g_pso_q6k: ^MTL.ComputePipelineState  // Q6_K MMQ

GemmDims :: struct {
	M, N, K: u32,
}

@(private = "file")
rng_f :: proc(seed: ^u32) -> f32 {
	// xorshift32 -> [0,1)
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
copy_f32 :: proc(buf: ^MTL.Buffer, src: []f32) {
	dst := buf->contentsAsSlice([]u8)
	sb := ([^]u8)(raw_data(src))[: len(src) * 4]
	copy(dst, sb)
}

@(private = "file")
bytes_of :: proc(p: rawptr, n: int) -> []u8 {
	return ([^]u8)(p)[:n]
}

setup_metal :: proc() -> bool {
	g_dev = MTL.CreateSystemDefaultDevice()
	if g_dev == nil {
		fmt.eprintln("metal: no device")
		return false
	}
	fmt.printf("metal: %s\n", g_dev->name()->odinString())
	g_queue = g_dev->newCommandQueue()
	src := NS.String.alloc()->initWithOdinString(MSL_SRC)
	defer src->release()
	lib, err := g_dev->newLibraryWithSource(src, nil)
	if err != nil {
		fmt.eprintfln("metal: shader compile failed: %s", err->localizedDescription()->odinString())
		return false
	}
	defer lib->release()
	fn_ := lib->newFunctionWithName(NS.String.alloc()->initWithOdinString("gemm_f16_f32"))
	if fn_ == nil {
		fmt.eprintln("metal: kernel 'gemm_f16_f32' not found")
		return false
	}
	defer fn_->release()
	pso, perr := g_dev->newComputePipelineStateWithFunction(fn_)
	if perr != nil {
		fmt.eprintfln("metal: pipeline failed: %s", perr->localizedDescription()->odinString())
		return false
	}
	g_pso = pso
	// diagnostic kernel
	fn2 := lib->newFunctionWithName(NS.String.alloc()->initWithOdinString("gemm_f16_f32_kstep16"))
	defer fn2->release()
	pso2, err2 := g_dev->newComputePipelineStateWithFunction(fn2)
	if err2 != nil {
		fmt.eprintfln("metal: k16 pipeline failed: %s", err2->localizedDescription()->odinString())
		return false
	}
	g_pso_k16 = pso2
	// Q4_K MMQ kernel
	fn3 := lib->newFunctionWithName(NS.String.alloc()->initWithOdinString("gemm_q4k_f32"))
	defer fn3->release()
	pso3, err3 := g_dev->newComputePipelineStateWithFunction(fn3)
	if err3 != nil {
		fmt.eprintfln("metal: q4k pipeline failed: %s", err3->localizedDescription()->odinString())
		return false
	}
	g_pso_q4k = pso3
	fn4 := lib->newFunctionWithName(NS.String.alloc()->initWithOdinString("gemm_q6k_f32"))
	defer fn4->release()
	pso4, err4 := g_dev->newComputePipelineStateWithFunction(fn4)
	if err4 != nil {
		fmt.eprintfln("metal: q6k pipeline failed: %s", err4->localizedDescription()->odinString())
		return false
	}
	g_pso_q6k = pso4
	return true
}

// ---------- data generation ----------
Fill_Mode :: enum { AllOnes, Identity_A, Row_Distinct, Col_Distinct, Random }

// Fill A[M*K] and B[K*N] (host f16) + matching f32 reference values used by CPU
// reference (so quantization rounding is shared by both sides; remaining error
// is purely kernel correctness).
fill :: proc(a_h, b_h: []f16, a_ref, b_ref: []f32, M, N, K: int, mode: Fill_Mode, seed: ^u32) {
	rng :: proc(seed: ^u32) -> f32 {
		// xorshift32 -> [0,1)
		x := seed^
		x = x ~ (x << 13)
		x = x ~ (x >> 17)
		x = x ~ (x << 5)
		seed^ = x
		return f32((x >> 8) & 0xFFFFFF) / f32(0x1000000)
	}
	for i in 0 ..< M * K {
		v: f32 = 0
		switch mode {
		case .AllOnes: v = 1.0
		case .Identity_A:
			m := i / K
			k := i % K
			v = m == k ? 1.0 : 0.0
		case .Row_Distinct:
			m := i / K
			v = f32(m) + 1.0
		case .Col_Distinct:
			v = 1.0
		case .Random: v = rng(seed) * 2.0 - 1.0
		}
		a_h[i] = f16(v)
		a_ref[i] = f32(a_h[i])
	}
	for i in 0 ..< K * N {
		v: f32 = 0
		switch mode {
		case .AllOnes: v = 1.0
		case .Identity_A:
			k := i / N
			n := i % N
			v = f32(k * N + n + 1) * 0.0625 // distinct per (k,n)
		case .Row_Distinct: v = 1.0
		case .Col_Distinct:
			n := i % N
			v = f32(n) + 1.0
		case .Random: v = rng(seed) * 2.0 - 1.0
		}
		b_h[i] = f16(v)
		b_ref[i] = f32(b_h[i])
	}
}

// CPU reference C_ref[M*N] = A[M,K] @ B[K,N] computed in f64.
cpu_ref :: proc(c_ref: []f64, a_ref, b_ref: []f32, M, N, K: int) {
	for m in 0 ..< M {
		for n in 0 ..< N {
			s: f64 = 0
			for k in 0 ..< K {
				s += f64(a_ref[m*K+k]) * f64(b_ref[k*N+n])
			}
			c_ref[m*N+n] = s
		}
	}
}

// ---------- dispatch ----------
run_gemm :: proc(bufA, bufB, bufC: ^MTL.Buffer, M, N, K: int, pso: ^MTL.ComputePipelineState = g_pso) {
	cmd := g_queue->commandBuffer()
	enc := cmd->computeCommandEncoder()
	enc->setComputePipelineState(pso)
	enc->setBuffer(bufA, 0, 0)
	enc->setBuffer(bufB, 0, 1)
	enc->setBuffer(bufC, 0, 2)
	dims := GemmDims{u32(M), u32(N), u32(K)}
	enc->setBytes(bytes_of(&dims, size_of(GemmDims)), 3)
	grid := MTL.Size{NS.Integer(N / 8), NS.Integer(M / 8), 1}
	per := MTL.Size{32, 1, 1} // one simdgroup = 32 threads
	enc->dispatchThreadgroups(grid, per)
	enc->endEncoding()
	cmd->commit()
	cmd->waitUntilCompleted()
}

@(private = "file")
new_buf :: proc(n_bytes: int) -> ^MTL.Buffer {
	return g_dev->newBufferWithLength(NS.UInteger(n_bytes), MTL.ResourceStorageModeShared)
}

// ---------- one correctness test ----------
correctness_test :: proc(name: string, M, N, K: int, mode: Fill_Mode, tol: f64) -> bool {
	a_h := make([]f16, M*K)
	b_h := make([]f16, K*N)
	a_ref := make([]f32, M*K)
	b_ref := make([]f32, K*N)
	defer delete(a_h); defer delete(b_h); defer delete(a_ref); defer delete(b_ref)

	seed := u32(0xC0FFEE) ~ (u32(M) * 2654435761) ~ (u32(N) * 40503) ~ (u32(K) * 7919)
	fill(a_h, b_h, a_ref, b_ref, M, N, K, mode, &seed)

	bufA := new_buf(M * K * 2)
	bufB := new_buf(K * N * 2)
	bufC := new_buf(M * N * 4)
	defer bufA->release(); defer bufB->release(); defer bufC->release()

	copy_to(bufA, a_h); copy_to(bufB, b_h)

	run_gemm(bufA, bufB, bufC, M, N, K)
	c_gpu := bufC->contentsAsSlice([]f32)[: M * N]

	// build reference
	if mode == .AllOnes {
		expected := f32(K)
		max_abs: f64 = 0
		bad := 0
		for i in 0 ..< M * N {
			d := f64(math.abs(c_gpu[i] - expected))
			if d > max_abs {
				max_abs = d
			}
			if d > tol {
				bad += 1
			}
		}
		ok := max_abs <= tol && bad == 0
		fmt.printfln("  [{}] {} M={} N={} K={}  expect={}  max_abs={:.4e} bad={}", ok ? "OK" : "FAIL", name, M, N, K, expected, max_abs, bad)
		return ok
	}

	c_ref := make([]f64, M*N)
	defer delete(c_ref)
	cpu_ref(c_ref, a_ref, b_ref, M, N, K)

	max_abs: f64 = 0
	max_rel: f64 = 0
	worst_i := 0
	for i in 0 ..< M * N {
		d := math.abs(f64(c_gpu[i]) - c_ref[i])
		if d > max_abs {
			max_abs = d
			worst_i = i
		}
		ref := math.abs(c_ref[i])
		if ref > 1e-6 {
			rel := d / ref
			if rel > max_rel {
				max_rel = rel
			}
		}
	}
	ok := max_abs <= tol
	fmt.printfln("  [{}] {} M={} N={} K={}  max_abs={:.4e} max_rel={:.4e} (gpu={:.5f} ref={:.5f} @{})", ok ? "OK" : "FAIL", name, M, N, K, max_abs, max_rel, c_gpu[worst_i], f32(c_ref[worst_i]), worst_i)
	return ok
}

@(private = "file")
copy_to :: proc(buf: ^MTL.Buffer, src: []f16) {
	dst := buf->contentsAsSlice([]u8)
	sb := ([^]u8)(raw_data(src))[: len(src) * 2]
	for i in 0 ..< len(sb) {
		dst[i] = sb[i]
	}
}

// ---------- perf test (timing only, all-ones so correctness is self-evident) ----------
perf_test :: proc(name: string, M, N, K: int, iters: int) {
	a_h := make([]f16, M*K)
	b_h := make([]f16, K*N)
	for i in 0 ..< M * K {
		a_h[i] = 1.0
	}
	for i in 0 ..< K * N {
		b_h[i] = 1.0
	}
	bufA := new_buf(M * K * 2)
	bufB := new_buf(K * N * 2)
	bufC := new_buf(M * N * 4)
	defer bufA->release(); defer bufB->release(); defer bufC->release()
	copy_to(bufA, a_h); copy_to(bufB, b_h)

	// warmup
	for _ in 0 ..< 3 {
		run_gemm(bufA, bufB, bufC, M, N, K)
	}
	c_gpu := bufC->contentsAsSlice([]f32)[: M * N]
	sample := c_gpu[0]
	expected := f32(K)
	fmt.printfln("    correctness spot-check: C[0]={:.1f} (expect {})", sample, expected)

	// timed
	best: time.Duration = cast(time.Duration)0x7FFFFFFFFFFFFFFF
	for _ in 0 ..< iters {
		t0 := time.tick_now()
		run_gemm(bufA, bufB, bufC, M, N, K)
		dt := time.tick_since(t0)
		if dt < best {
			best = dt
		}
	}
	flops := f64(2) * f64(M) * f64(N) * f64(K)
	secs := f64(i64(best)) / 1e9
	gflops := flops / secs / 1e9
	fmt.printfln("  [PERF] {} M={} N={} K={}  best={:.3f} ms  {:.0f} GFLOPS", name, M, N, K, secs * 1e3, gflops)
}

// ---------- root-cause confirmation: prove K-step-16 yields 2048 on all-ones ----------
confirm_k16_bug :: proc() {
	M, N, K := 4096, 64, 4096
	a_h := make([]f16, M*K); b_h := make([]f16, K*N)
	for i in 0 ..< M*K { a_h[i] = 1.0 }
	for i in 0 ..< K*N { b_h[i] = 1.0 }
	bufA := new_buf(M*K*2); bufB := new_buf(K*N*2); bufC := new_buf(M*N*4)
	defer bufA->release(); defer bufB->release(); defer bufC->release()
	copy_to(bufA, a_h); copy_to(bufB, b_h)
	run_gemm(bufA, bufB, bufC, M, N, K, g_pso_k16)
	c := bufC->contentsAsSlice([]f32)[:M*N]
	first := c[0]
	expected_if_bug := f32(K / 2) // 2048
	ok := first == expected_if_bug
	fmt.printfln("  [{}] k+=16 variant on 4096^3x64 all-ones: C[0]={:.1f}  (old bug predicts {})  => {}",
		ok ? "CONFIRMED" : "SURPRISE", first, expected_if_bug,
		ok ? "K-step-16 is the root cause of the 2048 bug" : "k16 did NOT reproduce 2048; investigate")
}

// ---------- Q4_K MMQ: build random Q4_K blocks, CPU-ref via ggml.dequant_row ----------
@(private = "file")
run_q4k :: proc(bufA, bufB, bufC: ^MTL.Buffer, M, N, K: int) {
	cmd := g_queue->commandBuffer()
	enc := cmd->computeCommandEncoder()
	enc->setComputePipelineState(g_pso_q4k)
	enc->setBuffer(bufA, 0, 0)
	enc->setBuffer(bufB, 0, 1)
	enc->setBuffer(bufC, 0, 2)
	dims := GemmDims{u32(M), u32(N), u32(K)}
	enc->setBytes(bytes_of(&dims, size_of(GemmDims)), 3)
	enc->dispatchThreadgroups(MTL.Size{NS.Integer(N / 8), NS.Integer(M / 8), 1}, MTL.Size{32, 1, 1})
	enc->endEncoding()
	cmd->commit()
	cmd->waitUntilCompleted()
}

q4k_test :: proc(name: string, M, N, K: int, seed_in: u32, tol: f64) -> bool {
	nb := K / 256
	a_bytes := make([]u8, M * nb * 144)
	b_h := make([]f16, K * N)
	defer delete(a_bytes); defer delete(b_h)
	seed := seed_in
	// random Q4_K blocks: d in [0.02,0.2], dmin in [0,0.05], random scales/qs
	for b in 0 ..< M * nb {
		base := b * 144
		write_f16_le(a_bytes, base, rng_f(&seed)*0.18 + 0.02)
		write_f16_le(a_bytes, base + 2, rng_f(&seed)*0.05)
		for s in 0 ..< 12 {
			a_bytes[base + 4 + s] = u8(rng_f(&seed) * 256.0)
		}
		for q in 0 ..< 128 {
			a_bytes[base + 16 + q] = u8(rng_f(&seed) * 256.0)
		}
	}
	for i in 0 ..< K * N {
		b_h[i] = f16(rng_f(&seed)*2.0 - 1.0)
	}

	// CPU reference: dequant A to f32, round to half (matches MMQ sa staging),
	// then matmul in f64. Isolates kernel bugs from half-rounding noise.
	a_ref := make([]f32, M * K)
	defer delete(a_ref)
	ggml.dequant_row(.Q4_K, a_bytes, M * K, a_ref)
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

	bufA := new_buf(M * nb * 144)
	bufB := new_buf(K * N * 2)
	bufC := new_buf(M * N * 4)
	defer bufA->release(); defer bufB->release(); defer bufC->release()
	copy_bytes(bufA, a_bytes)
	copy_to(bufB, b_h)
	run_q4k(bufA, bufB, bufC, M, N, K)
	c_gpu := bufC->contentsAsSlice([]f32)[: M * N]

	max_abs: f64 = 0
	max_rel: f64 = 0
	scale: f64 = 0
	worst := 0
	for i in 0 ..< M * N {
		d := math.abs(f64(c_gpu[i]) - c_ref[i])
		if d > max_abs {
			max_abs = d
			worst = i
		}
		a := math.abs(c_ref[i])
		if a > scale {
			scale = a
		}
		ref := a
		if ref > 1e-3 {
			rel := d / ref
			if rel > max_rel {
				max_rel = rel
			}
		}
	}
	// meaningful metric: error scaled by output magnitude (per-element relative
	// blows up near zero outputs; absolute is meaningless across scales).
	scaled := scale > 1e-6 ? max_abs / scale : max_abs
	ok := scaled <= tol
	fmt.printfln("  [{}] {} M={} N={} K={}  max_abs={:.4e} scaled={:.4e} (max_rel_nonzero={:.4e}, out_scale={:.1f})", ok ? "OK" : "FAIL", name, M, N, K, max_abs, scaled, max_rel, scale)
	return ok
}

q4k_perf :: proc(name: string, M, N, K: int, iters: int) {
	nb := K / 256
	a_bytes := make([]u8, M * nb * 144)
	b_h := make([]f16, K * N)
	defer delete(a_bytes); defer delete(b_h)
	seed := u32(0xABCDE)
	for b in 0 ..< M * nb {
		base := b * 144
		write_f16_le(a_bytes, base, rng_f(&seed)*0.18 + 0.02)
		write_f16_le(a_bytes, base + 2, rng_f(&seed)*0.05)
		for s in 0 ..< 12 {
			a_bytes[base + 4 + s] = u8(rng_f(&seed) * 256.0)
		}
		for q in 0 ..< 128 {
			a_bytes[base + 16 + q] = u8(rng_f(&seed) * 256.0)
		}
	}
	for i in 0 ..< K * N {
		b_h[i] = f16(rng_f(&seed)*2.0 - 1.0)
	}
	bufA := new_buf(M * nb * 144); bufB := new_buf(K * N * 2); bufC := new_buf(M * N * 4)
	defer bufA->release(); defer bufB->release(); defer bufC->release()
	copy_bytes(bufA, a_bytes); copy_to(bufB, b_h)
	for _ in 0 ..< 3 {
		run_q4k(bufA, bufB, bufC, M, N, K)
	}
	best: time.Duration = cast(time.Duration)0x7FFFFFFFFFFFFFFF
	for _ in 0 ..< iters {
		t0 := time.tick_now()
		run_q4k(bufA, bufB, bufC, M, N, K)
		dt := time.tick_since(t0)
		if dt < best {
			best = dt
		}
	}
	flops := f64(2) * f64(M) * f64(N) * f64(K)
	secs := f64(i64(best)) / 1e9
	gflops := flops / secs / 1e9
	fmt.printfln("  [PERF] {} M={} N={} K={}  best={:.3f} ms  {:.0f} GFLOPS", name, M, N, K, secs * 1e3, gflops)
}

// ---------- Q6_K MMQ (ffn_down w2 is Q6_K in Q4_K_M) ----------
@(private = "file")
run_q6k :: proc(bufA, bufB, bufC: ^MTL.Buffer, M, N, K: int) {
	cmd := g_queue->commandBuffer()
	enc := cmd->computeCommandEncoder()
	enc->setComputePipelineState(g_pso_q6k)
	enc->setBuffer(bufA, 0, 0); enc->setBuffer(bufB, 0, 1); enc->setBuffer(bufC, 0, 2)
	dims := GemmDims{u32(M), u32(N), u32(K)}
	enc->setBytes(bytes_of(&dims, size_of(GemmDims)), 3)
	enc->dispatchThreadgroups(MTL.Size{NS.Integer(N / 8), NS.Integer(M / 8), 1}, MTL.Size{32, 1, 1})
	enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted()
}

q6k_test :: proc(name: string, M, N, K: int, seed_in: u32, tol: f64) -> bool {
	nb := K / 256
	a_bytes := make([]u8, M * nb * 210)
	b_h := make([]f16, K * N)
	defer delete(a_bytes); defer delete(b_h)
	seed := seed_in
	for b in 0 ..< M * nb {
		base := b * 210
		write_f16_le(a_bytes, base + 208, rng_f(&seed)*0.18 + 0.02) // d
		for i in 0 ..< 128 { a_bytes[base + i] = u8(rng_f(&seed) * 256.0) }       // ql
		for i in 0 ..< 64 { a_bytes[base + 128 + i] = u8(rng_f(&seed) * 256.0) }  // qh
		for i in 0 ..< 16 { a_bytes[base + 192 + i] = u8(rng_f(&seed) * 256.0) }  // sc (int8)
	}
	for i in 0 ..< K * N { b_h[i] = f16(rng_f(&seed)*2.0 - 1.0) }
	a_ref := make([]f32, M * K)
	defer delete(a_ref)
	ggml.dequant_row(.Q6_K, a_bytes, M * K, a_ref)
	for i in 0 ..< M * K { a_ref[i] = f32(f16(a_ref[i])) } // match MMQ half staging
	b_ref := make([]f32, K * N)
	defer delete(b_ref)
	for i in 0 ..< K * N { b_ref[i] = f32(b_h[i]) }
	c_ref := make([]f64, M * N)
	defer delete(c_ref)
	for m in 0 ..< M {
		for n in 0 ..< N {
			s: f64 = 0
			for k in 0 ..< K { s += f64(a_ref[m*K+k]) * f64(b_ref[k*N+n]) }
			c_ref[m*N+n] = s
		}
	}
	bufA := new_buf(M * nb * 210); bufB := new_buf(K * N * 2); bufC := new_buf(M * N * 4)
	defer bufA->release(); defer bufB->release(); defer bufC->release()
	copy_bytes(bufA, a_bytes); copy_to(bufB, b_h)
	run_q6k(bufA, bufB, bufC, M, N, K)
	c_gpu := bufC->contentsAsSlice([]f32)[: M * N]
	max_abs, scale: f64 = 0, 0
	for i in 0 ..< M * N {
		d := math.abs(f64(c_gpu[i]) - c_ref[i])
		if d > max_abs { max_abs = d }
		a := math.abs(c_ref[i])
		if a > scale { scale = a }
	}
	scaled := scale > 1e-6 ? max_abs / scale : max_abs
	ok := scaled <= tol
	fmt.printfln("  [{}] {} M={} N={} K={}  max_abs={:.4e} scaled={:.4e} (out_scale={:.1f})", ok ? "OK" : "FAIL", name, M, N, K, max_abs, scaled, scale)
	return ok
}

main :: proc() {
	if !setup_metal() {
		os.exit(1)
	}

	fmt.println("\n=== Stage 1a correctness ladder (gemm_f16_f32, raw simdgroup 8x8) ===")
	all_ok := true
	all_ok = correctness_test("Step1: 8x8 all-ones -> 8", 8, 8, 8, .AllOnes, 1e-4) && all_ok
	all_ok = correctness_test("Step2: 16x16 all-ones -> 16 (K loop)", 16, 16, 16, .AllOnes, 1e-4) && all_ok
	all_ok = correctness_test("Step3a: 8x8 identity(A) x seq(B)", 8, 8, 8, .Identity_A, 1e-2) && all_ok
	all_ok = correctness_test("Step3b: 8x8 row-distinct(A) (A not transposed)", 8, 8, 8, .Row_Distinct, 1e-2) && all_ok
	all_ok = correctness_test("Step3c: 8x8 col-distinct(B) (B not transposed)", 8, 8, 8, .Col_Distinct, 1e-2) && all_ok
	all_ok = correctness_test("Step4: 4096x4096 x 4096x64 all-ones -> 4096", 4096, 64, 4096, .AllOnes, 1e-3) && all_ok
	all_ok = correctness_test("random 32x32x32 vs CPU ref", 32, 32, 32, .Random, 1e-2) && all_ok
	all_ok = correctness_test("random 64x64x256 vs CPU ref", 64, 64, 256, .Random, 1e-2) && all_ok

	fmt.println("\n=== root-cause confirmation: K-step-16 reproduces the old 2048 bug ===")
	confirm_k16_bug()

	fmt.println("\n=== Q4_K MMQ correctness (dequant fused into GEMM; CPU ref = ggml.dequant_row) ===")
	q_ok := true
	q_ok = q4k_test("Q4K 8x8x256 (1 block/row)", 8, 8, 256, 0x111, 2e-3) && q_ok
	q_ok = q4k_test("Q4K 16x16x512", 16, 16, 512, 0x222, 2e-3) && q_ok
	q_ok = q4k_test("Q4K 32x32x256", 32, 32, 256, 0x333, 2e-3) && q_ok
	q_ok = q4k_test("Q4K 64x64x512", 64, 64, 512, 0x444, 2e-3) && q_ok
	q_ok = q4k_test("Q4K 256x64x4096 (proj-ish)", 256, 64, 4096, 0x555, 5e-3) && q_ok
	q_ok = q4k_test("Q4K 4096x64x4096 (real proj shape)", 4096, 64, 4096, 0x666, 1e-2) && q_ok
	all_ok = all_ok && q_ok

	fmt.println("\n=== Q6_K MMQ correctness (ffn_down w2 is Q6_K) ===")
	q6 := true
	q6 = q6k_test("Q6K 8x8x256", 8, 8, 256, 0x701, 2e-3) && q6
	q6 = q6k_test("Q6K 64x64x512", 64, 64, 512, 0x702, 2e-3) && q6
	q6 = q6k_test("Q6K 4096x64x12288 (w2 shape)", 4096, 64, 12288, 0x703, 1e-2) && q6
	all_ok = all_ok && q6

	fmt.println("\n=== Q4_K MMQ perf ===")
	q4k_perf("real prefill projection", 4096, 64, 4096, 10)

	fmt.println("\n=== perf (Stage 1a F16 target >= 300 GFLOPS) ===")
	perf_test("prefill projection shape", 4096, 64, 4096, 20)
	perf_test("square 4096x4096x4096", 4096, 4096, 4096, 5)

	fmt.printfln("\n=== RESULT: {} ===", all_ok ? "ALL CORRECTNESS TESTS PASSED" : "FAILURES PRESENT")
	if !all_ok {
		os.exit(1)
	}
}
