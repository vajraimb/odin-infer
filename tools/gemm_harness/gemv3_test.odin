#+build darwin
package gemv3_test

import "core:fmt"
import "core:math"
import "core:os"
import NS "core:sys/darwin/Foundation"
import MTL "vendor:darwin/Metal"

MSL_SRC :: `
#include <metal_stdlib>
using namespace metal;

struct Dims { uint n; uint d; };

constant uint GEMV3_NXPSG = 8;    // threads cooperating on one row
constant uint GEMV3_R1PTG = 2;    // rows per thread
constant uint GEMV3_ROWS  = 64;   // rows per TG = 8 simdgroups * (32/8) * 2

inline void get_scale_min_k4(int j, device const uchar *q, thread uchar &d, thread uchar &m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j + 0] >> 6) << 4); }
}


inline void deq_q4_k_chunk16(device const uchar *blk, uint il, thread float4 reg[4]) {
    device const uchar *qs = blk + 16;
    device const uchar *scales = blk + 4;
    float dall = (float)(*(device const half*)blk);
    float dmin = (float)(*(device const half*)(blk + 2));
    uchar sc, m; get_scale_min_k4((int)(il >> 1), scales, sc, m);
    float d = dall * sc, mn = dmin * m;
    uint qo = (il >> 2) * 32 + (il & 1) * 16;
    if ((il & 3) < 2) {
        for (uint j = 0; j < 16; j += 4) {
            reg[j/4] = float4((float)(qs[qo+j]&0xF)*d - mn, (float)(qs[qo+j+1]&0xF)*d - mn,
                              (float)(qs[qo+j+2]&0xF)*d - mn, (float)(qs[qo+j+3]&0xF)*d - mn);
        }
    } else {
        for (uint j = 0; j < 16; j += 4) {
            reg[j/4] = float4((float)(qs[qo+j]>>4)*d - mn, (float)(qs[qo+j+1]>>4)*d - mn,
                              (float)(qs[qo+j+2]>>4)*d - mn, (float)(qs[qo+j+3]>>4)*d - mn);
        }
    }
}


inline void deq_q6_k_chunk16(device const uchar *blk, uint il, thread float4 reg[4]) {
    // 16-element chunk c of half hf (il = 8*hf + c). Element r = 16c+j of the half:
    //   ql byte = (r < 64) ? r : r-64, nibble = low (r<64) / high (r>=64)
    //   qh byte = 32*hf + 16*(c&1) + j, 2-bit field shift = (c & 6)
    //   value = ((nibble | (qh_bits << 4)) - 32) * dall * scales[8*hf + c]
    device const uchar *ql = blk;
    device const uchar *qh = blk + 128;
    device const char *scales = (device const char *)(blk + 192);
    float dall = (float)(*(device const half*)(blk + 208));
    uint hf = il >> 3, c = il & 7;
    float dsc = dall * (float)scales[8*hf + c];
    uint qhshift = (c & 6);
    int qloff = 64 * (int)hf + 16 * (int)c - ((c >= 4) ? 64 : 0);
    uint qhoff = 32 * hf + 16 * (c & 1);
    bool hi = c >= 4;
    for (uint j = 0; j < 16; j += 4) {
        float4 hb = float4((float)((qh[qhoff+j] >> qhshift) & 3), (float)((qh[qhoff+j+1] >> qhshift) & 3),
                           (float)((qh[qhoff+j+2] >> qhshift) & 3), (float)((qh[qhoff+j+3] >> qhshift) & 3));
        float4 lb;
        if (!hi) {
            lb = float4((float)(ql[qloff+j]&0xF), (float)(ql[qloff+j+1]&0xF), (float)(ql[qloff+j+2]&0xF), (float)(ql[qloff+j+3]&0xF));
        } else {
            lb = float4((float)(ql[qloff+j]>>4), (float)(ql[qloff+j+1]>>4), (float)(ql[qloff+j+2]>>4), (float)(ql[qloff+j+3]>>4));
        }
        reg[j/4] = (lb + hb * 16.0f - 32.0f) * dsc;
    }
}


kernel void gemv_q4_k_v3(device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant Dims &dim [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint tx = lane % GEMV3_NXPSG, ty = lane / GEMV3_NXPSG;
    uint row0 = tg * GEMV3_ROWS + sg * ((32 / GEMV3_NXPSG) * GEMV3_R1PTG) + ty * GEMV3_R1PTG;
    if (row0 >= dim.d) return;
    uint nb = dim.n / 256;
    device const uchar *wr0 = wb + (ulong)row0 * nb * 144;
    device const uchar *wr1 = (row0 + 1 < dim.d) ? wr0 + nb * 144 : wr0;
    device const float4 *y4 = (device const float4 *)x + tx * 4;
    float sum0 = 0.0, sum1 = 0.0;
    for (uint ich = tx; 16*ich < dim.n; ich += GEMV3_NXPSG) {
        uint b = ich >> 4, il = ich & 15;
        float4 r0[4], r1[4];
        deq_q4_k_chunk16(wr0 + b * 144, il, r0);
        deq_q4_k_chunk16(wr1 + b * 144, il, r1);
        float4 y0 = y4[0], y1 = y4[1], y2 = y4[2], y3 = y4[3];
        sum0 += dot(r0[0],y0) + dot(r0[1],y1) + dot(r0[2],y2) + dot(r0[3],y3);
        sum1 += dot(r1[0],y0) + dot(r1[1],y1) + dot(r1[2],y2) + dot(r1[3],y3);
        y4 += 32;   // NXPSG * 4 float4 per step
    }
    sum0 += simd_shuffle_down(sum0, 4);
    sum0 += simd_shuffle_down(sum0, 2);
    sum0 += simd_shuffle_down(sum0, 1);
    sum1 += simd_shuffle_down(sum1, 4);
    sum1 += simd_shuffle_down(sum1, 2);
    sum1 += simd_shuffle_down(sum1, 1);
    if (tx == 0) {
        if (row0 < dim.d) out[row0] = sum0;
        if (row0 + 1 < dim.d) out[row0 + 1] = sum1;
    }
}


kernel void gemv_q6_k_v3(device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant Dims &dim [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint tx = lane % GEMV3_NXPSG, ty = lane / GEMV3_NXPSG;
    uint row0 = tg * GEMV3_ROWS + sg * ((32 / GEMV3_NXPSG) * GEMV3_R1PTG) + ty * GEMV3_R1PTG;
    if (row0 >= dim.d) return;
    uint nb = dim.n / 256;
    device const uchar *wr0 = wb + (ulong)row0 * nb * 210;
    device const uchar *wr1 = (row0 + 1 < dim.d) ? wr0 + nb * 210 : wr0;
    device const float4 *y4 = (device const float4 *)x + tx * 4;
    float sum0 = 0.0, sum1 = 0.0;
    for (uint ich = tx; 16*ich < dim.n; ich += GEMV3_NXPSG) {
        uint b = ich >> 4, il = ich & 15;
        float4 r0[4], r1[4];
        deq_q6_k_chunk16(wr0 + b * 210, il, r0);
        deq_q6_k_chunk16(wr1 + b * 210, il, r1);
        float4 y0 = y4[0], y1 = y4[1], y2 = y4[2], y3 = y4[3];
        sum0 += dot(r0[0],y0) + dot(r0[1],y1) + dot(r0[2],y2) + dot(r0[3],y3);
        sum1 += dot(r1[0],y0) + dot(r1[1],y1) + dot(r1[2],y2) + dot(r1[3],y3);
        y4 += 32;
    }
    sum0 += simd_shuffle_down(sum0, 4);
    sum0 += simd_shuffle_down(sum0, 2);
    sum0 += simd_shuffle_down(sum0, 1);
    sum1 += simd_shuffle_down(sum1, 4);
    sum1 += simd_shuffle_down(sum1, 2);
    sum1 += simd_shuffle_down(sum1, 1);
    if (tx == 0) {
        if (row0 < dim.d) out[row0] = sum0;
        if (row0 + 1 < dim.d) out[row0 + 1] = sum1;
    }
}
`

g_dev: ^MTL.Device
g_queue: ^MTL.CommandQueue
g_pso4, g_pso6: ^MTL.ComputePipelineState

Dims :: struct { n: u32, d: u32 }

main :: proc() {
	g_dev = MTL.CreateSystemDefaultDevice()
	fmt.printf("metal: %s\n", g_dev->name()->odinString())
	g_queue = g_dev->newCommandQueue()
	src := NS.String.alloc()->initWithOdinString(MSL_SRC)
	defer src->release()
	lib, err := g_dev->newLibraryWithSource(src, nil)
	if err != nil {
		fmt.eprintfln("metal: shader compile failed: %s", err->localizedDescription()->odinString())
		os.exit(1)
	}
	defer lib->release()

	mk_pso :: proc(lib: ^MTL.Library, name: string) -> ^MTL.ComputePipelineState {
		ns := NS.String.alloc()->initWithOdinString(name)
		defer ns->release()
		fn := lib->newFunctionWithName(ns)
		if fn == nil { fmt.eprintfln("kernel %s not found", name); os.exit(1) }
		defer fn->release()
		pso, perr := g_dev->newComputePipelineStateWithFunction(fn)
		if perr != nil { fmt.eprintfln("pipeline %s failed: %s", name, perr->localizedDescription()->odinString()); os.exit(1) }
		return pso
	}
	g_pso4 = mk_pso(lib, "gemv_q4_k_v3")
	g_pso6 = mk_pso(lib, "gemv_q6_k_v3")

	// ===== Q4_K test =====
	K := 4096; D := 64; nb := K / 256
	weights := make([]u8, D * nb * 144)
	x := make([]f32, K)
	rng: u64 = 12345
	rand_u8 :: proc(r: ^u64) -> u8 { r^ = r^ * 6364136223846793005 + 1442695040888963407; return u8(r^ >> 56) }
	for i in 0 ..< len(weights) { weights[i] = rand_u8(&rng) }
	// make dall/dmin valid halves: set first 4 bytes of each block to 1.0 and 0.1 halfs
	for b in 0 ..< nb {
		for r in 0 ..< D {
			off := r * nb * 144 + b * 144
			// half 1.0 = 0x3C00, half 0.1 = 0x2E66
			weights[off+0] = 0x00; weights[off+1] = 0x3C
			weights[off+2] = 0x66; weights[off+3] = 0x2E
			// clamp scale bytes to 6-bit range so scale/min look sane
			for j in 4 ..< 16 { weights[off+j] &= 0x3F }
		}
	}
	for i in 0 ..< K { x[i] = (f32(rand_u8(&rng)) / 255.0 - 0.5) * 0.2 }

	// CPU reference (mirrors gemm_test k4_dequant_elem)
	k4_ref :: proc(w: []u8, x: []f32, r, K: int) -> f32 {
		sum: f32 = 0.0
		nb := K / 256
		for e in 0 ..< K {
			blk := w[r * nb * 144 + (e / 256) * 144 :]
			d := f32(transmute(f16)(u16(blk[0]) | u16(blk[1]) << 8))
			dmin := f32(transmute(f16)(u16(blk[2]) | u16(blk[3]) << 8))
			scales := blk[4:16]; qs := blk[16:]
			eb := e % 256
			sup := eb >> 6; ws := eb & 63; hf := ws >> 5; l := ws & 31
			is := sup * 2 + hf; qoff := sup * 32
			sc, m: u8
			if is < 4 { sc = scales[is] & 63; m = scales[is+4] & 63 }
			else { sc = (scales[is+4] & 0xF) | ((scales[is-4] >> 6) << 4); m = (scales[is+4] >> 4) | ((scales[is] >> 6) << 4) }
			nib := (qs[qoff + l] >> uint(4 * hf)) & 0xF
			sum += (d * f32(sc) * f32(nib) - dmin * f32(m)) * x[e]
		}
		return sum
	}

	wbuf := g_dev->newBufferWithBytes(weights[:], MTL.ResourceStorageModeShared)
	xbuf := g_dev->newBufferWithBytes((([^]u8)(raw_data(x)))[:len(x)*4], MTL.ResourceStorageModeShared)
	obuf := g_dev->newBufferWithLength(NS.UInteger(D * 4), MTL.ResourceStorageModeShared)

	run4 :: proc(wbuf, xbuf, obuf: ^MTL.Buffer, D, K: int) -> []f32 {
		cmd := g_queue->commandBuffer()
		enc := cmd->computeCommandEncoder()
		enc->setComputePipelineState(g_pso4)
		enc->setBuffer(wbuf, 0, 0); enc->setBuffer(xbuf, 0, 1); enc->setBuffer(obuf, 0, 2)
		dims := Dims{u32(K), u32(D)}
		enc->setBytes(([^]u8)(&dims)[:size_of(Dims)], 3)
		tg_x := (D + 63) / 64
		enc->dispatchThreadgroups(MTL.Size{NS.Integer(tg_x), 1, 1}, MTL.Size{256, 1, 1})
		enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted()
		return obuf->contentsAsSlice([]f32)[:D]
	}

	out := run4(wbuf, xbuf, obuf, D, K)
	max_err: f32 = 0.0
	bad := 0
	for r in 0 ..< D {
		ref := k4_ref(weights, x, r, K)
		err := math.abs(out[r] - ref)
		max_err = math.max(max_err, err)
		if err > 1e-3 * math.max(math.abs(ref), 1.0) {
			bad += 1
			if bad <= 4 { fmt.printfln("row %d: gpu=%f ref=%f err=%f", r, out[r], ref, err) }
		}
	}
	fmt.printfln("Q4_K: rows=%d max_err=%f bad_rows=%d -> %s", D, max_err, bad, bad == 0 ? "OK" : "FAIL")

	// ===== Q6_K test =====
	K6 := 4096; D6 := 64; nb6 := K6 / 256
	w6 := make([]u8, D6 * nb6 * 210)
	x6 := make([]f32, K6)
	rng6: u64 = 98765
	for i in 0 ..< len(w6) { w6[i] = rand_u8(&rng6) }
	for b in 0 ..< nb6 {
		for r in 0 ..< D6 {
			off := r * nb6 * 210 + b * 210
			w6[off+208] = 0x00; w6[off+209] = 0x3C  // d = half 1.0
			for j in 192 ..< 208 { w6[off+j] &= 0x7F } // scales int8 sane-ish
		}
	}
	for i in 0 ..< K6 { x6[i] = (f32(rand_u8(&rng6)) / 255.0 - 0.5) * 0.2 }

	k6_ref :: proc(w: []u8, x: []f32, r, K: int) -> f32 {
		sum: f32 = 0.0
		nb := K / 256
		for e in 0 ..< K {
			blk := w[r * nb * 210 + (e / 256) * 210 :]
			d := f32(transmute(f16)(u16(blk[208]) | u16(blk[209]) << 8))
			ql := blk; qh := blk[128:]; sc := blk[192:208]
			eb := e % 256
			hf := eb >> 7; within := eb & 127; l := within & 31; sub := within >> 5; is := l >> 4
			q: int
			scidx: int
			if sub == 0 { q = (int(ql[hf*64 + l] & 0xF) | (int(qh[hf*32 + l] & 3) << 4)) - 32; scidx = hf*8 + is + 0 }
			else if sub == 1 { q = (int(ql[hf*64 + l + 32] & 0xF) | (int(qh[hf*32 + l] >> 2 & 3) << 4)) - 32; scidx = hf*8 + is + 2 }
			else if sub == 2 { q = (int(ql[hf*64 + l] >> 4) | (int(qh[hf*32 + l] >> 4 & 3) << 4)) - 32; scidx = hf*8 + is + 4 }
			else { q = (int(ql[hf*64 + l + 32] >> 4) | (int(qh[hf*32 + l] >> 6 & 3) << 4)) - 32; scidx = hf*8 + is + 6 }
			sum += d * f32(int(sc[scidx])) * f32(q) * x[e]
		}
		return sum
	}

	w6buf := g_dev->newBufferWithBytes(w6[:], MTL.ResourceStorageModeShared)
	x6buf := g_dev->newBufferWithBytes((([^]u8)(raw_data(x6)))[:len(x6)*4], MTL.ResourceStorageModeShared)
	o6buf := g_dev->newBufferWithLength(NS.UInteger(D6 * 4), MTL.ResourceStorageModeShared)

	run6 :: proc(wbuf, xbuf, obuf: ^MTL.Buffer, D, K: int) -> []f32 {
		cmd := g_queue->commandBuffer()
		enc := cmd->computeCommandEncoder()
		enc->setComputePipelineState(g_pso6)
		enc->setBuffer(wbuf, 0, 0); enc->setBuffer(xbuf, 0, 1); enc->setBuffer(obuf, 0, 2)
		dims := Dims{u32(K), u32(D)}
		enc->setBytes(([^]u8)(&dims)[:size_of(Dims)], 3)
		tg_x := (D + 63) / 64
		enc->dispatchThreadgroups(MTL.Size{NS.Integer(tg_x), 1, 1}, MTL.Size{256, 1, 1})
		enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted()
		return obuf->contentsAsSlice([]f32)[:D]
	}

	out6 := run6(w6buf, x6buf, o6buf, D6, K6)
	max_err6: f32 = 0.0
	bad6 := 0
	for r in 0 ..< D6 {
		ref := k6_ref(w6, x6, r, K6)
		err := math.abs(out6[r] - ref)
		max_err6 = math.max(max_err6, err)
		if err > 1e-3 * math.max(math.abs(ref), 1.0) {
			bad6 += 1
			if bad6 <= 4 { fmt.printfln("q6 row %d: gpu=%f ref=%f err=%f", r, out6[r], ref, err) }
		}
	}
	fmt.printfln("Q6_K: rows=%d max_err=%f bad_rows=%d -> %s", D6, max_err6, bad6, bad6 == 0 ? "OK" : "FAIL")
}

