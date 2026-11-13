#+build darwin

/* Wide-N tile sweep for the 27B's dominant IQ matmuls.
   Shapes are the real GGUF ones: up 17408x5120 and down 5120x17408.
   Wall-clock around command-buffer wait. Gate is ggml.dequant_row,
   values rounded to f16, then an f64 dot.

   odin run tools/gemm_harness/iq_n_sweep.odin -file -o:speed \
     -collection:ggml=packages
*/

package iq_n_sweep

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

Kind :: enum { XXS, I1S, I2S, I3X, I2XS, I3S, I4XS, I1M }

kind_block :: proc(k: Kind) -> int {
	switch k {
	case .XXS: return 66
	case .I1S: return 50
	case .I2S: return 82
	case .I3X: return 98
	case .I2XS: return 74
	case .I3S: return 110
	case .I4XS: return 136
	case .I1M: return 56
	}
	return 0
}

kind_ggml :: proc(k: Kind) -> ggml.GGML_Type {
	switch k {
	case .XXS: return .IQ2_XXS
	case .I1S: return .IQ1_S
	case .I2S: return .IQ2_S
	case .I3X: return .IQ3_XXS
	case .I2XS: return .IQ2_XS
	case .I3S: return .IQ3_S
	case .I4XS: return .IQ4_XS
	case .I1M: return .IQ1_M
	}
	return .IQ2_XXS
}

kind_tag :: proc(k: Kind) -> string {
	switch k {
	case .XXS: return "iq2xxs"
	case .I1S: return "iq1s"
	case .I2S: return "iq2s"
	case .I3X: return "iq3xxs"
	case .I2XS: return "iq2xs"
	case .I3S: return "iq3s"
	case .I4XS: return "iq4xs"
	case .I1M: return "iq1m"
	}
	return "?"
}

// Dequant of one 8-wide chunk. sa index uses @@BK@@.
DEQUANT_XXS :: `
                device const uchar * blk = A + (ulong)(r0 + row) * row_bytes + (uint)(e_base / 256) * 66;
                int ew = int(e_base & 255);
                float d = (float)(*(device const half *)blk);
                uint ib32 = (uint)ew >> 5;
                uint qb = 2 + 8 * ib32;
                uint aux1 = (uint)blk[qb + 4] | ((uint)blk[qb + 5] << 8) | ((uint)blk[qb + 6] << 16) | ((uint)blk[qb + 7] << 24);
                float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
                uint l = ((uint)ew >> 3) & 3;
                ulong grid = iq2xxs_grid[blk[qb + l]];
                uchar signs = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
                for (uint j = 0; j < 8; j++) {
                    float v = (float)((grid >> (8 * j)) & 0xff);
                    float sgn = (signs & kmask_iq2xs[j]) == 0 ? 1.0f : -1.0f;
                    sa[row * @@BK@@ + col_base + j] = (half)(db * v * sgn);
                }`

DEQUANT_I1S :: `
                device const uchar * blk = A + (ulong)(r0 + row) * row_bytes + (uint)(e_base / 256) * 50;
                int ew = int(e_base & 255);
                float d = (float)(*(device const half *)blk);
                device const uchar * qs = blk + 2; device const uchar * qh = blk + 34;
                uint ib = (uint)ew >> 5;
                uint qhv = (uint)qh[2 * ib] | ((uint)qh[2 * ib + 1] << 8);
                float dl = d * (float)(2 * (int)((qhv >> 12) & 7) + 1);
                float delta = (qhv & 0x8000) == 0 ? 0.125f : -0.125f;
                uint l = ((uint)ew >> 3) & 3;
                uint idx = (uint)qs[4 * ib + l] | (((qhv >> (3 * l)) & 7) << 8);
                ulong grid = iq1s_grid[idx];
                for (uint j = 0; j < 8; j++) {
                    float gv = (float)((char)((grid >> (8 * j)) & 0xff));
                    sa[row * @@BK@@ + col_base + j] = (half)(dl * (gv + delta));
                }`

DEQUANT_I2S :: `
                device const uchar * blk = A + (ulong)(r0 + row) * row_bytes + (uint)(e_base / 256) * 82;
                int ew = int(e_base & 255);
                float d = (float)(*(device const half *)blk);
                device const uchar * qs = blk + 2; device const uchar * signs = blk + 34;
                device const uchar * qh = blk + 66; device const uchar * scales = blk + 74;
                uint ib32 = (uint)ew >> 5;
                uchar sc = scales[ib32];
                float db0 = d * (0.5f + (float)(sc & 0xF)) * 0.25f;
                float db1 = d * (0.5f + (float)(sc >> 4)) * 0.25f;
                uint l = ((uint)ew >> 3) & 3;
                uint idx = (uint)qs[4 * ib32 + l] | (((uint)qh[ib32] << (8 - 2 * l)) & 0x300);
                ulong grid = iq2s_grid[idx];
                uchar sn = signs[4 * ib32 + l];
                float dl = (l < 2) ? db0 : db1;
                for (uint j = 0; j < 8; j++) {
                    float v = (float)((grid >> (8 * j)) & 0xff);
                    float sgn = (sn & kmask_iq2xs[j]) == 0 ? 1.0f : -1.0f;
                    sa[row * @@BK@@ + col_base + j] = (half)(dl * v * sgn);
                }`

DEQUANT_I3X :: `
                device const uchar * blk = A + (ulong)(r0 + row) * row_bytes + (uint)(e_base / 256) * 98;
                int ew = int(e_base & 255);
                float d = (float)(*(device const half *)blk);
                device const uchar * qs = blk + 2; device const uchar * sas = blk + 66;
                uint ib32 = (uint)ew >> 5;
                uint sbo = 4 * ib32;
                uint aux = (uint)sas[sbo] | ((uint)sas[sbo + 1] << 8) | ((uint)sas[sbo + 2] << 16) | ((uint)sas[sbo + 3] << 24);
                float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
                uint l = ((uint)ew >> 3) & 3;
                uchar signs = ksigns_iq2xs[(aux >> (7 * l)) & 127];
                uint g1 = iq3xxs_grid[qs[8 * ib32 + 2 * l + 0]];
                uint g2 = iq3xxs_grid[qs[8 * ib32 + 2 * l + 1]];
                for (uint j = 0; j < 4; j++) {
                    float v1 = (float)((g1 >> (8 * j)) & 0xff);
                    float v2 = (float)((g2 >> (8 * j)) & 0xff);
                    float s1 = (signs & kmask_iq2xs[j + 0]) == 0 ? 1.0f : -1.0f;
                    float s2 = (signs & kmask_iq2xs[j + 4]) == 0 ? 1.0f : -1.0f;
                    sa[row * @@BK@@ + col_base + j + 0] = (half)(db * v1 * s1);
                    sa[row * @@BK@@ + col_base + j + 4] = (half)(db * v2 * s2);
                }`

DEQUANT_I2XS :: `
                device const uchar * blk = A + (ulong)(r0 + row) * row_bytes + (uint)(e_base / 256) * 74;
                int ew = int(e_base & 255);
                float d = (float)(*(device const half *)blk);
                uint ib32 = (uint)ew >> 5;
                uchar sc = blk[66 + ib32];
                float db0 = d * (0.5f + (float)(sc & 0xF)) * 0.25f;
                float db1 = d * (0.5f + (float)(sc >> 4)) * 0.25f;
                uint l = ((uint)ew >> 3) & 3;
                uint qoff = 2 + 2 * (4 * ib32 + l);
                uint q = (uint)blk[qoff] | ((uint)blk[qoff + 1] << 8);
                ulong grid = iq2xs_grid[q & 511];
                uchar signs = ksigns_iq2xs[q >> 9];
                float dl = (l < 2) ? db0 : db1;
                for (uint j = 0; j < 8; j++) {
                    float v = (float)((grid >> (8 * j)) & 0xff);
                    float sgn = (signs & kmask_iq2xs[j]) == 0 ? 1.0f : -1.0f;
                    sa[row * @@BK@@ + col_base + j] = (half)(dl * v * sgn);
                }`

DEQUANT_I3S :: `
                device const uchar * blk = A + (ulong)(r0 + row) * row_bytes + (uint)(e_base / 256) * 110;
                int ew = int(e_base & 255);
                float d = (float)(*(device const half *)blk);
                device const uchar * qs = blk + 2; device const uchar * qh = blk + 66;
                device const uchar * signs = blk + 74; device const uchar * scales = blk + 106;
                uint ib32 = (uint)ew >> 5;
                uint qoff = 8 * ib32; uint signoff = 4 * ib32; uint qhoff = 2 * (ib32 >> 1);
                float db1 = d * (float)(1 + 2 * (int)(scales[qhoff / 2] & 0xF));
                float db2 = d * (float)(1 + 2 * (int)(scales[qhoff / 2] >> 4));
                uint l = ((uint)ew >> 3) & 3;
                uint qhi = qhoff + (ib32 & 1);
                uint idx1 = (uint)qs[qoff + 2 * l + 0] | (((uint)qh[qhi] << (8 - 2 * l)) & 256);
                uint idx2 = (uint)qs[qoff + 2 * l + 1] | (((uint)qh[qhi] << (7 - 2 * l)) & 256);
                uint g1 = iq3s_grid[idx1]; uint g2 = iq3s_grid[idx2];
                uchar sn = signs[signoff + l];
                float dl = (ib32 & 1) == 0 ? db1 : db2;
                for (uint j = 0; j < 4; j++) {
                    float v1 = (float)((g1 >> (8 * j)) & 0xff);
                    float v2 = (float)((g2 >> (8 * j)) & 0xff);
                    float s1 = (sn & kmask_iq2xs[j + 0]) == 0 ? 1.0f : -1.0f;
                    float s2 = (sn & kmask_iq2xs[j + 4]) == 0 ? 1.0f : -1.0f;
                    sa[row * @@BK@@ + col_base + j + 0] = (half)(dl * v1 * s1);
                    sa[row * @@BK@@ + col_base + j + 4] = (half)(dl * v2 * s2);
                }`

DEQUANT_I4XS :: `
                device const uchar * blk = A + (ulong)(r0 + row) * row_bytes + (uint)(e_base / 256) * 136;
                int ew = int(e_base & 255);
                float d = (float)(*(device const half *)blk);
                uint scales_h = (uint)blk[2] | ((uint)blk[3] << 8);
                device const uchar * scales_l = blk + 4;
                device const uchar * qs = blk + 8;
                uint ib = (uint)ew >> 5;
                int ls = (int)((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) | (int)(((scales_h >> (2 * ib)) & 3) << 4);
                float dl = d * (float)(ls - 32);
                uint e32 = (uint)ew & 31;
                uint qoff = 16 * ib;
                for (uint j = 0; j < 8; j++) {
                    uchar qb = qs[qoff + (e32 & 15) + j];
                    float v = (float)kvalues_iq4nl[(e32 < 16) ? (qb & 0xF) : (qb >> 4)];
                    sa[row * @@BK@@ + col_base + j] = (half)(dl * v);
                }`

DEQUANT_I1M :: `
                device const uchar * blk = A + (ulong)(r0 + row) * row_bytes + (uint)(e_base / 256) * 56;
                int ew = int(e_base & 255);
                device const uchar * qs = blk; device const uchar * qh = blk + 32;
                device const uchar * scales = blk + 48;
                uint sc0 = (uint)scales[0] | ((uint)scales[1] << 8);
                uint sc1 = (uint)scales[2] | ((uint)scales[3] << 8);
                uint sc2 = (uint)scales[4] | ((uint)scales[5] << 8);
                uint sc3 = (uint)scales[6] | ((uint)scales[7] << 8);
                uint scale = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000);
                float d = half_to_float((ushort)scale);
                uint ib = (uint)ew >> 5;
                uint sca[4] = {sc0, sc1, sc2, sc3};
                uint scv = (ib & 1) == 0 ? sca[ib / 2] : (sca[ib / 2] >> 6);
                float dl1 = d * (float)(2 * (int)((scv >> 0) & 0x7) + 1);
                float dl2 = d * (float)(2 * (int)((scv >> 3) & 0x7) + 1);
                uint l = ((uint)ew >> 3) & 3;
                uint qoff = 4 * ib; uint qhoff = 2 * ib;
                uint idx; float delta;
                if (l == 0)      { idx = (uint)qs[qoff + 0] | (((uint)qh[qhoff + 0] << 8) & 0x700); delta = (qh[qhoff + 0] & 0x08) == 0 ? 0.125f : -0.125f; }
                else if (l == 1) { idx = (uint)qs[qoff + 1] | (((uint)qh[qhoff + 0] << 4) & 0x700); delta = (qh[qhoff + 0] & 0x80) == 0 ? 0.125f : -0.125f; }
                else if (l == 2) { idx = (uint)qs[qoff + 2] | (((uint)qh[qhoff + 1] << 8) & 0x700); delta = (qh[qhoff + 1] & 0x08) == 0 ? 0.125f : -0.125f; }
                else             { idx = (uint)qs[qoff + 3] | (((uint)qh[qhoff + 1] << 4) & 0x700); delta = (qh[qhoff + 1] & 0x80) == 0 ? 0.125f : -0.125f; }
                float dl = (l < 2) ? dl1 : dl2;
                ulong grid = iq1s_grid[idx];
                for (uint j = 0; j < 8; j++) {
                    float gv = (float)((char)((grid >> (8 * j)) & 0xff));
                    sa[row * @@BK@@ + col_base + j] = (half)(dl * (gv + delta));
                }`

dequant_of :: proc(k: Kind) -> string {
	switch k {
	case .XXS: return DEQUANT_XXS
	case .I1S: return DEQUANT_I1S
	case .I2S: return DEQUANT_I2S
	case .I3X: return DEQUANT_I3X
	case .I2XS: return DEQUANT_I2XS
	case .I3S: return DEQUANT_I3S
	case .I4XS: return DEQUANT_I4XS
	case .I1M: return DEQUANT_I1M
	}
	return ""
}

SHELL :: `kernel void @@NAME@@(
    device const uchar * A [[buffer(0)]], device const half * B [[buffer(1)]],
    device float * C [[buffer(2)]], constant GemmDims & dims [[buffer(3)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiitg [[thread_index_in_threadgroup]]) {
    const uint M = dims.M, N = dims.N, K = dims.K;
    const uint r0 = tgpig.y * @@BM@@, c0 = tgpig.x * @@BN@@;
    const ulong row_bytes = (ulong)(K / 256) * @@BS@@;
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
            if (r0 + row < M) {
@@DEQUANT@@
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

Config :: struct {
	kind:       Kind,
	bm, bn, bk: int,
}

g_dev: ^MTL.Device
g_queue: ^MTL.CommandQueue

kernel_src :: proc(c: Config) -> (string, string, int) {
	nsg_n := c.bn / 8
	nt := (c.bm / 8) * nsg_n * 32
	tpr := c.bk / 8
	rstep := nt / tpr
	name := fmt.tprintf("gemm_{}_{}_{}_{}", kind_tag(c.kind), c.bm, c.bn, c.bk)
	body := SHELL
	body = sub(body, "@@NAME@@", name)
	body = sub(body, "@@DEQUANT@@", dequant_of(c.kind))
	body = sub(body, "@@BM@@", fmt.tprintf("{}", c.bm))
	body = sub(body, "@@BN@@", fmt.tprintf("{}", c.bn))
	body = sub(body, "@@BK@@", fmt.tprintf("{}", c.bk))
	body = sub(body, "@@BS@@", fmt.tprintf("{}", kind_block(c.kind)))
	body = sub(body, "@@NT@@", fmt.tprintf("{}", nt))
	body = sub(body, "@@TPR@@", fmt.tprintf("{}", tpr))
	body = sub(body, "@@RSTEP@@", fmt.tprintf("{}", rstep))
	body = sub(body, "@@NSGN@@", fmt.tprintf("{}", nsg_n))
	return name, body, nt
}

load_tables :: proc() -> string {
	raw, err := os.read_entire_file_from_path("packages/qwen3_5/msl_tables.odin", context.allocator)
	if err != os.ERROR_NONE {
		fmt.eprintln("cannot read msl_tables.odin")
		os.exit(1)
	}
	s := string(raw)
	marker := "MSL_TABLES :: `"
	i := strings.index(s, marker)
	s = s[i + len(marker):]
	j := strings.last_index(s, "`")
	return s[:j]
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

gen_quant :: proc(k: Kind, M, K: int, seed_in: u32) -> []u8 {
	bs := kind_block(k)
	nb := K / 256
	a := make([]u8, M * nb * bs)
	seed := seed_in
	for b in 0 ..< M * nb {
		base := b * bs
		write_f16_le(a, base, rng_f(&seed)*0.2 + 0.02)
		for i in 2 ..< bs {
			a[base + i] = u8(rng_f(&seed) * 256.0)
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
	bs := kind_block(c.kind)
	a_bytes := gen_quant(c.kind, M, K, 0xBEEF)
	b_h := gen_b(K, N, 0xF00D)
	defer delete(a_bytes)
	defer delete(b_h)
	a_ref := make([]f32, M * K)
	defer delete(a_ref)
	ggml.dequant_row(kind_ggml(c.kind), a_bytes, M * K, a_ref)
	for i in 0 ..< M * K {
		a_ref[i] = f32(f16(a_ref[i]))
	}
	c_ref := make([]f64, M * N)
	defer delete(c_ref)
	for m in 0 ..< M {
		for n in 0 ..< N {
			s: f64 = 0
			for kk in 0 ..< K {
				s += f64(a_ref[m*K + kk]) * f64(b_h[kk*N + n])
			}
			c_ref[m*N + n] = s
		}
	}
	bufA := g_dev->newBufferWithLength(NS.UInteger(M * (K / 256) * bs), MTL.ResourceStorageModeShared)
	bufB := g_dev->newBufferWithLength(NS.UInteger(K * N * 2), MTL.ResourceStorageModeShared)
	bufC := g_dev->newBufferWithLength(NS.UInteger(M * N * 4), MTL.ResourceStorageModeShared)
	defer bufA->release()
	defer bufB->release()
	defer bufC->release()
	copy(bufA->contentsAsSlice([]u8), a_bytes)
	copy(bufB->contentsAsSlice([]u8), ([^]u8)(raw_data(b_h))[:len(b_h) * 2])
	run_kernel(pso, bufA, bufB, bufC, M, N, K, c.bm, c.bn, nt)
	c_gpu := bufC->contentsAsSlice([]f32)[:M * N]
	max_abs, scale: f64 = 0, 0
	for i in 0 ..< M * N {
		d := math.abs(f64(c_gpu[i]) - c_ref[i])
		if d > max_abs { max_abs = d }
		a := math.abs(c_ref[i])
		if a > scale { scale = a }
	}
	rel := scale > 1e-6 ? max_abs / scale : max_abs
	return rel, rel <= GATE_TOL
}

Shape :: struct { m, n, k: int, name: string }

time_shape :: proc(pso: ^MTL.ComputePipelineState, c: Config, nt: int, s: Shape, a: []u8, b: []f16) -> (f64, f64) {
	bs := kind_block(c.kind)
	bufA := g_dev->newBufferWithLength(NS.UInteger(s.m * (s.k / 256) * bs), MTL.ResourceStorageModeShared)
	bufB := g_dev->newBufferWithLength(NS.UInteger(s.k * s.n * 2), MTL.ResourceStorageModeShared)
	bufC := g_dev->newBufferWithLength(NS.UInteger(s.m * s.n * 4), MTL.ResourceStorageModeShared)
	defer bufA->release()
	defer bufB->release()
	defer bufC->release()
	copy(bufA->contentsAsSlice([]u8), a)
	copy(bufB->contentsAsSlice([]u8), ([^]u8)(raw_data(b))[:len(b) * 2])
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
	if os.get_env("WRITE", context.temp_allocator) == "1" {
		all := []Kind{.XXS, .I1S, .I2S, .I3X, .I2XS, .I3S, .I4XS, .I1M}
		out: strings.Builder
		strings.builder_init(&out, context.allocator)
		for k in all {
			_, body, _ := kernel_src(Config{k, 32, 32, 64})
			strings.write_string(&out, body)
			strings.write_string(&out, "\n")
		}
		werr := os.write_entire_file("/tmp/iq_wide.msl", transmute([]u8)strings.to_string(out))
		if werr != os.ERROR_NONE {
			fmt.eprintfln("write failed {}", werr)
			os.exit(1)
		}
		fmt.eprintln("wrote /tmp/iq_wide.msl")
		return
	}
	if g_dev == nil {
		fmt.eprintln("metal: no device")
		os.exit(1)
	}
	g_queue = g_dev->newCommandQueue()
	fmt.eprintfln("metal: {}", g_dev->name()->odinString())

	kinds := []Kind{.I2XS, .I3S, .I4XS, .I1M}
	tiles := [][3]int{{32, 32, 64}}
	configs := make([dynamic]Config)
	for k in kinds {
		for t in tiles {
			append(&configs, Config{k, t[0], t[1], t[2]})
		}
	}
	parts := make([dynamic]string)
	append(&parts, "#include <metal_stdlib>\nusing namespace metal;\nstruct GemmDims { uint M; uint N; uint K; };\n")
	append(&parts, "inline float half_to_float(ushort h) {\n    uint sign = (uint)(h >> 15) << 31;\n    uint exp = (h >> 10) & 0x1f;\n    uint man = (uint)h & 0x3ff;\n    if (exp == 0) {\n        if (man == 0) return as_type<float>(sign);\n        uint m = man; int e = -15;\n        while ((m & 0x400) == 0) { m <<= 1; e -= 1; }\n        m &= 0x3ff;\n        return as_type<float>(sign | ((uint)(127 + e) << 23) | (m << 13));\n    } else if (exp == 31) {\n        return as_type<float>(sign | (0xffu << 23) | (man << 13));\n    }\n    return as_type<float>(sign | ((exp + 112) << 23) | (man << 13));\n}\n")
	append(&parts, load_tables())
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
	fmt.eprintln("compiling...")
	src_ns := NS.String.alloc()->initWithOdinString(src)
	defer src_ns->release()
	opts := MTL.CompileOptions.alloc()->init()
	defer opts->release()
	opts->setFastMathEnabled(true)
	lib, err := g_dev->newLibraryWithSource(src_ns, opts)
	if err != nil {
		fmt.eprintfln("compile: {}", err->localizedDescription()->odinString())
		os.exit(1)
	}
	defer lib->release()
	fmt.eprintln("compiled")

	shapes := []Shape{
		{17408, 512, 5120, "up512"},
		{5120, 512, 17408, "down512"},
		{17408, 64, 5120, "up64"},
	}
	// One weight buffer per kind per shape. Activations shared per shape.
	// b depends only on K,N so per shape; a depends on kind and shape.
	bs := make([][]f16, len(shapes))
	for s, i in shapes {
		bs[i] = gen_b(s.k, s.n, 0x12345)
	}
	as := make([][][]u8, len(kinds))
	for ki in 0 ..< len(kinds) {
		as[ki] = make([][]u8, len(shapes))
		for s, si in shapes {
			as[ki][si] = gen_quant(kinds[ki], s.m, s.k, 0xABCDE)
		}
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
			fmt.eprintfln("FAIL {} pipeline {}", names[i], perr->localizedDescription()->odinString())
			continue
		}
		rel, ok := gate(pso, c, nts[i])
		if !ok {
			fmt.eprintfln("FAIL {} bm={} bn={} bk={} rel={:.3e}", kind_tag(c.kind), c.bm, c.bn, c.bk, rel)
			pso->release()
			continue
		}
		fmt.eprintfln("OK   {} bm={} bn={} bk={} rel={:.3e}", kind_tag(c.kind), c.bm, c.bn, c.bk, rel)
		ki := 0
		for kk, idx in kinds {
			if kk == c.kind { ki = idx }
		}
		for s, si in shapes {
			if s.m % c.bm != 0 || s.n % c.bn != 0 || s.k % c.bk != 0 {
				continue
			}
			gflops, ms := time_shape(pso, c, nts[i], s, as[ki][si], bs[si])
			fmt.eprintfln("     {}  {:.0f} GFLOPS  {:.1f} ms", s.name, gflops, ms)
		}
		pso->release()
	}
}
