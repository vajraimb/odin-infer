#+build darwin
package gemv_sweep

import "core:fmt"
import "core:math"
import "core:os"
import NS "core:sys/darwin/Foundation"
import MTL "vendor:darwin/Metal"

MSL_SRC :: #load("gemv_sweep_msl.txt", string)

g_dev: ^MTL.Device
g_queue: ^MTL.CommandQueue

Dims :: struct { n: u32, d: u32 }

Kernel_Info :: struct { name: string, rows_tg: int, nx: int, r1: int, qt: string }

KERNELS := []Kernel_Info{
	{ "gemv_q4_k", 32, 8, 0, "q4" },
	{ "gemv_q6_k", 32, 8, 0, "q6" },
	{ "gemv_q4_k_n4r1", 64, 4, 1, "q4" },
	{ "gemv_q4_k_n4r2", 128, 4, 2, "q4" },
	{ "gemv_q4_k_n4r4", 256, 4, 4, "q4" },
	{ "gemv_q4_k_n8r1", 32, 8, 1, "q4" },
	{ "gemv_q4_k_n8r2", 64, 8, 2, "q4" },
	{ "gemv_q4_k_n8r4", 128, 8, 4, "q4" },
	{ "gemv_q4_k_n16r1", 16, 16, 1, "q4" },
	{ "gemv_q4_k_n16r2", 32, 16, 2, "q4" },
	{ "gemv_q4_k_n16r4", 64, 16, 4, "q4" },
	{ "gemv_q4_k_n32r1", 8, 32, 1, "q4" },
	{ "gemv_q4_k_n32r2", 16, 32, 2, "q4" },
	{ "gemv_q4_k_n32r4", 32, 32, 4, "q4" },
	{ "gemv_q6_k_n4r1", 64, 4, 1, "q6" },
	{ "gemv_q6_k_n4r2", 128, 4, 2, "q6" },
	{ "gemv_q6_k_n4r4", 256, 4, 4, "q6" },
	{ "gemv_q6_k_n8r1", 32, 8, 1, "q6" },
	{ "gemv_q6_k_n8r2", 64, 8, 2, "q6" },
	{ "gemv_q6_k_n8r4", 128, 8, 4, "q6" },
	{ "gemv_q6_k_n16r1", 16, 16, 1, "q6" },
	{ "gemv_q6_k_n16r2", 32, 16, 2, "q6" },
	{ "gemv_q6_k_n16r4", 64, 16, 4, "q6" },
	{ "gemv_q6_k_n32r1", 8, 32, 1, "q6" },
	{ "gemv_q6_k_n32r2", 16, 32, 2, "q6" },
	{ "gemv_q6_k_n32r4", 32, 32, 4, "q6" },
}

k4_ref_row :: proc(w: []u8, x: []f32, r, K: int) -> f32 {
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

rand_u8 :: proc(r: ^u64) -> u8 { r^ = r^ * 6364136223846793005 + 1442695040888963407; return u8(r^ >> 56) }

time_kernel :: proc(lib: ^MTL.Library, name: string, wbuf, xbuf, obuf: ^MTL.Buffer, D, K, rows_tg: int) -> f64 {
	ns := NS.String.alloc()->initWithOdinString(name)
	defer ns->release()
	fn := lib->newFunctionWithName(ns)
	if fn == nil { fmt.eprintfln("kernel %s not found", name); return -1 }
	defer fn->release()
	pso, perr := g_dev->newComputePipelineStateWithFunction(fn)
	if perr != nil { fmt.eprintfln("pso %s failed: %s", name, perr->localizedDescription()->odinString()); return -1 }
	best := math.INF_F64
	for iter in 0 ..< 5 {
		cmd := g_queue->commandBuffer()
		enc := cmd->computeCommandEncoder()
		enc->setComputePipelineState(pso)
		enc->setBuffer(wbuf, 0, 0); enc->setBuffer(xbuf, 0, 1); enc->setBuffer(obuf, 0, 2)
		dims := Dims{u32(K), u32(D)}
		enc->setBytes(([^]u8)(&dims)[:size_of(Dims)], 3)
		tg_x := (D + rows_tg - 1) / rows_tg
		enc->dispatchThreadgroups(MTL.Size{NS.Integer(tg_x), 1, 1}, MTL.Size{256, 1, 1})
		enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted()
		ms := f64(cmd->GPUEndTime() - cmd->GPUStartTime()) * 1000.0
		if ms > 0 && ms < best { best = ms }
	}
	return best
}

main :: proc() {
	g_dev = MTL.CreateSystemDefaultDevice()
	fmt.printf("metal: %s\n", g_dev->name()->odinString())
	g_queue = g_dev->newCommandQueue()
	src := NS.String.alloc()->initWithOdinString(MSL_SRC)
	defer src->release()
	lib, err := g_dev->newLibraryWithSource(src, nil)
	if err != nil {
		fmt.eprintfln("shader compile failed: %s", err->localizedDescription()->odinString())
		os.exit(1)
	}
	defer lib->release()

	shapes_q4 := [][2]int{ {12288, 4096}, {248320, 4096}, {8192, 4096} }
	shapes_q6 := [][2]int{ {4096, 12288}, {8192, 4096} }

	for shape in shapes_q4 {
		D, K := shape[0], shape[1]
		nb := K / 256
		w := make([]u8, D * nb * 144)
		rng: u64 = 99
		for i in 0 ..< len(w) { w[i] = rand_u8(&rng) }
		for r in 0 ..< D {
			for b in 0 ..< nb {
				off := r * nb * 144 + b * 144
				w[off+0] = 0x00; w[off+1] = 0x3C; w[off+2] = 0x66; w[off+3] = 0x2E
				for j in 4 ..< 16 { w[off+j] &= 0x3F }
			}
		}
		x := make([]f32, K)
		for i in 0 ..< K { x[i] = (f32(rand_u8(&rng)) / 255.0 - 0.5) * 0.2 }
		wbuf := g_dev->newBufferWithBytes(w[:], MTL.ResourceStorageModeShared)
		xbuf := g_dev->newBufferWithBytes((([^]u8)(raw_data(x)))[:len(x)*4], MTL.ResourceStorageModeShared)
		obuf := g_dev->newBufferWithLength(NS.UInteger(D * 4), MTL.ResourceStorageModeShared)
		fmt.printfln("=== Q4_K %d x %d ===", D, K)
		for k in KERNELS {
			if k.qt != "q4" { continue }
			ms := time_kernel(lib, k.name, wbuf, xbuf, obuf, D, K, k.rows_tg)
			if ms < 0 { continue }
			bw := f64(D) * f64(K) * 0.5625 / (ms / 1000.0) / 1e9
			verr := "?"
			if D == 248320 {
				// verify rows 0, 1, 123456, 248318 against reference
				out := obuf->contentsAsSlice([]f32)[:D]
				max_e: f32 = 0.0
				check_rows := [7]int{0, 1, 2, 123456, 248317, 248318, 248319}
				for r in check_rows {
					ref := k4_ref_row(w, x, r, K)
					e := math.abs(out[r] - ref) / math.max(math.abs(ref), 1.0)
					max_e = math.max(max_e, e)
				}
				verr = fmt.tprintf("rel_err=%.2e", f64(max_e))
			}
			fmt.printfln("  %-18s %7.2f ms  %6.1f GB/s  %s", k.name, ms, bw, verr)
		}
	}
	for shape in shapes_q6 {
		D, K := shape[0], shape[1]
		nb := K / 256
		w := make([]u8, D * nb * 210)
		rng: u64 = 77
		for i in 0 ..< len(w) { w[i] = rand_u8(&rng) }
		for r in 0 ..< D {
			for b in 0 ..< nb {
				off := r * nb * 210 + b * 210
				w[off+208] = 0x00; w[off+209] = 0x3C
			}
		}
		x := make([]f32, K)
		for i in 0 ..< K { x[i] = (f32(rand_u8(&rng)) / 255.0 - 0.5) * 0.2 }
		wbuf := g_dev->newBufferWithBytes(w[:], MTL.ResourceStorageModeShared)
		xbuf := g_dev->newBufferWithBytes((([^]u8)(raw_data(x)))[:len(x)*4], MTL.ResourceStorageModeShared)
		obuf := g_dev->newBufferWithLength(NS.UInteger(D * 4), MTL.ResourceStorageModeShared)
		fmt.printfln("=== Q6_K %d x %d ===", D, K)
		for k in KERNELS {
			if k.qt != "q6" { continue }
			ms := time_kernel(lib, k.name, wbuf, xbuf, obuf, D, K, k.rows_tg)
			if ms < 0 { continue }
			bw := f64(D) * f64(K) * 0.8125 / (ms / 1000.0) / 1e9
			fmt.printfln("  %-18s %7.2f ms  %6.1f GB/s", k.name, ms, bw)
		}
	}
}
