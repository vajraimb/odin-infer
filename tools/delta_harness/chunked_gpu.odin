#+build darwin

/* Stage 2 GPU chunked gated-delta kernel — standalone validation vs the CPU
   prototype (delta_ref.odin:chunked_delta_chunk). Proves the Metal kernel is
   correct BEFORE wiring it into the engine.

   Architecture (v1): one threadgroup per (chunk, head), TG = hvd threads (one
   per v-col). S_in/S_out stay in global memory (each thread reads/writes only its
   own v-col -> 128 floats, no 32KB threadgroup-memory limit issue). KKT/KQT
   (C x C) staged in threadgroup memory. The delta forward-substitution is
   per-v-col independent -> no barriers in the solve. No inverse, no hkd x hkd
   matrix. Matches the CPU prototype's math exactly.

   Run: odin run delta_harness/chunked_gpu.odin -file -o:speed
*/

package chunked_gpu

import "core:fmt"
import "core:math"
import "core:os"
import NS "core:sys/darwin/Foundation"
import MTL "vendor:darwin/Metal"

hkd :: 128
hvd :: 128

MSL_SRC := `
#include <metal_stdlib>
using namespace metal;
struct DeltaP { uint hkd; uint hvd; uint C; };

kernel void chunked_delta(
    device const float * q_ch [[buffer(0)]], device const float * k_ch [[buffer(1)]],
    device const float * v_ch [[buffer(2)]], device const float * beta [[buffer(3)]],
    device const float * gdec [[buffer(4)]], device const float * S_in [[buffer(5)]],
    device float * outs [[buffer(6)]], device float * S_out [[buffer(7)]],
    constant DeltaP & P [[buffer(8)]],
    uint vc [[thread_position_in_threadgroup]]) {
    const uint hkd = P.hkd, hvd = P.hvd, C = P.C;
    threadgroup float KKT[256]; threadgroup float KQT[256];   // C <= 16
    for (uint idx = vc; idx < C*C; idx += hvd) {
        uint t = idx / C, i = idx - t*C;
        device const float * ki = k_ch + (ulong)i*hkd;
        device const float * kt = k_ch + (ulong)t*hkd;
        device const float * qt = q_ch + (ulong)t*hkd;
        float dk = 0, dq = 0;
        for (uint a = 0; a < hkd; a++) { dk += ki[a]*kt[a]; dq += ki[a]*qt[a]; }
        KKT[idx] = dk; KQT[idx] = dq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (vc >= hvd) return;
    float cp[16]; cp[0] = gdec[0]; for (uint t = 1; t < C; t++) cp[t] = cp[t-1]*gdec[t];
    float delta[16];
    for (uint t = 0; t < C; t++) {
        device const float * kt = k_ch + (ulong)t*hkd;
        float sik = 0; for (uint a = 0; a < hkd; a++) sik += S_in[a*hvd+vc]*kt[a];
        float d = beta[t]*v_ch[t*hvd+vc] - beta[t]*cp[t]*sik;   // rhs[t][vc]
        for (uint i = 0; i < t; i++) {
            float L = beta[t]*(cp[t]/cp[i])*KKT[t*C+i];          // P(i+1,t)=cp[t]/cp[i]
            d -= L*delta[i];
        }
        delta[t] = d;
        device const float * qt = q_ch + (ulong)t*hkd;
        float siq = 0; for (uint a = 0; a < hkd; a++) siq += S_in[a*hvd+vc]*qt[a];
        float o = cp[t]*siq;
        for (uint i = 0; i <= t; i++) {
            float pfac = (i < t) ? cp[t]/cp[i] : 1.0f;            // i==t -> P(t+1,t)=1
            o += pfac*KQT[t*C+i]*delta[i];
        }
        outs[t*hvd+vc] = o;
    }
    float cpC = cp[C-1];
    for (uint a = 0; a < hkd; a++) {
        float s = cpC * S_in[a*hvd+vc];
        for (uint i = 0; i < C; i++) s += (cpC/cp[i]) * k_ch[(ulong)i*hkd+a] * delta[i];
        S_out[a*hvd+vc] = s;
    }
}
`

@(private = "file")
g_dev: ^MTL.Device
@(private = "file")
g_queue: ^MTL.CommandQueue
@(private = "file")
g_pso: ^MTL.ComputePipelineState

DeltaP :: struct { hkd, hvd, C: u32 }

@(private = "file")
bytes_of :: proc(p: rawptr, n: int) -> []u8 { return ([^]u8)(p)[:n] }

setup :: proc() -> bool {
	g_dev = MTL.CreateSystemDefaultDevice()
	g_queue = g_dev->newCommandQueue()
	src := NS.String.alloc()->initWithOdinString(MSL_SRC); defer src->release()
	lib, err := g_dev->newLibraryWithSource(src, nil)
	if err != nil { fmt.eprintfln("compile failed: %s", err->localizedDescription()->odinString()); return false }
	defer lib->release()
	fn_ := lib->newFunctionWithName(NS.String.alloc()->initWithOdinString("chunked_delta"))
	defer fn_->release()
	pso, perr := g_dev->newComputePipelineStateWithFunction(fn_)
	if perr != nil { fmt.eprintfln("pipeline failed"); return false }
	g_pso = pso
	fmt.printfln("maxThreadgroupMemoryLength: {} KB", g_dev->maxThreadgroupMemoryLength() / 1024)
	return true
}

@(private = "file")
new_buf :: proc(n: int) -> ^MTL.Buffer {
	return g_dev->newBufferWithLength(NS.UInteger(n), MTL.ResourceStorageModeShared)
}
@(private = "file")
fill_f32 :: proc(b: ^MTL.Buffer, src: []f32) {
	dst := b->contentsAsSlice([]f32); copy(dst, src)
}

// dispatch ONE chunk on the GPU. S_in -> outs[C*hvd], S_out.
run_chunk_gpu :: proc(q, k, v, beta, g, S_in, outs, S_out: ^MTL.Buffer, C: int) {
	cmd := g_queue->commandBuffer()
	enc := cmd->computeCommandEncoder()
	enc->setComputePipelineState(g_pso)
	enc->setBuffer(q, 0, 0); enc->setBuffer(k, 0, 1); enc->setBuffer(v, 0, 2)
	enc->setBuffer(beta, 0, 3); enc->setBuffer(g, 0, 4); enc->setBuffer(S_in, 0, 5)
	enc->setBuffer(outs, 0, 6); enc->setBuffer(S_out, 0, 7)
	P := DeltaP{u32(hkd), u32(hvd), u32(C)}
	enc->setBytes(bytes_of(&P, size_of(P)), 8)
	enc->dispatchThreadgroups(MTL.Size{1, 1, 1}, MTL.Size{hvd, 1, 1})
	enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted()
}

// ---- CPU reference (verbatim copy of delta_ref.odin:chunked_delta_chunk, f32) ----
cpu_chunk :: proc(S_in, q_ch, k_ch, v_ch: []f32, beta, g: []f32, C: int, outs, S_out: []f32) {
	cp := make([]f32, C); defer delete(cp)
	acc: f32 = 1.0; for t in 0 ..< C { acc *= g[t]; cp[t] = acc }
	KKT := make([]f32, C * C); KQT := make([]f32, C * C); defer delete(KKT); defer delete(KQT)
	for t in 0 ..< C {
		for i in 0 ..< C {
			ki := k_ch[i * hkd:]; kt := k_ch[t * hkd:]; qt := q_ch[t * hkd:]
			dk: f32 = 0; dq: f32 = 0
			for a in 0 ..< hkd { dk += ki[a] * kt[a]; dq += ki[a] * qt[a] }
			KKT[t * C + i] = dk; KQT[t * C + i] = dq
		}
	}
	s0k := make([]f32, C * hvd); s0q := make([]f32, C * hvd); defer delete(s0k); defer delete(s0q)
	for t in 0 ..< C {
		kt := k_ch[t * hkd:]; qt := q_ch[t * hkd:]
		for j in 0 ..< hvd {
			sk: f32 = 0; sq: f32 = 0
			for a in 0 ..< hkd { sk += S_in[a * hvd + j] * kt[a]; sq += S_in[a * hvd + j] * qt[a] }
			s0k[t * hvd + j] = sk; s0q[t * hvd + j] = sq
		}
	}
	rhs := make([]f32, C * hvd); delta := make([]f32, C * hvd); defer delete(rhs); defer delete(delta)
	for t in 0 ..< C {
		for j in 0 ..< hvd {
			rhs[t * hvd + j] = beta[t] * v_ch[t * hvd + j] - beta[t] * cp[t] * s0k[t * hvd + j]
			delta[t * hvd + j] = rhs[t * hvd + j]
		}
	}
	for t in 0 ..< C {
		bt := beta[t]; cpt := cp[t]; dt := delta[t * hvd:]
		for i in 0 ..< t { L := bt * (cpt / cp[i]) * KKT[t * C + i]; di := delta[i * hvd:]; for j in 0 ..< hvd { dt[j] -= L * di[j] } }
	}
	for t in 0 ..< C {
		cpt := cp[t]; ot := outs[t * hvd:]
		for j in 0 ..< hvd { ot[j] = cpt * s0q[t * hvd + j] }
		for i in 0 ..< t + 1 {
			pfac := (i <= t - 1) ? cpt / cp[i] : 1.0; w := pfac * KQT[t * C + i]; di := delta[i * hvd:]
			for j in 0 ..< hvd { ot[j] += w * di[j] }
		}
	}
	cpC := cp[C - 1]
	for a in 0 ..< hkd { for j in 0 ..< hvd { S_out[a * hvd + j] = cpC * S_in[a * hvd + j] } }
	for i in 0 ..< C {
		pfac := cpC / cp[i]; ki := k_ch[i * hkd:]; di := delta[i * hvd:]
		for a in 0 ..< hkd { kaa := pfac * ki[a]; base := a * hvd; for j in 0 ..< hvd { S_out[base + j] += kaa * di[j] } }
	}
}

rng :: proc(seed: ^u64) -> f64 {
	x := seed^; x = x ~ (x << 13); x = x ~ (x >> 7); x = x ~ (x << 17); seed^ = x
	return f64((x >> 11) & 0xFFFFFF) / f64(0x1000000)
}

// Build a chunk's inputs (l2norm'd q/k, scaled q, raw v), run GPU + CPU, compare.
test_one_chunk :: proc(C: int, seed_in: u64) -> bool {
	n := hkd * hvd
	seed := seed_in
	q := make([]f32, C * hkd); k := make([]f32, C * hkd); v := make([]f32, C * hvd)
	beta := make([]f32, C); g := make([]f32, C)
	defer delete(q); defer delete(k); defer delete(v); defer delete(beta); defer delete(g)
	q_scale := f32(1.0 / math.sqrt(f64(hkd)))
	for t in 0 ..< C {
		qd := make([]f64, hkd); kd := make([]f64, hkd)
		for i in 0 ..< hkd { qd[i] = rng(&seed) * 2.0 - 1.0; kd[i] = rng(&seed) * 2.0 - 1.0 }
		ks: f64 = 0; for i in 0 ..< hkd { ks += kd[i] * kd[i] }
		k_inv := 1.0 / math.max(math.sqrt(ks), 1e-6); for i in 0 ..< hkd { kd[i] *= k_inv }
		qs: f64 = 0; for i in 0 ..< hkd { qs += qd[i] * qd[i] }
		q_inv := q_scale / f32(math.max(math.sqrt(qs), 1e-6)); for i in 0 ..< hkd { qd[i] *= f64(q_inv) }
		for i in 0 ..< hkd { q[t*hkd+i] = f32(qd[i]); k[t*hkd+i] = f32(kd[i]) }
		for j in 0 ..< hvd { v[t*hvd+j] = f32(rng(&seed) * 2.0 - 1.0) }
		beta[t] = f32(0.2 + rng(&seed) * 0.6) // VARYING beta (was constant 0.55)
		g[t] = f32(0.3 + rng(&seed) * 0.65)   // VARYING g in [0.3,0.95] (was constant 0.99)
		delete(qd); delete(kd)
	}
	S_in := make([]f32, n)
	for i in 0 ..< n { S_in[i] = f32(rng(&seed) * 2.0 - 1.0) * 0.1 }
	defer delete(S_in)

	// CPU
	outs_cpu := make([]f32, C * hvd); S_out_cpu := make([]f32, n)
	defer delete(outs_cpu); defer delete(S_out_cpu)
	cpu_chunk(S_in, q, k, v, beta, g, C, outs_cpu, S_out_cpu)

	// GPU
	bq := new_buf(C * hkd * 4); bk := new_buf(C * hkd * 4); bv := new_buf(C * hvd * 4)
	bbeta := new_buf(C * 4); bg := new_buf(C * 4); bS_in := new_buf(n * 4)
	bouts := new_buf(C * hvd * 4); bS_out := new_buf(n * 4)
	defer bq->release(); defer bk->release(); defer bv->release(); defer bbeta->release()
	defer bg->release(); defer bS_in->release(); defer bouts->release(); defer bS_out->release()
	fill_f32(bq, q); fill_f32(bk, k); fill_f32(bv, v); fill_f32(bbeta, beta); fill_f32(bg, g); fill_f32(bS_in, S_in)
	run_chunk_gpu(bq, bk, bv, bbeta, bg, bS_in, bouts, bS_out, C)
	outs_gpu := bouts->contentsAsSlice([]f32)[: C * hvd]
	S_out_gpu := bS_out->contentsAsSlice([]f32)[: n]

	max_out, max_state, scale: f64 = 0, 0, 0
	for i in 0 ..< C * hvd {
		d := math.abs(f64(outs_gpu[i]) - f64(outs_cpu[i])); if d > max_out { max_out = d }
		a := math.abs(f64(outs_cpu[i])); if a > scale { scale = a }
	}
	for i in 0 ..< n {
		d := math.abs(f64(S_out_gpu[i]) - f64(S_out_cpu[i])); if d > max_state { max_state = d }
	}
	// NaN/Inf check
	nan_inf := 0
	for i in 0 ..< C * hvd { if math.is_nan(f64(outs_gpu[i])) || math.is_inf(f64(outs_gpu[i])) { nan_inf += 1 } }
	ok := max_out < 1e-3 && nan_inf == 0
	fmt.printfln("  [{}] C={}  GPU-vs-CPU: max_out_abs={:.3e} (out_scale={:.2f})  max_state_abs={:.3e}  NaN/Inf={}",
		ok ? "OK" : "FAIL", C, max_out, scale, max_state, nan_inf)
	return ok
}

main :: proc() {
	if !setup() { os.exit(1) }
	fmt.println("=== Stage 2 GPU chunked delta: GPU kernel vs CPU prototype ===")
	all_ok := true
	all_ok = test_one_chunk(1, 0x111) && all_ok    // identity
	all_ok = test_one_chunk(2, 0x222) && all_ok
	all_ok = test_one_chunk(4, 0x333) && all_ok
	all_ok = test_one_chunk(8, 0x444) && all_ok
	all_ok = test_one_chunk(16, 0x555) && all_ok
	fmt.printfln("\n=== RESULT: {} ===", all_ok ? "GPU chunked kernel matches CPU prototype" : "MISMATCH — investigate (check fast-math / layout)")
}
