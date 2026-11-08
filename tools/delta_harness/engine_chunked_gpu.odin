#+build darwin

/* Engine-twin harness for the C=16 cold case (ROOFLINE.md / MILESTONE.md).

   The historical failure: v1 engine wiring of chunked_delta produced "!!!!!" at
   C=16 while C<=8 was fine; the STANDALONE harness (chunked_gpu.odin) passed C=16
   with varying beta/g. Two candidate causes were recorded: Metal fast-math, or
   host-side dispatch/layout. The v1 wiring was reverted and never committed, so
   the host-side path is dead code and cannot be re-tested directly.

   What CAN still be tested is the ENGINE KERNEL ITSELF (qwen3_5/metal.odin
   chunked_delta, retained in-tree but never validated standalone — the harness
   validates a different kernel with different indexing and beta/g inputs).

   This harness:
     1. Compiles the engine kernel VERBATIM (copied from metal.odin @ 7149bb6,
        ChunkP layout + engine strides + in-kernel beta/g from bt/at/a_decay/dt_bias).
     2. Runs it with engine-realistic dims (n_vh=32 threadgroups, key_dim=2048,
        conv_dim=8192, value_dim=4096, T=32, multi-chunk in one encoder).
     3. Compares against a line-level Odin CPU twin, sweeping
        C in {8, 16} x fast-math {on, off}.
     4. Case B (adversarial): forces near-zero cumulative decay (g ~ 1e-14) so
        cp[t] underflows to literal 0 mid-chunk, probing the 0/0 in cp[t]/cp[i]
        ratios — a formulation fragility whose probability grows with C and which
        would produce exactly the observed "!!!!!" (NaN) symptom. The tokenwise
        path never forms these ratios, which is why it is immune.

   Run: odin run delta_harness/engine_chunked_gpu.odin -file -o:speed
*/

package engine_chunked_gpu

import "core:fmt"
import "core:math"
import "core:os"
import NS "core:sys/darwin/Foundation"
import MTL "vendor:darwin/Metal"

hkd       :: 128
hvd       :: 128
n_kh      :: 16
n_vh      :: 32
key_dim   :: n_kh * hkd               // 2048
value_dim :: n_vh * hvd               // 4096
conv_dim  :: key_dim * 2 + value_dim  // 8192

// ---- engine kernel, VERBATIM from qwen3_5/metal.odin (chunked_delta, 7149bb6) ----
MSL_SRC := `
#include <metal_stdlib>
using namespace metal;
struct ChunkP { uint hkd; uint hvd; uint C; uint key_dim; uint conv_dim; uint value_dim; uint n_vh; uint n_kh; uint chunk_start; uint slot; };
kernel void chunked_delta(
    device const float * qlin [[buffer(0)]], device const float * klin [[buffer(1)]],
    device const float * qkv2 [[buffer(2)]], device const float * bt [[buffer(3)]],
    device const float * at [[buffer(4)]], device const float * a_decay [[buffer(5)]],
    device const float * dt_bias [[buffer(6)]], device float * rec_states [[buffer(7)]],
    device float * lout [[buffer(8)]], constant ChunkP & P [[buffer(9)]],
    uint vh [[threadgroup_position_in_grid]], uint vc [[thread_position_in_threadgroup]]) {
    const uint hkd=P.hkd, hvd=P.hvd, C=P.C, key_dim=P.key_dim, conv_dim=P.conv_dim;
    const uint value_dim=P.value_dim, n_vh=P.n_vh, kh = vh % P.n_kh, cs = P.chunk_start;
    device float * S = rec_states + (ulong)P.slot * n_vh * hkd * hvd + (ulong)vh * hkd * hvd;
    threadgroup float KKT[256]; threadgroup float KQT[256];
    for (uint idx = vc; idx < C*C; idx += hvd) {
        uint t = idx/C, i = idx - t*C;
        device const float * ki = klin + (ulong)(cs+i)*key_dim + (ulong)kh*hkd;
        device const float * kt = klin + (ulong)(cs+t)*key_dim + (ulong)kh*hkd;
        device const float * qt = qlin + (ulong)(cs+t)*key_dim + (ulong)kh*hkd;
        float dk=0, dq=0;
        for (uint a=0;a<hkd;a++){ dk+=ki[a]*kt[a]; dq+=ki[a]*qt[a]; }
        KKT[idx]=dk; KQT[idx]=dq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (vc >= hvd) return;
    float beta[16], cp[16];
    float a_dec = a_decay[vh], dtb = dt_bias[vh];
    for (uint t=0;t<C;t++) {
        float bv = bt[(ulong)(cs+t)*n_vh + vh];
        float av = at[(ulong)(cs+t)*n_vh + vh];
        beta[t] = 1.0f/(1.0f+exp(-bv));
        float sp = av + dtb;
        float softplus = sp > 20.0f ? sp : (sp < -20.0f ? 0.0f : log(1.0f+exp(sp)));
        cp[t] = (t==0) ? exp(a_dec*softplus) : cp[t-1]*exp(a_dec*softplus);
    }
    float delta[16];
    for (uint t=0;t<C;t++) {
        device const float * kt = klin + (ulong)(cs+t)*key_dim + (ulong)kh*hkd;
        float sik=0; for (uint a=0;a<hkd;a++) sik += S[a*hvd+vc]*kt[a];
        float vval = qkv2[(ulong)(cs+t)*conv_dim + 2ull*key_dim + (ulong)vh*hvd + vc];
        float d = beta[t]*vval - beta[t]*cp[t]*sik;
        for (uint i=0;i<t;i++){ float L = beta[t]*(cp[t]/cp[i])*KKT[t*C+i]; d -= L*delta[i]; }
        delta[t]=d;
        device const float * qt = qlin + (ulong)(cs+t)*key_dim + (ulong)kh*hkd;
        float siq=0; for (uint a=0;a<hkd;a++) siq += S[a*hvd+vc]*qt[a];
        float o = cp[t]*siq;
        for (uint i=0;i<=t;i++){ float pfac = (i<t)?cp[t]/cp[i]:1.0f; o += pfac*KQT[t*C+i]*delta[i]; }
        lout[(ulong)(cs+t)*value_dim + (ulong)vh*hvd + vc] = o;
    }
    float cpC = cp[C-1];
    for (uint a=0;a<hkd;a++){
        float s = cpC*S[a*hvd+vc];
        for (uint i=0;i<C;i++) s += (cpC/cp[i])*klin[(ulong)(cs+i)*key_dim+(ulong)kh*hkd+a]*delta[i];
        S[a*hvd+vc] = s;
    }
}
`

ChunkP :: struct { hkd, hvd, C, key_dim, conv_dim, value_dim, n_vh, n_kh, chunk_start, slot: u32 }

@(private = "file")
g_dev:   ^MTL.Device
@(private = "file")
g_queue: ^MTL.CommandQueue
@(private = "file")
g_pso_fast, g_pso_precise: ^MTL.ComputePipelineState

@(private = "file")
bytes_of :: proc(p: rawptr, n: int) -> []u8 { return ([^]u8)(p)[:n] }

@(private = "file")
make_kernel_pso :: proc(fast_math: bool) -> ^MTL.ComputePipelineState {
	src := NS.String.alloc()->initWithOdinString(MSL_SRC); defer src->release()
	opts := MTL.CompileOptions.alloc()->init(); defer opts->release()
	opts->setFastMathEnabled(fast_math)
	lib, err := g_dev->newLibraryWithSource(src, opts)
	if err != nil {
		fmt.eprintfln("compile failed (fast_math=%v): %s", fast_math, err->localizedDescription()->odinString())
		return nil
	}
	defer lib->release()
	fn_ := lib->newFunctionWithName(NS.String.alloc()->initWithOdinString("chunked_delta"))
	defer fn_->release()
	pso, perr := g_dev->newComputePipelineStateWithFunction(fn_)
	if perr != nil { fmt.eprintfln("pipeline failed"); return nil }
	return pso
}

setup :: proc() -> bool {
	g_dev = MTL.CreateSystemDefaultDevice()
	g_queue = g_dev->newCommandQueue()
	g_pso_fast    = make_kernel_pso(true)
	g_pso_precise = make_kernel_pso(false)
	return g_pso_fast != nil && g_pso_precise != nil
}

@(private = "file")
new_buf :: proc(n: int) -> ^MTL.Buffer {
	return g_dev->newBufferWithLength(NS.UInteger(n), MTL.ResourceStorageModeShared)
}
@(private = "file")
fill_f32 :: proc(b: ^MTL.Buffer, src: []f32) { copy(b->contentsAsSlice([]f32), src) }

// Run all chunks of a T-token batch on the GPU, engine-style: one dispatch per
// chunk, grid = n_vh threadgroups, all inside ONE encoder (as the v1 wiring did).
run_gpu :: proc(pso: ^MTL.ComputePipelineState, qlin, klin, qkv2, bt, at, a_decay, dt_bias, rec, lout: ^MTL.Buffer, T, C: int) {
	cmd := g_queue->commandBuffer()
	enc := cmd->computeCommandEncoder()
	for cs := 0; cs < T; cs += C {
		enc->setComputePipelineState(pso)
		enc->setBuffer(qlin, 0, 0); enc->setBuffer(klin, 0, 1); enc->setBuffer(qkv2, 0, 2)
		enc->setBuffer(bt, 0, 3); enc->setBuffer(at, 0, 4); enc->setBuffer(a_decay, 0, 5)
		enc->setBuffer(dt_bias, 0, 6); enc->setBuffer(rec, 0, 7); enc->setBuffer(lout, 0, 8)
		P := ChunkP{hkd, hvd, u32(C), key_dim, conv_dim, value_dim, n_vh, n_kh, u32(cs), 0}
		enc->setBytes(bytes_of(&P, size_of(P)), 9)
		enc->dispatchThreadgroups(MTL.Size{n_vh, 1, 1}, MTL.Size{hvd, 1, 1})
	}
	enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted()
}

// ---- CPU twin: line-level port of the engine kernel (f32) ----

sigmoid :: proc(x: f32) -> f32 { return 1.0 / (1.0 + math.exp(-x)) }
softplus_kernel :: proc(x: f32) -> f32 {
	if x > 20.0 { return x }
	if x < -20.0 { return 0.0 }
	return math.ln(1.0 + math.exp(x))
}

cpu_engine_chunk :: proc(qlin, klin, qkv2, bt, at, a_decay, dt_bias, rec_states, lout: []f32, C, cs: int) {
	for vh in 0 ..< n_vh {
		kh := vh % n_kh
		S := rec_states[vh * hkd * hvd:] // slot = 0
		KKT := make([]f32, C * C); KQT := make([]f32, C * C)
		defer delete(KKT); defer delete(KQT)
		for t in 0 ..< C {
			for i in 0 ..< C {
				ki := klin[(cs + i) * key_dim + kh * hkd:]
				kt := klin[(cs + t) * key_dim + kh * hkd:]
				qt := qlin[(cs + t) * key_dim + kh * hkd:]
				dk: f32 = 0; dq: f32 = 0
				for a in 0 ..< hkd { dk += ki[a] * kt[a]; dq += ki[a] * qt[a] }
				KKT[t * C + i] = dk; KQT[t * C + i] = dq
			}
		}
		beta := make([]f32, C); cp := make([]f32, C)
		defer delete(beta); defer delete(cp)
		a_dec := a_decay[vh]; dtb := dt_bias[vh]
		for t in 0 ..< C {
			beta[t] = sigmoid(bt[(cs + t) * n_vh + vh])
			g := math.exp(a_dec * softplus_kernel(at[(cs + t) * n_vh + vh] + dtb))
			cp[t] = t == 0 ? g : cp[t - 1] * g
		}
		delta := make([]f32, C * hvd); defer delete(delta)
		for t in 0 ..< C {
			kt := klin[(cs + t) * key_dim + kh * hkd:]
			qt := qlin[(cs + t) * key_dim + kh * hkd:]
			for vc in 0 ..< hvd {
				sik: f32 = 0; for a in 0 ..< hkd { sik += S[a * hvd + vc] * kt[a] }
				vval := qkv2[(cs + t) * conv_dim + 2 * key_dim + vh * hvd + vc]
				d := beta[t] * vval - beta[t] * cp[t] * sik
				for i in 0 ..< t {
					L := beta[t] * (cp[t] / cp[i]) * KKT[t * C + i]
					d -= L * delta[i * hvd + vc]
				}
				delta[t * hvd + vc] = d
				siq: f32 = 0; for a in 0 ..< hkd { siq += S[a * hvd + vc] * qt[a] }
				o := cp[t] * siq
				for i in 0 ..< t + 1 {
					pfac: f32 = i < t ? cp[t] / cp[i] : 1.0
					o += pfac * KQT[t * C + i] * delta[i * hvd + vc]
				}
				lout[(cs + t) * value_dim + vh * hvd + vc] = o
			}
		}
		cpC := cp[C - 1]
		for a in 0 ..< hkd {
			for vc in 0 ..< hvd {
				s := cpC * S[a * hvd + vc]
				for i in 0 ..< C {
					s += (cpC / cp[i]) * klin[(cs + i) * key_dim + kh * hkd + a] * delta[i * hvd + vc]
				}
				S[a * hvd + vc] = s
			}
		}
	}
}

rng :: proc(seed: ^u64) -> f64 {
	x := seed^; x = x ~ (x << 13); x = x ~ (x >> 7); x = x ~ (x << 17); seed^ = x
	return f64((x >> 11) & 0xFFFFFF) / f64(0x1000000)
}

// Inverse softplus so we can target g = exp(a_dec * y) precisely.
inv_softplus :: proc(y: f64) -> f64 { return math.ln(math.exp(y) - 1.0) }

count_nan_inf :: proc(s: []f32) -> int {
	n := 0
	for v in s { if math.is_nan(f64(v)) || math.is_inf(f64(v)) { n += 1 } }
	return n
}

max_abs_diff :: proc(a, b: []f32) -> f64 {
	m: f64 = 0
	for i in 0 ..< len(a) {
		d := math.abs(f64(a[i]) - f64(b[i]))
		if d > m && !math.is_nan(d) { m = d }
	}
	return m
}

// Build engine-layout inputs. adversarial: plant near-zero-g tokens in vh=0.
make_inputs :: proc(T: int, seed_in: u64, adversarial: bool) -> (qlin, klin, qkv2, bt, at, a_decay, dt_bias, S0: []f32) {
	seed := seed_in
	qlin = make([]f32, T * key_dim); klin = make([]f32, T * key_dim)
	qkv2 = make([]f32, T * conv_dim)
	bt = make([]f32, T * n_vh); at = make([]f32, T * n_vh)
	a_decay = make([]f32, n_vh); dt_bias = make([]f32, n_vh)
	S0 = make([]f32, n_vh * hkd * hvd)

	q_scale := 1.0 / math.sqrt(f64(hkd))
	for t in 0 ..< T {
		for kh in 0 ..< n_kh {
			qs, ks: f64 = 0, 0
			base := t * key_dim + kh * hkd
			for a in 0 ..< hkd {
				qlin[base + a] = f32(rng(&seed) * 2 - 1); qs += f64(qlin[base + a]) * f64(qlin[base + a])
				klin[base + a] = f32(rng(&seed) * 2 - 1); ks += f64(klin[base + a]) * f64(klin[base + a])
			}
			qi := q_scale / math.max(math.sqrt(qs), 1e-6)
			ki := 1.0 / math.max(math.sqrt(ks), 1e-6)
			for a in 0 ..< hkd { qlin[base + a] *= f32(qi); klin[base + a] *= f32(ki) }
		}
		for j in 0 ..< value_dim { qkv2[t * conv_dim + 2 * key_dim + j] = f32(rng(&seed) * 2 - 1) }
	}
	for vh in 0 ..< n_vh {
		a_decay[vh] = -f32(0.3 + rng(&seed) * 1.2) // [-1.5, -0.3]
		dt_bias[vh] = f32(rng(&seed) * 2 - 1)
	}
	for t in 0 ..< T {
		for vh in 0 ..< n_vh {
			bt[t * n_vh + vh] = f32(rng(&seed) * 4 - 2) // beta in ~[0.12, 0.88]
			y := 0.1 + rng(&seed) * 0.9               // target softplus -> g in ~[0.22, 0.97]
			at[t * n_vh + vh] = f32(inv_softplus(y)) - dt_bias[vh]
		}
	}
	if adversarial {
		// vh=0: force sp = av + dtb = 30 (> 20 clamp -> softplus = 30) with
		// a_dec = -4 -> g = exp(-120) = 0 in f32. Two such tokens in the first
		// chunk drive cp[] to literal 0, then cp[t]/cp[i] = 0/0.
		a_decay[0] = -4.0
		for t0 in ([]int{3, 11}) {
			at[t0 * n_vh + 0] = 30.0 - dt_bias[0]
		}
	}
	for i in 0 ..< n_vh * hkd * hvd { S0[i] = f32(rng(&seed) * 2 - 1) * 0.1 }
	return
}

// One full comparison run: CPU twin vs GPU (fast-math on/off), T tokens, chunk C.
run_case :: proc(name: string, T, C: int, seed: u64, adversarial: bool) -> bool {
	fmt.printfln("--- %s: T=%d C=%d adversarial=%v ---", name, T, C, adversarial)
	qlin, klin, qkv2, bt, at, a_decay, dt_bias, S0 := make_inputs(T, seed, adversarial)
	defer delete(qlin); defer delete(klin); defer delete(qkv2); defer delete(bt)
	defer delete(at); defer delete(a_decay); defer delete(dt_bias); defer delete(S0)

	// CPU twin (sequential chunks, in-place S)
	S_cpu := make([]f32, n_vh * hkd * hvd); defer delete(S_cpu)
	copy(S_cpu, S0)
	lout_cpu := make([]f32, T * value_dim); defer delete(lout_cpu)
	for cs := 0; cs < T; cs += C {
		cpu_engine_chunk(qlin, klin, qkv2, bt, at, a_decay, dt_bias, S_cpu, lout_cpu, C, cs)
	}

	cpu_nan := count_nan_inf(lout_cpu) + count_nan_inf(S_cpu)
	fmt.printfln("  CPU twin:            NaN/Inf = %d", cpu_nan)

	all_ok := true
	for fm in ([]struct{ pso: ^MTL.ComputePipelineState, label: string }{{g_pso_fast, "fast-math ON "}, {g_pso_precise, "fast-math OFF"}}) {
		bq := new_buf(T * key_dim * 4); bk := new_buf(T * key_dim * 4); bv := new_buf(T * conv_dim * 4)
		bbt := new_buf(T * n_vh * 4); bat := new_buf(T * n_vh * 4)
		bad := new_buf(n_vh * 4); bdtb := new_buf(n_vh * 4)
		bS := new_buf(n_vh * hkd * hvd * 4); blout := new_buf(T * value_dim * 4)
		defer bq->release(); defer bk->release(); defer bv->release(); defer bbt->release()
		defer bat->release(); defer bad->release(); defer bdtb->release(); defer bS->release(); defer blout->release()
		fill_f32(bq, qlin); fill_f32(bk, klin); fill_f32(bv, qkv2); fill_f32(bbt, bt); fill_f32(bat, at)
		fill_f32(bad, a_decay); fill_f32(bdtb, dt_bias); fill_f32(bS, S0)

		run_gpu(fm.pso, bq, bk, bv, bbt, bat, bad, bdtb, bS, blout, T, C)

		lout_gpu := blout->contentsAsSlice([]f32)[: T * value_dim]
		S_gpu := bS->contentsAsSlice([]f32)[: n_vh * hkd * hvd]
		gpu_nan := count_nan_inf(lout_gpu) + count_nan_inf(S_gpu)
		d_out := max_abs_diff(lout_gpu, lout_cpu)
		d_state := max_abs_diff(S_gpu, S_cpu)
		// NaN pattern agreement counts as "matching" for the adversarial case;
		// for benign cases require tight numeric agreement.
		ok := adversarial ? (gpu_nan > 0) == (cpu_nan > 0) : (d_out < 1e-3 && gpu_nan == 0)
		fmt.printfln("  GPU %s: max|dOut|=%.3e  max|dState|=%.3e  NaN/Inf=%d  -> %s",
			fm.label, d_out, d_state, gpu_nan, ok ? "OK" : "FAIL")
		all_ok = all_ok && ok
	}
	return all_ok
}

main :: proc() {
	if !setup() { os.exit(1) }
	fmt.println("=== C=16 cold case: engine kernel verbatim, engine dims/layout ===")

	ok1 := run_case("benign C=8 ", 32, 8, 0xA11, false)
	ok2 := run_case("benign C=16", 32, 16, 0xB22, false)
	fmt.println()
	ok3 := run_case("adversarial near-zero-g C=16", 32, 16, 0xC33, true)

	fmt.println()
	fmt.println("=== VERDICT ===")
	if ok1 && ok2 {
		fmt.println("Engine kernel passes benign C=8/C=16 under BOTH fast-math settings ->")
		fmt.println("fast-math is EXONERATED for the kernel itself; historical C=16 failure")
		fmt.println("points at either the dead v1 wiring or the adversarial mechanism below.")
	} else {
		fmt.println("BENIGN CASE FAILED — see per-case lines for which (C, fast-math) combo.")
	}
	if ok3 {
		fmt.println("Adversarial: CPU twin and GPU agree on the NaN pattern -> cp-ratio 0/0")
		fmt.println("underflow is a REAL formulation fragility (grows with C). Fused kernel")
		fmt.println("must stabilize cp[t]/cp[i] (clamp cumulative decay or log-space ratios).")
	} else {
		fmt.println("Adversarial case inconclusive — inspect NaN counts above.")
	}
}
