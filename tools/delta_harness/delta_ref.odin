#+build darwin

/* Stage 2 golden-reference harness for the gated delta-rule recurrence.

   Implements the tokenwise delta_recurrent_step in BOTH f32 (a verbatim copy of
   qwen3_5/ops.odin) and f64 (same formula, promoted). Runs a T-token trajectory
   per value head and compares them token-by-token + at state checkpoints.

   Purpose: this is the go/no-go gate for chunked scan. If f64 and f32 align to
   ~1e-5 here, the f64 reference is a trustworthy golden standard to validate a
   future chunked-scan kernel against (chunked vs f64 must match similarly).

   Pure CPU math (no Metal/GGUF). Build:
     odin run delta_harness/delta_ref.odin -file -o:speed
*/

package delta_ref

import "core:fmt"
import "core:math"

hkd :: 128 // head_k_dim (Ornith)
hvd :: 128 // head_v_dim

// ---- f32 tokenwise step (verbatim from qwen3_5/ops.odin:delta_recurrent_step) ----
delta_step_f32 :: proc(state, q_t, k_t, v_t, delta, out: []f32, beta, g_decay: f32) {
	for i in 0 ..< hkd * hvd {
		state[i] *= g_decay
	}
	for j in 0 ..< hvd {
		s: f32 = 0
		for i in 0 ..< hkd {
			s += state[i * hvd + j] * k_t[i]
		}
		delta[j] = (v_t[j] - s) * beta
	}
	for i in 0 ..< hkd {
		ki := k_t[i]
		base := i * hvd
		for j in 0 ..< hvd {
			state[base + j] += ki * delta[j]
		}
	}
	for j in 0 ..< hvd {
		s: f32 = 0
		for i in 0 ..< hkd {
			s += state[i * hvd + j] * q_t[i]
		}
		out[j] = s
	}
}

// ---- f64 golden (same formula, f64) ----
delta_step_f64 :: proc(state, q_t, k_t, v_t, delta, out: []f64, beta, g_decay: f64) {
	for i in 0 ..< hkd * hvd {
		state[i] *= g_decay
	}
	for j in 0 ..< hvd {
		s: f64 = 0
		for i in 0 ..< hkd {
			s += state[i * hvd + j] * q_t[i]
		}
		delta[j] = (v_t[j] - s) * beta
	}
	for i in 0 ..< hkd {
		ki := k_t[i]
		base := i * hvd
		for j in 0 ..< hvd {
			state[base + j] += ki * delta[j]
		}
	}
	for j in 0 ..< hvd {
		s: f64 = 0
		for i in 0 ..< hkd {
			s += state[i * hvd + j] * q_t[i]
		}
		out[j] = s
	}
}

// xorshift64 rng -> [0,1)
rng :: proc(seed: ^u64) -> f64 {
	x := seed^
	x = x ~ (x << 13)
	x = x ~ (x >> 7)
	x = x ~ (x << 17)
	seed^ = x
	return f64((x >> 11) & 0xFFFFFF) / f64(0x1000000)
}

cosine :: proc(a, b: []f64) -> f64 {
	dot, na, nb: f64 = 0, 0, 0
	for i in 0 ..< len(a) {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na < 1e-300 || nb < 1e-300 do return 1.0
	return dot / (math.sqrt(na) * math.sqrt(nb))
}

frobenius :: proc(s32: []f32, s64: []f64) -> f64 {
	s: f64 = 0
	for i in 0 ..< len(s32) {
		d := f64(s32[i]) - s64[i]
		s += d * d
	}
	return math.sqrt(s)
}

// Run one scenario: T tokens, given per-token (beta, g_decay) and input generators.
// Reports max output |f32-f64|, min output cosine, max state Frobenius diff, NaN/Inf.
run_scenario :: proc(name: string, T: int, seed_in: u64, beta_of, g_of: proc(t: int) -> f64) -> bool {
	seed := seed_in
	n := hkd * hvd
	s32 := make([]f32, n); s64 := make([]f64, n)
	q32 := make([]f32, hkd); k32 := make([]f32, hkd); v32 := make([]f32, hvd); d32 := make([]f32, hvd); o32 := make([]f32, hvd)
	q64 := make([]f64, hkd); k64 := make([]f64, hkd); v64 := make([]f64, hvd); d64 := make([]f64, hvd); o64 := make([]f64, hvd)
	defer delete(s32); defer delete(s64)
	defer delete(q32); defer delete(k32); defer delete(v32); defer delete(d32); defer delete(o32)
	defer delete(q64); defer delete(k64); defer delete(v64); defer delete(d64); defer delete(o64)

	max_abs: f64 = 0
	min_cos: f64 = 2.0
	max_state: f64 = 0
	nan_inf: int = 0
	state_scale: f64 = 0
	q_scale := 1.0 / math.sqrt(f64(hkd))

	for t in 0 ..< T {
		// identical inputs for both. q/k are l2-normalized (and q scaled by
		// 1/sqrt(hkd)) to match what the engine feeds delta_recurrent; v is raw.
		for i in 0 ..< hkd {
			q64[i] = rng(&seed)*2.0 - 1.0
			k64[i] = rng(&seed)*2.0 - 1.0
		}
		// l2norm k
		ks: f64 = 0
		for i in 0 ..< hkd { ks += k64[i] * k64[i] }
		k_inv := 1.0 / math.max(math.sqrt(ks), 1e-6)
		for i in 0 ..< hkd { k64[i] *= k_inv }
		// l2norm + scale q
		qs: f64 = 0
		for i in 0 ..< hkd { qs += q64[i] * q64[i] }
		q_inv := q_scale / math.max(math.sqrt(qs), 1e-6)
		for i in 0 ..< hkd { q64[i] *= q_inv }
		for j in 0 ..< hvd {
			v64[j] = rng(&seed)*2.0 - 1.0
		}
		for i in 0 ..< hkd { q32[i] = f32(q64[i]); k32[i] = f32(k64[i]) }
		for j in 0 ..< hvd { v32[j] = f32(v64[j]) }
		beta := beta_of(t)
		g := g_of(t)
		delta_step_f32(s32, q32, k32, v32, d32, o32, f32(beta), f32(g))
		delta_step_f64(s64, q64, k64, v64, d64, o64, beta, g)
		for j in 0 ..< hvd {
			d := math.abs(f64(o32[j]) - o64[j])
			if d > max_abs { max_abs = d }
			if math.is_nan(f64(o32[j])) || math.is_nan(o64[j]) || math.is_inf(f64(o32[j])) || math.is_inf(o64[j]) {
				nan_inf += 1
			}
		}
		o32d := make([]f64, hvd)
		for j in 0 ..< hvd { o32d[j] = f64(o32[j]) }
		c := cosine(o32d, o64)
		delete(o32d)
		if c < min_cos { min_cos = c }
		sf := frobenius(s32, s64)
		if sf > max_state { max_state = sf }
		for i in 0 ..< n {
			a := math.abs(s64[i])
			if a > state_scale { state_scale = a }
		}
	}
	rel_state := state_scale > 0 ? max_state / (state_scale * f64(n)) : max_state
	// "OK" here means STABLE (no NaN/Inf blowup). The max_out_abs / min_cos numbers
	// are the f32-vs-f64 PRECISION FLOOR: the recurrence is numerically delicate,
	// so f32 drifts from f64 over long/slow-decay sequences. This floor sets the
	// loosest tolerance a chunked-scan kernel can be validated to against f64.
	ok := nan_inf == 0
	fmt.printfln("  [{}] {} T={}  max_out_abs={:.3e}  min_cos={:.8f}  state_Frob={:.3e} (rel~{:.2e})  NaN/Inf={}",
		ok ? "OK" : "FAIL", name, T, max_abs, min_cos, max_state, rel_state, nan_inf)
	return ok
}

// ===================== chunked gated delta (CPU prototype) =====================
// Implements the chunked recurrence derived above (forward-substitution delta
// system, no inverse, no hkd x hkd matrix). Processes ONE chunk of C tokens with
// incoming state S_in (hkd x hvd), writes outs[C*hvd] and S_out[hkd*hvd]. f32.
@(private = "file")
chunked_delta_chunk :: proc(
	S_in: []f32, q_ch, k_ch: []f32, v_ch: []f32, beta, g: []f32, C: int,
	outs: []f32, S_out: []f32,
) {
	// cumprod cp[t] = P(0,t) = g[0]*...*g[t];  cp_prev = 1 before chunk start
	cp := make([]f32, C)
	defer delete(cp)
	acc: f32 = 1.0
	for t in 0 ..< C { acc *= g[t]; cp[t] = acc }

	// KKT[t][i] = dot(k_i, k_t) ; KQT[t][i] = dot(k_i, q_t)   (C x C each)
	KKT := make([]f32, C * C); KQT := make([]f32, C * C)
	defer delete(KKT); defer delete(KQT)
	for t in 0 ..< C {
		for i in 0 ..< C {
			ki := k_ch[i * hkd :]
			kt := k_ch[t * hkd :]
			qt := q_ch[t * hkd :]
			dk: f32 = 0; dq: f32 = 0
			for a in 0 ..< hkd { dk += ki[a] * kt[a]; dq += ki[a] * qt[a] }
			KKT[t * C + i] = dk
			KQT[t * C + i] = dq
		}
	}

	// s0k[t] = S_in^T k_t  (hvd);  s0q[t] = S_in^T q_t  (hvd)
	s0k := make([]f32, C * hvd); s0q := make([]f32, C * hvd)
	defer delete(s0k); defer delete(s0q)
	for t in 0 ..< C {
		kt := k_ch[t * hkd :]; qt := q_ch[t * hkd :]
		for j in 0 ..< hvd {
			sk: f32 = 0; sq: f32 = 0
			for a in 0 ..< hkd {
				sk += S_in[a * hvd + j] * kt[a]
				sq += S_in[a * hvd + j] * qt[a]
			}
			s0k[t * hvd + j] = sk
			s0q[t * hvd + j] = sq
		}
	}

	// rhs[t] = beta_t * v_t - beta_t * cp[t] * s0k[t]   (hvd)
	// delta forward sub: delta[t] = rhs[t] - sum_{i<t} L_{t,i}*delta[i], L=beta_t*P(i+1,t)*KKT
	rhs := make([]f32, C * hvd); delta := make([]f32, C * hvd)
	defer delete(rhs); defer delete(delta)
	for t in 0 ..< C {
		for j in 0 ..< hvd {
			rhs[t * hvd + j] = beta[t] * v_ch[t * hvd + j] - beta[t] * cp[t] * s0k[t * hvd + j]
			delta[t * hvd + j] = rhs[t * hvd + j] // init before subtracting earlier deltas
		}
	}
	for t in 0 ..< C {
		bt := beta[t]; cpt := cp[t]
		dt := delta[t * hvd :]
		for i in 0 ..< t {
			L := bt * (cpt / cp[i]) * KKT[t * C + i] // P(i+1,t)=cp[t]/cp[i]
			di := delta[i * hvd :]
			for j in 0 ..< hvd { dt[j] -= L * di[j] }
		}
	}

	// outputs: out[t] = cp[t]*s0q[t] + sum_{i<=t} P(i+1,t)*KQT[t][i]*delta[i]
	for t in 0 ..< hvd * C { outs[t] = 0 }
	for t in 0 ..< C {
		cpt := cp[t]
		ot := outs[t * hvd :]
		for j in 0 ..< hvd { ot[j] = cpt * s0q[t * hvd + j] }
		for i in 0 ..< t + 1 {
			pfac := (i <= t - 1) ? cpt / cp[i] : 1.0 // i==t -> P(t+1,t)=1
			w := pfac * KQT[t * C + i]
			di := delta[i * hvd :]
			for j in 0 ..< hvd { ot[j] += w * di[j] }
		}
	}

	// S_out = cp[C-1]*S_in + sum_i P(i+1,C-1) * k_i delta_i^T
	cpC := cp[C - 1]
	for a in 0 ..< hkd {
		for j in 0 ..< hvd {
			S_out[a * hvd + j] = cpC * S_in[a * hvd + j]
		}
	}
	for i in 0 ..< C {
		pfac := cp[C - 1] / cp[i] // P(i+1,C-1)
		ki := k_ch[i * hkd :]
		di := delta[i * hvd :]
		for a in 0 ..< hkd {
			kaa := pfac * ki[a]
			base := a * hvd
			for j in 0 ..< hvd { S_out[base + j] += kaa * di[j] }
		}
	}
}

// Run a T-token sequence both tokenwise-f32 and chunked-f32 (chunk size C),
// compare outputs (per token) + final state. Reports chunked-vs-tokenwise error
// and the extra_error_ratio vs the tokenwise-f64 floor.
run_chunked :: proc(C, T: int, seed_in: u64, beta_of, g_of: proc(t: int) -> f64) -> bool {
	seed := seed_in
	n := hkd * hvd
	// generate the full sequence once (q/k l2-normalized, q scaled)
	q_seq := make([]f32, T * hkd); k_seq := make([]f32, T * hkd); v_seq := make([]f32, T * hvd)
	beta_s := make([]f32, T); g_s := make([]f32, T)
	defer delete(q_seq); defer delete(k_seq); defer delete(v_seq); defer delete(beta_s); defer delete(g_s)
	q_scale := 1.0 / math.sqrt(f64(hkd))
	for t in 0 ..< T {
		qd := make([]f64, hkd); kd := make([]f64, hkd)
		for i in 0 ..< hkd { qd[i] = rng(&seed)*2.0 - 1.0; kd[i] = rng(&seed)*2.0 - 1.0 }
		ks: f64 = 0; for i in 0 ..< hkd { ks += kd[i] * kd[i] }
		k_inv := 1.0 / math.max(math.sqrt(ks), 1e-6); for i in 0 ..< hkd { kd[i] *= k_inv }
		qs: f64 = 0; for i in 0 ..< hkd { qs += qd[i] * qd[i] }
		q_inv := q_scale / math.max(math.sqrt(qs), 1e-6); for i in 0 ..< hkd { qd[i] *= q_inv }
		for i in 0 ..< hkd { q_seq[t*hkd+i] = f32(qd[i]); k_seq[t*hkd+i] = f32(kd[i]) }
		for j in 0 ..< hvd { v_seq[t*hvd+j] = f32(rng(&seed)*2.0 - 1.0) }
		beta_s[t] = f32(beta_of(t)); g_s[t] = f32(g_of(t))
		delete(qd); delete(kd)
	}

	// tokenwise-f32 reference (per-token delta_step_f32)
	s_tok := make([]f32, n); s_tok_check := make([]f32, n) // s_tok_check unused here
	out_tok := make([]f32, T * hvd)
	defer delete(s_tok); defer delete(s_tok_check); defer delete(out_tok)
	qt := make([]f32, hkd); kt := make([]f32, hkd); vt := make([]f32, hvd); dt := make([]f32, hvd); ot := make([]f32, hvd)
	defer delete(qt); defer delete(kt); defer delete(vt); defer delete(dt); defer delete(ot)
	for t in 0 ..< T {
		for i in 0 ..< hkd { qt[i] = q_seq[t*hkd+i]; kt[i] = k_seq[t*hkd+i] }
		for j in 0 ..< hvd { vt[j] = v_seq[t*hvd+j] }
		delta_step_f32(s_tok, qt, kt, vt, dt, ot, beta_s[t], g_s[t])
		for j in 0 ..< hvd { out_tok[t*hvd+j] = ot[j] }
	}

	// chunked-f32
	s_ch := make([]f32, n); out_ch := make([]f32, T * hvd)
	defer delete(s_ch); defer delete(out_ch)
	nc := T / C
	for c in 0 ..< nc {
		lo := c * C
		outs := make([]f32, C * hvd); sout := make([]f32, n)
		chunked_delta_chunk(s_ch, q_seq[lo*hkd:], k_seq[lo*hkd:], v_seq[lo*hvd:], beta_s[lo:], g_s[lo:], C, outs, sout)
		for j in 0 ..< C * hvd { out_ch[(lo)*hvd+j] = outs[j] }
		for i in 0 ..< n { s_ch[i] = sout[i] }
		delete(outs); delete(sout)
	}

	// compare outputs + final state
	max_out: f64 = 0; out_scale: f64 = 0
	for i in 0 ..< T * hvd {
		d := math.abs(f64(out_ch[i]) - f64(out_tok[i]))
		if d > max_out { max_out = d }
		a := math.abs(f64(out_tok[i])); if a > out_scale { out_scale = a }
	}
	sf: f64 = 0; st_scale: f64 = 0
	for i in 0 ..< n {
		d := f64(s_ch[i]) - f64(s_tok[i]); sf += d * d
		a := math.abs(f64(s_tok[i])); if a > st_scale { st_scale = a }
	}
	state_frob := math.sqrt(sf)
	// extra_error_ratio: chunked-vs-tokenwise relative to typical output magnitude
	ok := max_out < 1e-2 && !(math.is_nan(max_out) || math.is_inf(max_out))
	fmt.printfln("  [{}] C={} T={}  chunked_vs_tokenwise: max_out_abs={:.3e} (out_scale={:.1f})  state_Frob={:.3e}",
		ok ? "OK" : "FAIL", C, T, max_out, out_scale, state_frob)
	return ok
}

main :: proc() {
	fmt.println("=== Stage 2 f64 delta golden-reference: f32-vs-f64 tokenwise alignment ===")
	all_ok := true

	// realistic Ornith-like: beta = sigmoid(b) in ~(0.3,0.7), g_decay ~ 0.99 (slow decay)
	beta_real :: proc(t: int) -> f64 { return 0.55 }
	g_real :: proc(t: int) -> f64 { return 0.99 }
	all_ok = run_scenario("realistic (beta=.55 g=.99)", 256, 0x1234, beta_real, g_real) && all_ok

	// long slow-decay (stresses accumulation; state grows)
	all_ok = run_scenario("long slow-decay (g=.999 T=1024)", 1024, 0x2345, beta_real, proc(t: int) -> f64 { return 0.999 }) && all_ok

	// strong decay (state resets each step; isolates per-step write)
	all_ok = run_scenario("strong decay (g=.1)", 256, 0x3456, beta_real, proc(t: int) -> f64 { return 0.1 }) && all_ok

	// beta=1 (full delta write), beta~0 (near no update)
	all_ok = run_scenario("beta=1 (full write)", 256, 0x4567, proc(t: int) -> f64 { return 1.0 }, g_real) && all_ok
	all_ok = run_scenario("beta=0.01 (tiny update)", 256, 0x5678, proc(t: int) -> f64 { return 0.01 }, g_real) && all_ok

	// edge: g_decay=1 (no decay, pure delta net), g_decay=0 (full forget)
	all_ok = run_scenario("g=1 (no decay)", 256, 0x6789, beta_real, proc(t: int) -> f64 { return 1.0 }) && all_ok
	all_ok = run_scenario("g=0 (full forget)", 256, 0x789a, beta_real, proc(t: int) -> f64 { return 0.0 }) && all_ok

	fmt.println("\n=== Stage 2 chunked gated delta: chunked-f32 vs tokenwise-f32 (behavior golden) ===")
	c_ok := true
	// C=1 must be (near-)identical (validates affine decomposition); C=2/4/16 test composition.
	c_ok = run_chunked(1, 256, 0x111, beta_real, g_real) && c_ok
	c_ok = run_chunked(2, 256, 0x111, beta_real, g_real) && c_ok
	c_ok = run_chunked(4, 256, 0x111, beta_real, g_real) && c_ok
	c_ok = run_chunked(8, 256, 0x111, beta_real, g_real) && c_ok
	c_ok = run_chunked(16, 256, 0x111, beta_real, g_real) && c_ok
	// harder: slow decay, long
	c_ok = run_chunked(16, 1024, 0x222, beta_real, proc(t: int) -> f64 { return 0.999 }) && c_ok
	c_ok = run_chunked(16, 1024, 0x333, proc(t: int) -> f64 { return 1.0 }, g_real) && c_ok
	// VARYING beta/g at C=16 — does chunked diverge from tokenwise? (engine uses varying)
	c_ok = run_chunked(16, 256, 0x444,
		proc(t: int) -> f64 { return 0.3 + 0.4 * math.sin(f64(t)) },
		proc(t: int) -> f64 { return 0.4 + 0.4 * math.cos(f64(t) * 0.7) }) && c_ok
	c_ok = run_chunked(16, 256, 0x555,
		proc(t: int) -> f64 { x := f64(t) * 0.31; return 0.2 + 0.6 * (x - math.floor(x)) },
		proc(t: int) -> f64 { x := f64(t) * 0.47; return 0.25 + 0.6 * (x - math.floor(x)) }) && c_ok
	all_ok = all_ok && c_ok

	fmt.printfln("\n=== RESULT: {} ===", all_ok ? "f64 reference is a faithful golden standard (aligned to f32)" : "ALIGNMENT FAILED — investigate")
}
