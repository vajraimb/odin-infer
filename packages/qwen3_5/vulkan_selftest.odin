/* Kernel self-test for the Vulkan backend. No model file: boots the device,
   runs GEMV against ggml.dequant_row, then the elementwise kernels against
   the CPU procs in ops.odin. Integer-dot and cooperative-matrix are optional
   and print "skip" when the device does not expose them. */

package qwen3_5

import "core:fmt"
import "core:math"
import "core:mem"
import ggml "ggml:ggml"
import vkr "vulkan:vulkan"

@(private)
_vulkan_selftest_keep :: proc() {
	_ = vkr.device_info
}

when GPU_VULKAN {

vulkan_selftest :: proc() -> bool {
	info, okb := vkr.boot()
	if !okb {
		fmt.eprintln("vulkan_selftest: device init failed")
		return false
	}
	g_info = info
	g_max_wg = int(info.max_wg)
	if g_max_wg <= 0 {
		g_max_wg = 65535
	}
	if !make_pipes() {
		vulkan_destroy()
		return false
	}
	wbuf, oka := vkr.make_buffer(256 * 1024)
	abuf, okb2 := vkr.make_buffer(256 * 1024)
	if !oka || !okb2 {
		fmt.eprintln("vulkan_selftest: buffer alloc failed")
		vulkan_destroy()
		return false
	}
	g_act = abuf
	bufs := [1]vkr.Buffer{wbuf}
	if !vkr.bind_storage(bufs[:], abuf) {
		vulkan_destroy()
		return false
	}
	pass := true
	pass = test_gemv_all(wbuf) && pass
	pass = test_ops(wbuf) && pass
	pass = test_optional(wbuf) && pass
	if pass {
		fmt.eprintln("vulkan_selftest: PASS")
	} else {
		fmt.eprintln("vulkan_selftest: FAIL")
	}
	vulkan_destroy()
	return pass
}

@(private)
test_gemv_all :: proc(wbuf: vkr.Buffer) -> bool {
	kinds := [?]ggml.GGML_Type {
		.F32, .F16, .Q4_0, .Q4_1, .Q5_0, .Q5_1, .Q8_0,
		.Q2_K, .Q3_K, .Q4_K, .Q5_K, .Q6_K, .Q8_K,
	}
	pass := true
	for k in kinds {
		for T in 1 ..= 2 {
			mx, ok := gemv_case(wbuf, k, T, g_gemv_exact)
			fmt.eprintfln("vulkan_selftest: gemv exact {} T={} max_abs={}", k, T, mx)
			if !ok || mx > 1e-5 {
				pass = false
			}
			mxp, okp := gemv_case(wbuf, k, T, g_gemv_par)
			fmt.eprintfln("vulkan_selftest: gemv parallel {} T={} max_abs={}", k, T, mxp)
			if !okp || mxp > 1e-2 {
				pass = false
			}
			if g_gemv_sub != 0 {
				mxs, oks := gemv_case(wbuf, k, T, g_gemv_sub)
				fmt.eprintfln("vulkan_selftest: gemv subgroup {} T={} max_abs={}", k, T, mxs)
				if !oks || mxs > 1e-2 {
					pass = false
				}
			}
		}
	}
	if g_gemv_sub == 0 {
		fmt.eprintln("vulkan_selftest: gemv subgroup skip (pipeline or device)")
	}
	return pass
}

@(private)
gemv_case :: proc(wbuf: vkr.Buffer, kind: ggml.GGML_Type, T: int, pipe: type_of(g_gemv)) -> (f32, bool) {
	_ = pipe
	rows := 3
	cols := 256
	elems, blen := ggml.ggml_block_info(kind)
	if cols % elems != 0 {
		return 0, false
	}
	row_bytes := (cols / elems) * blen
	nbytes := rows * row_bytes
	raw := ([^]u8)(wbuf.ptr)[:nbytes]
	fill_quant(kind, raw, row_bytes)
	x_n := T * cols
	y_off := 4096
	xs := fslice(0, x_n)
	for i in 0 ..< x_n {
		xs[i] = f32(i % 17) * 0.01 - 0.08
	}
	ys := fslice(y_off, T * rows)
	for i in 0 ..< T * rows {
		ys[i] = 0
	}
	vkr.flush_range(wbuf, 0, nbytes)
	vkr.flush_range(g_act, 0, (y_off + T * rows) * 4)
	saved := g_gemv
	g_gemv = pipe
	w: Wref
	append(&w.parts, Wpart {
		chunk = 0, off = 0, row0 = 0, rows = rows, cols = cols,
		row_bytes = row_bytes, kind = u32(kind),
	})
	if !vkr.begin() {
		delete(w.parts)
		g_gemv = saved
		return 0, false
	}
	gemv_rows(w, 0, y_off, 0, rows, T, cols, rows)
	delete(w.parts)
	g_gemv = saved
	if !vkr.submit_wait() {
		return 0, false
	}
	vkr.invalidate_range(g_act, y_off * 4, T * rows * 4)
	mx: f32 = 0
	deq := make([]f32, cols)
	defer delete(deq)
	for t in 0 ..< T {
		for r in 0 ..< rows {
			ggml.dequant_row(kind, raw[r * row_bytes:], cols, deq)
			sum: f32 = 0
			for i in 0 ..< cols {
				sum += deq[i] * xs[t * cols + i]
			}
			d := ys[t * rows + r] - sum
			if d < 0 { d = -d }
			if d > mx { mx = d }
		}
	}
	return mx, true
}

@(private)
fill_quant :: proc(kind: ggml.GGML_Type, raw: []u8, row_bytes: int) {
	s: u32 = 0x12345678
	for i in 0 ..< len(raw) {
		s = s * 1664525 + 1013904223
		raw[i] = u8(s >> 24)
	}
	rows := len(raw) / row_bytes
	h := transmute(u16)f16(f32(0.02))
	lo := u8(h)
	hi := u8(h >> 8)
	put_f16 :: proc(b: []u8, off: int, lo, hi: u8) {
		b[off] = lo
		b[off + 1] = hi
	}
	for r in 0 ..< rows {
		b := raw[r * row_bytes:]
		#partial switch kind {
		case .F32:
			for i in 0 ..< row_bytes / 4 {
				v := f32(i + r) * 0.01 - 0.05
				bits := transmute(u32)v
				b[i * 4] = u8(bits)
				b[i * 4 + 1] = u8(bits >> 8)
				b[i * 4 + 2] = u8(bits >> 16)
				b[i * 4 + 3] = u8(bits >> 24)
			}
		case .F16:
			for i in 0 ..< row_bytes / 2 {
				put_f16(b, i * 2, lo, hi)
				// vary the payload a little so the dot is not a single scale
				if i % 3 == 0 {
					hh := transmute(u16)f16(f32(0.02) * f32(1 + i % 5))
					b[i * 2] = u8(hh)
					b[i * 2 + 1] = u8(hh >> 8)
				}
			}
		case .Q4_0, .Q5_0, .Q8_0:
			_, bl := ggml.ggml_block_info(kind)
			for o := 0; o + bl <= row_bytes; o += bl {
				put_f16(b, o, lo, hi)
			}
		case .Q4_1, .Q5_1:
			_, bl := ggml.ggml_block_info(kind)
			for o := 0; o + bl <= row_bytes; o += bl {
				put_f16(b, o, lo, hi)
				put_f16(b, o + 2, lo, hi)
			}
		case .Q2_K:
			for o := 0; o + 84 <= row_bytes; o += 84 {
				put_f16(b, o + 80, lo, hi)
				put_f16(b, o + 82, lo, hi)
			}
		case .Q3_K:
			for o := 0; o + 110 <= row_bytes; o += 110 {
				put_f16(b, o + 108, lo, hi)
			}
		case .Q4_K, .Q5_K:
			_, bl := ggml.ggml_block_info(kind)
			for o := 0; o + bl <= row_bytes; o += bl {
				put_f16(b, o, lo, hi)
				put_f16(b, o + 2, lo, hi)
			}
		case .Q6_K:
			for o := 0; o + 210 <= row_bytes; o += 210 {
				put_f16(b, o + 208, lo, hi)
			}
		case .Q8_K:
			for o := 0; o + 292 <= row_bytes; o += 292 {
				bits := transmute(u32)f32(0.02)
				b[o] = u8(bits)
				b[o + 1] = u8(bits >> 8)
				b[o + 2] = u8(bits >> 16)
				b[o + 3] = u8(bits >> 24)
			}
		}
	}
}

@(private)
max_abs_diff :: proc(a, b: []f32) -> f32 {
	m: f32 = 0
	n := min(len(a), len(b))
	for i in 0 ..< n {
		d := a[i] - b[i]
		if d < 0 { d = -d }
		if d > m { m = d }
	}
	return m
}

@(private)
test_ops :: proc(wbuf: vkr.Buffer) -> bool {
	pass := true
	// rmsnorm, two rows of 4
	x := []f32{1, 2, 3, 4, 0.5, -1, 0.25, 2}
	w := []f32{1, 1.5, 0.5, 2}
	mem.copy(raw_data(fslice(0, 8)), raw_data(x), 32)
	mem.copy(raw_data(fslice(32, 4)), raw_data(w), 16)
	vkr.flush_range(g_act, 0, 64 * 4)
	if !vkr.begin() { return false }
	rms(0, 64, 32, 2, 4, 4, 4, 1e-6)
	if !vkr.submit_wait() { return false }
	vkr.invalidate_range(g_act, 64 * 4, 8 * 4)
	ref := make([]f32, 8)
	defer delete(ref)
	rmsnorm(ref[0:4], x[0:4], w, 1e-6)
	rmsnorm(ref[4:8], x[4:8], w, 1e-6)
	got := fslice(64, 8)
	mx := max_abs_diff(got, ref)
	fmt.eprintfln("vulkan_selftest: rmsnorm max_abs={}", mx)
	if mx > 1e-5 { pass = false }

	// l2norm + scale
	mem.copy(raw_data(fslice(0, 4)), raw_data(x), 16)
	vkr.flush_range(g_act, 0, 16)
	if !vkr.begin() { return false }
	l2(0, 64, 1, 4, 4, 4, 0.5, 1e-6)
	if !vkr.submit_wait() { return false }
	vkr.invalidate_range(g_act, 64 * 4, 16)
	l2norm_into(ref[0:4], x[0:4], 4, 1e-6)
	for i in 0 ..< 4 { ref[i] *= 0.5 }
	mx = max_abs_diff(fslice(64, 4), ref[0:4])
	fmt.eprintfln("vulkan_selftest: l2norm max_abs={}", mx)
	if mx > 1e-5 { pass = false }

	// rope, rotary 4 inside a head of 4, pos 3
	rv := []f32{0.1, -0.2, 0.3, -0.4}
	inv := []f32{0.01, 0.001}
	mem.copy(raw_data(fslice(0, 4)), raw_data(rv), 16)
	mem.copy(raw_data(fslice(8, 2)), raw_data(inv), 8)
	vkr.flush_range(g_act, 0, 64)
	if !vkr.begin() { return false }
	rope(0, 8, 1, 4, 4, 3, 1)
	if !vkr.submit_wait() { return false }
	vkr.invalidate_range(g_act, 0, 16)
	copy(ref[0:4], rv)
	apply_mrope_text_head(ref[0:4], 3, inv, 4)
	mx = max_abs_diff(fslice(0, 4), ref[0:4])
	fmt.eprintfln("vulkan_selftest: rope max_abs={}", mx)
	if mx > 1e-5 { pass = false }

	// attention, 1 head, hd 4, pos 2, seq 4, n_kv 1
	q := []f32{0.2, -0.1, 0.0, 0.3}
	kv := make([]f32, 3 * 4)
	defer delete(kv)
	for i in 0 ..< 12 { kv[i] = f32(i) * 0.05 - 0.1 }
	mem.copy(raw_data(fslice(0, 4)), raw_data(q), 16)
	mem.copy(raw_data(fslice(16, 12)), raw_data(kv), 48)
	vkr.flush_range(g_act, 0, 128 * 4)
	if !vkr.begin() { return false }
	p := Push{}
	p.u[0] = 0
	p.u[1] = 16
	p.u[2] = 16
	p.u[3] = 64
	p.u[4] = 128
	p.u[5] = 4
	p.u[6] = 1
	p.u[7] = 1
	p.u[8] = 4
	p.u[9] = 2
	p.u[10] = 4
	p.f[0] = 0.5
	vkr.barrier()
	vkr.dispatch(g_attn, 1, 1, 1, &p, size_of(p), 0)
	if !vkr.submit_wait() { return false }
	vkr.invalidate_range(g_act, 64 * 4, 16)
	// CPU reference: scores, softmax, mix. K and V are the same buffer here.
	att := make([]f32, 3)
	defer delete(att)
	for t in 0 ..< 3 {
		s: f32 = 0
		for i in 0 ..< 4 { s += q[i] * kv[t * 4 + i] }
		att[t] = s * 0.5
	}
	softmax(att)
	out := make([]f32, 4)
	defer delete(out)
	for t in 0 ..< 3 {
		for i in 0 ..< 4 { out[i] += att[t] * kv[t * 4 + i] }
	}
	mx = max_abs_diff(fslice(64, 4), out)
	fmt.eprintfln("vulkan_selftest: attn max_abs={}", mx)
	if mx > 1e-4 { pass = false }

	// conv, 2 channels, kernel 3. Weight is F32 in the weight buffer.
	cw := []f32{0.1, 0.2, 0.3, -0.1, 0.0, 0.4}
	mem.copy(wbuf.ptr, raw_data(cw), 24)
	inn := []f32{0.5, -0.25}
	st := []f32{0.1, 0.2, -0.1, 0.05}
	mem.copy(raw_data(fslice(0, 2)), raw_data(inn), 8)
	mem.copy(raw_data(fslice(8, 4)), raw_data(st), 16)
	vkr.flush_range(wbuf, 0, 24)
	vkr.flush_range(g_act, 0, 64)
	ww: Wref
	append(&ww.parts, Wpart{chunk = 0, off = 0, row0 = 0, rows = 2, cols = 3, row_bytes = 12, kind = 0})
	if !vkr.begin() { delete(ww.parts); return false }
	conv_dispatch(ww, 0, 32, 8, 2, 3)
	delete(ww.parts)
	if !vkr.submit_wait() { return false }
	vkr.invalidate_range(g_act, 0, 64 * 4)
	cref := make([]f32, 2)
	defer delete(cref)
	cst := make([]f32, 4)
	defer delete(cst)
	copy(cst, st)
	conv1d_step(cref, inn, cst, cw, 2, 3)
	mx = max_abs_diff(fslice(32, 2), cref)
	st_mx := max_abs_diff(fslice(8, 4), cst)
	fmt.eprintfln("vulkan_selftest: conv max_abs={} state_max_abs={}", mx, st_mx)
	if mx > 1e-5 || st_mx > 1e-5 { pass = false }

	// delta, one value head, hk=hv=4
	qq := []f32{0.1, 0.0, -0.1, 0.2}
	kk := []f32{0.2, 0.1, 0.0, -0.1}
	vv := []f32{0.3, -0.2, 0.1, 0.0}
	ss := make([]f32, 16)
	defer delete(ss)
	for i in 0 ..< 16 { ss[i] = f32(i) * 0.01 }
	// b, a, a_decay, dt as scalars in the float buffer
	mem.copy(raw_data(fslice(0, 4)), raw_data(qq), 16)
	mem.copy(raw_data(fslice(4, 4)), raw_data(kk), 16)
	mem.copy(raw_data(fslice(8, 4)), raw_data(vv), 16)
	mem.copy(raw_data(fslice(16, 16)), raw_data(ss), 64)
	sc := fslice(40, 4)
	sc[0] = 0.0   // b logit -> sigmoid 0.5
	sc[1] = 0.0   // a
	sc[2] = -0.1  // a_decay
	sc[3] = 0.0   // dt
	vkr.flush_range(g_act, 0, 64 * 4)
	if !vkr.begin() { return false }
	dp := Push{}
	dp.u[0] = 0
	dp.u[1] = 4
	dp.u[2] = 8
	dp.u[3] = 16
	dp.u[4] = 64
	dp.u[5] = 40
	dp.u[6] = 41
	dp.u[7] = 42
	dp.u[8] = 43
	dp.u[9] = 4
	dp.u[10] = 4
	dp.u[11] = 1
	vkr.barrier()
	vkr.dispatch(g_deltap, 1, 1, 1, &dp, size_of(dp), 0)
	if !vkr.submit_wait() { return false }
	vkr.invalidate_range(g_act, 0, 80 * 4)
	beta := sigmoid_f32(0)
	gval := f32(-0.1) * softplus_f32(0)
	gdec := math.exp_f32(gval)
	dst := make([]f32, 16)
	defer delete(dst)
	copy(dst, ss)
	dout := make([]f32, 4)
	defer delete(dout)
	dscr := make([]f32, 4)
	defer delete(dscr)
	delta_recurrent_step(dout, dst, qq, kk, vv, dscr, beta, gdec, 4, 4)
	mx = max_abs_diff(fslice(64, 4), dout)
	smx := max_abs_diff(fslice(16, 16), dst)
	fmt.eprintfln("vulkan_selftest: delta max_abs={} state_max_abs={}", mx, smx)
	// softplus on GPU is f32 log; CPU is f64. Allow a small gap.
	if mx > 1e-4 || smx > 1e-4 { pass = false }
	return pass
}

@(private)
quant_i8 :: proc(v: f32) -> int {
	x := v
	if x > 127 { x = 127 }
	if x < -127 { x = -127 }
	if x >= 0 { return int(math.floor(x + 0.5)) }
	return -int(math.floor(-x + 0.5))
}

@(private)
test_optional :: proc(wbuf: vkr.Buffer) -> bool {
	pass := true
	if g_q8 == 0 {
		fmt.eprintln("vulkan_selftest: intdot skip")
	} else {
		ikinds := [?]ggml.GGML_Type{.Q8_0, .Q4_0}
		for kind in ikinds {
			mx, fx, ok := intdot_case(wbuf, kind)
			fmt.eprintfln("vulkan_selftest: intdot {} max_abs_vs_int={} max_abs_vs_f32={}", kind, mx, fx)
			if !ok || mx > 1e-4 { pass = false }
		}
	}
	if g_coop == 0 {
		fmt.eprintln("vulkan_selftest: coopmat skip")
	} else {
		mx, ok := coop_case(wbuf)
		fmt.eprintfln("vulkan_selftest: coopmat max_abs={}", mx)
		if !ok || mx > 1e-2 { pass = false }
	}
	return pass
}

@(private)
intdot_case :: proc(wbuf: vkr.Buffer, kind: ggml.GGML_Type) -> (vs_int, vs_f32: f32, ok: bool) {
	cols := 32
	rows := 2
	_, bl := ggml.ggml_block_info(kind)
	row_bytes := bl
	raw := ([^]u8)(wbuf.ptr)[:rows * row_bytes]
	fill_quant(kind, raw, row_bytes)
	xs := fslice(0, cols)
	for i in 0 ..< cols { xs[i] = f32(i % 9) * 0.05 - 0.2 }
		yz := fslice(100, rows)
		for i in 0 ..< rows { yz[i] = 0 }
	vkr.flush_range(wbuf, 0, rows * row_bytes)
	vkr.flush_range(g_act, 0, 128 * 4)
	p := Push{}
	p.u[0] = 0
	p.u[1] = 0
	p.u[2] = 100
	p.u[3] = u32(rows)
	p.u[4] = u32(cols)
	p.u[5] = 1
	p.u[6] = u32(kind)
	p.u[11] = u32(row_bytes)
	if !vkr.begin() { return 0, 0, false }
	vkr.barrier()
	vkr.dispatch(g_q8, u32(rows), 1, 1, &p, size_of(p), 0)
	if !vkr.submit_wait() { return 0, 0, false }
	vkr.invalidate_range(g_act, 100 * 4, rows * 4)
	deq := make([]f32, cols)
	defer delete(deq)
	for r in 0 ..< rows {
		ggml.dequant_row(kind, raw[r * row_bytes:], cols, deq)
		fsum: f32 = 0
		for i in 0 ..< cols { fsum += deq[i] * xs[i] }
		// Integer reference: one block of 32.
		amax: f32 = 0
		for i in 0 ..< 32 {
			a := xs[i]
			if a < 0 { a = -a }
			if a > amax { amax = a }
		}
		sx: f32 = 1
		if amax > 0 { sx = amax / 127 }
		idot := 0
		for i in 0 ..< 32 {
			qv: int
			if kind == .Q8_0 {
				bits := int(raw[r * row_bytes + 2 + i])
				qv = bits >= 128 ? bits - 256 : bits
			} else if i < 16 {
				qs := raw[r * row_bytes + 2 + i]
				qv = int(qs & 0xF) - 8
			} else {
				qs := raw[r * row_bytes + 2 + (i - 16)]
				qv = int(qs >> 4) - 8
			}
			id := quant_i8(xs[i] / sx)
			idot += qv * id
		}
		d := ggml_f16_at(raw[r * row_bytes:])
		iref := d * sx * f32(idot)
		got := fslice(100 + r, 1)[0]
		di := got - iref
		if di < 0 { di = -di }
		df := got - fsum
		if df < 0 { df = -df }
		if di > vs_int { vs_int = di }
		if df > vs_f32 { vs_f32 = df }
	}
	return vs_int, vs_f32, true
}

@(private)
ggml_f16_at :: proc(b: []u8) -> f32 {
	h := transmute(f16)(u16(b[0]) | u16(b[1]) << 8)
	return f32(h)
}

@(private)
coop_case :: proc(wbuf: vkr.Buffer) -> (f32, bool) {
	n := 16 * 16
	af := make([]f32, n)
	bf := make([]f32, n)
	defer delete(af)
	defer delete(bf)
	dstb := ([^]u8)(wbuf.ptr)
	for i in 0 ..< n {
		af[i] = f32(i % 7) * 0.01 - 0.02
		bf[i] = f32((i * 3) % 5) * 0.01
		ha := transmute(u16)f16(af[i])
		hb := transmute(u16)f16(bf[i])
		af[i] = f32(transmute(f16)ha)
		bf[i] = f32(transmute(f16)hb)
		dstb[i * 2] = u8(ha)
		dstb[i * 2 + 1] = u8(ha >> 8)
		dstb[512 + i * 2] = u8(hb)
		dstb[512 + i * 2 + 1] = u8(hb >> 8)
	}
	vkr.flush_range(wbuf, 0, 1024)
	if !vkr.begin() { return 0, false }
	vkr.barrier()
	vkr.dispatch(g_coop, 1, 1, 1, nil, 0, 0)
	if !vkr.submit_wait() { return 0, false }
	vkr.invalidate_range(g_act, 0, n * 4)
	got := fslice(0, n)
	mx: f32 = 0
	for r in 0 ..< 16 {
		for c in 0 ..< 16 {
			s: f32 = 0
			for k in 0 ..< 16 {
				s += af[r * 16 + k] * bf[k * 16 + c]
			}
			d := got[r * 16 + c] - s
			if d < 0 { d = -d }
			if d > mx { mx = d }
		}
	}
	return mx, true
}

} // GPU_VULKAN
