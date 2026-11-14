/* CUDA correctness checks against the CPU reference in packages/ggml and ops.odin.
   cuda_nvrtc_check compiles PTX and does not need a GPU.
   cuda_selftest needs a driver and a device (sm_70+). */

package qwen3_5

import cu "cuda:cuda"
import ggml "ggml:ggml"

import "core:c"
import "core:fmt"
import "core:math"
import "core:os"
import "core:strconv"
import "core:time"

@(private = "file")
_cuda_self_import_keep :: proc() {
	_ = cu.BACKEND_TAG
	_ = ggml.QK_K
	_ = c.int(0)
	_ = fmt.tprintf("")
	_ = math.abs(f32(0))
	_ = os.O_RDONLY
	_, _ = strconv.parse_int("0")
	_ = time.now()
}

when GPU_CUDA {

@(private)
s_dev: cu.Device
@(private)
s_ctx: bool
@(private)
s_mod: cu.Module

cuda_nvrtc_check :: proc(arch: string) -> bool {
	opt := fmt.tprintf("--gpu-architecture=%s", arch)
	ptx, log, err := cu.compile_ptx(CUDA_KERNEL_SRC, "odinfer_cuda.cu", []string{opt})
	if len(log) > 0 {
		fmt.eprintfln("nvrtc: %s", log)
	}
	if err != cu.OK {
		fmt.eprintf("cuda: NVRTC compile %s failed: %s\n", arch, cu.error_string(err))
		return false
	}
	fmt.eprintfln("cuda: NVRTC compiled %s (%d bytes PTX) via %s", arch, len(ptx), cu.nv_loaded_from)
	return true
}

@(private)
s_fn :: proc(name: string) -> cu.Function {
	fn, err := cu.module_function(s_mod, name)
	if err != cu.OK || fn == nil {
		fmt.eprintf("cuda_selftest: missing %s (%s)\n", name, cu.error_string(err))
		return nil
	}
	return fn
}

@(private)
s_close :: proc() {
	if s_mod != nil {
		cu.drv.ModuleUnload(s_mod)
		s_mod = nil
	}
	if s_ctx {
		cu.drv.DevicePrimaryCtxRelease_v2(s_dev)
		s_ctx = false
	}
}

@(private)
s_open :: proc() -> bool {
	if cu.load_driver() != cu.OK {
		fmt.eprintln("cuda_selftest: libcuda not found")
		return false
	}
	if cu.drv.Init(0) != cu.OK {
		fmt.eprintln("cuda_selftest: cuInit failed")
		return false
	}
	n: c.int
	if cu.drv.DeviceGetCount(&n) != cu.OK || n < 1 {
		fmt.eprintln("cuda_selftest: no CUDA device")
		return false
	}
	if cu.drv.DeviceGet(&s_dev, 0) != cu.OK {
		return false
	}
	major, minor: c.int
	cu.drv.DeviceGetAttribute(&major, cu.ATTR_COMPUTE_CAPABILITY_MAJOR, s_dev)
	cu.drv.DeviceGetAttribute(&minor, cu.ATTR_COMPUTE_CAPABILITY_MINOR, s_dev)
	ctx: cu.Context
	if cu.drv.DevicePrimaryCtxRetain(&ctx, s_dev) != cu.OK {
		return false
	}
	s_ctx = true
	if cu.drv.CtxSetCurrent(ctx) != cu.OK {
		s_close()
		return false
	}
	arch := fmt.tprintf("compute_%d%d", int(major), int(minor))
	opt := fmt.tprintf("--gpu-architecture=%s", arch)
	ptx, log, err := cu.compile_ptx(CUDA_KERNEL_SRC, "odinfer_cuda.cu", []string{opt})
	if len(log) > 0 {
		fmt.eprintfln("nvrtc: %s", log)
	}
	if err != cu.OK {
		fmt.eprintf("cuda_selftest: compile %s failed: %s\n", arch, cu.error_string(err))
		s_close()
		return false
	}
	mod, jit, merr := cu.module_load_ptx(ptx)
	if len(jit) > 0 {
		fmt.eprintfln("jit: %s", jit)
	}
	if merr != cu.OK {
		fmt.eprintf("cuda_selftest: module load failed: %s (%d)\n", cu.error_string(merr), int(merr))
		s_close()
		return false
	}
	s_mod = mod
	fmt.eprintfln("cuda_selftest: device sm_%d%d arch %s", int(major), int(minor), arch)
	return true
}

Case_Err :: struct {
	max_abs: f32,
	tol:     f32,
	bad:     bool,
	printed: int,
}

@(private)
s_note :: proc(st: ^Case_Err, name: string, got, want, atol, rtol: f32) {
	d := math.abs(got - want)
	tol := atol + rtol * math.abs(want)
	if math.is_nan(d) || d > st.max_abs {
		st.max_abs = d
		st.tol = tol
	}
	if math.is_nan(got) || math.is_inf(got) || d > tol {
		st.bad = true
		if st.printed < 4 {
			fmt.eprintf("cuda_selftest: %s got %.6g want %.6g diff %.3g tol %.3g\n", name, got, want, d, tol)
			st.printed += 1
		}
	}
}

// Printed for every case, pass or fail. tol is the allowance at the element
// with the largest absolute error (atol + rtol*|ref|).
@(private)
s_report :: proc(label: string, st: Case_Err, atol, rtol: f32) -> bool {
	status := "PASS"
	if st.bad {
		status = "FAIL"
	}
	fmt.eprintfln("cuda_selftest: %s %s  max_abs=%.6g  tol=%.6g  (atol=%.3g rtol=%.3g)", status, label, st.max_abs, st.tol, atol, rtol)
	return !st.bad
}

@(private)
fill_weight :: proc(kind: ggml.GGML_Type, rows, cols: int) -> []u8 {
	elems, bsz := ggml.ggml_block_info(kind)
	nb := cols / elems
	raw := make([]u8, rows * nb * bsz)
	if kind == .F32 {
		f := (cast([^]f32)raw_data(raw))[:rows * cols]
		for i in 0 ..< rows * cols {
			f[i] = (f32(i % 7) - 3) * 0.02
		}
		return raw
	}
	if kind == .F16 {
		for i in 0 ..< rows * cols {
			write_f16_le(raw[i * 2:], (f32(i % 7) - 3) * 0.02)
		}
		return raw
	}
	for i in 0 ..< len(raw) {
		raw[i] = u8((i * 17 + 9) & 0xFF)
	}
	for b in 0 ..< rows * nb {
		base := b * bsz
		#partial switch kind {
		case .Q8_K:
			bits := transmute(u32)(f32(0.02))
			raw[base + 0] = u8(bits)
			raw[base + 1] = u8(bits >> 8)
			raw[base + 2] = u8(bits >> 16)
			raw[base + 3] = u8(bits >> 24)
		case .Q2_K:
			write_f16_le(raw[base + 80:], 0.05)
			write_f16_le(raw[base + 82:], 0.01)
		case .Q3_K:
			write_f16_le(raw[base + 108:], 0.05)
		case .Q6_K:
			write_f16_le(raw[base + 208:], 0.02)
		case .Q4_1, .Q5_1:
			write_f16_le(raw[base:], 0.05)
			write_f16_le(raw[base + 2:], 0.01)
		case .Q4_K, .Q5_K:
			write_f16_le(raw[base:], 0.05)
			write_f16_le(raw[base + 2:], 0.01)
		case:
			write_f16_le(raw[base:], 0.05)
		}
	}
	return raw
}

@(private)
cpu_gemv :: proc(kind: ggml.GGML_Type, raw: []u8, x: []f32, rows, cols: int) -> []f32 {
	elems, bsz := ggml.ggml_block_info(kind)
	rb := (cols / elems) * bsz
	out := make([]f32, rows)
	scratch := make([]f32, cols)
	defer delete(scratch)
	for r in 0 ..< rows {
		ggml.dequant_row(kind, raw[r * rb:], cols, scratch)
		sum: f32 = 0
		for j in 0 ..< cols {
			sum += scratch[j] * x[j]
		}
		out[r] = sum
	}
	return out
}

@(private)
run_gemv :: proc(kind: ggml.GGML_Type, rows, cols: int, atol, rtol: f32) -> bool {
	tag := cuda_kind_tag(kind)
	fn := s_fn(fmt.tprintf("gemv_%s", tag))
	if fn == nil {
		return false
	}
	raw := fill_weight(kind, rows, cols)
	defer delete(raw)
	x := make([]f32, cols)
	defer delete(x)
	for i in 0 ..< cols {
		x[i] = (f32(i % 5) - 2) * 0.05
	}
	ref := cpu_gemv(kind, raw, x, rows, cols)
	defer delete(ref)
	dw, e1 := cu.alloc(len(raw))
	dx, e2 := cu.alloc(cols * 4)
	dy, e3 := cu.alloc(rows * 4)
	defer cu.free(dw)
	defer cu.free(dx)
	defer cu.free(dy)
	if e1 != cu.OK || e2 != cu.OK || e3 != cu.OK {
		fmt.eprintln("cuda_selftest: alloc failed")
		return false
	}
	cu.upload(dw, raw_data(raw), len(raw))
	cu.upload(dx, raw_data(x), cols * 4)
	wb, xx, yy := dw, dx, dy
	nn, dd := i32(cols), i32(rows)
	args := [?]rawptr{&wb, &xx, &yy, &nn, &dd}
	if cu.launch(fn, u32(rows), 1, 1, 32, 1, 1, 0, args[:]) != cu.OK {
		fmt.eprintf("cuda_selftest: gemv %s launch failed\n", tag)
		return false
	}
	got := make([]f32, rows)
	defer delete(got)
	cu.download(raw_data(got), dy, rows * 4)
	st: Case_Err
	for r in 0 ..< rows {
		s_note(&st, fmt.tprintf("gemv %s row %d", tag, r), got[r], ref[r], atol, rtol)
	}
	return s_report(fmt.tprintf("gemv %s", tag), st, atol, rtol)
}

@(private)
run_gemv2 :: proc(kind: ggml.GGML_Type, rows, cols: int, atol, rtol: f32) -> bool {
	tag := cuda_kind_tag(kind)
	fn := s_fn(fmt.tprintf("gemv2_%s", tag))
	qfn := s_fn("quant_q8")
	if fn == nil || qfn == nil {
		fmt.eprintfln("cuda_selftest: gemv2 %s missing", tag)
		return false
	}
	raw := fill_weight(kind, rows, cols)
	defer delete(raw)
	x := make([]f32, cols)
	defer delete(x)
	for i in 0 ..< cols {
		x[i] = (f32(i % 5) - 2) * 0.05
	}
	ref := cpu_gemv(kind, raw, x, rows, cols)
	defer delete(ref)
	dw, e1 := cu.alloc(len(raw) + 64)
	dx, e2 := cu.alloc(cols * 4)
	dy, e3 := cu.alloc(rows * 4)
	ng := cols / 32
	dqs, e4 := cu.alloc(cols)
	dsc, e5 := cu.alloc(ng * 4)
	dsm, e6 := cu.alloc(ng * 8)
	defer cu.free(dw)
	defer cu.free(dx)
	defer cu.free(dy)
	defer cu.free(dqs)
	defer cu.free(dsc)
	defer cu.free(dsm)
	if e1 != cu.OK || e2 != cu.OK || e3 != cu.OK || e4 != cu.OK || e5 != cu.OK || e6 != cu.OK {
		fmt.eprintln("cuda_selftest: gemv2 alloc failed")
		return false
	}
	cu.upload(dw, raw_data(raw), len(raw))
	cu.upload(dx, raw_data(x), cols * 4)
	xx, qs, sc, sm := dx, dqs, dsc, dsm
	nn := i32(cols)
	qargs := [?]rawptr{&xx, &qs, &sc, &sm, &nn}
	if cu.launch(qfn, u32((ng + 127) / 128), 1, 1, 128, 1, 1, 0, qargs[:]) != cu.OK {
		fmt.eprintfln("cuda_selftest: quant_q8 launch failed (%s)", tag)
		return false
	}
	wb, yy := dw, dy
	dd := i32(rows)
	acc := i32(0)
	warps := 4
	gx := u32((rows + warps * 4 - 1) / (warps * 4))
	bx := u32(warps * 32)
	launched: bool
	if kind == .Q4_K || kind == .Q5_K {
		args := [?]rawptr{&wb, &qs, &sc, &sm, &yy, &nn, &dd, &acc}
		launched = cu.launch(fn, gx, 1, 1, bx, 1, 1, 0, args[:]) == cu.OK
	} else {
		args := [?]rawptr{&wb, &qs, &sc, &yy, &nn, &dd, &acc}
		launched = cu.launch(fn, gx, 1, 1, bx, 1, 1, 0, args[:]) == cu.OK
	}
	if !launched || cu.sync() != cu.OK {
		fmt.eprintfln("cuda_selftest: gemv2 launch failed (%s)", tag)
		return false
	}
	got := make([]f32, rows)
	defer delete(got)
	cu.download(raw_data(got), dy, rows * 4)
	st: Case_Err
	for r in 0 ..< rows {
		s_note(&st, fmt.tprintf("gemv2 %s row %d", tag, r), got[r], ref[r], atol, rtol)
	}
	ok := s_report(fmt.tprintf("gemv2 %s %dx%d", tag, rows, cols), st, atol, rtol)
	if !ok {
		return false
	}
	if kind == .Q4_K {
		nfn := s_fn("gemv2_q4nc_k")
		xfn := s_fn("gemv2_q4x_k")
		if nfn == nil || xfn == nil {
			fmt.eprintln("cuda_selftest: gemv2_q4nc_k or gemv2_q4x_k missing")
			return false
		}
		acc = 0
		nargs := [?]rawptr{&wb, &qs, &sc, &sm, &yy, &nn, &dd, &acc}
		if cu.launch(nfn, gx, 1, 1, bx, 1, 1, 0, nargs[:]) != cu.OK || cu.sync() != cu.OK {
			fmt.eprintln("cuda_selftest: gemv2_q4nc_k launch failed")
			return false
		}
		cu.download(raw_data(got), dy, rows * 4)
		st = {}
		for r in 0 ..< rows {
			s_note(&st, fmt.tprintf("gemv2_q4nc row %d", r), got[r], ref[r], atol, rtol)
		}
		if !s_report(fmt.tprintf("gemv2_q4nc %dx%d", rows, cols), st, atol, rtol) {
			return false
		}
		gx8 := u32((rows + warps * 8 - 1) / (warps * 8))
		acc = 0
		xargs := [?]rawptr{&wb, &qs, &sc, &sm, &yy, &nn, &dd, &acc}
		if cu.launch(xfn, gx8, 1, 1, bx, 1, 1, 0, xargs[:]) != cu.OK || cu.sync() != cu.OK {
			fmt.eprintln("cuda_selftest: gemv2_q4x_k launch failed")
			return false
		}
		cu.download(raw_data(got), dy, rows * 4)
		st = {}
		for r in 0 ..< rows {
			s_note(&st, fmt.tprintf("gemv2_q4x row %d", r), got[r], ref[r], atol, rtol)
		}
		if !s_report(fmt.tprintf("gemv2_q4x %dx%d", rows, cols), st, atol, rtol) {
			return false
		}
		if cols / 256 >= 2 {
			if cu.memset(dy, 0, rows * 4) != cu.OK {
				fmt.eprintln("cuda_selftest: q4x split memset failed")
				return false
			}
			acc = 0
			sargs := [?]rawptr{&wb, &qs, &sc, &sm, &yy, &nn, &dd, &acc}
			if cu.launch(xfn, gx8, 2, 1, bx, 1, 1, 0, sargs[:]) != cu.OK || cu.sync() != cu.OK {
				fmt.eprintln("cuda_selftest: gemv2_q4x split launch failed")
				return false
			}
			cu.download(raw_data(got), dy, rows * 4)
			st = {}
			for r in 0 ..< rows {
				s_note(&st, fmt.tprintf("gemv2_q4x split row %d", r), got[r], ref[r], atol, rtol)
			}
			if !s_report(fmt.tprintf("gemv2_q4x split %dx%d", rows, cols), st, atol, rtol) {
				return false
			}
		}
	}
	if kind == .Q6_K {
		sfn := s_fn("gemv2_q6s_k")
		if sfn == nil {
			fmt.eprintln("cuda_selftest: gemv2_q6s_k missing")
			return false
		}
		args := [?]rawptr{&wb, &qs, &sc, &yy, &nn, &dd, &acc}
		if cu.launch(sfn, gx, 1, 1, bx, 1, 1, 0, args[:]) != cu.OK || cu.sync() != cu.OK {
			fmt.eprintln("cuda_selftest: gemv2_q6s_k launch failed")
			return false
		}
		cu.download(raw_data(got), dy, rows * 4)
		st = {}
		for r in 0 ..< rows {
			s_note(&st, fmt.tprintf("gemv2_q6s row %d", r), got[r], ref[r], atol, rtol)
		}
		if !s_report(fmt.tprintf("gemv2_q6s %dx%d", rows, cols), st, atol, rtol) {
			return false
		}
	}
	if kind == .Q4_K || kind == .Q6_K {
		rname := "gemv2r1_q4_k"
		if kind == .Q6_K {
			rname = "gemv2r1_q6_k"
		}
		rfn := s_fn(rname)
		if rfn == nil {
			fmt.eprintfln("cuda_selftest: %s missing", rname)
			return false
		}
		gx1 := u32((rows + warps - 1) / warps)
		acc = 0
		launched = false
		if kind == .Q4_K {
			args := [?]rawptr{&wb, &qs, &sc, &sm, &yy, &nn, &dd, &acc}
			launched = cu.launch(rfn, gx1, 1, 1, bx, 1, 1, 0, args[:]) == cu.OK
		} else {
			args := [?]rawptr{&wb, &qs, &sc, &yy, &nn, &dd, &acc}
			launched = cu.launch(rfn, gx1, 1, 1, bx, 1, 1, 0, args[:]) == cu.OK
		}
		if !launched || cu.sync() != cu.OK {
			fmt.eprintfln("cuda_selftest: %s launch failed", rname)
			return false
		}
		cu.download(raw_data(got), dy, rows * 4)
		st = {}
		for r in 0 ..< rows {
			s_note(&st, fmt.tprintf("%s row %d", rname, r), got[r], ref[r], atol, rtol)
		}
		if !s_report(fmt.tprintf("%s %dx%d", rname, rows, cols), st, atol, rtol) {
			return false
		}
		// accumulate=1 adds onto whatever is already in the output.
		bias := make([]f32, rows)
		defer delete(bias)
		for r in 0 ..< rows {
			bias[r] = 0.25
		}
		cu.upload(dy, raw_data(bias), rows * 4)
		acc = 1
		if kind == .Q4_K {
			args := [?]rawptr{&wb, &qs, &sc, &sm, &yy, &nn, &dd, &acc}
			launched = cu.launch(rfn, gx1, 1, 1, bx, 1, 1, 0, args[:]) == cu.OK
		} else {
			args := [?]rawptr{&wb, &qs, &sc, &yy, &nn, &dd, &acc}
			launched = cu.launch(rfn, gx1, 1, 1, bx, 1, 1, 0, args[:]) == cu.OK
		}
		if !launched || cu.sync() != cu.OK {
			fmt.eprintfln("cuda_selftest: %s accumulate launch failed", rname)
			return false
		}
		cu.download(raw_data(got), dy, rows * 4)
		st = {}
		for r in 0 ..< rows {
			s_note(&st, fmt.tprintf("%s accum row %d", rname, r), got[r], ref[r] + 0.25, atol, rtol)
		}
		if !s_report(fmt.tprintf("%s accumulate %dx%d", rname, rows, cols), st, atol, rtol) {
			return false
		}
		if cols / 256 >= 2 {
			if cu.memset(dy, 0, rows * 4) != cu.OK {
				fmt.eprintln("cuda_selftest: split memset failed")
				return false
			}
			acc = 0
			launched = false
			if kind == .Q4_K {
				args := [?]rawptr{&wb, &qs, &sc, &sm, &yy, &nn, &dd, &acc}
				launched = cu.launch(fn, gx, 2, 1, bx, 1, 1, 0, args[:]) == cu.OK
			} else {
				args := [?]rawptr{&wb, &qs, &sc, &yy, &nn, &dd, &acc}
				launched = cu.launch(fn, gx, 2, 1, bx, 1, 1, 0, args[:]) == cu.OK
			}
			if !launched || cu.sync() != cu.OK {
				fmt.eprintfln("cuda_selftest: gemv2 split launch failed (%s)", tag)
				return false
			}
			cu.download(raw_data(got), dy, rows * 4)
			st = {}
			for r in 0 ..< rows {
				s_note(&st, fmt.tprintf("gemv2 split %s row %d", tag, r), got[r], ref[r], atol, rtol)
			}
			if !s_report(fmt.tprintf("gemv2 split %s %dx%d", tag, rows, cols), st, atol, rtol) {
				return false
			}
		}
	}
	return true
}

@(private)
run_mmq :: proc(kind: ggml.GGML_Type, rows, cols, T: int) -> bool {
	tag := cuda_kind_tag(kind)
	fn := s_fn(fmt.tprintf("mmq_%s", tag))
	if fn == nil {
		return false
	}
	raw := fill_weight(kind, rows, cols)
	defer delete(raw)
	x := make([]f32, T * cols)
	defer delete(x)
	for t in 0 ..< T {
		for j in 0 ..< cols {
			x[t * cols + j] = (f32((t + 1) * (j + 3) % 9) - 4) * 0.03
		}
	}
	ref := make([]f32, T * rows)
	defer delete(ref)
	for t in 0 ..< T {
		row := cpu_gemv(kind, raw, x[t * cols:][:cols], rows, cols)
		copy(ref[t * rows:][:rows], row)
		delete(row)
	}
	dw, _ := cu.alloc(len(raw))
	dx, _ := cu.alloc(T * cols * 4)
	dy, _ := cu.alloc(T * rows * 4)
	defer cu.free(dw)
	defer cu.free(dx)
	defer cu.free(dy)
	cu.upload(dw, raw_data(raw), len(raw))
	cu.upload(dx, raw_data(x), T * cols * 4)
	wb, xx, yy := dw, dx, dy
	nn, dd, tt := i32(cols), i32(rows), i32(T)
	args := [?]rawptr{&wb, &xx, &yy, &nn, &dd, &tt}
	if cu.launch(fn, u32(rows), 1, 1, 32, 1, 1, 0, args[:]) != cu.OK {
		fmt.eprintf("cuda_selftest: mmq %s launch failed\n", tag)
		return false
	}
	got := make([]f32, T * rows)
	defer delete(got)
	cu.download(raw_data(got), dy, T * rows * 4)
	st: Case_Err
	atol, rtol := f32(2e-2), f32(1e-3)
	for i in 0 ..< T * rows {
		s_note(&st, fmt.tprintf("mmq %s [%d]", tag, i), got[i], ref[i], atol, rtol)
	}
	return s_report(fmt.tprintf("mmq %s T=%d", tag, T), st, atol, rtol)
}

@(private)
run_tc :: proc(kind: ggml.GGML_Type, rows, cols, T: int, atol, rtol: f32) -> bool {
	tag := cuda_kind_tag(kind)
	fn := s_fn(fmt.tprintf("tc_%s", tag))
	if fn == nil {
		fmt.eprintf("cuda_selftest: missing tc_%s\n", tag)
		return false
	}
	raw := fill_weight(kind, rows, cols)
	defer delete(raw)
	x := make([]f32, T * cols)
	defer delete(x)
	for t in 0 ..< T {
		for j in 0 ..< cols {
			x[t * cols + j] = (f32((t + 1) * (j + 3) % 9) - 4) * 0.03
		}
	}
	ref := make([]f32, T * rows)
	defer delete(ref)
	for t in 0 ..< T {
		row := cpu_gemv(kind, raw, x[t * cols:][:cols], rows, cols)
		copy(ref[t * rows:][:rows], row)
		delete(row)
	}
	dw, _ := cu.alloc(len(raw))
	dx, _ := cu.alloc(T * cols * 4)
	dy, _ := cu.alloc(T * rows * 4)
	defer cu.free(dw)
	defer cu.free(dx)
	defer cu.free(dy)
	cu.upload(dw, raw_data(raw), len(raw))
	cu.upload(dx, raw_data(x), T * cols * 4)
	wb, xx, yy := dw, dx, dy
	nn, dd, tt := i32(cols), i32(rows), i32(T)
	args := [?]rawptr{&wb, &xx, &yy, &nn, &dd, &tt}
	gx := u32((rows + 63) / 64)
	gy := u32((T + 63) / 64)
	if cu.launch(fn, gx, gy, 1, 512, 1, 1, 0, args[:]) != cu.OK {
		fmt.eprintf("cuda_selftest: tc %s launch failed\n", tag)
		return false
	}
	got := make([]f32, T * rows)
	defer delete(got)
	cu.download(raw_data(got), dy, T * rows * 4)
	st: Case_Err
	for i in 0 ..< T * rows {
		s_note(&st, fmt.tprintf("tc %s [%d]", tag, i), got[i], ref[i], atol, rtol)
	}
	return s_report(fmt.tprintf("tc %s T=%d rows=%d cols=%d", tag, T, rows, cols), st, atol, rtol)
}

@(private)
run_tc2 :: proc(kind: ggml.GGML_Type, rows, cols, T: int, atol, rtol: f32, wide: bool) -> bool {
	name := "tc2s_q4_k"
	if wide {
		name = "tc2_q4_k"
	}
	if kind == .Q6_K {
		name = "tc2s_q6_k"
		if wide {
			name = "tc2_q6_k"
		}
	}
	fn := s_fn(name)
	if fn == nil {
		fmt.eprintf("cuda_selftest: missing %s\n", name)
		return false
	}
	raw := fill_weight(kind, rows, cols)
	defer delete(raw)
	x := make([]f32, T * cols)
	defer delete(x)
	for t in 0 ..< T {
		for j in 0 ..< cols {
			x[t * cols + j] = (f32((t + 1) * (j + 3) % 9) - 4) * 0.03
		}
	}
	ref := make([]f32, T * rows)
	defer delete(ref)
	for t in 0 ..< T {
		row := cpu_gemv(kind, raw, x[t * cols:][:cols], rows, cols)
		copy(ref[t * rows:][:rows], row)
		delete(row)
	}
	dw, _ := cu.alloc(len(raw))
	dx, _ := cu.alloc(T * cols * 4)
	dy, _ := cu.alloc(T * rows * 4)
	defer cu.free(dw)
	defer cu.free(dx)
	defer cu.free(dy)
	cu.upload(dw, raw_data(raw), len(raw))
	cu.upload(dx, raw_data(x), T * cols * 4)
	wb, xx, yy := dw, dx, dy
	nn, dd, tt := i32(cols), i32(rows), i32(T)
	args := [?]rawptr{&wb, &xx, &yy, &nn, &dd, &tt}
	gx := u32((rows + 31) / 32)
	gy := u32((T + 31) / 32)
	if wide {
		gy = u32((T + 127) / 128)
	}
	if cu.launch(fn, gx, gy, 1, 256, 1, 1, 0, args[:]) != cu.OK {
		fmt.eprintf("cuda_selftest: %s launch failed\n", name)
		return false
	}
	got := make([]f32, T * rows)
	defer delete(got)
	cu.download(raw_data(got), dy, T * rows * 4)
	st: Case_Err
	for i in 0 ..< T * rows {
		s_note(&st, fmt.tprintf("%s [%d]", name, i), got[i], ref[i], atol, rtol)
	}
	return s_report(fmt.tprintf("%s T=%d rows=%d cols=%d", name, T, rows, cols), st, atol, rtol)
}

@(private)
run_tc2m :: proc(kind: ggml.GGML_Type, rows, cols, T: int, atol, rtol: f32) -> bool {
	name := "tc2m_q4_k"
	if kind == .Q6_K {
		name = "tc2m_q6_k"
	}
	fn := s_fn(name)
	if fn == nil {
		fmt.eprintf("cuda_selftest: missing %s\n", name)
		return false
	}
	raw := fill_weight(kind, rows, cols)
	defer delete(raw)
	x := make([]f32, T * cols)
	defer delete(x)
	for t in 0 ..< T {
		for j in 0 ..< cols {
			x[t * cols + j] = (f32((t + 1) * (j + 3) % 9) - 4) * 0.03
		}
	}
	ref := make([]f32, T * rows)
	defer delete(ref)
	for t in 0 ..< T {
		row := cpu_gemv(kind, raw, x[t * cols:][:cols], rows, cols)
		copy(ref[t * rows:][:rows], row)
		delete(row)
	}
	dw, _ := cu.alloc(len(raw))
	dx, _ := cu.alloc(T * cols * 4)
	dy, _ := cu.alloc(T * rows * 4)
	defer cu.free(dw)
	defer cu.free(dx)
	defer cu.free(dy)
	cu.upload(dw, raw_data(raw), len(raw))
	cu.upload(dx, raw_data(x), T * cols * 4)
	wb, xx, yy := dw, dx, dy
	nn, dd, tt := i32(cols), i32(rows), i32(T)
	args := [?]rawptr{&wb, &xx, &yy, &nn, &dd, &tt}
	gx := u32((rows + 63) / 64)
	gy := u32((T + 63) / 64)
	if cu.launch(fn, gx, gy, 1, 512, 1, 1, 0, args[:]) != cu.OK {
		fmt.eprintf("cuda_selftest: %s launch failed\n", name)
		return false
	}
	got := make([]f32, T * rows)
	defer delete(got)
	cu.download(raw_data(got), dy, T * rows * 4)
	st: Case_Err
	for i in 0 ..< T * rows {
		s_note(&st, fmt.tprintf("%s [%d]", name, i), got[i], ref[i], atol, rtol)
	}
	return s_report(fmt.tprintf("%s T=%d rows=%d cols=%d", name, T, rows, cols), st, atol, rtol)
}

@(private)
run_elem :: proc() -> bool {
	// RMSNorm
	n := 64
	x := make([]f32, n)
	w := make([]f32, n)
	ref := make([]f32, n)
	defer delete(x)
	defer delete(w)
	defer delete(ref)
	for i in 0 ..< n {
		x[i] = (f32(i % 9) - 4) * 0.1
		w[i] = 0.8 + f32(i % 5) * 0.05
	}
	rmsnorm(ref, x, w, 1e-6)
	dx, _ := cu.alloc(n * 4)
	dw, _ := cu.alloc(n * 4)
	dy, _ := cu.alloc(n * 4)
	defer cu.free(dx)
	defer cu.free(dw)
	defer cu.free(dy)
	cu.upload(dx, raw_data(x), n * 4)
	cu.upload(dw, raw_data(w), n * 4)
	fn := s_fn("rmsnorm_k")
	if fn == nil {
		return false
	}
	a, b, c0 := dx, dy, dw
	sz := i32(n)
	ep := f32(1e-6)
	args := [?]rawptr{&a, &b, &c0, &sz, &ep}
	if cu.launch(fn, 1, 1, 1, 128, 1, 1, 0, args[:]) != cu.OK {
		fmt.eprintln("cuda_selftest: rmsnorm launch failed")
		return false
	}
	got := make([]f32, n)
	defer delete(got)
	cu.download(raw_data(got), dy, n * 4)
	st_rms: Case_Err
	for i in 0 ..< n {
		s_note(&st_rms, fmt.tprintf("rmsnorm[%d]", i), got[i], ref[i], 1e-4, 1e-4)
	}
	if !s_report("rmsnorm", st_rms, 1e-4, 1e-4) {
		return false
	}

	// residual add then the same RMSNorm, one kernel. n is a multiple of 32.
	yv := make([]f32, n)
	sumv := make([]f32, n)
	defer delete(yv)
	defer delete(sumv)
	for i in 0 ..< n {
		yv[i] = (f32(i % 7) - 3) * 0.05
		sumv[i] = x[i] + yv[i]
	}
	ref2 := make([]f32, n)
	defer delete(ref2)
	rmsnorm(ref2, sumv, w, 1e-6)
	dy2, _ := cu.alloc(n * 4)
	dqs, _ := cu.alloc(n)
	dsc, _ := cu.alloc((n / 32) * 4)
	dsm, _ := cu.alloc((n / 32) * 8)
	defer cu.free(dy2)
	defer cu.free(dqs)
	defer cu.free(dsc)
	defer cu.free(dsm)
	cu.upload(dx, raw_data(x), n * 4)
	cu.upload(dy2, raw_data(yv), n * 4)
	rn := s_fn("residual_rmsnorm_q8")
	if rn == nil {
		fmt.eprintln("cuda_selftest: residual_rmsnorm_q8 missing")
		return false
	}
	rx, ry, ro, rw := dx, dy2, dy, dw
	rqs, rsc, rsm := dqs, dsc, dsm
	rn_args := [?]rawptr{&rx, &ry, &ro, &rw, &rqs, &rsc, &rsm, &sz, &ep}
	if cu.launch(rn, 1, 1, 1, 128, 1, 1, 0, rn_args[:]) != cu.OK {
		fmt.eprintln("cuda_selftest: residual_rmsnorm_q8 launch failed")
		return false
	}
	got2 := make([]f32, n)
	gotx := make([]f32, n)
	defer delete(got2)
	defer delete(gotx)
	cu.download(raw_data(got2), dy, n * 4)
	cu.download(raw_data(gotx), dx, n * 4)
	st_rn: Case_Err
	for i in 0 ..< n {
		s_note(&st_rn, fmt.tprintf("residual_norm.x[%d]", i), gotx[i], sumv[i], 1e-5, 1e-5)
		s_note(&st_rn, fmt.tprintf("residual_norm[%d]", i), got2[i], ref2[i], 1e-4, 1e-4)
	}
	if !s_report("residual_rmsnorm_q8", st_rn, 1e-4, 1e-4) {
		return false
	}

	// RoPE, 2 heads, head_dim 8, rotary 4
	hd, rot, nh, pos := 8, 4, 2, 3
	vec := make([]f32, nh * hd)
	cpuv := make([]f32, nh * hd)
	defer delete(vec)
	defer delete(cpuv)
	for i in 0 ..< len(vec) {
		vec[i] = (f32(i % 5) - 2) * 0.3
		cpuv[i] = vec[i]
	}
	inv := make([]f32, rot / 2)
	defer delete(inv)
	inv_freq_default(10000, rot, inv)
	for h in 0 ..< nh {
		apply_mrope_text_head(cpuv[h * hd:][:hd], pos, inv, rot)
	}
	dv, _ := cu.alloc(len(vec) * 4)
	defer cu.free(dv)
	cu.upload(dv, raw_data(vec), len(vec) * 4)
	rf := s_fn("rope_partial")
	if rf == nil {
		return false
	}
	vv := dv
	ihd, ipos, irot, inh := i32(hd), i32(pos), i32(rot), i32(nh)
	theta := f32(10000)
	slot: cu.Device_Ptr
	rargs := [?]rawptr{&vv, &ihd, &ipos, &theta, &irot, &inh, &slot}
	total := nh * (rot / 2)
	if cu.launch(rf, u32((total + 127) / 128), 1, 1, 128, 1, 1, 0, rargs[:]) != cu.OK {
		fmt.eprintln("cuda_selftest: rope launch failed")
		return false
	}
	gotv := make([]f32, len(vec))
	defer delete(gotv)
	cu.download(raw_data(gotv), dv, len(vec) * 4)
	st_rope: Case_Err
	for i in 0 ..< len(vec) {
		s_note(&st_rope, fmt.tprintf("rope[%d]", i), gotv[i], cpuv[i], 1e-4, 1e-4)
	}
	if !s_report("rope", st_rope, 1e-4, 1e-4) {
		return false
	}

	// SwiGLU
	m := 17
	hb := make([]f32, m)
	hb2 := make([]f32, m)
	defer delete(hb)
	defer delete(hb2)
	for i in 0 ..< m {
		hb[i] = (f32(i) - 8) * 0.4
		hb2[i] = (f32(i % 3) - 1) * 0.7
	}
	want := make([]f32, m)
	defer delete(want)
	for i in 0 ..< m {
		want[i] = silu_f32(hb[i]) * hb2[i]
	}
	dh, _ := cu.alloc(m * 4)
	dh2, _ := cu.alloc(m * 4)
	defer cu.free(dh)
	defer cu.free(dh2)
	cu.upload(dh, raw_data(hb), m * 4)
	cu.upload(dh2, raw_data(hb2), m * 4)
	sf := s_fn("swiglu")
	if sf == nil {
		return false
	}
	ha, hb2p := dh, dh2
	nn := i32(m)
	sargs := [?]rawptr{&ha, &hb2p, &nn}
	if cu.launch(sf, 1, 1, 1, 32, 1, 1, 0, sargs[:]) != cu.OK {
		return false
	}
	gout := make([]f32, m)
	defer delete(gout)
	cu.download(raw_data(gout), dh, m * 4)
	st_swi: Case_Err
	for i in 0 ..< m {
		s_note(&st_swi, fmt.tprintf("swiglu[%d]", i), gout[i], want[i], 1e-5, 1e-5)
	}
	return s_report("swiglu", st_swi, 1e-5, 1e-5)
}

@(private)
run_kv_attn :: proc() -> bool {
	// One head, head_dim 32, two positions. Mixed V store + attention.
	hd := 32
	k := make([]f32, hd)
	v := make([]f32, hd)
	q := make([]f32, hd)
	defer delete(k)
	defer delete(v)
	defer delete(q)
	for i in 0 ..< hd {
		k[i] = (f32(i % 7) - 3) * 0.2
		v[i] = (f32(i % 5) - 1) * 0.5
		q[i] = (f32(i % 4) - 1.5) * 0.3
	}
	// Spike an outlier so the mixed-V index is not trivial.
	v[3] = 4.5
	seq := 4
	k_pos := 34
	v_pos := 40
	kc := make([]u8, seq * k_pos)
	vc := make([]u8, seq * v_pos)
	defer delete(kc)
	defer delete(vc)
	dk, _ := cu.alloc(hd * 4)
	dv, _ := cu.alloc(hd * 4)
	dq, _ := cu.alloc(hd * 4)
	dkc, _ := cu.alloc(len(kc))
	dvc, _ := cu.alloc(len(vc))
	dsc, _ := cu.alloc(seq * 4)
	dxb, _ := cu.alloc(hd * 4)
	defer cu.free(dk)
	defer cu.free(dv)
	defer cu.free(dq)
	defer cu.free(dkc)
	defer cu.free(dvc)
	defer cu.free(dsc)
	defer cu.free(dxb)
	cu.upload(dk, raw_data(k), hd * 4)
	cu.upload(dv, raw_data(v), hd * 4)
	cu.memset(dkc, 0, len(kc))
	cu.memset(dvc, 0, len(vc))
	store := s_fn("store_kv_q8v")
	if store == nil {
		return false
	}
	// pos 0 = k, pos 1 = a second copy scaled, so attention has two keys.
	for pos in 0 ..< 2 {
		scale := f32(1)
		if pos == 1 {
			scale = 0.5
		}
		ks := make([]f32, hd)
		vs := make([]f32, hd)
		for i in 0 ..< hd {
			ks[i] = k[i] * scale
			vs[i] = v[i] * scale
		}
		cu.upload(dk, raw_data(ks), hd * 4)
		cu.upload(dv, raw_data(vs), hd * 4)
		a, b, c0, d := dk, dv, dkc, dvc
		nb, kb, vb, ps := i32(1), i32(k_pos), i32(v_pos), i32(pos)
		slot: cu.Device_Ptr
		args := [?]rawptr{&a, &b, &c0, &d, &nb, &kb, &vb, &ps, &slot}
		if cu.launch(store, 1, 1, 1, 1, 1, 1, 0, args[:]) != cu.OK {
			fmt.eprintln("cuda_selftest: store_kv_q8v failed")
			delete(ks)
			delete(vs)
			return false
		}
		delete(ks)
		delete(vs)
	}
	cu.download(raw_data(kc), dkc, len(kc))
	cu.download(raw_data(vc), dvc, len(vc))
	// CPU quant of pos 0 and compare dequant (f16 scale may differ by an ulp).
	refk := make([]u8, 34)
	refv := make([]u8, 40)
	defer delete(refk)
	defer delete(refv)
	quantize_q8_block(k, refk)
	quantize_q8_v_mix_block(v, refv)
	dkf := make([]f32, 32)
	dvf := make([]f32, 32)
	rkf := make([]f32, 32)
	rvf := make([]f32, 32)
	defer delete(dkf)
	defer delete(dvf)
	defer delete(rkf)
	defer delete(rvf)
	dequantize_q8_block(kc[:34], dkf)
	dequantize_q8_v_mix_block(vc[:40], dvf)
	dequantize_q8_block(refk, rkf)
	dequantize_q8_v_mix_block(refv, rvf)
	st_q8: Case_Err
	for i in 0 ..< 32 {
		s_note(&st_q8, fmt.tprintf("q8k[%d]", i), dkf[i], rkf[i], 2e-2, 1e-3)
		s_note(&st_q8, fmt.tprintf("q8v[%d]", i), dvf[i], rvf[i], 2e-2, 1e-3)
	}
	if !s_report("q8v store", st_q8, 2e-2, 1e-3) {
		return false
	}

	cu.upload(dq, raw_data(q), hd * 4)
	attn := s_fn("attention_q8v")
	if attn == nil {
		return false
	}
	qa, kca, vca, xb, sc := dq, dkc, dvc, dxb, dsc
	ihd, ikb, ivb, mul, ps, sl := i32(hd), i32(k_pos), i32(v_pos), i32(1), i32(1), i32(seq)
	slot: cu.Device_Ptr
	stride := i32(0)
	aargs := [?]rawptr{&qa, &kca, &vca, &xb, &sc, &ihd, &ikb, &ivb, &mul, &ps, &sl, &slot, &stride}
	if cu.launch(attn, 1, 1, 1, 128, 1, 1, 0, aargs[:]) != cu.OK {
		fmt.eprintln("cuda_selftest: attention_q8v failed")
		return false
	}
	got := make([]f32, hd)
	defer delete(got)
	cu.download(raw_data(got), dxb, hd * 4)

	// CPU attention over the dequantized cache the GPU just wrote.
	k0 := make([]f32, hd)
	k1 := make([]f32, hd)
	v0 := make([]f32, hd)
	v1 := make([]f32, hd)
	defer delete(k0)
	defer delete(k1)
	defer delete(v0)
	defer delete(v1)
	dequantize_q8_block(kc[0:34], k0)
	dequantize_q8_block(kc[k_pos:k_pos + 34], k1)
	dequantize_q8_v_mix_block(vc[0:40], v0)
	dequantize_q8_v_mix_block(vc[v_pos:v_pos + 40], v1)
	dot := proc(a, b: []f32) -> f32 {
		s: f32 = 0
		for i in 0 ..< len(a) {
			s += a[i] * b[i]
		}
		return s
	}
	scale := 1 / math.sqrt_f32(f32(hd))
	s0 := dot(q, k0) * scale
	s1 := dot(q, k1) * scale
	m := max(s0, s1)
	e0 := math.exp_f32(s0 - m)
	e1 := math.exp_f32(s1 - m)
	inv := 1 / (e0 + e1)
	st_attn: Case_Err
	for i in 0 ..< hd {
		want := (e0 * v0[i] + e1 * v1[i]) * inv
		s_note(&st_attn, fmt.tprintf("attn[%d]", i), got[i], want, 2e-2, 1e-3)
	}
	return s_report("attention_q8v", st_attn, 2e-2, 1e-3)
}

@(private)
run_delta :: proc() -> bool {
	hkd, hvd := 4, 4
	state := make([]f32, hkd * hvd)
	q := make([]f32, hkd)
	k := make([]f32, hkd)
	v := make([]f32, hvd)
	defer delete(state)
	defer delete(q)
	defer delete(k)
	defer delete(v)
	for i in 0 ..< len(state) {
		state[i] = (f32(i % 5) - 2) * 0.1
	}
	for i in 0 ..< hkd {
		q[i] = 0.2
		k[i] = (f32(i) - 1) * 0.3
	}
	for i in 0 ..< hvd {
		v[i] = (f32(i) - 0.5) * 0.4
	}
	bv := f32(0.3)
	av := f32(0.2)
	dec := f32(-0.7)
	dt := f32(0.1)
	beta := sigmoid_f32(bv)
	g := math.exp_f32(dec * softplus_f32(av + dt))
	cpu_state := make([]f32, len(state))
	copy(cpu_state, state)
	defer delete(cpu_state)
	out := make([]f32, hvd)
	scratch := make([]f32, hvd)
	defer delete(out)
	defer delete(scratch)
	delta_recurrent_step(out, cpu_state, q, k, v, scratch, beta, g, hkd, hvd)

	ds, _ := cu.alloc(len(state) * 4)
	dq, _ := cu.alloc(hkd * 4)
	dk, _ := cu.alloc(hkd * 4)
	dv, _ := cu.alloc(hvd * 4)
	db, _ := cu.alloc(4)
	da, _ := cu.alloc(4)
	dd, _ := cu.alloc(4)
	dtb, _ := cu.alloc(4)
	dout, _ := cu.alloc(hvd * 4)
	defer cu.free(ds)
	defer cu.free(dq)
	defer cu.free(dk)
	defer cu.free(dv)
	defer cu.free(db)
	defer cu.free(da)
	defer cu.free(dd)
	defer cu.free(dtb)
	defer cu.free(dout)
	cu.upload(ds, raw_data(state), len(state) * 4)
	cu.upload(dq, raw_data(q), hkd * 4)
	cu.upload(dk, raw_data(k), hkd * 4)
	cu.upload(dv, raw_data(v), hvd * 4)
	b1 := [1]f32{bv}
	a1 := [1]f32{av}
	d1 := [1]f32{dec}
	t1 := [1]f32{dt}
	cu.upload(db, raw_data(b1[:]), 4)
	cu.upload(da, raw_data(a1[:]), 4)
	cu.upload(dd, raw_data(d1[:]), 4)
	cu.upload(dtb, raw_data(t1[:]), 4)
	fn := s_fn("delta_recurrent")
	if fn == nil {
		return false
	}
	o, st, qq, kk, vv, bb, aa, decp, dtp := dout, ds, dq, dk, dv, db, da, dd, dtb
	ik, iv, nk := i32(hkd), i32(hvd), i32(1)
	args := [?]rawptr{&o, &st, &qq, &kk, &vv, &bb, &aa, &decp, &dtp, &ik, &iv, &nk}
	if cu.launch(fn, 1, 1, 1, 128, 1, 1, 0, args[:]) != cu.OK {
		fmt.eprintln("cuda_selftest: delta launch failed")
		return false
	}
	gout := make([]f32, hvd)
	gstate := make([]f32, len(state))
	defer delete(gout)
	defer delete(gstate)
	cu.download(raw_data(gout), dout, hvd * 4)
	cu.download(raw_data(gstate), ds, len(state) * 4)
	st_out: Case_Err
	for i in 0 ..< hvd {
		s_note(&st_out, fmt.tprintf("delta out[%d]", i), gout[i], out[i], 1e-4, 1e-4)
	}
	if !s_report("delta out", st_out, 1e-4, 1e-4) {
		return false
	}
	st_state: Case_Err
	for i in 0 ..< len(state) {
		s_note(&st_state, fmt.tprintf("delta state[%d]", i), gstate[i], cpu_state[i], 1e-4, 1e-4)
	}
	return s_report("delta state", st_state, 1e-4, 1e-4)
}

// Two tokens, two value heads, one key head. The chunk kernel must match
// stepping delta_recurrent_step in token order.
@(private)
run_delta_chunk :: proc() -> bool {
	fn := s_fn("delta_recurrent_chunk")
	if fn == nil {
		fmt.eprintln("cuda_selftest: delta_recurrent_chunk missing")
		return false
	}
	hkd, hvd, nkh, nvh, T := 4, 4, 1, 2, 2
	state := make([]f32, nvh * hkd * hvd)
	q := make([]f32, T * nkh * hkd)
	k := make([]f32, T * nkh * hkd)
	v := make([]f32, T * nvh * hvd)
	b := make([]f32, T * nvh)
	a := make([]f32, T * nvh)
	dec := make([]f32, nvh)
	dt := make([]f32, nvh)
	defer delete(state)
	defer delete(q)
	defer delete(k)
	defer delete(v)
	defer delete(b)
	defer delete(a)
	defer delete(dec)
	defer delete(dt)
	for i in 0 ..< len(state) {
		state[i] = (f32(i % 5) - 2) * 0.05
	}
	for t in 0 ..< T {
		for i in 0 ..< nkh * hkd {
			q[t * nkh * hkd + i] = 0.1 * f32(t + 1) + 0.01 * f32(i)
			k[t * nkh * hkd + i] = 0.05 * f32(i + 1) - 0.02 * f32(t)
		}
		for h in 0 ..< nvh {
			b[t * nvh + h] = 0.2 * f32(h + 1) + 0.1 * f32(t)
			a[t * nvh + h] = -0.3 + 0.1 * f32(t + h)
			for j in 0 ..< hvd {
				v[(t * nvh + h) * hvd + j] = 0.15 * f32(j + 1) - 0.05 * f32(t)
			}
		}
	}
	for h in 0 ..< nvh {
		dec[h] = -0.4 - 0.1 * f32(h)
		dt[h] = 0.05 * f32(h + 1)
	}
	cpu_state := make([]f32, len(state))
	copy(cpu_state, state)
	defer delete(cpu_state)
	cpu_out := make([]f32, T * nvh * hvd)
	defer delete(cpu_out)
	scratch := make([]f32, hvd)
	defer delete(scratch)
	for t in 0 ..< T {
		for h in 0 ..< nvh {
			beta := sigmoid_f32(b[t * nvh + h])
			g := math.exp_f32(dec[h] * softplus_f32(a[t * nvh + h] + dt[h]))
			delta_recurrent_step(
				cpu_out[(t * nvh + h) * hvd:][:hvd],
				cpu_state[h * hkd * hvd:][:hkd * hvd],
				q[t * nkh * hkd:][:hkd],
				k[t * nkh * hkd:][:hkd],
				v[(t * nvh + h) * hvd:][:hvd],
				scratch, beta, g, hkd, hvd,
			)
		}
	}
	ds, _ := cu.alloc(len(state) * 4)
	dq, _ := cu.alloc(len(q) * 4)
	dk, _ := cu.alloc(len(k) * 4)
	dv, _ := cu.alloc(len(v) * 4)
	db, _ := cu.alloc(len(b) * 4)
	da, _ := cu.alloc(len(a) * 4)
	dd, _ := cu.alloc(len(dec) * 4)
	dtb, _ := cu.alloc(len(dt) * 4)
	dout, _ := cu.alloc(len(cpu_out) * 4)
	defer cu.free(ds)
	defer cu.free(dq)
	defer cu.free(dk)
	defer cu.free(dv)
	defer cu.free(db)
	defer cu.free(da)
	defer cu.free(dd)
	defer cu.free(dtb)
	defer cu.free(dout)
	cu.upload(ds, raw_data(state), len(state) * 4)
	cu.upload(dq, raw_data(q), len(q) * 4)
	cu.upload(dk, raw_data(k), len(k) * 4)
	cu.upload(dv, raw_data(v), len(v) * 4)
	cu.upload(db, raw_data(b), len(b) * 4)
	cu.upload(da, raw_data(a), len(a) * 4)
	cu.upload(dd, raw_data(dec), len(dec) * 4)
	cu.upload(dtb, raw_data(dt), len(dt) * 4)
	o, stp, qq, kk, vv, bb, aa, decp, dtp := dout, ds, dq, dk, dv, db, da, dd, dtb
	ik, iv, nk, nv := i32(hkd), i32(hvd), i32(nkh), i32(nvh)
	tt, qs, vs := i32(T), i32(nkh * hkd), i32(nvh * hvd)
	args := [?]rawptr{&o, &stp, &qq, &kk, &vv, &bb, &aa, &decp, &dtp, &ik, &iv, &nk, &nv, &tt, &qs, &vs}
	if cu.launch(fn, u32(nvh), 1, 1, 128, 1, 1, 0, args[:]) != cu.OK || cu.sync() != cu.OK {
		fmt.eprintln("cuda_selftest: delta chunk launch failed")
		return false
	}
	gout := make([]f32, len(cpu_out))
	gstate := make([]f32, len(state))
	defer delete(gout)
	defer delete(gstate)
	cu.download(raw_data(gout), dout, len(cpu_out) * 4)
	cu.download(raw_data(gstate), ds, len(state) * 4)
	st_out: Case_Err
	for i in 0 ..< len(cpu_out) {
		s_note(&st_out, fmt.tprintf("delta chunk out[%d]", i), gout[i], cpu_out[i], 1e-4, 1e-4)
	}
	if !s_report("delta chunk out", st_out, 1e-4, 1e-4) {
		return false
	}
	st_state: Case_Err
	for i in 0 ..< len(state) {
		s_note(&st_state, fmt.tprintf("delta chunk state[%d]", i), gstate[i], cpu_state[i], 1e-4, 1e-4)
	}
	return s_report("delta chunk state", st_state, 1e-4, 1e-4)
}

cuda_selftest :: proc() -> bool {
	if !s_open() {
		return false
	}
	defer s_close()
	ok := true
	kinds := [?]ggml.GGML_Type{
		.F32, .F16, .Q8_0, .Q4_0, .Q4_1, .Q5_0, .Q5_1,
		.Q4_K, .Q6_K, .Q2_K, .Q3_K, .Q5_K, .Q8_K,
	}
	for kind in kinds {
		atol, rtol: f32 = 2e-2, 1e-3
		if kind == .F32 || kind == .F16 {
			atol, rtol = 1e-4, 1e-5
		}
		if !run_gemv(kind, 3, 256, atol, rtol) {
			ok = false
		}
	}
	// dp4a GEMV vs the same CPU reference. q8 of x is approximate, so the
	// tolerance is wider than the scalar kernel; a swapped scale is still O(1).
	gemv2 := [?]ggml.GGML_Type{.Q4_K, .Q6_K, .Q8_0, .Q5_0, .Q5_K}
	for kind in gemv2 {
		// q8 of x is the only approximation. A swapped nibble or scale is O(|ref|).
		if !run_gemv2(kind, 64, 512, 1, 5e-2) {
			ok = false
		}
		// A row count that is not a multiple of 4 exercises the tail.
		if !run_gemv2(kind, 6, 256, 1, 5e-2) {
			ok = false
		}
	}
	mmq_kinds := [?]ggml.GGML_Type{.F32, .Q8_0, .Q4_K, .Q6_K, .Q5_0}
	for kind in mmq_kinds {
		if !run_mmq(kind, 3, 256, 5) {
			ok = false
		}
	}
	// fp16 tiles vs the fp32 CPU dot. K=256, values O(0.1). A wrong WMMA
	// layout is off by O(1), not by this tolerance.
	tc_kinds := [?]ggml.GGML_Type{
		.F32, .F16, .Q8_0, .Q4_0, .Q5_0, .Q4_K, .Q6_K, .Q5_K, .Q8_K,
	}
	for kind in tc_kinds {
		atol, rtol: f32 = 5e-2, 5e-2
		if kind == .F32 || kind == .F16 {
			atol, rtol = 2e-2, 1e-2
		}
		if !run_tc(kind, 64, 256, 32, atol, rtol) {
			ok = false
		}
	}
	// Partial tile: T and rows are not multiples of 16.
	if !run_tc(.Q4_K, 40, 256, 20, 5e-2, 5e-2) {
		ok = false
	}
	if !run_tc(.Q8_0, 40, 64, 20, 5e-2, 5e-2) {
		ok = false
	}
	if !run_tc(.Q5_0, 40, 64, 20, 5e-2, 5e-2) {
		ok = false
	}
	if !run_tc2(.Q4_K, 64, 256, 32, 5e-2, 5e-2, false) || !run_tc2(.Q6_K, 64, 256, 32, 5e-2, 5e-2, false) {
		ok = false
	}
	if !run_tc2(.Q4_K, 40, 256, 20, 5e-2, 5e-2, false) || !run_tc2(.Q6_K, 40, 512, 20, 5e-2, 5e-2, false) {
		ok = false
	}
	// 80 tokens forces the second 64-token chunk that reuses the fp16 weight tile.
	if !run_tc2(.Q4_K, 64, 256, 80, 5e-2, 5e-2, true) || !run_tc2(.Q6_K, 48, 512, 160, 5e-2, 5e-2, true) {
		ok = false
	}
	if !run_tc2(.Q4_K, 64, 256, 32, 5e-2, 5e-2, true) || !run_tc2(.Q6_K, 40, 256, 20, 5e-2, 5e-2, true) {
		ok = false
	}
	// tc2m is the opt-in 64x64 block-scale tile. T=80 is a partial second
	// token tile; 40 rows is a partial row tile; K=512 crosses two superblocks.
	if !run_tc2m(.Q4_K, 64, 256, 64, 5e-2, 5e-2) || !run_tc2m(.Q4_K, 40, 256, 80, 5e-2, 5e-2) {
		ok = false
	}
	if !run_tc2m(.Q6_K, 64, 512, 64, 5e-2, 5e-2) || !run_tc2m(.Q6_K, 48, 256, 96, 5e-2, 5e-2) {
		ok = false
	}
	if !run_tc2m(.Q6_K, 40, 512, 128, 5e-2, 5e-2) {
		ok = false
	}
	if !run_delta_chunk() {
		ok = false
	}
	if !run_elem() {
		ok = false
	}
	if !run_kv_attn() {
		ok = false
	}
	if !run_delta() {
		ok = false
	}
	if ok {
		fmt.eprintln("cuda_selftest: PASS")
	} else {
		fmt.eprintln("cuda_selftest: FAIL")
	}
	return ok
}

// Representative 9B prefill GEMMs (T=64) plus a small Q8_0 / Q5_0 pair for
// the 0.8B MoE quants. Times tensor-core WMMA against the MMQ fallback.
// FLOPs count a multiply and an add: 2*T*rows*cols.
cuda_tc_bench :: proc() -> bool {
	if !s_open() {
		return false
	}
	defer s_close()
	Shape :: struct {
		kind:       ggml.GGML_Type,
		rows, cols: int,
		T:          int,
		name:       string,
	}
	shapes := [?]Shape{
		{.Q4_K, 12288, 4096, 64, "9B w1/w3"},
		{.Q4_K, 8192, 4096, 64, "9B wq"},
		{.Q4_K, 4096, 4096, 64, "9B wo"},
		{.Q6_K, 4096, 12288, 64, "9B w2"},
		{.Q8_0, 1024, 1024, 32, "0.8B Q8_0"},
		{.Q5_0, 1024, 1024, 32, "0.8B Q5_0"},
	}
	fmt.eprintln("cuda_bench: TC is WMMA 16x16x16 fp16, fp32 acc. MMQ is the warp fallback.")
	for s in shapes {
		tag := cuda_kind_tag(s.kind)
		tc := s_fn(fmt.tprintf("tc_%s", tag))
		mmq := s_fn(fmt.tprintf("mmq_%s", tag))
		if tc == nil || mmq == nil {
			fmt.eprintf("cuda_bench: missing kernel for %s\n", tag)
			return false
		}
		raw := fill_weight(s.kind, s.rows, s.cols)
		x := make([]f32, s.T * s.cols)
		for i in 0 ..< len(x) {
			x[i] = (f32(i % 11) - 5) * 0.02
		}
		dw, e1 := cu.alloc(len(raw))
		dx, e2 := cu.alloc(len(x) * 4)
		dy, e3 := cu.alloc(s.T * s.rows * 4)
		if e1 != cu.OK || e2 != cu.OK || e3 != cu.OK {
			fmt.eprintln("cuda_bench: alloc failed")
			delete(raw)
			delete(x)
			return false
		}
		cu.upload(dw, raw_data(raw), len(raw))
		cu.upload(dx, raw_data(x), len(x) * 4)
		delete(raw)
		delete(x)
		wb, xx, yy := dw, dx, dy
		nn, dd, tt := i32(s.cols), i32(s.rows), i32(s.T)
		args := [?]rawptr{&wb, &xx, &yy, &nn, &dd, &tt}
		gx := u32((s.rows + 63) / 64)
		gy := u32((s.T + 63) / 64)
		tc_ms := bench_ms(tc, args[:], gx, gy, 512)
		mmq_ms := bench_ms(mmq, args[:], u32(s.rows), 1, 32)
		cu.free(dw)
		cu.free(dx)
		cu.free(dy)
		flops := 2 * f64(s.T) * f64(s.rows) * f64(s.cols)
		tc_tflops := flops / (tc_ms * 1e-3) / 1e12
		mmq_tflops := flops / (mmq_ms * 1e-3) / 1e12
		fmt.eprintfln(
			"cuda_bench: %s %s  %dx%dx%d  TC %.2f ms  %.2f TFLOPS   MMQ %.2f ms  %.2f TFLOPS   TC/MMQ %.2fx",
			s.name, tag, s.rows, s.T, s.cols, tc_ms, tc_tflops, mmq_ms, mmq_tflops, mmq_ms / tc_ms,
		)
	}
	return true
}

@(private)
bench_ms :: proc(fn: cu.Function, args: []rawptr, gx, gy, bx: u32) -> f64 {
	for _ in 0 ..< 2 {
		cu.launch(fn, gx, gy, 1, bx, 1, 1, 0, args)
	}
	cu.sync()
	iters := 5
	t0 := time.now()
	for _ in 0 ..< iters {
		cu.launch(fn, gx, gy, 1, bx, 1, 1, 0, args)
		cu.sync()
	}
	return f64(time.duration_milliseconds(time.since(t0))) / f64(iters)
}

Gemv_Shape :: struct {
	name: string,
	kind: ggml.GGML_Type,
	rows: int,
	cols: int,
}

@(private)
gemv_gbs :: proc(kind: ggml.GGML_Type, rows, cols: int, ms: f64) -> (bytes: f64, gbs: f64) {
	elems, bsz := ggml.ggml_block_info(kind)
	bytes = f64(rows) * f64(cols / elems) * f64(bsz)
	if ms > 0 {
		gbs = bytes / ms / 1e6
	}
	return
}

// cuda_gemv_bench times the scalar GEMV and the bandwidth GEMV on the
// T=1 shapes that dominate 9B and 27B decode. The q8 of x is done once,
// outside the timed loop, matching a cached activation.
cuda_gemv_bench :: proc() {
	if !s_open() {
		return
	}
	defer s_close()
	peak := 1134.0
	if s := os.get_env("ODINFER_CUDA_PEAK_GBS", context.temp_allocator); s != "" {
		if v, ok := strconv.parse_f64(s); ok && v > 0 {
			peak = v
		}
	}
	fmt.eprintfln("cuda_gemv: T=1  peak %.0f GB/s (ODINFER_CUDA_PEAK_GBS). GB/s counts the weight bytes only.", peak)
	shapes := [?]Gemv_Shape{
		{"9B w1", .Q4_K, 12288, 4096},
		{"9B wq", .Q4_K, 8192, 4096},
		{"9B wo", .Q4_K, 4096, 4096},
		{"9B w2", .Q6_K, 4096, 12288},
		{"9B out", .Q6_K, 248320, 4096},
		{"27B w1", .Q4_K, 17408, 5120},
		{"27B wq", .Q4_K, 12288, 5120},
		{"27B wo", .Q4_K, 5120, 6144},
		{"27B w2", .Q6_K, 5120, 17408},
		{"27B out", .Q6_K, 248320, 5120},
		{"q8_0", .Q8_0, 2048, 2048},
		{"q5_0", .Q5_0, 2048, 2048},
		{"q5_k", .Q5_K, 5120, 5120},
	}
	for sh in shapes {
		gemv_bench_one(sh, peak)
	}
}

@(private)
gemv_bench_one :: proc(sh: Gemv_Shape, peak: f64) {
	tag := cuda_kind_tag(sh.kind)
	old := s_fn(fmt.tprintf("gemv_%s", tag))
	nw := s_fn(fmt.tprintf("gemv2_%s", tag))
	qfn := s_fn("quant_q8")
	if old == nil || nw == nil || qfn == nil {
		fmt.eprintfln("  %s %s  kernel missing", sh.name, tag)
		return
	}
	elems, bsz := ggml.ggml_block_info(sh.kind)
	nbytes := sh.rows * (sh.cols / elems) * bsz
	raw := fill_weight(sh.kind, sh.rows, sh.cols)
	defer delete(raw)
	x := make([]f32, sh.cols)
	defer delete(x)
	for i in 0 ..< sh.cols {
		x[i] = (f32(i % 7) - 3) * 0.02
	}
	dw, e1 := cu.alloc(nbytes + 64)
	dx, e2 := cu.alloc(sh.cols * 4)
	dy, e3 := cu.alloc(sh.rows * 4)
	ng := sh.cols / 32
	dqs, e4 := cu.alloc(sh.cols)
	dsc, e5 := cu.alloc(ng * 4)
	dsm, e6 := cu.alloc(ng * 8)
	defer cu.free(dw)
	defer cu.free(dx)
	defer cu.free(dy)
	defer cu.free(dqs)
	defer cu.free(dsc)
	defer cu.free(dsm)
	if e1 != cu.OK || e2 != cu.OK || e3 != cu.OK || e4 != cu.OK || e5 != cu.OK || e6 != cu.OK {
		fmt.eprintfln("  %s %s  %dx%d  alloc failed (%.1f MiB weights)", sh.name, tag, sh.rows, sh.cols, f64(nbytes) / (1024.0 * 1024.0))
		return
	}
	cu.upload(dw, raw_data(raw), nbytes)
	cu.upload(dx, raw_data(x), sh.cols * 4)
	xx, qs, sc, sm := dx, dqs, dsc, dsm
	nn := i32(sh.cols)
	qargs := [?]rawptr{&xx, &qs, &sc, &sm, &nn}
	cu.launch(qfn, u32((ng + 127) / 128), 1, 1, 128, 1, 1, 0, qargs[:])
	cu.sync()
	wb, yy := dw, dy
	dd := i32(sh.rows)
	old_args := [?]rawptr{&wb, &xx, &yy, &nn, &dd}
	warm, iters := 2, 5
	if sh.rows > 100000 {
		warm, iters = 1, 3
	}
	old_ms := gemv_time(old, old_args[:], u32(sh.rows), 1, 32, warm, iters)
	warps := 8
	gx := u32((sh.rows + warps * 4 - 1) / (warps * 4))
	bx := u32(warps * 32)
	acc := i32(0)
	new_ms: f64
	if sh.kind == .Q4_K || sh.kind == .Q5_K {
		args := [?]rawptr{&wb, &qs, &sc, &sm, &yy, &nn, &dd, &acc}
		new_ms = gemv_time(nw, args[:], gx, 1, bx, warm, iters)
	} else {
		args := [?]rawptr{&wb, &qs, &sc, &yy, &nn, &dd, &acc}
		new_ms = gemv_time(nw, args[:], gx, 1, bx, warm, iters)
	}
	r1_ms := f64(-1)
	if sh.kind == .Q4_K || sh.kind == .Q6_K {
		rname := "gemv2r1_q4_k"
		if sh.kind == .Q6_K {
			rname = "gemv2r1_q6_k"
		}
		rfn := s_fn(rname)
		if rfn != nil {
			gx1 := u32((sh.rows + warps - 1) / warps)
			if sh.kind == .Q4_K {
				args := [?]rawptr{&wb, &qs, &sc, &sm, &yy, &nn, &dd, &acc}
				r1_ms = gemv_time(rfn, args[:], gx1, 1, bx, warm, iters)
			} else {
				args := [?]rawptr{&wb, &qs, &sc, &yy, &nn, &dd, &acc}
				r1_ms = gemv_time(rfn, args[:], gx1, 1, bx, warm, iters)
			}
		}
	}
	_, old_gbs := gemv_gbs(sh.kind, sh.rows, sh.cols, old_ms)
	bytes, new_gbs := gemv_gbs(sh.kind, sh.rows, sh.cols, new_ms)
	fmt.eprintfln("  %s %s  %dx%d  %.1f MiB", sh.name, tag, sh.rows, sh.cols, bytes / (1024.0 * 1024.0))
	fmt.eprintfln("    r4   %.3f ms  %.1f GB/s  %.1f%% of peak", new_ms, new_gbs, new_gbs / peak * 100)
	if sh.kind == .Q4_K {
		nfn := s_fn("gemv2_q4nc_k")
		xfn := s_fn("gemv2_q4x_k")
		if nfn != nil {
			args := [?]rawptr{&wb, &qs, &sc, &sm, &yy, &nn, &dd, &acc}
			nc_ms := gemv_time(nfn, args[:], gx, 1, bx, warm, iters)
			if nc_ms > 0 {
				_, nc_gbs := gemv_gbs(sh.kind, sh.rows, sh.cols, nc_ms)
				fmt.eprintfln("    nc   %.3f ms  %.1f GB/s  %.1f%% of peak", nc_ms, nc_gbs, nc_gbs / peak * 100)
			}
		}
		if xfn != nil {
			gx8 := u32((sh.rows + warps * 8 - 1) / (warps * 8))
			args := [?]rawptr{&wb, &qs, &sc, &sm, &yy, &nn, &dd, &acc}
			x_ms := gemv_time(xfn, args[:], gx8, 1, bx, warm, iters)
			if x_ms > 0 {
				_, x_gbs := gemv_gbs(sh.kind, sh.rows, sh.cols, x_ms)
				fmt.eprintfln("    r8   %.3f ms  %.1f GB/s  %.1f%% of peak", x_ms, x_gbs, x_gbs / peak * 100)
			}
		}
	}
	if r1_ms > 0 {
		_, r1_gbs := gemv_gbs(sh.kind, sh.rows, sh.cols, r1_ms)
		fmt.eprintfln("    r1   %.3f ms  %.1f GB/s  %.1f%% of peak", r1_ms, r1_gbs, r1_gbs / peak * 100)
	}
	fmt.eprintfln("    old  %.3f ms  %.1f GB/s  %.1f%% of peak", old_ms, old_gbs, old_gbs / peak * 100)
}

@(private)
gemv_time :: proc(fn: cu.Function, args: []rawptr, gx, gy, bx: u32, warm, iters: int) -> f64 {
	for _ in 0 ..< warm {
		cu.launch(fn, gx, gy, 1, bx, 1, 1, 0, args)
	}
	cu.sync()
	t0 := time.now()
	for _ in 0 ..< iters {
		if cu.launch(fn, gx, gy, 1, bx, 1, 1, 0, args) != cu.OK {
			return -1
		}
		cu.sync()
	}
	return f64(time.duration_milliseconds(time.since(t0))) / f64(iters)
}

} // GPU_CUDA
