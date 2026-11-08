/* GGML quant block formats: dequantization + SIMD dot products */

package ggml

import "core:simd"

QK_K :: 256

// elements-per-block and bytes-per-block for each ggml type
ggml_block_info :: proc(k: GGML_Type) -> (elems: int, bytes: int) {
	switch k {
	case .F32:
		return 1, 4
	case .F16:
		return 1, 2
	case .Q4_0:
		return 32, 18
	case .Q4_1:
		return 32, 20
	case .Q5_0:
		return 32, 22
	case .Q5_1:
		return 32, 24
	case .Q8_0:
		return 32, 34
	case .Q8_1:
		return 32, 36
	case .Q2_K:
		return 256, 84
	case .Q3_K:
		return 256, 110
	case .Q4_K:
		return 256, 144
	case .Q5_K:
		return 256, 176
	case .Q6_K:
		return 256, 210
	case .Q8_K:
		return 256, 292
	}
	return 1, 4
}

row_byte_size :: proc(k: GGML_Type, n: int) -> int {
	elems, bytes := ggml_block_info(k)
	return (n / elems) * bytes
}

is_supported_quant :: proc(k: GGML_Type) -> bool {
	#partial switch k {
	case .F32, .F16, .Q8_0, .Q4_0, .Q4_1, .Q5_0, .Q5_1, .Q4_K, .Q6_K:
		return true
	}
	return false
}

@(private = "file")
f16_to_f32 :: #force_inline proc(b: []u8) -> f32 {
	h := transmute(f16)(u16(b[0]) | (u16(b[1]) << 8))
	return f32(h)
}

@(private = "file")
get_scale_min_k4 :: #force_inline proc(j: int, q: []u8) -> (sc: u8, m: u8) {
	if j < 4 {
		sc = q[j] & 63
		m = q[j + 4] & 63
	} else {
		sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)
		m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4)
	}
	return
}

// Dequantize a single row of `n` elements from raw block bytes into dst[0:n].
dequant_row :: proc(k: GGML_Type, src: []u8, n: int, dst: []f32) {
	#no_bounds_check switch k {
	case .F32:
		s := (cast([^]f32)raw_data(src))[:n]
		copy(dst, s)

	case .F16:
		for i in 0 ..< n {
			dst[i] = f16_to_f32(src[i * 2:])
		}

	case .Q8_0:
		nb := n / 32
		for b in 0 ..< nb {
			base := b * 34
			d := f16_to_f32(src[base:])
			o := b * 32
			for j in 0 ..< 32 {
				dst[o + j] = d * f32(i8(src[base + 2 + j]))
			}
		}

	case .Q4_0:
		nb := n / 32
		for b in 0 ..< nb {
			base := b * 18
			d := f16_to_f32(src[base:])
			qs := src[base + 2:]
			o := b * 32
			for j in 0 ..< 16 {
				x0 := f32(i32(qs[j] & 0xF) - 8)
				x1 := f32(i32(qs[j] >> 4) - 8)
				dst[o + j] = x0 * d
				dst[o + j + 16] = x1 * d
			}
		}

	case .Q4_1:
		nb := n / 32
		for b in 0 ..< nb {
			base := b * 20
			d := f16_to_f32(src[base:])
			m := f16_to_f32(src[base + 2:])
			qs := src[base + 4:]
			o := b * 32
			for j in 0 ..< 16 {
				dst[o + j] = f32(qs[j] & 0xF) * d + m
				dst[o + j + 16] = f32(qs[j] >> 4) * d + m
			}
		}

	case .Q5_0:
		nb := n / 32
		for b in 0 ..< nb {
			base := b * 22
			d := f16_to_f32(src[base:])
			qh := u32(src[base + 2]) | u32(src[base + 3]) << 8 | u32(src[base + 4]) << 16 | u32(src[base + 5]) << 24
			qs := src[base + 6:]
			o := b * 32
			for j in 0 ..< 16 {
				xh0 := u8((qh >> u32(j)) << 4) & 0x10
				xh1 := u8(qh >> (u32(j) + 12)) & 0x10
				x0 := i32((qs[j] & 0xF) | xh0) - 16
				x1 := i32((qs[j] >> 4) | xh1) - 16
				dst[o + j] = f32(x0) * d
				dst[o + j + 16] = f32(x1) * d
			}
		}

	case .Q5_1:
		nb := n / 32
		for b in 0 ..< nb {
			base := b * 24
			d := f16_to_f32(src[base:])
			m := f16_to_f32(src[base + 2:])
			qh := u32(src[base + 4]) | u32(src[base + 5]) << 8 | u32(src[base + 6]) << 16 | u32(src[base + 7]) << 24
			qs := src[base + 8:]
			o := b * 32
			for j in 0 ..< 16 {
				xh0 := u8((qh >> u32(j)) << 4) & 0x10
				xh1 := u8(qh >> (u32(j) + 12)) & 0x10
				x0 := i32((qs[j] & 0xF) | xh0)
				x1 := i32((qs[j] >> 4) | xh1)
				dst[o + j] = f32(x0) * d + m
				dst[o + j + 16] = f32(x1) * d + m
			}
		}

	case .Q4_K:
		dequant_q4_k(src, n, dst)

	case .Q6_K:
		dequant_q6_k(src, n, dst)

	case .Q8_1, .Q2_K, .Q3_K, .Q5_K, .Q8_K:
		for i in 0 ..< n {
			dst[i] = 0
		}
	}
}

@(private = "file")
dequant_q4_k :: proc(src: []u8, n: int, dst: []f32) {
	#no_bounds_check {
		nsb := n / QK_K
		for sb in 0 ..< nsb {
			base := sb * 144
			d := f16_to_f32(src[base:])
			dmin := f16_to_f32(src[base + 2:])
			scales := src[base + 4:base + 16]
			qs := src[base + 16:base + 144]
			yoff := sb * QK_K
			is := 0
			qoff := 0
			for j := 0; j < QK_K; j += 64 {
				sc0, m0 := get_scale_min_k4(is + 0, scales)
				sc1, m1 := get_scale_min_k4(is + 1, scales)
				d1 := d * f32(sc0)
				mm1 := dmin * f32(m0)
				d2 := d * f32(sc1)
				mm2 := dmin * f32(m1)
				for l in 0 ..< 32 {
					dst[yoff + l] = d1 * f32(qs[qoff + l] & 0xF) - mm1
				}
				for l in 0 ..< 32 {
					dst[yoff + 32 + l] = d2 * f32(qs[qoff + l] >> 4) - mm2
				}
				qoff += 32
				is += 2
				yoff += 64
			}
		}
	}
}

@(private = "file")
dequant_q6_k :: proc(src: []u8, n: int, dst: []f32) {
	#no_bounds_check {
		nsb := n / QK_K
		for sb in 0 ..< nsb {
			base := sb * 210
			ql := src[base:base + 128]
			qh := src[base + 128:base + 192]
			sc := src[base + 192:base + 208]
			d := f16_to_f32(src[base + 208:])

			yoff := sb * QK_K
			qloff := 0
			qhoff := 0
			scoff := 0
			for _ in 0 ..< 2 {
				for l in 0 ..< 32 {
					is := l / 16
					q1 := i32((ql[qloff + l] & 0xF) | (((qh[qhoff + l] >> 0) & 3) << 4)) - 32
					q2 :=
						i32((ql[qloff + l + 32] & 0xF) | (((qh[qhoff + l] >> 2) & 3) << 4)) - 32
					q3 := i32((ql[qloff + l] >> 4) | (((qh[qhoff + l] >> 4) & 3) << 4)) - 32
					q4 :=
						i32((ql[qloff + l + 32] >> 4) | (((qh[qhoff + l] >> 6) & 3) << 4)) - 32
					dst[yoff + l] = d * f32(i8(sc[scoff + is + 0])) * f32(q1)
					dst[yoff + l + 32] = d * f32(i8(sc[scoff + is + 2])) * f32(q2)
					dst[yoff + l + 64] = d * f32(i8(sc[scoff + is + 4])) * f32(q3)
					dst[yoff + l + 96] = d * f32(i8(sc[scoff + is + 6])) * f32(q4)
				}
				qloff += 64
				qhoff += 32
				scoff += 8
				yoff += 128
			}
		}
	}
}

// SIMD f32 dot product over n elements.
dot_f32 :: proc(a, b: []f32, n: int) -> f32 {
	#no_bounds_check {
		acc0: #simd[8]f32
		acc1: #simd[8]f32
		i := 0
		for ; i + 16 <= n; i += 16 {
			va0 := simd.from_slice(#simd[8]f32, a[i:i + 8])
			vb0 := simd.from_slice(#simd[8]f32, b[i:i + 8])
			va1 := simd.from_slice(#simd[8]f32, a[i + 8:i + 16])
			vb1 := simd.from_slice(#simd[8]f32, b[i + 8:i + 16])
			acc0 = simd.fma(va0, vb0, acc0)
			acc1 = simd.fma(va1, vb1, acc1)
		}
		sum := simd.reduce_add_ordered(simd.add(acc0, acc1))
		for ; i < n; i += 1 {
			sum += a[i] * b[i]
		}
		return sum
	}
}

// Dot of a weight row (possibly quantized) with activation x.
// For F32 we dot directly from mmap; otherwise dequant into scratch first.
dot_row :: proc(k: GGML_Type, src: []u8, x: []f32, n: int, scratch: []f32) -> f32 {
	if k == .F32 {
		#no_bounds_check {
			row := (cast([^]f32)raw_data(src))[:n]
			return dot_f32(row, x, n)
		}
	}
	dequant_row(k, src, n, scratch)
	return dot_f32(scratch, x, n)
}
