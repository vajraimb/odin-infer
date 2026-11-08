/* GGML quant block formats: dequantization + SIMD dot products */

package ggml

import "core:simd"

QK_K :: 256
QK4_NL :: 32

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
	case .IQ2_XXS:
		return 256, 66
	case .IQ2_XS:
		return 256, 74
	case .IQ3_XXS:
		return 256, 98
	case .IQ1_S:
		return 256, 50
	case .IQ4_NL:
		return 32, 18
	case .IQ3_S:
		return 256, 110
	case .IQ2_S:
		return 256, 82
	case .IQ4_XS:
		return 256, 136
	case .IQ1_M:
		return 256, 56
	}
	return 1, 4
}

row_byte_size :: proc(k: GGML_Type, n: int) -> int {
	elems, bytes := ggml_block_info(k)
	return (n / elems) * bytes
}

is_supported_quant :: proc(k: GGML_Type) -> bool {
	#partial switch k {
	case .F32, .F16, .Q8_0, .Q4_0, .Q4_1, .Q5_0, .Q5_1, .Q4_K, .Q6_K, .Q2_K, .Q3_K, .Q5_K, .Q8_K, .IQ2_XXS, .IQ2_XS, .IQ3_XXS, .IQ1_S, .IQ4_NL, .IQ3_S, .IQ2_S, .IQ4_XS, .IQ1_M:
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

	case .Q2_K:
		dequant_q2_k(src, n, dst)

	case .Q3_K:
		dequant_q3_k(src, n, dst)

	case .Q4_K:
		dequant_q4_k(src, n, dst)

	case .Q5_K:
		dequant_q5_k(src, n, dst)

	case .Q6_K:
		dequant_q6_k(src, n, dst)

	case .Q8_K:
		dequant_q8_k(src, n, dst)

	case .IQ2_XXS:
		dequant_iq2_xxs(src, n, dst)

	case .IQ2_XS:
		dequant_iq2_xs(src, n, dst)

	case .IQ2_S:
		dequant_iq2_s(src, n, dst)

	case .IQ3_XXS:
		dequant_iq3_xxs(src, n, dst)

	case .IQ3_S:
		dequant_iq3_s(src, n, dst)

	case .IQ1_S:
		dequant_iq1_s(src, n, dst)

	case .IQ1_M:
		dequant_iq1_m(src, n, dst)

	case .IQ4_NL:
		dequant_iq4_nl(src, n, dst)

	case .IQ4_XS:
		dequant_iq4_xs(src, n, dst)

	case .Q8_1:
		for i in 0 ..< n {
			dst[i] = 0
		}

	case:
		for i in 0 ..< n {
			dst[i] = 0
		}
	}
}

@(private = "file")
dequant_q2_k :: proc(src: []u8, n: int, dst: []f32) {
	#no_bounds_check {
		nsb := n / QK_K
		for sb in 0 ..< nsb {
			base := sb * 84
			// block_q2_K layout: scales[16], qs[64], d(2), dmin(2)
			scales := src[base:base + 16]
			qs := src[base + 16:base + 80]
			d := f16_to_f32(src[base + 80:])
			dmin := f16_to_f32(src[base + 82:])
			yoff := sb * QK_K
			is := 0
			qoff := 0
			for _ in 0 ..< 2 {
				shift := 0
				for _ in 0 ..< 4 {
					sc := scales[is]
					is += 1
					dl := d * f32(sc & 0xF)
					ml := dmin * f32(sc >> 4)
					for l in 0 ..< 16 {
						dst[yoff + l] = dl * f32(u8(qs[qoff + l] >> u32(shift)) & 3) - ml
					}
					sc = scales[is]
					is += 1
					dl = d * f32(sc & 0xF)
					ml = dmin * f32(sc >> 4)
					for l in 0 ..< 16 {
						dst[yoff + 16 + l] = dl * f32(u8(qs[qoff + 16 + l] >> u32(shift)) & 3) - ml
					}
					shift += 2
					yoff += 32
				}
				qoff += 32
			}
		}
	}
}

@(private = "file")
dequant_q3_k :: proc(src: []u8, n: int, dst: []f32) {
	#no_bounds_check {
		nsb := n / QK_K
		KMASK1 :: u32(0x03030303)
		KMASK2 :: u32(0x0f0f0f0f)
		aux: [4]u32
		for sb in 0 ..< nsb {
			base := sb * 110
			// block_q3_K layout: hmask[32], qs[64], scales[12], d(2)
			hm := src[base:base + 32]
			qs := src[base + 32:base + 96]
			scales := src[base + 96:base + 108]
			d := f16_to_f32(src[base + 108:])

			// decode 12 packed scale bytes into 16 int8 scales via the C
			// little-endian u32 shuffle (aux = x[i].scales as uint32_t[3])
			aux[0] = u32(scales[0]) | u32(scales[1]) << 8 | u32(scales[2]) << 16 | u32(scales[3]) << 24
			aux[1] = u32(scales[4]) | u32(scales[5]) << 8 | u32(scales[6]) << 16 | u32(scales[7]) << 24
			tmp := u32(scales[8]) | u32(scales[9]) << 8 | u32(scales[10]) << 16 | u32(scales[11]) << 24
			aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4)
			aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4)
			aux[0] = (aux[0] & KMASK2) | (((tmp >> 0) & KMASK1) << 4)
			aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4)

			yoff := sb * QK_K
			qoff := 0
			m := u8(1)
			is := 0
			for _ in 0 ..< 2 {
				shift := 0
				for _ in 0 ..< 4 {
					sc := i8(u8(aux[is / 4] >> u32(8 * (is % 4))))
					is += 1
					dl := d * f32(i32(sc) - 32)
					for l in 0 ..< 16 {
						q := i32(u8(qs[qoff + l] >> u32(shift)) & 3) - i32((hm[l] & m) == 0 ? 4 : 0)
						dst[yoff + l] = dl * f32(q)
					}
					sc = i8(u8(aux[is / 4] >> u32(8 * (is % 4))))
					is += 1
					dl = d * f32(i32(sc) - 32)
					for l in 0 ..< 16 {
						q := i32(u8(qs[qoff + 16 + l] >> u32(shift)) & 3) - i32((hm[l + 16] & m) == 0 ? 4 : 0)
						dst[yoff + 16 + l] = dl * f32(q)
					}
					shift += 2
					m <<= 1
					yoff += 32
				}
				qoff += 32
			}
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
			for _ in 0 ..< 4 {
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
dequant_q5_k :: proc(src: []u8, n: int, dst: []f32) {
	#no_bounds_check {
		nsb := n / QK_K
		for sb in 0 ..< nsb {
			base := sb * 176
			d := f16_to_f32(src[base:])
			dmin := f16_to_f32(src[base + 2:])
			scales := src[base + 4:base + 16]
			qh := src[base + 16:base + 48]
			ql := src[base + 48:base + 176]
			yoff := sb * QK_K
			is := 0
			qloff := 0
			u1 := u8(1)
			u2 := u8(2)
			for _ in 0 ..< 4 {
				sc0, m0 := get_scale_min_k4(is + 0, scales)
				sc1, m1 := get_scale_min_k4(is + 1, scales)
				d1 := d * f32(sc0)
				mm1 := dmin * f32(m0)
				d2 := d * f32(sc1)
				mm2 := dmin * f32(m1)
				for l in 0 ..< 32 {
					v := i32(ql[qloff + l] & 0xF) + i32((qh[l] & u1) == 0 ? 0 : 16)
					dst[yoff + l] = d1 * f32(v) - mm1
				}
				for l in 0 ..< 32 {
					v := i32(ql[qloff + l] >> 4) + i32((qh[l] & u2) == 0 ? 0 : 16)
					dst[yoff + 32 + l] = d2 * f32(v) - mm2
				}
				qloff += 32
				is += 2
				u1 <<= 2
				u2 <<= 2
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

@(private = "file")
dequant_q8_k :: proc(src: []u8, n: int, dst: []f32) {
	#no_bounds_check {
		nsb := n / QK_K
		for sb in 0 ..< nsb {
			base := sb * 292
			d := transmute(f32)(u32(src[base]) | u32(src[base + 1]) << 8 | u32(src[base + 2]) << 16 | u32(src[base + 3]) << 24)
			yoff := sb * QK_K
			for j in 0 ..< QK_K {
				dst[yoff + j] = d * f32(i8(src[base + 4 + j]))
			}
		}
	}
}

// ============================== IQ2_XXS ==============================

@(private = "file")
dequant_iq2_xxs :: proc(src: []u8, n: int, dst: []f32) {
	#no_bounds_check {
		nsb := n / QK_K
		for sb in 0 ..< nsb {
			base := sb * 66
			d := f16_to_f32(src[base:])
			yoff := sb * QK_K
			for ib32 in 0 ..< 8 {
				// qs is uint16_t[32]: qs + 4*ib32 = byte offset 8*ib32
				qb := base + 2 + 8 * ib32
				aux1 := u32(src[qb + 4]) | u32(src[qb + 5]) << 8 | u32(src[qb + 6]) << 16 | u32(src[qb + 7]) << 24
				db := d * (0.5 + f32(aux1 >> 28)) * 0.25
				for l in 0 ..< 4 {
					grid := iq2xxs_grid[src[qb + l]]
					signs := ksigns_iq2xs[(aux1 >> u32(7 * l)) & 127]
					for j in 0 ..< 8 {
						v := f32(u8(grid >> u64(8 * j)) & 0xff)
						s := (signs & kmask_iq2xs[j]) == 0 ? f32(1) : f32(-1)
						dst[yoff + j] = db * v * s
					}
					yoff += 8
				}
			}
		}
	}
}

// ============================== IQ2_XS ==============================

@(private = "file")
dequant_iq2_xs :: proc(src: []u8, n: int, dst: []f32) {
	#no_bounds_check {
		nsb := n / QK_K
		for sb in 0 ..< nsb {
			base := sb * 74
			d := f16_to_f32(src[base:])
			yoff := sb * QK_K
			for ib32 in 0 ..< 8 {
				sc := src[base + 66 + ib32]
				db: [2]f32
				db[0] = d * (0.5 + f32(sc & 0xF)) * 0.25
				db[1] = d * (0.5 + f32(sc >> 4)) * 0.25
				for l in 0 ..< 4 {
					qoff := base + 2 + 2 * (4 * ib32 + l)
					q := u16(src[qoff]) | u16(src[qoff + 1]) << 8
					grid := iq2xs_grid[q & 511]
					signs := ksigns_iq2xs[q >> 9]
					dl := db[l / 2]
					for j in 0 ..< 8 {
						v := f32(u8(grid >> u64(8 * j)) & 0xff)
						s := (signs & kmask_iq2xs[j]) == 0 ? f32(1) : f32(-1)
						dst[yoff + j] = dl * v * s
					}
					yoff += 8
				}
			}
		}
	}
}

// ============================== IQ2_S ==============================

@(private = "file")
dequant_iq2_s :: proc(src: []u8, n: int, dst: []f32) {
	#no_bounds_check {
		nsb := n / QK_K
		for sb in 0 ..< nsb {
			base := sb * 82
			d := f16_to_f32(src[base:])
			qs := src[base + 2:base + 34]
			signs := src[base + 34:base + 66]
			qh := src[base + 66:base + 74]
			scales := src[base + 74:base + 82]
			yoff := sb * QK_K
			qoff := 0
			signoff := 0
			for ib32 in 0 ..< 8 {
				sc := scales[ib32]
				db: [2]f32
				db[0] = d * (0.5 + f32(sc & 0xF)) * 0.25
				db[1] = d * (0.5 + f32(sc >> 4)) * 0.25
				for l in 0 ..< 4 {
					idx := u32(qs[qoff + l]) | ((u32(qh[ib32]) << u32(8 - 2 * l)) & 0x300)
					grid := iq2s_grid[idx]
					sn := signs[signoff + l]
					dl := db[l / 2]
					for j in 0 ..< 8 {
						v := f32(u8(grid >> u64(8 * j)) & 0xff)
						s := (sn & kmask_iq2xs[j]) == 0 ? f32(1) : f32(-1)
						dst[yoff + j] = dl * v * s
					}
					yoff += 8
				}
				qoff += 4
				signoff += 4
			}
		}
	}
}

// ============================== IQ3_XXS ==============================

@(private = "file")
dequant_iq3_xxs :: proc(src: []u8, n: int, dst: []f32) {
	#no_bounds_check {
		nsb := n / QK_K
		for sb in 0 ..< nsb {
			base := sb * 98
			d := f16_to_f32(src[base:])
			qs := src[base + 2:base + 66]
			sas := src[base + 66:base + 98]
			yoff := sb * QK_K
			qoff := 0
			for ib32 in 0 ..< 8 {
				sb_ := 4 * ib32
				aux := u32(sas[sb_]) | u32(sas[sb_ + 1]) << 8 | u32(sas[sb_ + 2]) << 16 | u32(sas[sb_ + 3]) << 24
				db := d * (0.5 + f32(aux >> 28)) * 0.5
				for l in 0 ..< 4 {
					signs := ksigns_iq2xs[(aux >> u32(7 * l)) & 127]
					g1 := iq3xxs_grid[qs[qoff + 2 * l + 0]]
					g2 := iq3xxs_grid[qs[qoff + 2 * l + 1]]
					for j in 0 ..< 4 {
						v1 := f32(u8(g1 >> u32(8 * j)) & 0xff)
						v2 := f32(u8(g2 >> u32(8 * j)) & 0xff)
						s1 := (signs & kmask_iq2xs[j + 0]) == 0 ? f32(1) : f32(-1)
						s2 := (signs & kmask_iq2xs[j + 4]) == 0 ? f32(1) : f32(-1)
						dst[yoff + j + 0] = db * v1 * s1
						dst[yoff + j + 4] = db * v2 * s2
					}
					yoff += 8
				}
				qoff += 8
			}
		}
	}
}

// ============================== IQ3_S ==============================

@(private = "file")
dequant_iq3_s :: proc(src: []u8, n: int, dst: []f32) {
	#no_bounds_check {
		nsb := n / QK_K
		for sb in 0 ..< nsb {
			base := sb * 110
			d := f16_to_f32(src[base:])
			qs := src[base + 2:base + 66]
			qh := src[base + 66:base + 74]
			signs := src[base + 74:base + 106]
			scales := src[base + 106:base + 110]
			yoff := sb * QK_K
			qoff := 0
			signoff := 0
			qhoff := 0
			for _ in 0 ..< 4 { // ib32 = 0,2,4,6 (scales index = ib32/2)
				db1 := d * f32(1 + 2 * i32(scales[qhoff / 2] & 0xF))
				db2 := d * f32(1 + 2 * i32(scales[qhoff / 2] >> 4))
				for l in 0 ..< 4 {
					idx1 := u32(qs[qoff + 2 * l + 0]) | ((u32(qh[qhoff + 0]) << u32(8 - 2 * l)) & 256)
					idx2 := u32(qs[qoff + 2 * l + 1]) | ((u32(qh[qhoff + 0]) << u32(7 - 2 * l)) & 256)
					g1 := iq3s_grid[idx1]
					g2 := iq3s_grid[idx2]
					sn := signs[signoff + l]
					for j in 0 ..< 4 {
						v1 := f32(u8(g1 >> u32(8 * j)) & 0xff)
						v2 := f32(u8(g2 >> u32(8 * j)) & 0xff)
						s1 := (sn & kmask_iq2xs[j + 0]) == 0 ? f32(1) : f32(-1)
						s2 := (sn & kmask_iq2xs[j + 4]) == 0 ? f32(1) : f32(-1)
						dst[yoff + j + 0] = db1 * v1 * s1
						dst[yoff + j + 4] = db1 * v2 * s2
					}
					yoff += 8
				}
				qoff += 8
				signoff += 4
				for l in 0 ..< 4 {
					idx1 := u32(qs[qoff + 2 * l + 0]) | ((u32(qh[qhoff + 1]) << u32(8 - 2 * l)) & 256)
					idx2 := u32(qs[qoff + 2 * l + 1]) | ((u32(qh[qhoff + 1]) << u32(7 - 2 * l)) & 256)
					g1 := iq3s_grid[idx1]
					g2 := iq3s_grid[idx2]
					sn := signs[signoff + l]
					for j in 0 ..< 4 {
						v1 := f32(u8(g1 >> u32(8 * j)) & 0xff)
						v2 := f32(u8(g2 >> u32(8 * j)) & 0xff)
						s1 := (sn & kmask_iq2xs[j + 0]) == 0 ? f32(1) : f32(-1)
						s2 := (sn & kmask_iq2xs[j + 4]) == 0 ? f32(1) : f32(-1)
						dst[yoff + j + 0] = db2 * v1 * s1
						dst[yoff + j + 4] = db2 * v2 * s2
					}
					yoff += 8
				}
				qhoff += 2
				qoff += 8
				signoff += 4
			}
		}
	}
}

// ============================== IQ1_S ==============================

IQ1S_DELTA :: f32(0.125)

@(private = "file")
dequant_iq1_s :: proc(src: []u8, n: int, dst: []f32) {
	#no_bounds_check {
		nsb := n / QK_K
		for sb in 0 ..< nsb {
			base := sb * 50
			d := f16_to_f32(src[base:])
			qs := src[base + 2:base + 34]
			qh := src[base + 34:base + 50]
			yoff := sb * QK_K
			qoff := 0
			for ib in 0 ..< 8 {
				qhv := u16(qh[2 * ib]) | u16(qh[2 * ib + 1]) << 8
				dl := d * f32(2 * i32((qhv >> 12) & 7) + 1)
				delta := (qhv & 0x8000) == 0 ? IQ1S_DELTA : -IQ1S_DELTA
				for l in 0 ..< 4 {
					idx := u32(qs[qoff + l]) | u32(((qhv >> u16(3 * l)) & 7) << 8)
					grid := iq1s_grid[idx]
					for j in 0 ..< 8 {
						gv := f32(i8(u8(grid >> u64(8 * j)) & 0xff))
						dst[yoff + j] = dl * (gv + delta)
					}
					yoff += 8
				}
				qoff += 4
			}
		}
	}
}

// ============================== IQ1_M ==============================

@(private = "file")
dequant_iq1_m :: proc(src: []u8, n: int, dst: []f32) {
	#no_bounds_check {
		nsb := n / QK_K
		for sb in 0 ..< nsb {
			base := sb * 56
			// block_iq1_m layout: qs[32], qh[16], scales[8]
			qs := src[base:base + 32]
			qh := src[base + 32:base + 48]
			scales := src[base + 48:base + 56]
			sc: [4]u16
			for i in 0 ..< 4 {
				sc[i] = u16(scales[2 * i]) | u16(scales[2 * i + 1]) << 8
			}
			scale := (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000)
			d := f32(transmute(f16)scale)
			yoff := sb * QK_K
			qoff := 0
			qhoff := 0
			for ib in 0 ..< 8 {
				dl1 := d * f32(2 * i32((sc[ib / 2] >> u16(6 * (ib % 2) + 0)) & 0x7) + 1)
				dl2 := d * f32(2 * i32((sc[ib / 2] >> u16(6 * (ib % 2) + 3)) & 0x7) + 1)
				idx: [4]u32
				idx[0] = u32(qs[qoff + 0]) | ((u32(qh[qhoff + 0]) << 8) & 0x700)
				idx[1] = u32(qs[qoff + 1]) | ((u32(qh[qhoff + 0]) << 4) & 0x700)
				idx[2] = u32(qs[qoff + 2]) | ((u32(qh[qhoff + 1]) << 8) & 0x700)
				idx[3] = u32(qs[qoff + 3]) | ((u32(qh[qhoff + 1]) << 4) & 0x700)
				delta: [4]f32
				delta[0] = (qh[qhoff + 0] & 0x08) == 0 ? IQ1S_DELTA : -IQ1S_DELTA
				delta[1] = (qh[qhoff + 0] & 0x80) == 0 ? IQ1S_DELTA : -IQ1S_DELTA
				delta[2] = (qh[qhoff + 1] & 0x08) == 0 ? IQ1S_DELTA : -IQ1S_DELTA
				delta[3] = (qh[qhoff + 1] & 0x80) == 0 ? IQ1S_DELTA : -IQ1S_DELTA
				for l in 0 ..< 2 {
					grid := iq1s_grid[idx[l]]
					for j in 0 ..< 8 {
						gv := f32(i8(u8(grid >> u64(8 * j)) & 0xff))
						dst[yoff + j] = dl1 * (gv + delta[l])
					}
					yoff += 8
				}
				for l in 2 ..< 4 {
					grid := iq1s_grid[idx[l]]
					for j in 0 ..< 8 {
						gv := f32(i8(u8(grid >> u64(8 * j)) & 0xff))
						dst[yoff + j] = dl2 * (gv + delta[l])
					}
					yoff += 8
				}
				qoff += 4
				qhoff += 2
			}
		}
	}
}

// ============================== IQ4_NL ==============================

@(private = "file")
dequant_iq4_nl :: proc(src: []u8, n: int, dst: []f32) {
	#no_bounds_check {
		nsb := n / QK4_NL
		for sb in 0 ..< nsb {
			base := sb * 18
			d := f16_to_f32(src[base:])
			qs := src[base + 2:]
			o := sb * QK4_NL
			for j in 0 ..< QK4_NL / 2 {
				dst[o + j] = d * f32(kvalues_iq4nl[qs[j] & 0xF])
				dst[o + j + QK4_NL / 2] = d * f32(kvalues_iq4nl[qs[j] >> 4])
			}
		}
	}
}

// ============================== IQ4_XS ==============================

@(private = "file")
dequant_iq4_xs :: proc(src: []u8, n: int, dst: []f32) {
	#no_bounds_check {
		nsb := n / QK_K
		for sb in 0 ..< nsb {
			base := sb * 136
			d := f16_to_f32(src[base:])
			scales_h := u16(src[base + 2]) | u16(src[base + 3]) << 8
			scales_l := src[base + 4:base + 8]
			qs := src[base + 8:base + 136]
			yoff := sb * QK_K
			qoff := 0
			for ib in 0 ..< 8 {
				ls := i32((scales_l[ib / 2] >> u32(4 * (ib % 2))) & 0xF) | i32(((scales_h >> u16(2 * ib)) & 3) << 4)
				dl := d * f32(ls - 32)
				for j in 0 ..< 16 {
					dst[yoff + j] = dl * f32(kvalues_iq4nl[qs[qoff + j] & 0xF])
					dst[yoff + j + 16] = dl * f32(kvalues_iq4nl[qs[qoff + j] >> 4])
				}
				yoff += 32
				qoff += 16
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
