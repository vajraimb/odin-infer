/* KV snapshot I/O for the CUDA backend. On-disk bytes are the same f32
   canonical layout Metal and the CPU path already share. The device cache
   stays in the Q8 / mixed-V / f16 layout selected at init. */

package qwen3_5

import cu "cuda:cuda"

import "core:fmt"
import "core:os"

@(private = "file")
_cuda_kv_import_keep :: proc() {
	_ = cu.BACKEND_TAG
	_ = fmt.tprintf("")
	_ = os.O_RDONLY
}

when GPU_CUDA {

@(private)
kv_header_ok :: proc(t: ^Transformer, h: KV_Header) -> bool {
	if h.magic != KV_MAGIC {
		fmt.eprintf("kv: bad magic 0x%08x\n", h.magic)
		return false
	}
	if h.version != KV_VERSION {
		fmt.eprintf(
			"kv: version mismatch (file=%d, supported=%d) — file written by an incompatible engine build (pre-Q8 KV format); delete it (or set KV_PERSIST=0)\n",
			h.version, KV_VERSION,
		)
		return false
	}
	if h.arch != KV_ARCH_QWEN3_5 {
		fmt.eprintf("kv: arch mismatch (file=%d, expected qwen3_5)\n", h.arch)
		return false
	}
	expected := model_fingerprint(t)
	if h.fingerprint != expected {
		fmt.eprintf("kv: fingerprint mismatch (file=0x%016x, model=0x%016x)\n", h.fingerprint, expected)
		return false
	}
	if int(h.seq_len) != t.config.seq_len {
		fmt.eprintf("kv: seq_len mismatch (file=%d, engine=%d) — use same -c\n", h.seq_len, t.config.seq_len)
		return false
	}
	if int(h.n_valid_pos) > t.config.seq_len {
		fmt.eprintf("kv: file n_valid_pos %d > engine seq_len %d\n", h.n_valid_pos, t.config.seq_len)
		return false
	}
	return true
}

cuda_save_kv :: proc(t: ^Transformer, path: string, n_valid_pos: int) -> bool {
	if !g_ready {
		fmt.eprintln("kv: cuda not enabled")
		return false
	}
	c := &t.config
	if n_valid_pos > c.seq_len {
		fmt.eprintf("kv: n_valid_pos %d > seq_len %d\n", n_valid_pos, c.seq_len)
		return false
	}
	h := build_header(t, n_valid_pos)
	f, err := os.open(path, os.O_CREATE | os.O_WRONLY | os.O_TRUNC)
	if err != os.ERROR_NONE {
		fmt.eprintf("kv: cannot open %s for write (%v)\n", path, err)
		return false
	}
	defer os.close(f)
	write_header(f, h)

	kv_dim := c.n_kv_heads * c.head_dim
	tmp := make([]f32, n_valid_pos * kv_dim)
	defer delete(tmp)
	k_layer := make([]u8, g_k_layer)
	defer delete(k_layer)
	v_layer := make([]u8, g_v_layer)
	defer delete(v_layer)

	for l in 0 ..< c.n_full {
		if !ck(cu.download(raw_data(k_layer), cu.ptr_add(d_kc, l * g_k_layer), g_k_layer), "kv K") {
			return false
		}
		if !ck(cu.download(raw_data(v_layer), cu.ptr_add(d_vc, l * g_v_layer), g_v_layer), "kv V") {
			return false
		}
		if g_kv_q8 {
			pos_bytes := g_k_pos
			for p in 0 ..< n_valid_pos {
				for b in 0 ..< kv_dim / QK8_0 {
					off := p * pos_bytes + b * Q8_0_BLOCK_BYTES
					dequantize_q8_block(k_layer[off:off + Q8_0_BLOCK_BYTES], tmp[p * kv_dim + b * QK8_0:][:QK8_0])
				}
			}
			if !write_f32_slice(f, tmp) {
				fmt.eprintf("kv: K write failed at layer %d\n", l)
				return false
			}
			if g_kv_v_mix {
				for p in 0 ..< n_valid_pos {
					for b in 0 ..< kv_dim / QK8_0 {
						off := p * g_v_pos + b * Q8_V_MIX_BLOCK_BYTES
						dequantize_q8_v_mix_block(v_layer[off:off + Q8_V_MIX_BLOCK_BYTES], tmp[p * kv_dim + b * QK8_0:][:QK8_0])
					}
				}
			} else {
				for p in 0 ..< n_valid_pos {
					for b in 0 ..< kv_dim / QK8_0 {
						off := p * pos_bytes + b * Q8_0_BLOCK_BYTES
						dequantize_q8_block(v_layer[off:off + Q8_0_BLOCK_BYTES], tmp[p * kv_dim + b * QK8_0:][:QK8_0])
					}
				}
			}
			if !write_f32_slice(f, tmp) {
				fmt.eprintf("kv: V write failed at layer %d\n", l)
				return false
			}
		} else {
			for which in 0 ..< 2 {
				src := k_layer
				if which == 1 {
					src = v_layer
				}
				for h_idx in 0 ..< c.n_kv_heads {
					for p in 0 ..< n_valid_pos {
						for d in 0 ..< c.head_dim {
							m_off := (h_idx * c.seq_len * c.head_dim + p * c.head_dim + d) * 2
							raw := u16(src[m_off]) | (u16(src[m_off + 1]) << 8)
							tmp[p * kv_dim + h_idx * c.head_dim + d] = f32(transmute(f16)(raw))
						}
					}
				}
				if !write_f32_slice(f, tmp) {
					fmt.eprintf("kv: f16 write failed at layer %d\n", l)
					return false
				}
			}
		}
	}

	conv_n := int(h.n_linear_layers) * int(h.conv_dim) * int(h.lin_conv_kernel - 1)
	conv := make([]f32, conv_n)
	defer delete(conv)
	if conv_n > 0 {
		if !ck(cu.download(raw_data(conv), d_conv, conv_n * 4), "kv conv") {
			return false
		}
	}
	if !write_f32_slice(f, conv) {
		fmt.eprintln("kv: conv_states write failed")
		return false
	}
	rec_n := int(h.n_linear_layers) * int(h.lin_n_v_heads) * int(h.lin_head_k_dim) * int(h.lin_head_v_dim)
	rec := make([]f32, rec_n)
	defer delete(rec)
	if rec_n > 0 {
		if !ck(cu.download(raw_data(rec), d_rec, rec_n * 4), "kv rec") {
			return false
		}
	}
	if !write_f32_slice(f, rec) {
		fmt.eprintln("kv: recurrent_states write failed")
		return false
	}
	if !append_crc32(path) {
		fmt.eprintln("kv: crc append failed")
		return false
	}
	fmt.eprintfln("kv: saved (CUDA) %s (%d valid pos, %d full L, kv_dim=%d)", path, n_valid_pos, c.n_full, kv_dim)
	return true
}

cuda_load_kv :: proc(t: ^Transformer, path: string) -> (int, bool) {
	if !g_ready {
		fmt.eprintln("kv: cuda not enabled")
		return 0, false
	}
	f, err := os.open(path, os.O_RDONLY)
	if err != os.ERROR_NONE {
		fmt.eprintf("kv: cannot open %s (%v)\n", path, err)
		return 0, false
	}
	defer os.close(f)
	h, ok := read_header(f)
	if !ok {
		fmt.eprintln("kv: header truncated")
		return 0, false
	}
	if !kv_header_ok(t, h) {
		return 0, false
	}
	if !verify_crc32(path) {
		fmt.eprintln("kv: CRC mismatch — file corrupted")
		return 0, false
	}
	cu.memset(d_kc, 0, g_k_bytes)
	cu.memset(d_vc, 0, g_v_bytes)

	c := &t.config
	kv_dim := c.n_kv_heads * c.head_dim
	n_valid := int(h.n_valid_pos)
	tmp := make([]f32, n_valid * kv_dim)
	defer delete(tmp)
	k_layer := make([]u8, g_k_layer)
	defer delete(k_layer)
	v_layer := make([]u8, g_v_layer)
	defer delete(v_layer)

	for l in 0 ..< c.n_full {
		if !read_f32_into(f, tmp) {
			fmt.eprintf("kv: K read failed at layer %d\n", l)
			return 0, false
		}
		if g_kv_q8 {
			for p in 0 ..< n_valid {
				for b in 0 ..< kv_dim / QK8_0 {
					off := p * g_k_pos + b * Q8_0_BLOCK_BYTES
					quantize_q8_block(tmp[p * kv_dim + b * QK8_0:][:QK8_0], k_layer[off:off + Q8_0_BLOCK_BYTES])
				}
			}
		} else {
			for h_idx in 0 ..< c.n_kv_heads {
				for p in 0 ..< n_valid {
					for d in 0 ..< c.head_dim {
						m_off := (h_idx * c.seq_len * c.head_dim + p * c.head_dim + d) * 2
						raw := transmute(u16)(f16(tmp[p * kv_dim + h_idx * c.head_dim + d]))
						k_layer[m_off] = u8(raw & 0xff)
						k_layer[m_off + 1] = u8((raw >> 8) & 0xff)
					}
				}
			}
		}
		if !ck(cu.upload(cu.ptr_add(d_kc, l * g_k_layer), raw_data(k_layer), g_k_layer), "kv K upload") {
			return 0, false
		}

		if !read_f32_into(f, tmp) {
			fmt.eprintf("kv: V read failed at layer %d\n", l)
			return 0, false
		}
		if g_kv_q8 && g_kv_v_mix {
			for p in 0 ..< n_valid {
				for b in 0 ..< kv_dim / QK8_0 {
					off := p * g_v_pos + b * Q8_V_MIX_BLOCK_BYTES
					quantize_q8_v_mix_block(tmp[p * kv_dim + b * QK8_0:][:QK8_0], v_layer[off:off + Q8_V_MIX_BLOCK_BYTES])
				}
			}
		} else if g_kv_q8 {
			for p in 0 ..< n_valid {
				for b in 0 ..< kv_dim / QK8_0 {
					off := p * g_v_pos + b * Q8_0_BLOCK_BYTES
					quantize_q8_block(tmp[p * kv_dim + b * QK8_0:][:QK8_0], v_layer[off:off + Q8_0_BLOCK_BYTES])
				}
			}
		} else {
			for h_idx in 0 ..< c.n_kv_heads {
				for p in 0 ..< n_valid {
					for d in 0 ..< c.head_dim {
						m_off := (h_idx * c.seq_len * c.head_dim + p * c.head_dim + d) * 2
						raw := transmute(u16)(f16(tmp[p * kv_dim + h_idx * c.head_dim + d]))
						v_layer[m_off] = u8(raw & 0xff)
						v_layer[m_off + 1] = u8((raw >> 8) & 0xff)
					}
				}
			}
		}
		if !ck(cu.upload(cu.ptr_add(d_vc, l * g_v_layer), raw_data(v_layer), g_v_layer), "kv V upload") {
			return 0, false
		}
	}

	conv_n := int(h.n_linear_layers) * int(h.conv_dim) * int(h.lin_conv_kernel - 1)
	conv := make([]f32, conv_n)
	defer delete(conv)
	if !read_f32_into(f, conv) {
		fmt.eprintln("kv: conv_states read failed")
		return 0, false
	}
	if conv_n > 0 && !ck(cu.upload(d_conv, raw_data(conv), conv_n * 4), "kv conv upload") {
		return 0, false
	}
	rec_n := int(h.n_linear_layers) * int(h.lin_n_v_heads) * int(h.lin_head_k_dim) * int(h.lin_head_v_dim)
	rec := make([]f32, rec_n)
	defer delete(rec)
	if !read_f32_into(f, rec) {
		fmt.eprintln("kv: recurrent_states read failed")
		return 0, false
	}
	if rec_n > 0 && !ck(cu.upload(d_rec, raw_data(rec), rec_n * 4), "kv rec upload") {
		return 0, false
	}
	fmt.eprintfln("kv: loaded (CUDA) %s (%d valid pos)", path, n_valid)
	return n_valid, true
}

cuda_save_full_state :: proc() -> (kv_k: []u8, kv_v: []u8, conv: []u8, rec: []u8) {
	if !g_ready {
		return
	}
	kv_k = make([]u8, g_k_bytes)
	kv_v = make([]u8, g_v_bytes)
	if g_k_bytes > 0 && !ck(cu.download(raw_data(kv_k), d_kc, g_k_bytes), "save K") {
		return nil, nil, nil, nil
	}
	if g_v_bytes > 0 && !ck(cu.download(raw_data(kv_v), d_vc, g_v_bytes), "save V") {
		return nil, nil, nil, nil
	}
	conv = make([]u8, g_conv_bytes)
	rec = make([]u8, g_rec_bytes)
	if g_conv_bytes > 0 && !ck(cu.download(raw_data(conv), d_conv, g_conv_bytes), "save conv") {
		return nil, nil, nil, nil
	}
	if g_rec_bytes > 0 && !ck(cu.download(raw_data(rec), d_rec, g_rec_bytes), "save rec") {
		return nil, nil, nil, nil
	}
	return
}

cuda_restore_full_state :: proc(kv_k, kv_v, conv, rec: []u8) {
	if !g_ready {
		return
	}
	if len(kv_k) != g_k_bytes || len(kv_v) != g_v_bytes {
		fmt.eprintf("cuda: kv snapshot %d/%d bytes, engine %d/%d\n", len(kv_k), len(kv_v), g_k_bytes, g_v_bytes)
		return
	}
	if len(kv_k) > 0 {
		cu.upload(d_kc, raw_data(kv_k), len(kv_k))
	}
	if len(kv_v) > 0 {
		cu.upload(d_vc, raw_data(kv_v), len(kv_v))
	}
	cuda_restore_spec_state(conv, rec)
}

cuda_save_spec_state :: proc() -> (conv: []u8, rec: []u8) {
	if !g_ready {
		return
	}
	conv = make([]u8, g_conv_bytes)
	rec = make([]u8, g_rec_bytes)
	if g_conv_bytes > 0 {
		cu.download(raw_data(conv), d_conv, g_conv_bytes)
	}
	if g_rec_bytes > 0 {
		cu.download(raw_data(rec), d_rec, g_rec_bytes)
	}
	return
}

cuda_restore_spec_state :: proc(conv, rec: []u8) {
	if !g_ready {
		return
	}
	if len(conv) == g_conv_bytes && g_conv_bytes > 0 {
		cu.upload(d_conv, raw_data(conv), g_conv_bytes)
	}
	if len(rec) == g_rec_bytes && g_rec_bytes > 0 {
		cu.upload(d_rec, raw_data(rec), g_rec_bytes)
	}
}

} // GPU_CUDA
