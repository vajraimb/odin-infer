/* qwen3_5/kv_persist.odin — KV + SSM state persistence for Qwen3.5 (Ornith).

   Persists the position-evolving state (key_cache, value_cache for full-attention
   layers, plus conv_states and recurrent_states for linear-attention layers) so
   a conversation can resume without re-prefill.

   On-disk format (all little-endian):

     Header (fixed, 80 bytes):
       magic            : u32 = 0x4F494B56   ('OIKV')
       version          : u32 = 1
       arch             : u32 = 2            (qwen3_5)
       fingerprint      : u64                FNV-1a of identifying Config fields
       n_valid_pos      : u32                positions actually populated
       seq_len          : u32                max_ctx at save time
       n_full_layers    : u32
       n_linear_layers  : u32
       kv_dim           : u32                n_kv_heads * head_dim
       conv_dim         : u32
       lin_conv_kernel  : u32
       lin_n_v_heads    : u32
       lin_head_k_dim   : u32
       lin_head_v_dim   : u32
       _pad             : u32                alignment to 8

     Body (all f32, CPU standard layout, K/V trimmed to n_valid_pos):
       key_cache        : [n_full_layers * n_valid_pos * kv_dim] f32
       value_cache      : [n_full_layers * n_valid_pos * kv_dim] f32
       conv_states      : [n_linear * conv_dim * (kernel-1)] f32
       recurrent_states : [n_linear * lin_n_v_heads * lin_head_k_dim * lin_head_v_dim] f32

     crc32             : u32                 IEEE CRC over header+body

   Total size for 100-token chat (Qwen3.5-9B-ish, kv_dim=1024, n_full=32):
     32 * 100 * 1024 * 4 * 2 ≈ 26 MB + small conv/recurrent blobs.

   The Metal backend stores KV in f16 head-major layout in GPU buffers; this
   module does f16↔f32 + layout conversion at save/load time (see metal.odin).
*/

package qwen3_5

import "core:fmt"
import "core:hash"
import "core:os"

KV_MAGIC :: u32(0x4F494B56) // 'O','I','K','V' little-endian
KV_VERSION :: u32(1)
KV_ARCH_QWEN3_5 :: u32(2)

KV_Header :: struct {
	magic:           u32,
	version:         u32,
	arch:            u32,
	fingerprint:     u64,
	n_valid_pos:     u32,
	seq_len:         u32,
	n_full_layers:   u32,
	n_linear_layers: u32,
	kv_dim:          u32,
	conv_dim:        u32,
	lin_conv_kernel: u32,
	lin_n_v_heads:   u32,
	lin_head_k_dim:  u32,
	lin_head_v_dim:  u32,
}

// ---------- helpers ----------

compute_conv_dim :: proc(c: ^Config) -> int {
	return c.lin_n_k_heads * c.lin_head_k_dim * 2 + c.lin_n_v_heads * c.lin_head_v_dim
}

model_fingerprint :: proc(t: ^Transformer) -> u64 {
	// FNV-1a 64-bit over the identifying Config fields. If any of these change
	// between save and load, the fingerprint mismatches and load refuses.
	c := &t.config
	h: u64 = 0xcbf29ce484222325 // FNV-1a offset basis
	mix :: proc(h: u64, x: u64) -> u64 {
		return (h ~ x) * 0x100000001b3
	}
	h = mix(h, u64(c.dim))
	h = mix(h, u64(c.n_layers))
	h = mix(h, u64(c.n_heads))
	h = mix(h, u64(c.n_kv_heads))
	h = mix(h, u64(c.head_dim))
	h = mix(h, u64(c.lin_n_k_heads))
	h = mix(h, u64(c.lin_n_v_heads))
	h = mix(h, u64(c.lin_head_k_dim))
	h = mix(h, u64(c.lin_head_v_dim))
	h = mix(h, u64(c.lin_conv_kernel))
	h = mix(h, u64(c.vocab_size))
	return h
}

build_header :: proc(t: ^Transformer, n_valid_pos: int) -> KV_Header {
	c := &t.config
	return KV_Header {
		magic = KV_MAGIC,
		version = KV_VERSION,
		arch = KV_ARCH_QWEN3_5,
		fingerprint = model_fingerprint(t),
		n_valid_pos = u32(n_valid_pos),
		seq_len = u32(c.seq_len),
		n_full_layers = u32(c.n_full),
		n_linear_layers = u32(c.n_linear),
		kv_dim = u32(c.n_kv_heads * c.head_dim),
		conv_dim = u32(compute_conv_dim(c)),
		lin_conv_kernel = u32(c.lin_conv_kernel),
		lin_n_v_heads = u32(c.lin_n_v_heads),
		lin_head_k_dim = u32(c.lin_head_k_dim),
		lin_head_v_dim = u32(c.lin_head_v_dim),
	}
}

// Body section byte sizes (for a given header)
kv_bytes :: proc(h: ^KV_Header) -> u64 {
	return u64(h.n_full_layers) * u64(h.n_valid_pos) * u64(h.kv_dim) * 4
}
conv_bytes :: proc(h: ^KV_Header) -> u64 {
	if h.lin_conv_kernel == 0 { return 0 }
	return u64(h.n_linear_layers) * u64(h.conv_dim) * u64(h.lin_conv_kernel - 1) * 4
}
rec_bytes :: proc(h: ^KV_Header) -> u64 {
	return u64(h.n_linear_layers) * u64(h.lin_n_v_heads) *
		u64(h.lin_head_k_dim) * u64(h.lin_head_v_dim) * 4
}

// ---------- little-endian file IO ----------

write_u32 :: proc(f: ^os.File, v: u32) {
	buf := [4]u8{u8(v & 0xff), u8((v >> 8) & 0xff), u8((v >> 16) & 0xff), u8((v >> 24) & 0xff)}
	_, _ = os.write(f, buf[:])
}

write_u64 :: proc(f: ^os.File, v: u64) {
	buf: [8]u8
	for i in 0..<8 { buf[i] = u8((v >> cast(u64)(i * 8)) & 0xff) }
	_, _ = os.write(f, buf[:])
}

write_header :: proc(f: ^os.File, h: KV_Header) {
	write_u32(f, h.magic)
	write_u32(f, h.version)
	write_u32(f, h.arch)
	write_u64(f, h.fingerprint)
	write_u32(f, h.n_valid_pos)
	write_u32(f, h.seq_len)
	write_u32(f, h.n_full_layers)
	write_u32(f, h.n_linear_layers)
	write_u32(f, h.kv_dim)
	write_u32(f, h.conv_dim)
	write_u32(f, h.lin_conv_kernel)
	write_u32(f, h.lin_n_v_heads)
	write_u32(f, h.lin_head_k_dim)
	write_u32(f, h.lin_head_v_dim)
}

write_f32_slice :: proc(f: ^os.File, s: []f32) -> bool {
	if len(s) == 0 { return true }
	bytes := (cast([^]u8)raw_data(s))[:len(s) * 4]
	n, err := os.write(f, bytes)
	return err == os.ERROR_NONE && n == len(bytes)
}

read_exact :: proc(f: ^os.File, buf: []u8) -> bool {
	n, err := os.read(f, buf)
	return err == os.ERROR_NONE && n == len(buf)
}

read_u32 :: proc(f: ^os.File) -> (u32, bool) {
	buf: [4]u8
	if !read_exact(f, buf[:]) { return 0, false }
	return u32(buf[0]) | (u32(buf[1]) << 8) | (u32(buf[2]) << 16) | (u32(buf[3]) << 24), true
}

read_u64 :: proc(f: ^os.File) -> (u64, bool) {
	buf: [8]u8
	if !read_exact(f, buf[:]) { return 0, false }
	v: u64 = 0
	for i in 0..<8 { v |= u64(buf[i]) << cast(u64)(i * 8) }
	return v, true
}

read_header :: proc(f: ^os.File) -> (KV_Header, bool) {
	h: KV_Header
	ok: bool
	h.magic, ok = read_u32(f); if !ok { return h, false }
	h.version, ok = read_u32(f); if !ok { return h, false }
	h.arch, ok = read_u32(f); if !ok { return h, false }
	h.fingerprint, ok = read_u64(f); if !ok { return h, false }
	h.n_valid_pos, ok = read_u32(f); if !ok { return h, false }
	h.seq_len, ok = read_u32(f); if !ok { return h, false }
	h.n_full_layers, ok = read_u32(f); if !ok { return h, false }
	h.n_linear_layers, ok = read_u32(f); if !ok { return h, false }
	h.kv_dim, ok = read_u32(f); if !ok { return h, false }
	h.conv_dim, ok = read_u32(f); if !ok { return h, false }
	h.lin_conv_kernel, ok = read_u32(f); if !ok { return h, false }
	h.lin_n_v_heads, ok = read_u32(f); if !ok { return h, false }
	h.lin_head_k_dim, ok = read_u32(f); if !ok { return h, false }
	h.lin_head_v_dim, ok = read_u32(f); if !ok { return h, false }
	return h, true
}

read_f32_into :: proc(f: ^os.File, dst: []f32) -> bool {
	if len(dst) == 0 { return true }
	bytes := (cast([^]u8)raw_data(dst))[:len(dst) * 4]
	return read_exact(f, bytes)
}

// ---------- CPU path ----------

// CPU KV layout (from Run_State):
//   key_cache   : [n_full, seq_len, kv_dim]   f32
//   value_cache : [n_full, seq_len, kv_dim]   f32
// conv_states / recurrent_states are already the canonical layout.

cpu_save_kv :: proc(t: ^Transformer, path: string, n_valid_pos: int) -> bool {
	c := &t.config
	s := &t.state
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

	// K/V — per-layer slice [n_valid_pos * kv_dim]
	kv_dim := c.n_kv_heads * c.head_dim
	kv_row_bytes := u64(n_valid_pos) * u64(kv_dim)
	for l in 0 ..< c.n_full {
		layer_off := u64(l) * u64(c.seq_len) * u64(kv_dim)
		k_slice := s.key_cache[layer_off:layer_off + kv_row_bytes]
		v_slice := s.value_cache[layer_off:layer_off + kv_row_bytes]
		if !write_f32_slice(f, k_slice) {
			fmt.eprintf("kv: write failed at K layer %d\n", l); return false
		}
		if !write_f32_slice(f, v_slice) {
			fmt.eprintf("kv: write failed at V layer %d\n", l); return false
		}
	}
	if !write_f32_slice(f, s.conv_states) {
		fmt.eprintln("kv: conv_states write failed"); return false
	}
	if !write_f32_slice(f, s.recurrent_states) {
		fmt.eprintln("kv: recurrent_states write failed"); return false
	}

	// CRC32 over header + body. We re-read the file because we wrote header
	// field-by-field and can't easily hash in-process.
	// (Alternative: compute incrementally — left for v2.)
	if !append_crc32(path) {
		fmt.eprintln("kv: crc append failed"); return false
	}
	fmt.printfln("kv: saved %s (%d valid pos, %d full L, kv_dim=%d)",
		path, n_valid_pos, c.n_full, kv_dim)
	return true
}

cpu_load_kv :: proc(t: ^Transformer, path: string) -> (int, bool) {
	f, err := os.open(path, os.O_RDONLY)
	if err != os.ERROR_NONE {
		fmt.eprintf("kv: cannot open %s (%v)\n", path, err)
		return 0, false
	}
	defer os.close(f)

	h, ok := read_header(f)
	if !ok { fmt.eprintln("kv: header truncated"); return 0, false }
	if h.magic != KV_MAGIC {
		fmt.eprintf("kv: bad magic 0x%08x\n", h.magic); return 0, false
	}
	if h.version != KV_VERSION {
		fmt.eprintf("kv: version mismatch (file=%d, supported=%d)\n", h.version, KV_VERSION); return 0, false
	}
	if h.arch != KV_ARCH_QWEN3_5 {
		fmt.eprintf("kv: arch mismatch (file=%d, expected qwen3_5)\n", h.arch); return 0, false
	}
	expected_fp := model_fingerprint(t)
	if h.fingerprint != expected_fp {
		fmt.eprintf("kv: fingerprint mismatch (file=0x%016x, model=0x%016x) — wrong model?\n",
			h.fingerprint, expected_fp)
		return 0, false
	}
	c := &t.config
	if int(h.seq_len) != c.seq_len {
		fmt.eprintf("kv: seq_len mismatch (file=%d, engine=%d) — use same -c\n",
			h.seq_len, c.seq_len)
		return 0, false
	}
	if int(h.n_valid_pos) > c.seq_len {
		fmt.eprintf("kv: file n_valid_pos %d > engine seq_len %d\n", h.n_valid_pos, c.seq_len)
		return 0, false
	}

	// Verify CRC32 before touching Run_State
	if !verify_crc32(path) {
		fmt.eprintln("kv: CRC mismatch — file corrupted"); return 0, false
	}

	s := &t.state
	// Zero entire run-state first (cheap relative to file IO; ensures no stale
	// tail beyond n_valid_pos).
	for i in 0 ..< len(s.key_cache)   { s.key_cache[i] = 0 }
	for i in 0 ..< len(s.value_cache) { s.value_cache[i] = 0 }

	kv_dim := c.n_kv_heads * c.head_dim
	kv_row := int(h.n_valid_pos) * kv_dim
	for l in 0 ..< c.n_full {
		layer_off := l * c.seq_len * kv_dim
		k_slice := s.key_cache[layer_off:layer_off + kv_row]
		v_slice := s.value_cache[layer_off:layer_off + kv_row]
		if !read_f32_into(f, k_slice) {
			fmt.eprintf("kv: read failed at K layer %d\n", l); return 0, false
		}
		if !read_f32_into(f, v_slice) {
			fmt.eprintf("kv: read failed at V layer %d\n", l); return 0, false
		}
	}
	if !read_f32_into(f, s.conv_states) {
		fmt.eprintln("kv: conv_states read failed"); return 0, false
	}
	if !read_f32_into(f, s.recurrent_states) {
		fmt.eprintln("kv: recurrent_states read failed"); return 0, false
	}

	fmt.printfln("kv: loaded %s (%d valid pos)", path, h.n_valid_pos)
	return int(h.n_valid_pos), true
}

// ---------- CRC32 over file contents (excluding the trailing 4 CRC bytes) ----------

append_crc32 :: proc(path: string) -> bool {
	data, err := os.read_entire_file_from_path(path, context.temp_allocator)
	if err != os.ERROR_NONE { return false }
	c := hash.crc32(data)
	f, oerr := os.open(path, os.O_APPEND | os.O_WRONLY)
	if oerr != os.ERROR_NONE { return false }
	defer os.close(f)
	write_u32(f, c)
	return true
}

verify_crc32 :: proc(path: string) -> bool {
	data, err := os.read_entire_file_from_path(path, context.temp_allocator)
	if err != os.ERROR_NONE { return false }
	if len(data) < 4 { return false }
	body := data[:len(data) - 4]
	stored_crc := u32(data[len(data) - 4]) |
		(u32(data[len(data) - 3]) << 8) |
		(u32(data[len(data) - 2]) << 16) |
		(u32(data[len(data) - 1]) << 24)
	return hash.crc32(body) == stored_crc
}

// ---------- public dispatch ----------

engine_save_kv :: proc(e: ^Engine, path: string, n_valid_pos: int) -> bool {
	when ODIN_OS == .Darwin {
		if e.metal_ready {
			return metal_save_kv(&e.transformer, path, n_valid_pos)
		}
	}
	return cpu_save_kv(&e.transformer, path, n_valid_pos)
}

engine_load_kv :: proc(e: ^Engine, path: string) -> (int, bool) {
	when ODIN_OS == .Darwin {
		if e.metal_ready {
			return metal_load_kv(&e.transformer, path)
		}
	}
	return cpu_load_kv(&e.transformer, path)
}
