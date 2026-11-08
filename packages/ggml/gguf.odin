/* Minimal GGUF v2/v3 binary parser (little-endian hosts) */

package ggml

import "core:fmt"
import "core:mem/virtual"
import "core:os"

GGUF_MAGIC :: 0x46554747 // "GGUF"

GGUF_Type :: enum u32 {
	UINT8   = 0,
	INT8    = 1,
	UINT16  = 2,
	INT16   = 3,
	UINT32  = 4,
	INT32   = 5,
	FLOAT32 = 6,
	BOOL    = 7,
	STRING  = 8,
	ARRAY   = 9,
	UINT64  = 10,
	INT64   = 11,
	FLOAT64 = 12,
}

GGML_Type :: enum u32 {
	F32  = 0,
	F16  = 1,
	Q4_0 = 2,
	Q4_1 = 3,
	Q5_0 = 6,
	Q5_1 = 7,
	Q8_0 = 8,
	Q8_1 = 9,
	Q2_K = 10,
	Q3_K = 11,
	Q4_K = 12,
	Q5_K = 13,
	Q6_K = 14,
	Q8_K = 15,
}

GGUF_Meta :: struct {
	type:     GGUF_Type,
	u:        u64, // ints, uints, bools
	f:        f64, // floats
	str:      string, // strings (slice into mmap)
	arr_type: GGUF_Type,
	arr_len:  u64,
}

GGUF_Tensor :: struct {
	name:   string,
	kind:   GGML_Type,
	dims:   []u64,
	offset: u64, // relative to data section
	data:   []u8, // resolved slice into mmap
}

GGUF_File :: struct {
	metadata:   map[string]GGUF_Meta,
	tensors:    []GGUF_Tensor,
	tensor_map: map[string]int,
	mmap:       []u8,
}

Reader :: struct {
	data: []u8,
	pos:  int,
}

read_u8 :: proc(r: ^Reader) -> u8 {
	v := r.data[r.pos]
	r.pos += 1
	return v
}

read_u16 :: proc(r: ^Reader) -> u16 {
	b := r.data[r.pos:]
	r.pos += 2
	return u16(b[0]) | u16(b[1]) << 8
}

read_u32 :: proc(r: ^Reader) -> u32 {
	b := r.data[r.pos:]
	r.pos += 4
	return u32(b[0]) | u32(b[1]) << 8 | u32(b[2]) << 16 | u32(b[3]) << 24
}

read_u64 :: proc(r: ^Reader) -> u64 {
	b := r.data[r.pos:]
	r.pos += 8
	return(
		u64(b[0]) |
		u64(b[1]) << 8 |
		u64(b[2]) << 16 |
		u64(b[3]) << 24 |
		u64(b[4]) << 32 |
		u64(b[5]) << 40 |
		u64(b[6]) << 48 |
		u64(b[7]) << 56 \
	)
}

read_gguf_string :: proc(r: ^Reader) -> string {
	l := int(read_u64(r))
	s := string(r.data[r.pos:r.pos + l])
	r.pos += l
	return s
}

skip_gguf_value :: proc(r: ^Reader, t: GGUF_Type) {
	switch t {
	case .UINT8, .INT8, .BOOL:
		r.pos += 1
	case .UINT16, .INT16:
		r.pos += 2
	case .UINT32, .INT32, .FLOAT32:
		r.pos += 4
	case .UINT64, .INT64, .FLOAT64:
		r.pos += 8
	case .STRING:
		l := int(read_u64(r))
		r.pos += l
	case .ARRAY:
		et := GGUF_Type(read_u32(r))
		c := read_u64(r)
		for _ in 0 ..< c {
			skip_gguf_value(r, et)
		}
	}
}

read_gguf_value :: proc(r: ^Reader, t: GGUF_Type) -> GGUF_Meta {
	m: GGUF_Meta
	m.type = t
	switch t {
	case .UINT8, .INT8, .BOOL:
		m.u = u64(read_u8(r))
	case .UINT16, .INT16:
		m.u = u64(read_u16(r))
	case .UINT32, .INT32:
		m.u = u64(read_u32(r))
	case .UINT64, .INT64:
		m.u = read_u64(r)
	case .FLOAT32:
		m.f = f64(transmute(f32)read_u32(r))
	case .FLOAT64:
		m.f = transmute(f64)read_u64(r)
	case .STRING:
		m.str = read_gguf_string(r)
	case .ARRAY:
		m.arr_type = GGUF_Type(read_u32(r))
		m.arr_len = read_u64(r)
		for _ in 0 ..< m.arr_len {
			skip_gguf_value(r, m.arr_type)
		}
	}
	return m
}

align_up :: proc(x, a: int) -> int {
	return (x + a - 1) & ~(a - 1)
}

parse_gguf :: proc(path: string, g: ^GGUF_File) {
	mmap, err := virtual.map_file_from_path(path, {.Read})
	if err != .None {
		fmt.eprintf("mmap failed for %s: %v\n", path, err)
		os.exit(1)
	}
	g.mmap = mmap

	r := Reader{mmap, 0}

	magic := read_u32(&r)
	if magic != GGUF_MAGIC {
		fmt.eprintf("not a GGUF file (magic=0x%x)\n", magic)
		os.exit(1)
	}
	version := read_u32(&r)
	if version != 2 && version != 3 {
		fmt.eprintf("unsupported GGUF version %d\n", version)
		os.exit(1)
	}

	tensor_count := int(read_u64(&r))
	metadata_count := int(read_u64(&r))

	g.metadata = make(map[string]GGUF_Meta)
	for _ in 0 ..< metadata_count {
		key := read_gguf_string(&r)
		vtype := GGUF_Type(read_u32(&r))
		g.metadata[key] = read_gguf_value(&r, vtype)
	}

	g.tensors = make([]GGUF_Tensor, tensor_count)
	g.tensor_map = make(map[string]int)
	for i in 0 ..< tensor_count {
		t: GGUF_Tensor
		t.name = read_gguf_string(&r)
		n_dims := int(read_u32(&r))
		t.dims = make([]u64, n_dims)
		for d in 0 ..< n_dims {
			t.dims[d] = read_u64(&r)
		}
		t.kind = GGML_Type(read_u32(&r))
		t.offset = read_u64(&r)
		g.tensors[i] = t
		g.tensor_map[t.name] = i
	}

	alignment := 32
	if mv, ok := g.metadata["general.alignment"]; ok {
		alignment = int(mv.u)
	}
	data_base := align_up(r.pos, alignment)

	for &t in g.tensors {
		ne := 1
		for d in t.dims {
			ne *= int(d)
		}
		size := row_byte_size(t.kind, ne)
		start := data_base + int(t.offset)
		t.data = mmap[start:start + size]
	}
}

free_gguf :: proc(g: ^GGUF_File) {
	for &t in g.tensors {
		delete(t.dims)
	}
	delete(g.tensors)
	delete(g.tensor_map)
	delete(g.metadata)
	if g.mmap != nil {
		virtual.release(raw_data(g.mmap), uint(len(g.mmap)))
		g.mmap = nil
	}
}

gguf_meta_u64 :: proc(g: ^GGUF_File, key: string) -> (u64, bool) {
	if m, ok := g.metadata[key]; ok {
		return m.u, true
	}
	return 0, false
}

gguf_meta_f32 :: proc(g: ^GGUF_File, key: string) -> (f32, bool) {
	if m, ok := g.metadata[key]; ok {
		return f32(m.f), true
	}
	return 0, false
}

gguf_meta_str :: proc(g: ^GGUF_File, key: string) -> (string, bool) {
	if m, ok := g.metadata[key]; ok {
		return m.str, true
	}
	return "", false
}

gguf_get_tensor :: proc(g: ^GGUF_File, name: string) -> (^GGUF_Tensor, bool) {
	if idx, ok := g.tensor_map[name]; ok {
		return &g.tensors[idx], true
	}
	return nil, false
}
