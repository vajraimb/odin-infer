/* qwen3_5/kv_q8.odin — Q8_0-quantized KV cache (llama.cpp symmetric Q8_0).

   Per 32-element block: d (f16, 2 bytes) = max(|v|)/127, then 32 int8
   q = round(v * (d ? 1/d : 0)). Dequant: v = d*q. Block bytes = 34, so a
   position costs (kv_dim/32)*34 bytes instead of kv_dim*4 (fp32) — 3.76x
   smaller — or kv_dim*2 (f16) — 1.88x smaller.

   Canonical in-memory layout (identical on Metal buffers and in CPU-created
   snapshots, so CPU- and Metal-saved state bytes are interchangeable):
     per full-attention layer, per buffer (K and V stored separately):
       [seq_len positions][kv_dim/32 blocks][f16 d + 32 int8]
   Position-major, matching Run_State.key_cache [n_full, seq_len, kv_dim] order
   (each 32-block is contiguous and aligned inside one head because
   head_dim % 32 == 0).

   KV_Q8=0 disables the quantized path at runtime (legacy f16 Metal KV kept for
   A/B). KV_V=q8 keeps V in the same symmetric Q8_0 blocks as K. The default
   (KV_V unset) stores V in a mixed block: the single largest-magnitude value
   is kept as f16 and the other 31 are Q8 against the residual amax. One
   outlier otherwise sets d for the whole block (V amax ~85 -> per-element
   error ~0.3). Mixed block = 40 bytes (f16 d, u8 index, pad, f16 outlier,
   32 int8, 2 pad). K stays 34-byte Q8_0.
   The CPU Run_State stays f32 in every mode: the CPU attention code
   (forward.odin) keeps its fp32 dot loops untouched, and the conversion to/from
   the quantized byte layout happens only at the snapshot boundary
   (engine_save_full_state / engine_restore_full_state in engine.odin). That is
   the chosen tradeoff: converting at the boundary is a few dozen lines, while
   dequantizing per dot inside the CPU attention would touch every hot loop in
   forward.odin for no CPU speedup. The boundary conversion reproduces the MSL
   store kernel bit-for-bit (same max|v| scan order, same round(), same f16
   cast — store_kv_q8 for symmetric blocks, store_kv_q8v for mixed V), so
   CPU- and Metal-saved snapshots are byte-identical.
*/

package qwen3_5

import "core:math"
import "core:os"

QK8_0 :: 32
Q8_0_BLOCK_BYTES :: QK8_0 + 2 // f16 d + 32 int8 qs
// Mixed V block: f16 d, u8 outlier index, u8 pad, f16 outlier, 32 int8, 2 pad.
Q8_V_MIX_BLOCK_BYTES :: 40

// Runtime A/B switches. KV_Q8=0 selects the legacy f16 Metal KV path.
// KV_V=q8 keeps symmetric Q8 V; anything else (including unset) uses the
// mixed V block while K stays Q8_0. Read once in build_transformer.
g_kv_q8: bool
g_kv_v_mix: bool

kv_q8_init :: proc() {
	g_kv_q8 = os.get_env("KV_Q8", context.temp_allocator) != "0"
	g_kv_v_mix = g_kv_q8 && os.get_env("KV_V", context.temp_allocator) != "q8"
}

// Bytes of one position's K (or V): (kv_dim/32) blocks x 34 bytes.
kv_pos_bytes :: proc(kv_dim: int) -> int {
	assert(kv_dim % QK8_0 == 0)
	return (kv_dim / QK8_0) * Q8_0_BLOCK_BYTES
}

// Bytes of one full-attention layer's K buffer (V has the same size).
kv_layer_bytes_q8 :: proc(seq_len, kv_dim: int) -> int {
	return seq_len * kv_pos_bytes(kv_dim)
}

// Total bytes of one KV buffer (K or V) across all full-attention layers.
kv_total_bytes_q8 :: proc(n_full, seq_len, kv_dim: int) -> int {
	return n_full * kv_layer_bytes_q8(seq_len, kv_dim)
}

// Bytes of one position's mixed V row.
kv_v_mix_pos_bytes :: proc(kv_dim: int) -> int {
	assert(kv_dim % QK8_0 == 0)
	return (kv_dim / QK8_0) * Q8_V_MIX_BLOCK_BYTES
}

kv_v_mix_layer_bytes :: proc(seq_len, kv_dim: int) -> int {
	return seq_len * kv_v_mix_pos_bytes(kv_dim)
}

kv_v_mix_total_bytes :: proc(n_full, seq_len, kv_dim: int) -> int {
	return n_full * kv_v_mix_layer_bytes(seq_len, kv_dim)
}

// Resident K+V bytes and the mem: tag for the format actually in use.
kv_resident_bytes :: proc(n_full, seq_len, kv_dim: int) -> int {
	if !g_kv_q8 {
		return 2 * n_full * seq_len * kv_dim * 2
	}
	k := kv_total_bytes_q8(n_full, seq_len, kv_dim)
	if g_kv_v_mix {
		return k + kv_v_mix_total_bytes(n_full, seq_len, kv_dim)
	}
	return k * 2
}

kv_resident_tag :: proc() -> string {
	if !g_kv_q8 { return "f16" }
	if g_kv_v_mix { return "Q8K+Vmix" }
	return "Q8_0"
}

write_f16_le :: proc(dst: []u8, v: f32) {
	raw := transmute(u16)(f16(v))
	dst[0] = u8(raw & 0xff)
	dst[1] = u8((raw >> 8) & 0xff)
}

read_f16_le :: proc(src: []u8) -> f32 {
	raw := u16(src[0]) | (u16(src[1]) << 8)
	return f32(transmute(f16)(raw))
}

// Quantize one V block into the mixed layout. The first maximum-|v| element
// (lowest index on a tie) is stored as f16; the other 31 are symmetric Q8
// against their own amax. Matches store_kv_q8v / store_kv_q8v_c.
quantize_q8_v_mix_block :: proc(src: []f32, dst: []u8) {
	assert(len(src) == QK8_0)
	assert(len(dst) == Q8_V_MIX_BLOCK_BYTES)
	for i in 0 ..< Q8_V_MIX_BLOCK_BYTES { dst[i] = 0 }
	jmax := 0
	amax := math.abs(src[0])
	for j in 1 ..< QK8_0 {
		a := math.abs(src[j])
		if a > amax {
			amax = a
			jmax = j
		}
	}
	amax2: f32 = 0
	for j in 0 ..< QK8_0 {
		if j == jmax { continue }
		a := math.abs(src[j])
		if a > amax2 { amax2 = a }
	}
	d := amax2 / 127.0
	id: f32 = 0
	if d != 0 { id = 1.0 / d }
	write_f16_le(dst[0:], d)
	dst[2] = u8(jmax)
	write_f16_le(dst[4:], src[jmax])
	for j in 0 ..< QK8_0 {
		if j == jmax { continue }
		dst[6 + j] = u8(i8(math.round(src[j] * id)))
	}
}

dequantize_q8_v_mix_block :: proc(src: []u8, dst: []f32) {
	assert(len(src) == Q8_V_MIX_BLOCK_BYTES)
	assert(len(dst) == QK8_0)
	d := read_f16_le(src[0:])
	idx := int(src[2])
	ov := read_f16_le(src[4:])
	for j in 0 ..< QK8_0 {
		if j == idx {
			dst[j] = ov
		} else {
			dst[j] = d * f32(i8(src[6 + j]))
		}
	}
}

// Quantize one 32-element block into dst[0..34]: f16 d then 32 int8 q.
// Matches llama.cpp quantize_row_q8_0 (and the MSL store_kv_q8 kernel)
// exactly: q = round(v * id) with id = 1/d when d != 0, else 0.
quantize_q8_block :: proc(src: []f32, dst: []u8) {
	assert(len(src) == QK8_0)
	assert(len(dst) == Q8_0_BLOCK_BYTES)
	amax: f32 = 0
	for j in 0 ..< QK8_0 {
		a := math.abs(src[j])
		if a > amax { amax = a }
	}
	d := amax / 127.0
	id: f32 = 0
	if d != 0 { id = 1.0 / d }
	f16_raw := transmute(u16)(f16(d))
	dst[0] = u8(f16_raw & 0xff)
	dst[1] = u8((f16_raw >> 8) & 0xff)
	for j in 0 ..< QK8_0 {
		dst[2 + j] = u8(i8(math.round(src[j] * id)))
	}
}

// Dequantize one block: dst[j] = d * qs[j].
dequantize_q8_block :: proc(src: []u8, dst: []f32) {
	assert(len(src) == Q8_0_BLOCK_BYTES)
	assert(len(dst) == QK8_0)
	f16_raw := u16(src[0]) | (u16(src[1]) << 8)
	d := f32(transmute(f16)(f16_raw))
	for j in 0 ..< QK8_0 {
		dst[j] = d * f32(i8(src[2 + j]))
	}
}

// Quantize a full kv_dim-wide row (one position's K or V) into kv_pos_bytes.
quantize_kv_row_q8 :: proc(src: []f32, dst: []u8) {
	assert(len(src) % QK8_0 == 0)
	assert(len(dst) == kv_pos_bytes(len(src)))
	for b in 0 ..< len(src) / QK8_0 {
		quantize_q8_block(src[b * QK8_0:(b + 1) * QK8_0], dst[b * Q8_0_BLOCK_BYTES:(b + 1) * Q8_0_BLOCK_BYTES])
	}
}

// Dequantize a full kv_dim-wide row back into f32.
dequantize_kv_row_q8 :: proc(src: []u8, dst: []f32) {
	assert(len(dst) % QK8_0 == 0)
	assert(len(src) == kv_pos_bytes(len(dst)))
	for b in 0 ..< len(dst) / QK8_0 {
		dequantize_q8_block(src[b * Q8_0_BLOCK_BYTES:(b + 1) * Q8_0_BLOCK_BYTES], dst[b * QK8_0:(b + 1) * QK8_0])
	}
}
