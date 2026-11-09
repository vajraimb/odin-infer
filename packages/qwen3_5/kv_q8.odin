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
   A/B). The CPU Run_State stays f32 in BOTH modes: the CPU attention code
   (forward.odin) keeps its fp32 dot loops untouched, and the conversion to/from
   the quantized byte layout happens only at the snapshot boundary
   (engine_save_full_state / engine_restore_full_state in engine.odin). That is
   the chosen tradeoff: converting at the boundary is a few dozen lines, while
   dequantizing per dot inside the CPU attention would touch every hot loop in
   forward.odin for no CPU speedup. The boundary conversion reproduces the MSL
   store_kv_q8 kernel bit-for-bit (same max|v| scan order, same round(), same
   f16 cast), so CPU- and Metal-saved snapshots are byte-identical.
*/

package qwen3_5

import "core:math"
import "core:os"

QK8_0 :: 32
Q8_0_BLOCK_BYTES :: QK8_0 + 2 // f16 d + 32 int8 qs

// Runtime A/B switch. KV_Q8=0 selects the legacy f16 Metal KV path.
// Read once in build_transformer (runs for every engine, CPU or Metal).
g_kv_q8: bool

kv_q8_init :: proc() {
	g_kv_q8 = os.get_env("KV_Q8", context.temp_allocator) != "0"
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
