/* Public inference engine API for Qwen3.5 (CPU + Metal on Apple Silicon). */

package qwen3_5

import "core:fmt"
import "core:os"
import ggml "ggml:ggml"

Engine_Opts :: struct {
	max_ctx:     int,
	num_threads: int,
	use_metal:   bool,
}

Engine :: struct {
	transformer: Transformer,
	metal_ready: bool,
	batch_mm_ok: bool, // dense batched-MMQ: kinds with MMQ kernels only (Q4_K/Q6_K/IQ*/Q2_K/Q3_K/Q8_0)
}

engine_load :: proc(path: string, opts: Engine_Opts) -> (Engine, bool) {
	e: Engine
	matmul_set_threads(max(opts.num_threads, 1))
	// Skip CPU Run_State allocation when Metal is requested — forward_gpu uses
	// Metal buffers exclusively, so the CPU KV/conv/recurrent state would be
	// dead memory (up to GBs at large -c).
	use_metal := opts.use_metal
	build_transformer(&e.transformer, path, opts.max_ctx, skip_cpu_state = use_metal)
	e.batch_mm_ok = model_batch_mm_ok(&e.transformer)
	if e.transformer.config.is_moe && os.get_env("MOE_CPU", context.temp_allocator) == "1" {
		fmt.eprintln("qwen35moe: MOE_CPU=1 forced CPU path")
		use_metal = false
		malloc_run_state(&e.transformer.state, e.transformer.config)
	}

	if use_metal {
		// Metal was explicitly requested. A failed init is a hard error by
		// default: v0.2.0 shipped a shader library that never compiled and every
		// model silently ran on CPU for a whole release. METAL_FALLBACK=1 opts
		// back into the old "warn and use CPU" behaviour.
		allow_fallback := os.get_env("METAL_FALLBACK", context.temp_allocator) == "1"
		when GPU_METAL {
			e.metal_ready = metal_init(&e.transformer, path)
		} else when GPU_CUDA {
			e.metal_ready = cuda_init(&e.transformer, path)
		} else when GPU_VULKAN {
			e.metal_ready = vulkan_init(&e.transformer, path)
		} else {
			fmt.eprintln("metal: only supported on macOS")
			e.metal_ready = false
		}
		if !e.metal_ready {
			when GPU_CUDA {
				if !allow_fallback {
					fmt.eprintln("cuda: init failed and GPU was requested; refusing to fall back to CPU (set METAL_FALLBACK=1 to allow)")
					return e, false
				}
				fmt.eprintln("cuda: init failed, METAL_FALLBACK=1 -> using CPU")
			} else when GPU_VULKAN {
				if !allow_fallback {
					fmt.eprintln("vulkan: init failed and GPU was requested; refusing to fall back to CPU (set METAL_FALLBACK=1 to allow)")
					return e, false
				}
				fmt.eprintln("vulkan: init failed, METAL_FALLBACK=1 -> using CPU")
			} else {
				if !allow_fallback {
					fmt.eprintln("metal: init failed and --metal was requested; refusing to fall back to CPU (set METAL_FALLBACK=1 to allow)")
					return e, false
				}
				fmt.eprintln("metal: init failed, METAL_FALLBACK=1 -> using CPU")
			}
			// GPU failed → allocate CPU Run_State for the CPU forward path
			malloc_run_state(&e.transformer.state, e.transformer.config)
		}
	}
	return e, true
}

engine_destroy :: proc(e: ^Engine) {
	when GPU_METAL {
		if e.metal_ready {
			metal_destroy()
		}
	} else when GPU_CUDA {
		if e.metal_ready {
			cuda_destroy()
		}
	} else when GPU_VULKAN {
		if e.metal_ready {
			vulkan_destroy()
		}
	}
	free_transformer(&e.transformer)
	destroy_matmul_pool()
}

engine_forward :: proc(e: ^Engine, token, pos: int) -> []f32 {
	when GPU_METAL {
		if e.metal_ready {
			return forward_gpu(&e.transformer, token, pos)
		}
	} else when GPU_CUDA {
		if e.metal_ready {
			return cuda_forward(&e.transformer, token, pos)
		}
	} else when GPU_VULKAN {
		if e.metal_ready {
			return vulkan_forward(&e.transformer, token, pos)
		}
	}
	return forward(&e.transformer, token, pos)
}

// Batched prefill (Metal only): process `tokens` from position `pos`, batching
// the MLP projections. Chunks of at least 64 tokens are multiples of 64 (wide-N
// MMQ); shorter tails are multiples of 8. A trailing 1-7 tokens run through the
// per-token path. Returns the logits for the LAST token. CPU path falls back to
// per-token forward.
// The dense batched-prefill MMQ kernels (enc_mm / enc_proj_fwd) handle
// Q4_K/Q6_K, the IQ family (IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/
// IQ4_XS), Q2_K/Q3_K and Q8_0. Models using any other kind (e.g. Q4_0/Q5_0/
// Q8_K) must prefill per-token on Metal or CPU.
model_batch_mm_ok :: proc(t: ^Transformer) -> bool {
	w := &t.weights
	ok_kind :: proc(k: ggml.GGML_Type) -> bool {
		#partial switch k {
		case .Q4_K, .Q6_K, .Q8_0, .Q2_K, .Q3_K, .IQ1_S, .IQ1_M, .IQ2_XXS, .IQ2_XS, .IQ2_S, .IQ3_XXS, .IQ3_S, .IQ4_XS:
			return true
		}
		return false
	}
	for l in 0 ..< t.config.n_layers {
		lw := &w.layers[l]
		if !ok_kind(lw.w1.kind) || !ok_kind(lw.w2.kind) || !ok_kind(lw.w3.kind) {
			return false
		}
		switch lw.layer_type {
		case .Full_Attention:
			if !ok_kind(lw.full.wq.kind) || !ok_kind(lw.full.wk.kind) || !ok_kind(lw.full.wv.kind) || !ok_kind(lw.full.wo.kind) {
				return false
			}
		case .Linear_Attention:
			if !ok_kind(lw.linear.in_qkv.kind) || !ok_kind(lw.linear.in_z.kind) || !ok_kind(lw.linear.in_b.kind) || !ok_kind(lw.linear.in_a.kind) || !ok_kind(lw.linear.out.kind) {
				return false
			}
		}
	}
	return true
}

engine_forward_batch :: proc(e: ^Engine, tokens: []int, pos_start: int) -> []f32 {
	when GPU_CUDA {
		if e.metal_ready {
			return cuda_forward_tokens(&e.transformer, tokens, pos_start)
		}
	} else when GPU_VULKAN {
		if e.metal_ready {
			return vulkan_forward_tokens(&e.transformer, tokens, pos_start)
		}
	}
	if e.transformer.config.is_moe {
		when GPU_METAL {
		if e.metal_ready {
			// Batched MoE prefill is the default on Metal (grouped GEMM + one
			// sync per layer vs two per token; the attention blocks GEMV the
			// real tokens so they stay bit-identical with the per-token path).
			// MOE_BATCH=0 falls back to per-token prefill (debug / A-B switch).
			if os.get_env("MOE_BATCH", context.temp_allocator) != "0" {
				logits: []f32 = nil
				i := 0
				n := len(tokens)
				for i < n {
					rem := n - i
					if rem >= 8 {
						chunk := min(MOE_BATCH_T, rem)
						chunk -= chunk % 8
						logits = forward_gpu_batch_moe(&e.transformer, tokens[i : i + chunk], pos_start + i, chunk)
						i += chunk
					} else if rem >= 3 {
						padded: [8]int
						for j in 0 ..< 8 {
							padded[j] = tokens[i + min(j, rem - 1)]
						}
						logits = forward_gpu_batch_moe(&e.transformer, padded[:], pos_start + i, rem)
						i += rem
					} else {
						logits = forward_gpu(&e.transformer, tokens[i], pos_start + i)
						i += 1
					}
				}
				return logits
			}
			logits: []f32 = nil
			for i in 0 ..< len(tokens) {
				logits = forward_gpu(&e.transformer, tokens[i], pos_start + i)
			}
			return logits
		}
		} // GPU_METAL
		// CPU path: per-token
		logits: []f32 = nil
		for i in 0 ..< len(tokens) {
			logits = engine_forward(e, tokens[i], pos_start + i)
		}
		return logits
	}
	when GPU_METAL {
		if e.metal_ready && e.batch_mm_ok {
			i := 0
			n := len(tokens)
			logits: []f32 = nil
			for i < n {
				rem := n - i
				// 64-aligned chunks hit the wide-N MMQ tiles (Q4 BN=32, Q6 BN=64).
				// The short tail stays a multiple of 8 so the BN=8 kernels still apply.
				if rem >= 64 {
					chunk := min(MAX_BATCH_T, rem)
					chunk -= chunk % 64
					logits = forward_gpu_batch(&e.transformer, tokens[i : i + chunk], pos_start + i, chunk)
					i += chunk
				} else if rem >= 8 {
					chunk := rem - rem % 8
					logits = forward_gpu_batch(&e.transformer, tokens[i : i + chunk], pos_start + i, chunk)
					i += chunk
				} else if rem >= 3 {
					// pad the tail to 8 with repeats of the last real token; the
					// stateful per-token loops inside skip the filler slots
					padded: [8]int
					for j in 0 ..< 8 {
						padded[j] = tokens[i + min(j, rem - 1)]
					}
					logits = forward_gpu_batch(&e.transformer, padded[:], pos_start + i, rem)
					i += rem
				} else {
					logits = forward_gpu(&e.transformer, tokens[i], pos_start + i)
					i += 1
				}
			}
			return logits
		}
	}
	// fallback: per-token (Metal when the engine has it, else CPU)
	logits: []f32 = nil
	for t in 0 ..< len(tokens) {
		when GPU_METAL {
			if e.metal_ready {
				logits = forward_gpu(&e.transformer, tokens[t], pos_start + t)
				continue
			}
		}
		logits = forward(&e.transformer, tokens[t], pos_start + t)
	}
	return logits
}

// Max tokens the GPU batch path handles per chunk. 0 on CPU.
engine_batch_max :: proc() -> int {
	when GPU_METAL {
		return MAX_BATCH_T
	} else when GPU_CUDA {
		return cuda_batch_max()
	} else when GPU_VULKAN {
		return vulkan_batch_max()
	} else {
		return 0
	}
}

engine_config :: proc(e: ^Engine) -> ^Config {
	return &e.transformer.config
}

// Reset all position-evolving state (conv_state, recurrent state, KV cache) to
// zero so a brand-new prompt can be prefilled from pos 0. Needed because the
// gated-delta recurrent state is not rewindable -- a divergent prompt requires
// a full recompute from scratch.
engine_reset_state :: proc(e: ^Engine) {
	when GPU_METAL {
		if e.metal_ready {
			metal_reset_state()
			return
		}
	} else when GPU_CUDA {
		if e.metal_ready {
			cuda_reset_state()
			return
		}
	} else when GPU_VULKAN {
		if e.metal_ready {
			vulkan_reset_state()
			return
		}
	}
	s := &e.transformer.state
	for i in 0 ..< len(s.conv_states) {
		s.conv_states[i] = 0
	}
	for i in 0 ..< len(s.recurrent_states) {
		s.recurrent_states[i] = 0
	}
	for i in 0 ..< len(s.key_cache) {
		s.key_cache[i] = 0
	}
	for i in 0 ..< len(s.value_cache) {
		s.value_cache[i] = 0
	}
}

// Speculative decoding: verify K draft tokens via batched forward.
// Returns per-position logits [K × vocab] and the number of tokens processed.
// Caller compares draft tokens against argmax of each row to accept/reject.
// Full-state snapshot for KV prefix caching (system prompt optimization).
// Saves KV cache + recurrent + conv states to memory. Restore is O(1) memcpy.
//
// The snapshot KV bytes use the SAME layout the Metal engine stores natively,
// so a snapshot saved by a CPU engine restores on a Metal engine and vice
// versa (byte-identical formats). See kv_q8.odin for the layout.
engine_save_full_state :: proc(e: ^Engine) -> (kv_k: []u8, kv_v: []u8, conv: []u8, rec: []u8) {
	when GPU_METAL {
		if e.metal_ready { return metal_save_full_state() }
	} else when GPU_CUDA {
		if e.metal_ready { return cuda_save_full_state() }
	} else when GPU_VULKAN {
		if e.metal_ready { return vulkan_save_full_state(&e.transformer) }
	}
	return cpu_save_full_state(&e.transformer)
}

engine_restore_full_state :: proc(e: ^Engine, kv_k, kv_v, conv, rec: []u8) {
	when GPU_METAL {
		if e.metal_ready { metal_restore_full_state(kv_k, kv_v, conv, rec); return }
	} else when GPU_CUDA {
		if e.metal_ready { cuda_restore_full_state(kv_k, kv_v, conv, rec); return }
	} else when GPU_VULKAN {
		if e.metal_ready { vulkan_restore_full_state(&e.transformer, kv_k, kv_v, conv, rec); return }
	}
	cpu_restore_full_state(&e.transformer, kv_k, kv_v, conv, rec)
}

// CPU full-state snapshot: the Run_State KV stays fp32 (the CPU attention dots
// read it directly), and the conversion to/from the Metal-native byte layout
// happens here at the snapshot boundary only. With KV_Q8=1 (default) that is
// Q8_0 position-major [n_full][seq_len][kv_dim/32 blocks][34 B] for K. V is
// the same Q8 layout when KV_V=q8, otherwise the 40-byte mixed block. With
// KV_Q8=0 both are the legacy f16 head-major layout. The quantizers reproduce
// the MSL store kernels bit-for-bit, and f16() reproduces the MSL (half) cast,
// so CPU- and Metal-saved snapshots of the same fp32 values are byte-identical.
cpu_save_full_state :: proc(t: ^Transformer) -> (kv_k: []u8, kv_v: []u8, conv: []u8, rec: []u8) {
	c := &t.config
	s := &t.state
	if len(s.key_cache) == 0 { return } // no full-attention layers (or Metal engine)
	kv_dim := c.n_kv_heads * c.head_dim
	if g_kv_q8 {
		kv_k = make([]u8, kv_total_bytes_q8(c.n_full, c.seq_len, kv_dim))
		pos_bytes := kv_pos_bytes(kv_dim)
		layer_bytes := kv_layer_bytes_q8(c.seq_len, kv_dim)
		v_pos := pos_bytes
		v_layer := layer_bytes
		v_block := Q8_0_BLOCK_BYTES
		if g_kv_v_mix {
			v_pos = kv_v_mix_pos_bytes(kv_dim)
			v_layer = kv_v_mix_layer_bytes(c.seq_len, kv_dim)
			v_block = Q8_V_MIX_BLOCK_BYTES
		}
		kv_v = make([]u8, c.n_full * v_layer)
		for l in 0 ..< c.n_full {
			layer_off := l * c.seq_len * kv_dim
			for p in 0 ..< c.seq_len {
				row_off := layer_off + p * kv_dim
				dst_off := l * layer_bytes + p * pos_bytes
				v_off := l * v_layer + p * v_pos
				for b in 0 ..< kv_dim / QK8_0 {
					quantize_q8_block(s.key_cache[row_off + b * QK8_0:row_off + (b + 1) * QK8_0], kv_k[dst_off + b * Q8_0_BLOCK_BYTES:dst_off + (b + 1) * Q8_0_BLOCK_BYTES])
					if g_kv_v_mix {
						quantize_q8_v_mix_block(s.value_cache[row_off + b * QK8_0:row_off + (b + 1) * QK8_0], kv_v[v_off + b * v_block:v_off + (b + 1) * v_block])
					} else {
						quantize_q8_block(s.value_cache[row_off + b * QK8_0:row_off + (b + 1) * QK8_0], kv_v[v_off + b * v_block:v_off + (b + 1) * v_block])
					}
				}
			}
		}
	} else {
		// legacy f16 head-major, matching the Metal KV_Q8=0 buffer layout
		kv_k = make([]u8, c.n_full * c.seq_len * kv_dim * 2)
		kv_v = make([]u8, c.n_full * c.seq_len * kv_dim * 2)
		layer_bytes := c.seq_len * kv_dim * 2
		for l in 0 ..< c.n_full {
			layer_off := l * c.seq_len * kv_dim
			for h_idx in 0 ..< c.n_kv_heads {
				for p in 0 ..< c.seq_len {
					for d in 0 ..< c.head_dim {
						m_off := l * layer_bytes +
							(h_idx * c.seq_len * c.head_dim + p * c.head_dim + d) * 2
						cpu_off := layer_off + p * kv_dim + h_idx * c.head_dim + d
						kraw := transmute(u16)(f16(s.key_cache[cpu_off]))
						vraw := transmute(u16)(f16(s.value_cache[cpu_off]))
						kv_k[m_off] = u8(kraw & 0xff); kv_k[m_off + 1] = u8((kraw >> 8) & 0xff)
						kv_v[m_off] = u8(vraw & 0xff); kv_v[m_off + 1] = u8((vraw >> 8) & 0xff)
					}
				}
			}
		}
	}
	conv_src := (cast([^]u8)raw_data(s.conv_states))[:len(s.conv_states) * 4]
	rec_src := (cast([^]u8)raw_data(s.recurrent_states))[:len(s.recurrent_states) * 4]
	conv = make([]u8, len(conv_src)); copy(conv, conv_src)
	rec = make([]u8, len(rec_src)); copy(rec, rec_src)
	return
}

// Reverse of cpu_save_full_state. Overwrites the whole Run_State KV
// (n_full * seq_len positions), so no separate zeroing pass is needed.
cpu_restore_full_state :: proc(t: ^Transformer, kv_k, kv_v, conv, rec: []u8) {
	c := &t.config
	s := &t.state
	if len(s.key_cache) == 0 { return }
	kv_dim := c.n_kv_heads * c.head_dim
	if g_kv_q8 {
		pos_bytes := kv_pos_bytes(kv_dim)
		layer_bytes := kv_layer_bytes_q8(c.seq_len, kv_dim)
		v_pos := pos_bytes
		v_layer := layer_bytes
		v_block := Q8_0_BLOCK_BYTES
		if g_kv_v_mix {
			v_pos = kv_v_mix_pos_bytes(kv_dim)
			v_layer = kv_v_mix_layer_bytes(c.seq_len, kv_dim)
			v_block = Q8_V_MIX_BLOCK_BYTES
		}
		for l in 0 ..< c.n_full {
			layer_off := l * c.seq_len * kv_dim
			for p in 0 ..< c.seq_len {
				row_off := layer_off + p * kv_dim
				src_off := l * layer_bytes + p * pos_bytes
				v_off := l * v_layer + p * v_pos
				for b in 0 ..< kv_dim / QK8_0 {
					dequantize_q8_block(kv_k[src_off + b * Q8_0_BLOCK_BYTES:src_off + (b + 1) * Q8_0_BLOCK_BYTES], s.key_cache[row_off + b * QK8_0:row_off + (b + 1) * QK8_0])
					if g_kv_v_mix {
						dequantize_q8_v_mix_block(kv_v[v_off + b * v_block:v_off + (b + 1) * v_block], s.value_cache[row_off + b * QK8_0:row_off + (b + 1) * QK8_0])
					} else {
						dequantize_q8_block(kv_v[v_off + b * v_block:v_off + (b + 1) * v_block], s.value_cache[row_off + b * QK8_0:row_off + (b + 1) * QK8_0])
					}
				}
			}
		}
	} else {
		layer_bytes := c.seq_len * kv_dim * 2
		for l in 0 ..< c.n_full {
			layer_off := l * c.seq_len * kv_dim
			for h_idx in 0 ..< c.n_kv_heads {
				for p in 0 ..< c.seq_len {
					for d in 0 ..< c.head_dim {
						m_off := l * layer_bytes +
							(h_idx * c.seq_len * c.head_dim + p * c.head_dim + d) * 2
						cpu_off := layer_off + p * kv_dim + h_idx * c.head_dim + d
						s.key_cache[cpu_off] = f32(transmute(f16)(u16(kv_k[m_off]) | (u16(kv_k[m_off + 1]) << 8)))
						s.value_cache[cpu_off] = f32(transmute(f16)(u16(kv_v[m_off]) | (u16(kv_v[m_off + 1]) << 8)))
					}
				}
			}
		}
	}
	copy((cast([^]u8)raw_data(s.conv_states))[:len(s.conv_states) * 4], conv)
	copy((cast([^]u8)raw_data(s.recurrent_states))[:len(s.recurrent_states) * 4], rec)
}

// Speculative decoding support
engine_save_spec_state :: proc(e: ^Engine) -> (conv: []u8, rec: []u8) {
	when GPU_METAL {
		if e.metal_ready { return metal_save_spec_state() }
	} else when GPU_CUDA {
		if e.metal_ready { return cuda_save_spec_state() }
	} else when GPU_VULKAN {
		if e.metal_ready { return vulkan_save_spec_state() }
	}
	return
}

engine_restore_spec_state :: proc(e: ^Engine, conv: []u8, rec: []u8) {
	when GPU_METAL {
		if e.metal_ready { metal_restore_spec_state(conv, rec); return }
	} else when GPU_CUDA {
		if e.metal_ready { cuda_restore_spec_state(conv, rec); return }
	} else when GPU_VULKAN {
		if e.metal_ready { vulkan_restore_spec_state(conv, rec); return }
	}
}

engine_verify_batch :: proc(e: ^Engine, tokens: []int, pos_start: int) -> []f32 {
	when GPU_CUDA {
		if e.metal_ready {
			fmt.eprintln("cuda: engine_verify_batch is not implemented")
			return {}
		}
	}
	when GPU_VULKAN {
		if e.metal_ready {
			fmt.eprintln("vulkan: engine_verify_batch is not implemented")
			return {}
		}
	}
	when GPU_METAL {
		if e.metal_ready {
			_ = forward_gpu_batch(&e.transformer, tokens, pos_start, len(tokens))
			return metal_verify_batch_logits(&e.transformer, len(tokens))
		}
	}
	return {}
}
