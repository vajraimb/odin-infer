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
		when ODIN_OS == .Darwin {
			e.metal_ready = metal_init(&e.transformer, path)
			if !e.metal_ready {
				fmt.eprintln("metal: init failed, falling back to CPU")
				// Metal failed → allocate CPU Run_State for the CPU forward path
				malloc_run_state(&e.transformer.state, e.transformer.config)
			}
		} else {
			fmt.eprintln("metal: only supported on macOS; using CPU")
			malloc_run_state(&e.transformer.state, e.transformer.config)
		}
	}
	return e, true
}

engine_destroy :: proc(e: ^Engine) {
	when ODIN_OS == .Darwin {
		if e.metal_ready {
			metal_destroy()
		}
	}
	free_transformer(&e.transformer)
	destroy_matmul_pool()
}

engine_forward :: proc(e: ^Engine, token, pos: int) -> []f32 {
	when ODIN_OS == .Darwin {
		if e.metal_ready {
			return forward_gpu(&e.transformer, token, pos)
		}
	}
	return forward(&e.transformer, token, pos)
}

// Batched prefill (Metal only): process `tokens` from position `pos`, batching
// the MLP projections. Chunks into <=MAX_BATCH_T blocks (multiples of 8); the
// trailing 1-7 tokens run through the per-token path. Returns the logits for the
// LAST token (the one used to predict the next token). CPU path falls back to
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
	if e.transformer.config.is_moe {
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
		// CPU path: per-token
		logits: []f32 = nil
		for i in 0 ..< len(tokens) {
			logits = engine_forward(e, tokens[i], pos_start + i)
		}
		return logits
	}
	when ODIN_OS == .Darwin {
		if e.metal_ready && e.batch_mm_ok {
			i := 0
			n := len(tokens)
			logits: []f32 = nil
			for i < n {
				rem := n - i
				if rem >= 8 {
					chunk := min(MAX_BATCH_T, rem)
					chunk -= chunk % 8
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
		if e.metal_ready {
			logits = forward_gpu(&e.transformer, tokens[t], pos_start + t)
		} else {
			logits = forward(&e.transformer, tokens[t], pos_start + t)
		}
	}
	return logits
}

// Max tokens the Metal batch path handles per chunk (multiples of 8). 0 on CPU.
engine_batch_max :: proc() -> int {
	when ODIN_OS == .Darwin {
		return MAX_BATCH_T
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
	when ODIN_OS == .Darwin {
		if e.metal_ready {
			metal_reset_state()
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
engine_save_full_state :: proc(e: ^Engine) -> (kv_k: []u8, kv_v: []u8, conv: []u8, rec: []u8) {
	when ODIN_OS == .Darwin {
		if e.metal_ready { return metal_save_full_state() }
	}
	return
}

engine_restore_full_state :: proc(e: ^Engine, kv_k, kv_v, conv, rec: []u8) {
	when ODIN_OS == .Darwin {
		if e.metal_ready { metal_restore_full_state(kv_k, kv_v, conv, rec); return }
	}
}

// Speculative decoding support
engine_save_spec_state :: proc(e: ^Engine) -> (conv: []u8, rec: []u8) {
	when ODIN_OS == .Darwin {
		if e.metal_ready { return metal_save_spec_state() }
	}
	return
}

engine_restore_spec_state :: proc(e: ^Engine, conv: []u8, rec: []u8) {
	when ODIN_OS == .Darwin {
		if e.metal_ready { metal_restore_spec_state(conv, rec); return }
	}
}

engine_verify_batch :: proc(e: ^Engine, tokens: []int, pos_start: int) -> []f32 {
	when ODIN_OS == .Darwin {
		if e.metal_ready {
			_ = forward_gpu_batch(&e.transformer, tokens, pos_start, len(tokens))
			return metal_verify_batch_logits(&e.transformer, len(tokens))
		}
	}
	return {}
}
