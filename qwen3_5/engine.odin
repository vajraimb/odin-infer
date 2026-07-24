/* Public inference engine API for Qwen3.5 (CPU + Metal on Apple Silicon). */

package qwen3_5

import "core:fmt"

Engine_Opts :: struct {
	max_ctx:     int,
	num_threads: int,
	use_metal:   bool,
}

Engine :: struct {
	transformer: Transformer,
	metal_ready: bool,
}

engine_load :: proc(path: string, opts: Engine_Opts) -> (Engine, bool) {
	e: Engine
	matmul_set_threads(max(opts.num_threads, 1))
	// Skip CPU Run_State allocation when Metal is requested — forward_gpu uses
	// Metal buffers exclusively, so the CPU KV/conv/recurrent state would be
	// dead memory (up to GBs at large -c).
	build_transformer(&e.transformer, path, opts.max_ctx, skip_cpu_state = opts.use_metal)

	if opts.use_metal {
		when ODIN_OS == .Darwin {
			e.metal_ready = metal_init(&e.transformer)
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
engine_forward_batch :: proc(e: ^Engine, tokens: []int, pos_start: int) -> []f32 {
	when ODIN_OS == .Darwin {
		if e.metal_ready {
			i := 0
			n := len(tokens)
			logits: []f32 = nil
			for i < n {
				rem := n - i
				if rem >= 8 {
					chunk := min(MAX_BATCH_T, rem)
					chunk -= chunk % 8
					logits = forward_gpu_batch(&e.transformer, tokens[i : i + chunk], pos_start + i)
					i += chunk
				} else {
					logits = forward_gpu(&e.transformer, tokens[i], pos_start + i)
					i += 1
				}
			}
			return logits
		}
	}
	// CPU fallback: per-token
	logits: []f32 = nil
	for t in 0 ..< len(tokens) {
		logits = forward(&e.transformer, tokens[t], pos_start + t)
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
			_ = forward_gpu_batch(&e.transformer, tokens, pos_start)
			return metal_verify_batch_logits(&e.transformer, len(tokens))
		}
	}
	return {}
}
