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
	build_transformer(&e.transformer, path, opts.max_ctx)

	if opts.use_metal {
		when ODIN_OS == .Darwin {
			e.metal_ready = metal_init(&e.transformer)
			if !e.metal_ready {
				fmt.eprintln("metal: init failed, falling back to CPU")
			}
		} else {
			fmt.eprintln("metal: only supported on macOS; using CPU")
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
