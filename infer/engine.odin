/* Public inference engine API */

package infer

import "core:fmt"

Engine_Opts :: struct {
	max_ctx:     int,
	use_metal:   bool,
	num_threads: int,
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

engine_metal_ready :: proc(e: ^Engine) -> bool {
	return e.metal_ready
}
