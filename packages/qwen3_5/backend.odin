/* Compile-time GPU backend selection.
   Darwin default is Metal (unchanged). Linux default is CPU.
   -define:ODINFER_BACKEND="cuda" compiles the CUDA path in and leaves Metal
   out of the hot path. The choice is a `when`, so a given binary never
   branches between Metal and CUDA on the forward path. */

package qwen3_5

import "core:testing"

ODINFER_BACKEND :: #config(ODINFER_BACKEND, "default")

when ODIN_OS == .Darwin {
	when ODINFER_BACKEND == "cuda" {
		GPU_METAL :: false
		GPU_CUDA :: true
	} else {
		GPU_METAL :: true
		GPU_CUDA :: false
	}
} else {
	when ODINFER_BACKEND == "cuda" {
		GPU_METAL :: false
		GPU_CUDA :: true
	} else {
		GPU_METAL :: false
		GPU_CUDA :: false
	}
}

when GPU_METAL && GPU_CUDA {
	#panic("ODINFER_BACKEND: Metal and CUDA are mutually exclusive")
}

// Name of the accelerator that will actually run when `ready` is set.
// "cpu" when the engine fell back or GPU was not requested.
gpu_backend_name :: proc(ready: bool) -> string {
	if !ready {
		return "cpu"
	}
	when GPU_CUDA {
		return "cuda"
	} else when GPU_METAL {
		return "metal"
	}
	return "cpu"
}

@(test)
test_gpu_backend_name :: proc(t: ^testing.T) {
	testing.expect_value(t, gpu_backend_name(false), "cpu")
	when GPU_CUDA {
		testing.expect_value(t, gpu_backend_name(true), "cuda")
	} else when GPU_METAL {
		testing.expect_value(t, gpu_backend_name(true), "metal")
	} else {
		testing.expect_value(t, gpu_backend_name(true), "cpu")
	}
}
