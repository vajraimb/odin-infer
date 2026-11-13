/* CUDA kernel correctness. Build with -define:ODINFER_BACKEND="cuda".
   --nvrtc-only [compute_XX] compiles the kernel source and does not need a GPU.
   --bench times tensor-core WMMA against MMQ, then the T=1 GEMV old vs new (GB/s).
   --bench-gemv times only the T=1 GEMV (9B and 27B shapes, GB/s and % of peak).
   With no flag, runs cuda_selftest against the CPU reference (needs a device).

   odin run tools/cuda_kernel_test.odin -file -define:ODINFER_BACKEND="cuda" \
     -collection:ggml=packages -collection:infer=packages \
     -collection:qwen3_5=packages -collection:qwen3_5_tokenizer=packages \
     -collection:sampler=packages -collection:tokenizer=packages \
     -collection:cuda=packages -- --nvrtc-only compute_70
*/

package main

import q35 "qwen3_5:qwen3_5"

import "core:fmt"
import "core:os"

main :: proc() {
	when !q35.GPU_CUDA {
		fmt.eprintln("cuda_kernel_test: rebuild with -define:ODINFER_BACKEND=\"cuda\"")
		os.exit(1)
	} else {
		if len(os.args) > 1 && os.args[1] == "--nvrtc-only" {
			arch := "compute_70"
			if len(os.args) > 2 {
				arch = os.args[2]
			}
			if !q35.cuda_nvrtc_check(arch) {
				os.exit(1)
			}
			return
		}
		if len(os.args) > 1 && os.args[1] == "--bench-gemv" {
			q35.cuda_gemv_bench()
			return
		}
		if len(os.args) > 1 && os.args[1] == "--bench" {
			if !q35.cuda_tc_bench() {
				os.exit(1)
			}
			q35.cuda_gemv_bench()
			return
		}
		if !q35.cuda_selftest() {
			os.exit(1)
		}
	}
}
