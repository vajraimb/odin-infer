/* Vulkan kernel self-test. No model weights.
   odin run tools/vulkan_kernel_test.odin -file -define:ODINFER_BACKEND=vulkan \
     -collection:ggml=packages -collection:infer=packages \
     -collection:qwen3_5=packages -collection:qwen3_5_tokenizer=packages \
     -collection:sampler=packages -collection:tokenizer=packages \
     -collection:cuda=packages -collection:vulkan=packages
   Point VK_ICD_FILENAMES at lavapipe when there is no GPU.
*/

package main

import q35 "qwen3_5:qwen3_5"

import "core:fmt"
import "core:os"

main :: proc() {
	when !q35.GPU_VULKAN {
		fmt.eprintln("vulkan_kernel_test: rebuild with -define:ODINFER_BACKEND=vulkan")
		os.exit(1)
	} else {
		if !q35.vulkan_selftest() {
			os.exit(1)
		}
	}
}
