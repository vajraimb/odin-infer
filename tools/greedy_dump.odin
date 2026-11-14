/* Greedy token dump for a CPU or Vulkan build of the same commit.
   Prints one line of token ids: the prompt, then n_decode argmax tokens.
   ODINFER_LOGITS_OUT=path writes the prefill logits as raw little-endian f32.

   odin build tools/greedy_dump.odin -file -out:greedy_cpu ...collections
   odin build tools/greedy_dump.odin -file -define:ODINFER_BACKEND=vulkan -out:greedy_vk ...
   greedy_cpu model.gguf max_ctx n_decode tok tok ...
*/

package main

import q35 "qwen3_5:qwen3_5"

import "core:fmt"
import "core:os"
import "core:strconv"

main :: proc() {
	if len(os.args) < 5 {
		fmt.eprintln("usage: greedy_dump <model.gguf> <max_ctx> <n_decode> <token>...")
		os.exit(1)
	}
	path := os.args[1]
	max_ctx, okc := strconv.parse_int(os.args[2])
	n_decode, okd := strconv.parse_int(os.args[3])
	if !okc || !okd || max_ctx < 2 || n_decode < 0 {
		fmt.eprintln("greedy_dump: bad max_ctx or n_decode")
		os.exit(1)
	}
	prompt := make([]int, len(os.args) - 4)
	defer delete(prompt)
	for i in 0 ..< len(prompt) {
		v, ok := strconv.parse_int(os.args[4 + i])
		if !ok {
			fmt.eprintln("greedy_dump: token id is not an integer")
			os.exit(1)
		}
		prompt[i] = int(v)
	}
	use_gpu := false
	when q35.GPU_VULKAN {
		use_gpu = true
	}
	engine, load_ok := q35.engine_load(path, q35.Engine_Opts{max_ctx = int(max_ctx), use_metal = use_gpu, num_threads = 4})
	if !load_ok {
		fmt.eprintln("greedy_dump: engine_load failed")
		os.exit(1)
	}
	defer q35.engine_destroy(&engine)
	when q35.GPU_VULKAN {
		if !engine.metal_ready {
			fmt.eprintln("greedy_dump: Vulkan was requested and did not initialize")
			os.exit(1)
		}
	}
	fmt.eprintfln("greedy_dump: backend=%s", q35.gpu_backend_name(engine.metal_ready))
	logits := q35.engine_forward_batch(&engine, prompt, 0)
	if len(logits) == 0 {
		fmt.eprintln("greedy_dump: empty logits")
		os.exit(1)
	}
	if outp := os.get_env("ODINFER_LOGITS_OUT", context.temp_allocator); outp != "" {
		f, err := os.open(outp, os.O_CREATE | os.O_WRONLY | os.O_TRUNC)
		if err != os.ERROR_NONE {
			fmt.eprintfln("greedy_dump: cannot write {}", outp)
			os.exit(1)
		}
		defer os.close(f)
		bytes := (cast([^]u8)raw_data(logits))[:len(logits) * 4]
		_, _ = os.write(f, bytes)
	}
	for t in prompt {
		fmt.printf("%d ", t)
	}
	pos := len(prompt)
	for _ in 0 ..< n_decode {
		next := 0
		best := logits[0]
		for i in 1 ..< len(logits) {
			if logits[i] > best {
				best = logits[i]
				next = i
			}
		}
		fmt.printf("%d ", next)
		if pos >= max_ctx - 1 {
			break
		}
		logits = q35.engine_forward(&engine, next, pos)
		pos += 1
	}
	fmt.println()
}
