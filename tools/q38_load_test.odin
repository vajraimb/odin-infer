/* Acceptance test for the IQ/K quant dequant port: loads the 27B dynamic
   mixed-quant Qwen3.5 GGUF, prints its quant histogram, then runs ONE forward
   on token 1 and reports logits finiteness + argmax + wall time.

   --compare mode: load the model TWICE (CPU engine + Metal engine), run
   engine_forward for 2 tokens on each (same tokens, pos 0 and 1), compare the
   logits elementwise (max_abs diff + count of |diff| > 1e-2) and report the
   per-token GPU wall time.

   Run:
     odin run tools/q38_load_test.odin -file \
       -collection:ggml=packages -collection:infer=packages \
       -collection:qwen3_5=packages -collection:qwen3_5_tokenizer=packages \
       -collection:sampler=packages -collection:tokenizer=packages \
       -- /Users/vajra/Downloads/Qwen3.8-27B-UD-IQ2_XXS.gguf [max_ctx] [--compare]
*/

package main

import ggml "ggml:ggml"
import q35 "qwen3_5:qwen3_5"

import "core:fmt"
import "core:math"
import "core:os"
import "core:strconv"
import "core:time"

kind_name :: proc(k: ggml.GGML_Type) -> string {
	#partial switch k {
	case .F32:      return "F32"
	case .F16:      return "F16"
	case .Q8_0:     return "Q8_0"
	case .Q4_0:     return "Q4_0"
	case .Q4_1:     return "Q4_1"
	case .Q5_0:     return "Q5_0"
	case .Q5_1:     return "Q5_1"
	case .Q2_K:     return "Q2_K"
	case .Q3_K:     return "Q3_K"
	case .Q4_K:     return "Q4_K"
	case .Q5_K:     return "Q5_K"
	case .Q6_K:     return "Q6_K"
	case .Q8_K:     return "Q8_K"
	case .IQ2_XXS:  return "IQ2_XXS"
	case .IQ2_XS:   return "IQ2_XS"
	case .IQ3_XXS:  return "IQ3_XXS"
	case .IQ1_S:    return "IQ1_S"
	case .IQ4_NL:   return "IQ4_NL"
	case .IQ3_S:    return "IQ3_S"
	case .IQ2_S:    return "IQ2_S"
	case .IQ4_XS:   return "IQ4_XS"
	case .IQ1_M:    return "IQ1_M"
	}
	return "?"
}

compare_logits :: proc(cpu, gpu: []f32, label: string) {
	max_abs := f32(0)
	over_count := 0
	n := min(len(cpu), len(gpu))
	for i in 0 ..< n {
		d := math.abs(cpu[i] - gpu[i])
		if d > max_abs {
			max_abs = d
		}
		if d > 1e-2 {
			over_count += 1
		}
	}
	fmt.printf(
		"%s: n=%d max_abs_diff=%.6e count(|diff|>1e-2)=%d (cpu_argmax=%d gpu_argmax=%d)\n",
		label,
		n,
		max_abs,
		over_count,
		argmax_of(cpu),
		argmax_of(gpu),
	)
}

argmax_of :: proc(v: []f32) -> int {
	best := f32(-3.4028235e38)
	idx := -1
	for i in 0 ..< len(v) {
		if v[i] > best {
			best = v[i]
			idx = i
		}
	}
	return idx
}

print_histogram :: proc(path: string) {
	g: ggml.GGUF_File
	ggml.parse_gguf(path, &g)
	hist := make(map[ggml.GGML_Type]int)
	for &t in g.tensors {
		hist[t.kind] += 1
	}
	order := []ggml.GGML_Type{.F32, .F16, .Q8_0, .Q4_0, .Q4_1, .Q5_0, .Q5_1, .Q2_K, .Q3_K, .Q4_K, .Q5_K, .Q6_K, .Q8_K, .IQ2_XXS, .IQ2_XS, .IQ3_XXS, .IQ1_S, .IQ4_NL, .IQ3_S, .IQ2_S, .IQ4_XS, .IQ1_M}
	fmt.eprintln("quant histogram (tensor counts):")
	total := 0
	for k in order {
		if c, ok := hist[k]; ok {
			fmt.eprintfln("  %-8s x %d", kind_name(k), c)
			total += c
		}
	}
	fmt.eprintfln("  total tensors: %d", total)
	ggml.free_gguf(&g)
}

main :: proc() {
	if len(os.args) < 2 {
		fmt.eprintln("usage: q38_load_test <model.gguf> [max_ctx] [--compare]")
		os.exit(2)
	}
	path := os.args[1]
	max_ctx := 256
	compare := false
	for a in os.args[2:] {
		if a == "--compare" {
			compare = true
		} else if v, ok := strconv.parse_int(a); ok {
			max_ctx = v
		}
	}

	t_start := time.now()

	// ---- pass 1: quant histogram straight from the GGUF file ----
	print_histogram(path)

	if !compare {
		// ---- engine load, CPU only ----
		engine, ok := q35.engine_load(
			path,
			q35.Engine_Opts {
				max_ctx = max_ctx,
				num_threads = os.get_processor_core_count(),
				use_metal = false,
			},
		)
		if !ok {
			fmt.eprintln("engine_load failed")
			os.exit(1)
		}
		fmt.eprintfln("model loaded (cpu, max_ctx=%d) in %v", max_ctx, time.since(t_start))

		// ---- ONE forward on token 1 at position 0 ----
		t_fwd := time.now()
		logits := q35.engine_forward(&engine, 1, 0)
		wall := time.since(t_fwd)

		finite := true
		argmax := -1
		best := f32(-3.4028235e38)
		nonfinite_count := 0
		for i in 0 ..< len(logits) {
			v := logits[i]
			if math.is_nan(v) || math.is_inf(v) {
				finite = false
				nonfinite_count += 1
				continue
			}
			if v > best {
				best = v
				argmax = i
			}
		}
		fmt.printf(
			"n_logits=%d finite=%v nonfinite_count=%d argmax_token=%d logits[argmax]=%v\n",
			len(logits),
			finite,
			nonfinite_count,
			argmax,
			best,
		)
		fmt.printf("forward_wall_time=%v\n", wall)

		q35.engine_destroy(&engine)
		return
	}

	// ---- compare mode: CPU engine vs Metal engine, 2 tokens each ----
	tokens := []int{1, 2}

	// CPU engine
	cpu_engine, cpu_ok := q35.engine_load(
		path,
		q35.Engine_Opts {
			max_ctx = max_ctx,
			num_threads = os.get_processor_core_count(),
			use_metal = false,
		},
	)
	if !cpu_ok {
		fmt.eprintln("CPU engine_load failed")
		os.exit(1)
	}
	fmt.eprintfln("model loaded (cpu, max_ctx=%d) in %v", max_ctx, time.since(t_start))

	// Metal engine (skip the first-run autotune so the compare run is
	// deterministic and fast; the 27B tune table would not be reused anyway).
	os.set_env("ODIN_TUNE", "0")
	gpu_engine, gpu_ok := q35.engine_load(
		path,
		q35.Engine_Opts {
			max_ctx = max_ctx,
			num_threads = 1,
			use_metal = true,
		},
	)
	if !gpu_ok {
		fmt.eprintln("Metal engine_load failed")
		os.exit(1)
	}
	if !gpu_engine.metal_ready {
		fmt.eprintln("Metal engine did not come up (metal_ready=false) — cannot compare")
		os.exit(1)
	}
	fmt.eprintfln("model loaded (metal, max_ctx=%d) in %v", max_ctx, time.since(t_start))

	// CPU forwards (slow on a loaded machine: ~15-50s/token on 27B)
	cpu_logits: [2][]f32
	cpu_wall: [2]time.Duration
	for i in 0 ..< 2 {
		t_fwd := time.now()
		logits := q35.engine_forward(&cpu_engine, tokens[i], i)
		cpu_wall[i] = time.since(t_fwd)
		cpu_logits[i] = make([]f32, len(logits))
		copy(cpu_logits[i], logits)
		fmt.eprintfln("cpu forward token %d (pos %d) done in %v", tokens[i], i, cpu_wall[i])
	}
	q35.engine_destroy(&cpu_engine)

	// GPU forwards
	gpu_logits: [2][]f32
	gpu_wall: [2]time.Duration
	for i in 0 ..< 2 {
		t_fwd := time.now()
		logits := q35.engine_forward(&gpu_engine, tokens[i], i)
		gpu_wall[i] = time.since(t_fwd)
		gpu_logits[i] = make([]f32, len(logits))
		copy(gpu_logits[i], logits)
		fmt.eprintfln("gpu forward token %d (pos %d) done in %v", tokens[i], i, gpu_wall[i])
	}

	compare_logits(cpu_logits[0], gpu_logits[0], "compare token 1 (pos 0)")
	compare_logits(cpu_logits[1], gpu_logits[1], "compare token 2 (pos 1)")
	fmt.printf("gpu_wall_token1=%v\n", gpu_wall[0])
	fmt.printf("gpu_wall_token2=%v\n", gpu_wall[1])

	q35.engine_destroy(&gpu_engine)
}
