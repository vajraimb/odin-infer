/* Acceptance test for the IQ/K quant dequant port: loads the 27B dynamic
   mixed-quant Qwen3.5 GGUF (CPU only), prints its quant histogram, then runs
   ONE forward on token 1 and reports logits finiteness + argmax + wall time.

   Run:
     odin run tools/q38_load_test.odin -file \
       -collection:ggml=packages -collection:infer=packages \
       -collection:qwen3_5=packages -collection:qwen3_5_tokenizer=packages \
       -collection:sampler=packages -collection:tokenizer=packages \
       -- /Users/vajra/Downloads/Qwen3.8-27B-UD-IQ2_XXS.gguf [max_ctx]
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

main :: proc() {
	if len(os.args) < 2 {
		fmt.eprintln("usage: q38_load_test <model.gguf> [max_ctx]")
		os.exit(2)
	}
	path := os.args[1]
	max_ctx := 256
	if len(os.args) > 2 {
		if v, ok := strconv.parse_int(os.args[2]); ok {
			max_ctx = v
		}
	}

	t_start := time.now()

	// ---- pass 1: quant histogram straight from the GGUF file ----
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
}
