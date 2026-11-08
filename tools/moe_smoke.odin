/* Real Qwen3.5-MoE smoke test: load a real MoE GGUF on CPU, prefill a short
   prompt, decode a few steps, print tokens + text and per-step timing.
   Numeric correctness is pinned by tools/moe_load_test.odin (bit-exact vs a
   Python reference); this proves the REAL model loads and talks.

   Usage:
     odin run tools/moe_smoke.odin -file \
       -collection:ggml=packages -collection:infer=packages \
       -collection:qwen3_5=packages -collection:qwen3_5_tokenizer=packages \
       -collection:sampler=packages -collection:tokenizer=packages \
       -- <model.gguf> [steps] [--prompt "text"]
*/
package main

import q35 "qwen3_5:qwen3_5"
import tok35 "qwen3_5_tokenizer:qwen3_5_tokenizer"

import "core:fmt"
import "core:os"
import "core:math"
import "core:strconv"
import "core:strings"
import "core:time"

main :: proc() {
	args := os.args[1:]
	if len(args) < 1 {
		fmt.eprintln("usage: moe_smoke <model.gguf> [steps] [--prompt text]")
		os.exit(1)
	}
	path := args[0]
	steps := 8
	prompt := "What is 2+2?"
	use_metal := false
	compare := false
	rest := args[1:]
	for len(rest) > 0 {
		if rest[0] == "--prompt" && len(rest) > 1 {
			prompt = rest[1]
			rest = rest[2:]
		} else if rest[0] == "--metal" {
			use_metal = true
			rest = rest[1:]
		} else if rest[0] == "--compare" {
			compare = true
			rest = rest[1:]
		} else if v, ok := strconv.parse_int(rest[0]); ok {
			steps = v
			rest = rest[1:]
		} else {
			rest = rest[1:]
		}
	}

	tok: tok35.Tokenizer
	tok35.build_tokenizer(&tok)
	defer tok35.free_tokenizer(&tok)

	run_engine := proc(path: string, use_metal: bool, tok: ^tok35.Tokenizer, ids: ^[dynamic]int, steps: int, label: string) -> [dynamic][dynamic]f32 {
		t0 := time.now()
		e, ok := q35.engine_load(path, q35.Engine_Opts{max_ctx = 512, num_threads = 8, use_metal = use_metal})
		if !ok {
			fmt.eprintln("engine_load failed")
			os.exit(1)
		}
		defer q35.engine_destroy(&e)
		c := q35.engine_config(&e)
		load_ms := time.duration_milliseconds(time.since(t0))
		fmt.eprintfln("[%s] loaded %.0f ms: moe=%v dim=%d layers=%d experts=%d used=%d ff_exp=%d shexp_ff=%d vocab=%d",
			label, load_ms, c.is_moe, c.dim, c.n_layers, c.n_expert, c.n_expert_used, c.n_ff_exp, c.n_ff_shexp, c.vocab_size)
		if !c.is_moe {
			fmt.eprintln("not a qwen35moe model")
			os.exit(1)
		}
		n_prompt := len(ids^) - 0
		// prefill prompt tokens, then decode
		prefill_start := time.now()
		for i in 0 ..< n_prompt {
			q35.engine_forward(&e, ids[i], i)
		}
		prefill_ms := time.duration_milliseconds(time.since(prefill_start))
		fmt.eprintfln("[%s] prefill: %d tokens in %.0f ms", label, n_prompt, prefill_ms)
		logits_saved := make([dynamic][dynamic]f32, 0, steps)
		pos := n_prompt
		for step in 0 ..< steps {
			ts := time.now()
			logits := q35.engine_forward(&e, ids[len(ids^) - 1], pos - 1 + step)
			ms := time.duration_milliseconds(time.since(ts))
			best := 0
			for i in 1 ..< len(logits) {
				if logits[i] > logits[best] {
					best = i
				}
			}
			text := tok35.decode_token_id(tok, best, context.temp_allocator)
			text = strings.trim_space(text)
			fmt.eprintfln("[%s] step %d: tok=%d text=%q %.0f ms", label, step, best, text, ms)
			saved := make([dynamic]f32, len(logits))
			for i in 0 ..< len(logits) {
				saved[i] = logits[i]
			}
			_, _ = append(&logits_saved, saved)
			_, _ = append(ids, best)
		}
		return logits_saved
	}

	ids_src, aerr := tok35.encode(&tok, prompt)
	if aerr != nil {
		fmt.eprintfln("encode failed: %v", aerr)
		os.exit(1)
	}
	ids := make([dynamic]int, 0, len(ids_src) + steps)
	for v in ids_src {
		_, _ = append(&ids, v)
	}
	fmt.eprintfln("prompt %q -> %d tokens", prompt, len(ids))

	if compare {
		cpu_ids := make([dynamic]int, 0, len(ids) + steps)
		for v in ids {
			_, _ = append(&cpu_ids, v)
		}
		cpu_logits := run_engine(path, false, &tok, &cpu_ids, steps, "cpu")
		gpu_ids := make([dynamic]int, 0, len(ids) + steps)
		for v in ids {
			_, _ = append(&gpu_ids, v)
		}
		gpu_logits := run_engine(path, true, &tok, &gpu_ids, steps, "gpu")
		worst: f32 = 0.0
		for step in 0 ..< steps {
			a := cpu_logits[step]
			b := gpu_logits[step]
			if len(a) != len(b) {
				fmt.eprintfln("step %d: vocab mismatch %d vs %d", step, len(a), len(b))
				os.exit(1)
			}
			max_abs: f32 = 0.0
			for i in 0 ..< len(a) {
				max_abs = max(max_abs, math.abs(a[i] - b[i]))
			}
			if step == 0 {
				fmt.eprintfln("  sanity: a[0]=%.4f b[0]=%.4f a[271]=%.4f b[271]=%.4f lens %d %d", a[0], b[0], a[271], b[271], len(a), len(b))
			}
			worst = max(worst, max_abs)
			fmt.eprintfln("compare step %d: max_abs=%.6f %s", step, max_abs, max_abs < 5e-3 ? "OK" : "MISMATCH")
		}
		fmt.eprintfln("worst max_abs over %d steps: %.6f", steps, worst)
		if worst < 5e-3 {
			fmt.println("METAL MOE COMPARE: ALL PASSED")
		} else {
			fmt.eprintln("METAL MOE COMPARE: FAILED")
			os.exit(1)
		}
	} else {
		run_engine(path, use_metal, &tok, &ids, steps, use_metal ? "metal" : "cpu")
		fmt.println("REAL MOE SMOKE: DONE")
	}
}
