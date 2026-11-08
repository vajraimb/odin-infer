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
	rest := args[1:]
	for len(rest) > 0 {
		if rest[0] == "--prompt" && len(rest) > 1 {
			prompt = rest[1]
			rest = rest[2:]
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

	t0 := time.now()
	e, ok := q35.engine_load(path, q35.Engine_Opts{max_ctx = 512, num_threads = 8, use_metal = false})
	if !ok {
		fmt.eprintln("engine_load failed")
		os.exit(1)
	}
	defer q35.engine_destroy(&e)
	c := q35.engine_config(&e)
	load_ms := time.duration_milliseconds(time.since(t0))
	fmt.eprintfln("loaded %.0f ms: moe=%v dim=%d layers=%d experts=%d used=%d ff_exp=%d shexp_ff=%d vocab=%d",
		load_ms, c.is_moe, c.dim, c.n_layers, c.n_expert, c.n_expert_used, c.n_ff_exp, c.n_ff_shexp, c.vocab_size)
	if !c.is_moe {
		fmt.eprintln("not a qwen35moe model")
		os.exit(1)
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

	// prefill: feed the prompt tokens sequentially, then decode
	prefill_start := time.now()
	for i in 0 ..< len(ids) {
		q35.engine_forward(&e, ids[i], i)
	}
	prefill_ms := time.duration_milliseconds(time.since(prefill_start))
	fmt.eprintfln("prefill: %d tokens in %.0f ms", len(ids), prefill_ms)

	pos := len(ids)
	for step in 0 ..< steps {
		ts := time.now()
		logits := q35.engine_forward(&e, ids[len(ids) - 1], pos - 1 + step)
		best := 0
		for i in 1 ..< len(logits) {
			if logits[i] > logits[best] {
				best = i
			}
		}
		ms := time.duration_milliseconds(time.since(ts))
		text := tok35.decode_token_id(&tok, best, context.temp_allocator)
		text = strings.trim_space(text)
		fmt.eprintfln("step %d: tok=%d text=%q %.0f ms", step, best, text, ms)
		_, _ = append(&ids, best)
	}
	fmt.println("REAL MOE SMOKE: DONE")
}
