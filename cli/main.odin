/* CLI and chat loop. Auto-detects Qwen3 vs Qwen3.5 (Ornith) from the GGUF and
   dispatches to the matching engine + tokenizer. `-g 1` enables the GPU:
   Metal on Apple Silicon, CUDA when the binary was built with
   -define:ODINFER_BACKEND="cuda". `-x` sets repetition penalty (off by default). */

package main

import ggml "ggml:ggml"
import infer "infer:infer"
import q35 "qwen3_5:qwen3_5"
import sampler "sampler:sampler"
import tokenizer "tokenizer:tokenizer"
import tok35 "qwen3_5_tokenizer:qwen3_5_tokenizer"

import "core:fmt"
import "core:os"
import "core:strconv"
import "core:strings"
import "core:time"
import "base:runtime"

EOS_QWEN3   :: 151645  // <|im_end|>
EOS_QWEN3_5 :: 248046  // <|im_end|>

debug_tokens: bool

Model_Kind :: enum { Qwen3, Qwen3_5 }

time_in_ms :: proc() -> i64 {
	return time.to_unix_nanoseconds(time.now()) / 1_000_000
}

read_stdin :: proc(guide: string, buffer: ^[8192]u8) -> string {
	fmt.print(guide)
	offset := 0
	for offset < 8191 {
		n, err := os.read(os.stdin, buffer[offset:offset + 1])
		if err != os.ERROR_NONE || n == 0 do break
		if buffer[offset] == '\n' do break
		offset += 1
	}
	if offset == 0 do return ""
	result := strings.clone(string(buffer[:offset]))
	for len(result) > 0 && result[len(result) - 1] == '\r' {
		result = result[:len(result) - 1]
	}
	return result
}

// ---- architecture detection (peek the GGUF metadata before loading) ----
detect_arch :: proc(path: string) -> Model_Kind {
	g: ggml.GGUF_File
	ggml.parse_gguf(path, &g)
	defer ggml.free_gguf(&g)
	arch, _ := ggml.gguf_meta_str(&g, "general.architecture")
	switch arch {
	case "qwen35", "qwen3_5", "qwen3.5", "qwen3_5_text":
		return .Qwen3_5
	}
	return .Qwen3
}

// ============================ Qwen3 chat loop ============================
chat_q3 :: proc(
	engine: ^infer.Engine,
	tok: ^tokenizer.Tokenizer,
	samp: ^sampler.Sampler,
	think_on, multi_turn, tps, ttft: bool,
	rep_penalty: f32,
	tb: ^tokenizer.Token_Buffer,
) {
	system_buf: [8192]u8
	user_buf: [8192]u8
	system_prompt: string
	user_prompt: string
	prompt_tokens: [dynamic]int
	defer {
		delete(system_prompt)
		delete(user_prompt)
		delete(prompt_tokens)
	}

	cfg := infer.engine_config(engine)
	user_turn := true
	next: int
	pos := 0
	num_prompt_tokens := 0
	timer: f64 = -1
	timer2: i64 = -1
	t_ttft: i64 = 0
	count := 0

	for {
		if user_turn {
			if pos == 0 {
				system_prompt = read_stdin("Enter system prompt (or Enter to skip): ", &system_buf)
			}
			user_prompt = read_stdin("Q: ", &user_buf)
			if len(user_prompt) == 0 do break

			if ttft && timer2 == -1 { timer2 = time_in_ms() }

			rendered: string
			if pos == 0 && len(system_prompt) > 0 {
				rendered = fmt.tprintf(
					"<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
					system_prompt, user_prompt)
			} else {
				rendered = fmt.tprintf("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", user_prompt)
			}
			if !think_on {
				rendered = fmt.tprintf("%s<think>\n\n</think>\n", rendered)
			}

			encoded, err := tokenizer.encode(tok, rendered)
			if err != nil { fmt.eprintf("encode failed: %v\n", err); break }
			clear(&prompt_tokens)
			append(&prompt_tokens, ..encoded)
			if rep_penalty > 1.0 {
				for id in encoded { sampler.record_token(samp, id) }
			}

			max_ctx := cfg.seq_len
			if len(prompt_tokens) >= max_ctx {
				fmt.fprintf(os.stderr, "Prompt is %d tokens but context limit is %d; truncating.\n", len(prompt_tokens), max_ctx)
				resize(&prompt_tokens, max_ctx - 1)
			}

			pos = 0
			user_turn = false
			if multi_turn {
				tokenizer.append_tokens(tb, encoded)
				num_prompt_tokens = len(tb.data)
			} else {
				num_prompt_tokens = len(prompt_tokens)
			}
			fmt.fprintf(os.stderr, "Prefilling %d tokens...\n", num_prompt_tokens)
			fmt.print("A: ")
			os.flush(os.stdout)
		}

		token: int
		if pos < num_prompt_tokens {
			token = multi_turn ? tb.data[pos] : prompt_tokens[pos]
		} else {
			token = next
		}

		if pos >= cfg.seq_len {
			fmt.fprintln(os.stderr, "\nContext limit reached; stopping generation.")
			fmt.println()
			user_turn = true
			if tps && timer >= 0 {
				elapsed := f64(time_in_ms()) - timer
				if elapsed > 0 { fmt.eprintf("tok/s: %f\n", f64(count) / elapsed * 1000.0) }
				timer = -1; count = 0
			}
			continue
		}

		logits := infer.engine_forward(engine, token, pos)
		next = sampler.sample(samp, logits)
		pos += 1

		if pos >= num_prompt_tokens {
			if multi_turn {
				tokenizer.append_tokens(tb, []int{next})
				num_prompt_tokens = len(tb.data)
			}
			if next == EOS_QWEN3 {
				fmt.println()
				user_turn = true
				if tps && timer >= 0 {
					elapsed := f64(time_in_ms()) - timer
					if elapsed > 0 { fmt.eprintf("tok/s: %f\n", f64(count) / elapsed * 1000.0) }
					timer = -1; count = 0
				}
				if ttft { fmt.eprintf("TTFT: %d ms\n", t_ttft); timer2 = -1; t_ttft = 0 }
			} else {
				if rep_penalty > 1.0 { sampler.record_token(samp, next) }
				decoded := tokenizer.decode_token_id(tok, next)
				if len(decoded) > 0 { fmt.print(decoded); os.flush(os.stdout) }
				delete(decoded)
				if ttft && t_ttft == 0 { t_ttft = time_in_ms() - timer2 }
				if tps {
					count += 1
					if timer < 0 { timer = f64(time_in_ms()) }
				}
			}
		}
	}
}

// ============================ Qwen3.5 (Ornith) chat loop ============================
chat_q35 :: proc(
	engine: ^q35.Engine,
	tok: ^tok35.Tokenizer,
	samp: ^sampler.Sampler,
	think_on, multi_turn, tps, ttft: bool,
	rep_penalty: f32,
	tb: ^tok35.Token_Buffer,
	initial_pos: int,
) -> int {
	system_buf: [8192]u8
	user_buf: [8192]u8
	system_prompt: string
	user_prompt: string
	prompt_tokens: [dynamic]int
	defer {
		delete(system_prompt)
		delete(user_prompt)
		delete(prompt_tokens)
	}

	cfg := q35.engine_config(engine)
	user_turn := true
	next: int
	pos := initial_pos
	num_prompt_tokens := 0
	timer: f64 = -1
	timer2: i64 = -1
	t_ttft: i64 = 0
	count := 0

	for {
		if user_turn {
			if pos == 0 {
				system_prompt = read_stdin("Enter system prompt (or Enter to skip): ", &system_buf)
			}
			user_prompt = read_stdin("Q: ", &user_buf)
			if len(user_prompt) == 0 do break

			if ttft && timer2 == -1 { timer2 = time_in_ms() }

			rendered: string
			if pos == 0 && len(system_prompt) > 0 {
				rendered = fmt.tprintf(
					"<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
					system_prompt, user_prompt)
			} else {
				rendered = fmt.tprintf("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", user_prompt)
			}
			if !think_on {
				rendered = fmt.tprintf("%s<think>\n\n</think>\n", rendered)
			}

			encoded, err := tok35.encode(tok, rendered)
			if err != nil { fmt.eprintf("encode failed: %v\n", err); break }
			clear(&prompt_tokens)
			append(&prompt_tokens, ..encoded)
			if rep_penalty > 1.0 {
				for id in encoded { sampler.record_token(samp, id) }
			}

			max_ctx := cfg.seq_len
			if len(prompt_tokens) >= max_ctx {
				fmt.fprintf(os.stderr, "Prompt is %d tokens but context limit is %d; truncating.\n", len(prompt_tokens), max_ctx)
				resize(&prompt_tokens, max_ctx - 1)
			}

			pos = 0
			user_turn = false
			if multi_turn {
				tok35.append_tokens(tb, encoded)
				num_prompt_tokens = len(tb.data)
			} else {
				num_prompt_tokens = len(prompt_tokens)
			}
			fmt.fprintf(os.stderr, "Prefilling %d tokens...\n", num_prompt_tokens)
			// BATCHED PREFILL (Stage 1a): process the first N-1 prompt tokens as
			// matrix-matrix (Q4_K MMQ) batches; the last prompt token runs through
			// the normal loop so the first generated token is emitted naturally.
			// (engine_forward_batch self-dispatches Metal batch vs CPU per-token.)
			if num_prompt_tokens > 1 {
				m := num_prompt_tokens - 1
				src := multi_turn ? tb.data[:m] : prompt_tokens[:m]
				_ = q35.engine_forward_batch(engine, src, 0)
				pos = m
			}
			fmt.print("A: ")
			os.flush(os.stdout)
		}

		token: int
		if pos < num_prompt_tokens {
			token = multi_turn ? tb.data[pos] : prompt_tokens[pos]
		} else {
			token = next
		}

		if pos >= cfg.seq_len {
			fmt.fprintln(os.stderr, "\nContext limit reached; stopping generation.")
			fmt.println()
			user_turn = true
			if tps && timer >= 0 {
				elapsed := f64(time_in_ms()) - timer
				if elapsed > 0 { fmt.eprintf("tok/s: %f\n", f64(count) / elapsed * 1000.0) }
				timer = -1; count = 0
			}
			continue
		}

		logits := q35.engine_forward(engine, token, pos)
		next = sampler.sample(samp, logits)
		pos += 1

		if pos >= num_prompt_tokens {
			if multi_turn {
				tok35.append_tokens(tb, []int{next})
				num_prompt_tokens = len(tb.data)
			}
			if next == EOS_QWEN3_5 {
				fmt.println()
				user_turn = true
				if tps && timer >= 0 {
					elapsed := f64(time_in_ms()) - timer
					if elapsed > 0 { fmt.eprintf("tok/s: %f\n", f64(count) / elapsed * 1000.0) }
					timer = -1; count = 0
				}
				if ttft { fmt.eprintf("TTFT: %d ms\n", t_ttft); timer2 = -1; t_ttft = 0 }
			} else {
				if rep_penalty > 1.0 { sampler.record_token(samp, next) }
				decoded := tok35.decode_token_id(tok, next)
				if len(decoded) > 0 { fmt.print(decoded); os.flush(os.stdout) }
				delete(decoded)
				if ttft && t_ttft == 0 { t_ttft = time_in_ms() - timer2 }
				if tps {
					count += 1
					if timer < 0 { timer = f64(time_in_ms()) }
				}
			}
		}
	}
	return pos
}

dump_gguf :: proc(path: string) {
	g: ggml.GGUF_File
	ggml.parse_gguf(path, &g)
	defer ggml.free_gguf(&g)
	fmt.printf("tensors=%d metadata=%d\n", len(g.tensors), len(g.metadata))
	if arch, ok := ggml.gguf_meta_str(&g, "general.architecture"); ok {
		fmt.printf("architecture=%s\n", arch)
	}
}

error_usage :: proc() {
	fmt.eprintln("Usage:   odin-cli <model.gguf> [options]")
	fmt.eprintln("Examples:")
	fmt.eprintln("  ./odin-cli Qwen3-0.6B-Q4_K_M.gguf -g 1 -t 0 -r 1")
	fmt.eprintln("  ./odin-cli ornith-1.0-9b-Q4_K_M.gguf -g 1 -c 2048 -x 1.15 -k 1 -r 1")
	fmt.eprintln("Auto-detects Qwen3 vs Qwen3.5 (Ornith) from the GGUF architecture.")
	fmt.eprintln("Options:")
	fmt.eprintln("  -t <float>  temperature [0,inf], default 0.6 (0 = greedy)")
	fmt.eprintln("  -p <float>  top-p in [0,1], default 0.95")
	fmt.eprintln("  -s <int>    random seed, default time()")
	fmt.eprintln("  -m <int>    multi-turn: 0 = off (default), 1 = on")
	fmt.eprintln("  -k <int>    reasoning: 0 = inject empty <think>, 1 = let the model think")
	fmt.eprintln("  -x <float>  repetition penalty, 1.0 = off (default); try 1.1-1.3")
	fmt.eprintln("  -r <int>    print tok/s: 0 = off (default), 1 = on")
	fmt.eprintln("  -f <int>    print TTFT (ms): 0 = off (default), 1 = on")
	fmt.eprintln("  -j <int>    CPU matmul threads (default: core count)")
	fmt.eprintln("  -c <int>    max context length (default 4096; caps KV/state memory)")
	when q35.GPU_VULKAN {
		fmt.eprintln("  -g <int>    GPU: 0 = off (default), 1 = on (Vulkan; build with -define:ODINFER_BACKEND=vulkan)")
	} else {
		fmt.eprintln("  -g <int>    GPU: 0 = off (default), 1 = on (Metal on macOS, CUDA if built with ODINFER_BACKEND=cuda)")
	}
	fmt.eprintln("  -L <path>   load KV+SSM state from .oikv file (Qwen3.5 only, resumes chat)")
	fmt.eprintln("  -S <path>   save KV+SSM state to .oikv on exit (Qwen3.5 only)")
	fmt.eprintln("  -J <int>    force JSON output (Qwen3.5 only): 0 = off (default), 1 = on")
	os.exit(1)
}

main :: proc() {
	temperature: f32 = 0.6
	topp: f32 = 0.95
	rng_seed: u64 = 0
	multi_turn := false
	think_on := false
	tps := false
	ttft := false
	rep_penalty: f32 = 1.0
	num_threads := os.get_processor_core_count()
	max_ctx := infer.DEFAULT_MAX_CONTEXT
	use_metal := false
	load_kv_path: string = ""
	save_kv_path: string = ""
	json_mode := false

	args := os.args
	if len(args) < 2 { error_usage() }
	checkpoint_path := args[1]
	debug_tokens = os.get_env("QDBG", context.temp_allocator) == "1"

	if len(args) >= 3 && args[2] == "--dump" {
		dump_gguf(checkpoint_path)
		return
	}

	i := 2
	for i < len(args) {
		if i + 1 >= len(args) do error_usage()
		if args[i][0] != '-' || len(args[i]) != 2 do error_usage()
		switch args[i][1] {
		case 't': if v, ok := strconv.parse_f64(args[i + 1]); ok { temperature = f32(v) }
		case 'p': if v, ok := strconv.parse_f64(args[i + 1]); ok { topp = f32(v) }
		case 's': if v, ok := strconv.parse_u64(args[i + 1]); ok { rng_seed = v }
		case 'm': multi_turn = args[i + 1] == "1"
		case 'k': think_on = args[i + 1] == "1"
		case 'x': if v, ok := strconv.parse_f64(args[i + 1]); ok { rep_penalty = f32(v) }
		case 'r': tps = args[i + 1] == "1"
		case 'f': ttft = args[i + 1] == "1"
		case 'j': if v, ok := strconv.parse_int(args[i + 1]); ok { num_threads = v }
		case 'c': if v, ok := strconv.parse_int(args[i + 1]); ok { max_ctx = v }
		case 'g': use_metal = args[i + 1] == "1"
		case 'L': load_kv_path = args[i + 1]
		case 'S': save_kv_path = args[i + 1]
		case 'J': json_mode = args[i + 1] == "1"
		case: error_usage()
		}
		i += 2
	}

	if rng_seed == 0 { rng_seed = u64(time.time_to_unix(time.now())) }
	if temperature < 0 do temperature = 0
	if topp < 0 || topp > 1 do topp = 0.9
	if rep_penalty < 1.0 do rep_penalty = 1.0

	kind := detect_arch(checkpoint_path)
	fmt.eprintf("architecture: %s\n", kind == .Qwen3_5 ? "qwen3_5 (Ornith)" : "qwen3")

	if kind == .Qwen3_5 {
		run_q35(checkpoint_path, temperature, topp, rng_seed, rep_penalty, think_on, multi_turn, tps, ttft, use_metal, num_threads, max_ctx, load_kv_path, save_kv_path, json_mode)
	} else {
		if len(load_kv_path) > 0 || len(save_kv_path) > 0 {
			fmt.eprintln("warning: -L/-S KV persistence only supported on Qwen3.5; ignoring")
		}
		if json_mode {
			fmt.eprintln("warning: -J JSON mode only supported on Qwen3.5; ignoring")
		}
		run_q3(checkpoint_path, temperature, topp, rng_seed, rep_penalty, think_on, multi_turn, tps, ttft, use_metal, num_threads, max_ctx)
	}
}

run_q3 :: proc(
	path: string, temperature, topp: f32, rng_seed: u64, rep_penalty: f32,
	think_on, multi_turn, tps, ttft, use_metal: bool, num_threads, max_ctx: int,
) {
	engine, load_ok := infer.engine_load(path, infer.Engine_Opts{max_ctx = max_ctx, use_metal = use_metal, num_threads = num_threads})
	if !load_ok {
		fmt.eprintln("cli: engine_load failed")
		os.exit(1)
	}
	defer infer.engine_destroy(&engine)

	tok: tokenizer.Tokenizer
	tokenizer.build_tokenizer(&tok)
	defer tokenizer.free_tokenizer(&tok)
	tb: tokenizer.Token_Buffer
	tokenizer.build_token_buffer(&tb)
	defer tokenizer.free_token_buffer(&tb)

	samp: sampler.Sampler
	cfg := infer.engine_config(&engine)
	sampler.build_sampler(&samp, int(cfg.vocab_size), temperature, topp, rng_seed)
	defer sampler.free_sampler(&samp)
	if rep_penalty > 1.0 { sampler.enable_repeat_penalty(&samp, rep_penalty) }

	fmt.printf(
		"think=%s multi=%s rep=%s tps=%s ttft=%s metal=%s threads=%d T=%.2f P=%.2f\n",
		think_on ? "on" : "off", multi_turn ? "on" : "off",
		rep_penalty > 1.0 ? "on" : "off", tps ? "on" : "off", ttft ? "on" : "off",
		use_metal ? "on" : "off", num_threads, temperature, topp,
	)
	fmt.println("Press Enter to exit the chat")
	chat_q3(&engine, &tok, &samp, think_on, multi_turn, tps, ttft, rep_penalty, &tb)
	infer.destroy_matmul_pool()
}

run_q35 :: proc(
	path: string, temperature, topp: f32, rng_seed: u64, rep_penalty: f32,
	think_on, multi_turn, tps, ttft, use_metal: bool, num_threads, max_ctx: int,
	load_kv_path, save_kv_path: string,
	json_mode: bool,
) {
	engine, load_ok := q35.engine_load(path, q35.Engine_Opts{max_ctx = max_ctx, use_metal = use_metal, num_threads = num_threads})
	if !load_ok {
		fmt.eprintln("cli: engine_load failed")
		os.exit(1)
	}
	defer q35.engine_destroy(&engine)

	// Optional KV+SSM state restore.
	initial_pos := 0
	if len(load_kv_path) > 0 {
		pos, ok := q35.engine_load_kv(&engine, load_kv_path)
		if ok {
			initial_pos = pos
		} else {
			fmt.eprintln("kv: load failed; starting fresh")
		}
	}

	tok: tok35.Tokenizer
	tok35.build_tokenizer(&tok)
	defer tok35.free_tokenizer(&tok)
	tb: tok35.Token_Buffer
	tok35.build_token_buffer(&tb)
	defer tok35.free_token_buffer(&tb)

	samp: sampler.Sampler
	cfg := q35.engine_config(&engine)
	sampler.build_sampler(&samp, int(cfg.vocab_size), temperature, topp, rng_seed)
	defer sampler.free_sampler(&samp)
	if rep_penalty > 1.0 { sampler.enable_repeat_penalty(&samp, rep_penalty) }

	// JSON grammar-constrained output (Qwen3.5 only)
	grammar: sampler.JSON_Grammar
	if json_mode {
		fmt.eprintln("json: building token-bytes cache (one-time, may take a few seconds)…")
		t0 := time_in_ms()
		// Inline build_token_cache: closures in Odin can't capture locals, so
		// we can't pass `tok` via the callback API. Iterate directly.
		vs := int(cfg.vocab_size)
		token_bytes := make([][]u8, vs)
		for id in 0 ..< vs {
			s := tok35.decode_token_id(&tok, id)
			token_bytes[id] = transmute([]u8)s
		}
		grammar = sampler.init_json()
		samp.grammar = &grammar
		samp.token_bytes = token_bytes
		samp.eos_token = EOS_QWEN3_5
		fmt.eprintfln("json: ready (%d ms, vocab=%d)", (time_in_ms() - t0), vs)
	}

	fmt.printf(
		"think=%s multi=%s rep=%s tps=%s ttft=%s metal=%s threads=%d T=%.2f P=%.2f\n",
		think_on ? "on" : "off", multi_turn ? "on" : "off",
		rep_penalty > 1.0 ? "on" : "off", tps ? "on" : "off", ttft ? "on" : "off",
		use_metal ? "on" : "off", num_threads, temperature, topp,
	)
	fmt.println("Press Enter to exit the chat")
	final_pos := chat_q35(&engine, &tok, &samp, think_on, multi_turn, tps, ttft, rep_penalty, &tb, initial_pos)
	if json_mode {
		sampler.free_json(&grammar)
	}
	if len(save_kv_path) > 0 {
		if !q35.engine_save_kv(&engine, save_kv_path, final_pos) {
			fmt.eprintln("kv: save failed")
		}
	}
	q35.destroy_matmul_pool()
}
