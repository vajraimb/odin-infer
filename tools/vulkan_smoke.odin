/* End-to-end Vulkan smoke: load a Qwen3.5 GGUF, prefill a short prompt, decode
   N tokens at temperature 0, print prefill and decode tok/s.

   odin run tools/vulkan_smoke.odin -file -define:ODINFER_BACKEND=vulkan \
     -collection:ggml=packages -collection:infer=packages \
     -collection:qwen3_5=packages -collection:qwen3_5_tokenizer=packages \
     -collection:sampler=packages -collection:tokenizer=packages \
     -collection:cuda=packages -collection:vulkan=packages \
     -- <model.gguf> [n_tokens] [max_ctx]

   This is the laptop command. CI compiles it and does not run it: the lavapipe
   numeric check is greedy_dump on the synthetic GGUF from make_vulkan_gguf.py.
*/

package main

import q35 "qwen3_5:qwen3_5"
import sampler "sampler:sampler"
import tok35 "qwen3_5_tokenizer:qwen3_5_tokenizer"

import "core:fmt"
import "core:os"
import "core:strconv"
import "core:strings"
import "core:time"

EOS_QWEN3_5 :: 248046

main :: proc() {
	when !q35.GPU_VULKAN {
		fmt.eprintln("vulkan_smoke: rebuild with -define:ODINFER_BACKEND=vulkan")
		os.exit(1)
	} else {
		if len(os.args) < 2 {
			fmt.eprintln("usage: vulkan_smoke <model.gguf> [n_tokens] [max_ctx]")
			os.exit(1)
		}
		path := os.args[1]
		n_tokens := 16
		max_ctx := 2048
		if len(os.args) > 2 {
			if v, ok := strconv.parse_int(os.args[2]); ok && v > 0 {
				n_tokens = int(v)
			}
		}
		if len(os.args) > 3 {
			if v, ok := strconv.parse_int(os.args[3]); ok && v > 0 {
				max_ctx = int(v)
			}
		}
		engine, load_ok := q35.engine_load(path, q35.Engine_Opts{max_ctx = max_ctx, use_metal = true, num_threads = 4})
		if !load_ok || !engine.metal_ready {
			fmt.eprintln("vulkan_smoke: engine_load failed (GPU init is required; METAL_FALLBACK is not a pass)")
			os.exit(1)
		}
		defer q35.engine_destroy(&engine)
		cfg := q35.engine_config(&engine)
		fmt.eprintfln("vulkan_smoke: backend=%s dim=%d layers=%d vocab=%d ctx=%d",
			q35.gpu_backend_name(engine.metal_ready), cfg.dim, cfg.n_layers, cfg.vocab_size, cfg.seq_len)

		tok: tok35.Tokenizer
		tok35.build_tokenizer(&tok)
		defer tok35.free_tokenizer(&tok)
		prompt := "<|im_start|>user\nWhat is 2+2? Answer with one number.<|im_end|>\n<|im_start|>assistant\n"
		encoded, eerr := tok35.encode(&tok, prompt)
		if eerr != nil || len(encoded) < 2 {
			fmt.eprintln("vulkan_smoke: encode failed")
			os.exit(1)
		}
		defer delete(encoded)

		fmt.eprintfln("vulkan_smoke: bench prefill={} decode_req={} ctx={} temp=0", len(encoded) - 1, n_tokens, max_ctx)
		t0 := time.now()
		logits := q35.engine_forward_batch(&engine, encoded[:len(encoded) - 1], 0)
		prefill_s := time.duration_seconds(time.diff(t0, time.now()))
		prefill_n := len(encoded) - 1
		fmt.eprintfln("vulkan_smoke: prefill %d tok in %.3f s (%.2f tok/s)", prefill_n, prefill_s, f64(prefill_n) / prefill_s)

		samp: sampler.Sampler
		sampler.build_sampler(&samp, cfg.vocab_size, 0, 1, 1)
		defer sampler.free_sampler(&samp)

		text: strings.Builder
		strings.builder_init(&text)
		defer strings.builder_destroy(&text)

		pos := prefill_n
		next := sampler.sample(&samp, logits)
		t1 := time.now()
		produced := 0
		for _ in 0 ..< n_tokens {
			piece := tok35.decode_token_id(&tok, next)
			strings.write_string(&text, piece)
			delete(piece)
			produced += 1
			if next == EOS_QWEN3_5 {
				break
			}
			if pos >= cfg.seq_len - 1 {
				break
			}
			logits = q35.engine_forward(&engine, next, pos)
			pos += 1
			next = sampler.sample(&samp, logits)
		}
		decode_s := time.duration_seconds(time.diff(t1, time.now()))
		rate := 0.0
		if decode_s > 0 {
			rate = f64(produced) / decode_s
		}
		fmt.eprintfln("vulkan_smoke: decode %d tok in %.3f s (%.2f tok/s)", produced, decode_s, rate)
		fmt.eprintfln("vulkan_smoke: text: %s", strings.to_string(text))
	}
}
