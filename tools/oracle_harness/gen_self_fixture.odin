/* gen_self_fixture.odin — produce a fixture using odin-infer itself, so the
   oracle harness can be plumbing-verified without needing torch/HF.

   The token sequences below are NOT meaningful text — they're chosen only to
   exercise different token-id ranges. Running `oracle run` against this
   fixture should report ~100% top-1 match and near-zero max-abs (limited by
   any FP-reduction non-determinism across the thread pool).

   Run (from repo root):
     odin run oracle_harness/gen_self_fixture.odin -file -- \
       /path/to/Qwen3-0.6B-Q4_K_M.gguf oracle_harness/fixtures/self-qwen3-0.6b.bin \
       -collection:ggml=. -collection:infer=. \
       -collection:tokenizer=. -collection:sampler=.
*/

package gen_self_fixture

import "core:fmt"
import "core:os"
import infer "infer:infer"

MAGIC :: u32(0x4F524331) // 'ORC1'

// Synthetic token sequences (arbitrary IDs in valid range; Qwen3 vocab=151646).
// 6 prompts to mirror gen_fixture.py's prompt set in shape.
SELF_PROMPTS : [6][32]u32 = {
	{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0, 0, 0, 0, 0, 0, 0, 0},
	{1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	{500, 1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	{42000, 43000, 44000, 45000, 46000, 47000, 48000, 49000, 50000, 51000, 52000, 53000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	{99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	{123, 456, 789, 1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 9012, 13579, 24680, 99999, 100000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
}
PROMPT_LENS : [6]int = {24, 16, 10, 12, 20, 15}

write_u32_le :: proc(f: ^os.File, v: u32) {
	buf := [4]u8{u8(v & 0xff), u8((v >> 8) & 0xff), u8((v >> 16) & 0xff), u8((v >> 24) & 0xff)}
	_, _ = os.write(f, buf[:])
}

write_f16_le :: proc(f: ^os.File, v: f32) {
	h := transmute(u16)(f16(v))
	b := [2]u8{u8(h & 0xff), u8((h >> 8) & 0xff)}
	_, _ = os.write(f, b[:])
}

main :: proc() {
	args := os.args
	if len(args) < 3 {
		fmt.eprintln("usage: gen_self_fixture <model.gguf> <out.bin>")
		os.exit(1)
	}
	model_path := args[1]
	out_path := args[2]

	cores := os.get_processor_core_count()
	e, ok := infer.engine_load(model_path, infer.Engine_Opts{
		max_ctx = 4096,
		use_metal = false,
		num_threads = cores,
	})
	if !ok { fmt.eprintln("engine_load failed"); os.exit(1) }
	defer infer.engine_destroy(&e)
	defer infer.destroy_matmul_pool()

	cfg := infer.engine_config(&e)
	vocab := u32(cfg.vocab_size)
	fmt.printfln("model: %s  vocab=%d  threads=%d", model_path, vocab, cores)

	f, err := os.open(out_path, os.O_CREATE | os.O_WRONLY | os.O_TRUNC)
	if err != os.ERROR_NONE { fmt.eprintf("cannot open %s (%v)\n", out_path, err); os.exit(1) }
	defer os.close(f)

	self_prompts := SELF_PROMPTS[:]
	prompt_lens  := PROMPT_LENS[:]

	write_u32_le(f, MAGIC)
	write_u32_le(f, vocab)
	write_u32_le(f, u32(len(self_prompts)))

	for i in 0 ..< len(self_prompts) {
		plen := prompt_lens[i]
		tokens := self_prompts[i][:plen]
		write_u32_le(f, u32(plen))
		for t in tokens { write_u32_le(f, t) }
		// teacher-force and dump logits per position
		for pos in 0 ..< plen {
			logits := infer.engine_forward(&e, int(tokens[pos]), pos)
			if len(logits) != int(vocab) {
				fmt.eprintf("vocab mismatch at prompt %d pos %d\n", i, pos)
				os.exit(1)
			}
			for v in logits { write_f16_le(f, v) }
		}
		fmt.printfln("  [%d] wrote %d positions × %d logits", i, plen, vocab)
	}
	fmt.printfln("wrote %s", out_path)
}
