/* End-to-end MoE loader + CPU forward validation against a Python reference.

   Usage:
     python3 tools/make_moe_gguf.py          # writes models/moe_test.gguf + ref
     odin run tools/moe_load_test.odin -file \
       -collection:ggml=packages -collection:infer=packages \
       -collection:qwen3_5=packages -collection:qwen3_5_tokenizer=packages \
       -collection:sampler=packages -collection:tokenizer=packages
*/
package main

import q35 "qwen3_5:qwen3_5"

import "core:fmt"
import "core:math"
import "core:os"
import "core:strings"
import "core:strconv"

TOL :: 2e-3 // CPU vs python F32 reference, loose but far above float noise

main :: proc() {
	e, ok := q35.engine_load(
		"models/moe_test.gguf",
		q35.Engine_Opts{max_ctx = 64, num_threads = 1, use_metal = false},
	)
	if !ok {
		fmt.eprintln("engine_load failed")
		os.exit(1)
	}
	defer q35.engine_destroy(&e)

	c := q35.engine_config(&e)
	fmt.eprintfln("loaded: arch=moe dim=%d layers=%d experts=%d used=%d ff_exp=%d shexp=%d",
		c.dim, c.n_layers, c.n_expert, c.n_expert_used, c.n_ff_exp, c.n_ff_shexp)
	assert(c.is_moe)
	assert(c.n_expert == 6 && c.n_expert_used == 2 && c.n_ff_exp == 16)

	// reference sidecar: {"tokens": [..], "logits": [[..],[..]]}
	ref_data, rerr := os.read_entire_file("models/moe_test_ref.json", context.allocator)
	if rerr != os.ERROR_NONE {
		fmt.eprintln("missing models/moe_test_ref.json (run tools/make_moe_gguf.py)")
		os.exit(1)
	}
	ref := parse_ref(string(ref_data))

	bad := 0
	for step in 0 ..< 2 {
		token := ref.tokens[step]
		logits := q35.engine_forward(&e, token, step)
		want := ref.logits[step]
		if len(logits) != len(want) {
			fmt.eprintfln("step %d: vocab mismatch %d vs %d", step, len(logits), len(want))
			os.exit(1)
		}
		max_abs: f32 = 0.0
		for i in 0 ..< len(logits) {
			max_abs = max(max_abs, math.abs(logits[i] - want[i]))
		}
		status := "OK"
		if max_abs > TOL {
			status = "MISMATCH"
			bad += 1
		}
		fmt.eprintfln("step %d: max_abs=%.6f %s", step, max_abs, status)
	}

	if bad == 0 {
		fmt.println("MOE LOAD+FORWARD: ALL PASSED")
	} else {
		fmt.eprintfln("%d step(s) out of tolerance", bad)
		os.exit(1)
	}
}

Ref :: struct {
	tokens: [dynamic]int,
	logits: [dynamic][dynamic]f32,
}

parse_ref :: proc(s: string) -> Ref {
	r: Ref
	// minimal JSON parse: expect {"tokens":[a,b],"logits":[[...],[...]]}
	tokens := substring_between_loose(s, "\"tokens\"", "]")
	r.tokens = parse_ints(tokens)
	body := substring_between_loose(s, "\"logits\"", "]]")
	parts := strings.split(body, "],[")
	fmt.eprintfln("ref parse: body_len=%d parts=%d p0_len=%d sep_idx=%d", len(body), len(parts), len(parts[0]), strings.index(body, "],["))
	for row in parts {
		vals := parse_floats(row)
		append(&r.logits, vals)
	}
	return r
}

substring_between_loose :: proc(s, key, b: string) -> string {
	i := strings.index(s, key)
	if i < 0 do return ""
	i += len(key)
	for i < len(s) && (s[i] == ':' || s[i] == ' ') do i += 1
	if i >= len(s) || s[i] != '[' do return ""
	i += 1
	j := strings.index(s[i:], b)
	if j < 0 do return ""
	return s[i:i + j]
}

substring_between :: proc(s, a, b: string) -> string {
	i := strings.index(s, a)
	if i < 0 do return ""
	i += len(a)
	j := strings.index(s[i:], b)
	if j < 0 do return ""
	return s[i:i + j]
}

parse_ints :: proc(s: string) -> [dynamic]int {
	out := make([dynamic]int)
	for part in strings.split(s, ",") {
		p := strings.trim_space(part)
		if len(p) == 0 do continue
		if v, ok := strconv.parse_int(p); ok {
			append(&out, v)
		}
	}
	return out
}

parse_floats :: proc(s: string) -> [dynamic]f32 {
	out := make([dynamic]f32)
	for part in strings.split(s, ",") {
		p := strings.trim_space(part)
		// strip JSON brackets that may be attached to the first/last number
		start := 0
		end := len(p)
		for start < end && (p[start] == '[' || p[start] == ']') do start += 1
		for end > start && (p[end - 1] == '[' || p[end - 1] == ']') do end -= 1
		p = p[start:end]
		if len(p) == 0 do continue
		if v, ok := strconv.parse_f32(p); ok {
			append(&out, v)
		}
	}
	return out
}
