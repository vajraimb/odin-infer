/* BPE tokenizer for Qwen3 */

package tokenizer

import "core:fmt"
import "core:mem"
import "core:os"
import "core:strings"
import "core:testing"

MAX_TOKENS :: 1024
MAX_TOKEN_LEN :: 32
MAX_VOCAB :: 151936

MERGE_NOT_FOUND :: 1_000_000_000
MAX_MERGES :: 151386

EMBEDDED_VOCAB :: string(#load("vocab.txt"))
EMBEDDED_MERGES :: string(#load("merges.txt"))

Merge_Rule :: struct {
	left:  string,
	right: string,
	rank:  int,
}

Tokenizer :: struct {
	vocab:        [dynamic]string,
	vocab_lookup: map[string]int,
	merges:       [dynamic]Merge_Rule,
}

byte_to_unicode: [256]u32
unicode_bytes: [256][5]u8
unicode_to_byte: map[string]u8

SPECIAL_TOKENS :: []string {
	"<|im_start|>",
	"<|im_end|>",
	"<think>",
	"</think>",
}

init_byte_unicode_map :: proc() {
	unicode_to_byte = make(map[string]u8)
	n := 0
	for b in 0 ..< 256 {
		if (b >= 33 && b <= 126) ||
		   (b >= 161 && b <= 172) ||
		   (b >= 174 && b <= 255) {
			byte_to_unicode[b] = u32(b)
		} else {
			byte_to_unicode[b] = u32(256 + n)
			n += 1
		}

		cp := byte_to_unicode[b]
		if cp < 128 {
			unicode_bytes[b][0] = u8(cp)
			unicode_bytes[b][1] = 0
		} else if cp < 2048 {
			unicode_bytes[b][0] = 0xC0 | u8(cp >> 6)
			unicode_bytes[b][1] = 0x80 | u8(cp & 0x3F)
			unicode_bytes[b][2] = 0
		} else {
			unicode_bytes[b][0] = 0xE0 | u8(cp >> 12)
			unicode_bytes[b][1] = 0x80 | u8((cp >> 6) & 0x3F)
			unicode_bytes[b][2] = 0x80 | u8(cp & 0x3F)
			unicode_bytes[b][3] = 0
		}

		end := 0
		for end < len(unicode_bytes[b]) && unicode_bytes[b][end] != 0 {
			end += 1
		}
		key := strings.clone(string(unicode_bytes[b][:end]))
		unicode_to_byte[key] = u8(b)
	}
}

load_vocab_data :: proc(t: ^Tokenizer, data: string) {
	t.vocab = make([dynamic]string, 0, MAX_VOCAB)
	t.vocab_lookup = make(map[string]int)

	for line in strings.split_lines(data) {
		if len(line) == 0 do continue
		if len(t.vocab) >= MAX_VOCAB {
			fmt.eprintf("vocab exceeds MAX_VOCAB (%d)\n", MAX_VOCAB)
			break
		}
		id := len(t.vocab)
		token := strings.clone(line)
		t.vocab_lookup[token] = id
		append(&t.vocab, token)
	}
}

load_merges_data :: proc(t: ^Tokenizer, data: string) {
	t.merges = make([dynamic]Merge_Rule, 0, MAX_MERGES)
	rank := 0

	for line in strings.split_lines(data) {
		if len(line) == 0 || line[0] == '#' do continue
		parts := strings.fields(line)
		if len(parts) < 2 do continue
		append(&t.merges, Merge_Rule{
			left  = strings.clone(parts[0]),
			right = strings.clone(parts[1]),
			rank  = rank,
		})
		rank += 1
	}
}

load_vocab_file :: proc(t: ^Tokenizer, path: string) -> bool {
	data, err := os.read_entire_file_from_path(path, context.temp_allocator)
	if err != os.ERROR_NONE do return false
	load_vocab_data(t, string(data))
	return true
}

load_merges_file :: proc(t: ^Tokenizer, path: string) -> bool {
	data, err := os.read_entire_file_from_path(path, context.temp_allocator)
	if err != os.ERROR_NONE do return false
	load_merges_data(t, string(data))
	return true
}

// Prefer vocab.txt / merges.txt in the current directory; fall back to data
// embedded at compile time (#load) so a lone binary still works for distribution.
build_tokenizer :: proc(t: ^Tokenizer, vocab_path: string = "vocab.txt", merges_path: string = "merges.txt") {
	init_byte_unicode_map()

	vocab_ok := load_vocab_file(t, vocab_path)
	merges_ok := load_merges_file(t, merges_path)

	if !vocab_ok {
		load_vocab_data(t, EMBEDDED_VOCAB)
	}
	if !merges_ok {
		load_merges_data(t, EMBEDDED_MERGES)
	}
	if !vocab_ok || !merges_ok {
		fmt.eprintln("tokenizer: using embedded vocab/merges (place vocab.txt merges.txt in cwd to override)")
	}
}

verify_tokenizer :: proc(t: ^Tokenizer) -> bool {
	test := "<|im_start|>user\nTest<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
	ids, err := encode(t, test)
	if err != nil || len(ids) == 0 do return false
	for id in ids {
		if id == 0 do return false
	}
	return true
}

free_tokenizer :: proc(t: ^Tokenizer) {
	for s in t.vocab {
		delete(s)
	}
	delete(t.vocab)
	delete(t.vocab_lookup)
	for &m in t.merges {
		delete(m.left)
		delete(m.right)
	}
	delete(t.merges)
	delete(unicode_to_byte)
}

get_merge_rank :: proc(t: ^Tokenizer, left, right: string) -> int {
	for &m in t.merges {
		if m.left == left && m.right == right {
			return m.rank
		}
	}
	return MERGE_NOT_FOUND
}

match_special_token :: proc(t: ^Tokenizer, str: string) -> (id: int, match_len: int, ok: bool) {
	for special in SPECIAL_TOKENS {
		if strings.has_prefix(str, special) {
			if token_id, found := t.vocab_lookup[special]; found {
				return token_id, len(special), true
			}
		}
	}
	return -1, 0, false
}

is_special_pair :: proc(a, b: string) -> bool {
	return len(a) > 0 && a[0] == '<' && strings.contains_rune(a, '|') &&
	       len(b) > 0 && b[0] == '<' && strings.contains_rune(b, '|')
}

encode :: proc(t: ^Tokenizer, text: string, allocator := context.allocator) -> ([]int, mem.Allocator_Error) {
	tokens := make([dynamic]string, allocator = context.temp_allocator)

	i := 0
	for i < len(text) {
		if id, match_len, found := match_special_token(t, text[i:]); found {
			append(&tokens, t.vocab[id])
			i += match_len
			continue
		}

		b := text[i]
		end := 0
		for end < len(unicode_bytes[b]) && unicode_bytes[b][end] != 0 {
			end += 1
		}
		append(&tokens, strings.clone(string(unicode_bytes[b][:end])))
		i += 1
	}

	// BPE merge
	changed := true
	for changed {
		changed = false
		best_rank := MERGE_NOT_FOUND
		best_pos := -1

		for i in 0 ..< len(tokens) - 1 {
			rank := get_merge_rank(t, tokens[i], tokens[i + 1])
			if rank < best_rank {
				best_rank = rank
				best_pos = i
			}
		}

		if best_pos == -1 do break
		if is_special_pair(tokens[best_pos], tokens[best_pos + 1]) do break

		merged := strings.clone(fmt.tprintf("%s%s", tokens[best_pos], tokens[best_pos + 1]))
		tokens[best_pos] = merged
		ordered_remove(&tokens, best_pos + 1)
		changed = true
	}

	result, err := make([]int, len(tokens), allocator)
	if err != nil do return nil, err

	for i in 0 ..< len(tokens) {
		if id, found := t.vocab_lookup[tokens[i]]; found {
			result[i] = id
		} else {
			result[i] = 0
		}
	}

	return result, nil
}

decode_token_id :: proc(t: ^Tokenizer, token_id: int, allocator := context.allocator) -> string {
	if token_id < 0 || token_id >= len(t.vocab) {
		return ""
	}
	encoded := t.vocab[token_id]

	builder: strings.Builder
	strings.builder_init(&builder, allocator)
	defer strings.builder_destroy(&builder)

	i := 0
	for i < len(encoded) {
		matched := false
		for length in 1 ..= 3 {
			if i + length > len(encoded) do break
			utf8_str := encoded[i:i + length]
			if b, ok := unicode_to_byte[utf8_str]; ok {
				strings.write_byte(&builder, b)
				i += length
				matched = true
				break
			}
		}
		if !matched do break
	}

	return strings.clone(strings.to_string(builder), allocator)
}

Token_Buffer :: struct {
	data:     [dynamic]int,
}

build_token_buffer :: proc(tb: ^Token_Buffer, initial_capacity: int = 1024) {
	tb.data = make([dynamic]int, 0, initial_capacity)
}

free_token_buffer :: proc(tb: ^Token_Buffer) {
	delete(tb.data)
}

append_tokens :: proc(tb: ^Token_Buffer, tokens: []int) {
	append(&tb.data, ..tokens)
}

@(test)
encode_decode_roundtrip :: proc(t: ^testing.T) {
	tok: Tokenizer
	build_tokenizer(&tok)
	defer free_tokenizer(&tok)

	text := "Hello, world!"
	ids, err := encode(&tok, text)
	testing.expect(t, err == nil, "encode failed")
	testing.expect(t, len(ids) > 0, "expected tokens")

	decoded_parts: [dynamic]string
	defer delete(decoded_parts)
	for id in ids {
		append(&decoded_parts, decode_token_id(&tok, id))
	}
	reconstructed := strings.concatenate(decoded_parts[:], context.temp_allocator)
	testing.expect(t, reconstructed == text, "roundtrip mismatch")
}

@(test)
encode_special_token :: proc(t: ^testing.T) {
	tok: Tokenizer
	build_tokenizer(&tok)
	defer free_tokenizer(&tok)

	prompt := "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
	ids, err := encode(&tok, prompt)
	testing.expect(t, err == nil, "encode failed")
	testing.expect(t, len(ids) > 0, "expected tokens")

	if id, ok := tok.vocab_lookup["<|im_start|>"]; ok {
		testing.expect(t, ids[0] == id, "first token should be im_start")
	} else {
		testing.expect(t, false, "im_start not in vocab lookup")
	}
}

@(test)
encode_chinese_template :: proc(t: ^testing.T) {
	tok: Tokenizer
	build_tokenizer(&tok)
	defer free_tokenizer(&tok)

	prompt := "<|im_start|>user\n你能做什么<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
	ids, err := encode(&tok, prompt)
	testing.expect(t, err == nil, "encode failed")
	for id in ids {
		if id == 0 {
			testing.expect(t, false, "token id 0 in chinese template")
		}
	}
}

@(test)
encode_chinese :: proc(t: ^testing.T) {
	tok: Tokenizer
	build_tokenizer(&tok)
	defer free_tokenizer(&tok)

	text := "你能做什么"
	ids, err := encode(&tok, text)
	testing.expect(t, err == nil, "encode failed")
	testing.expect(t, len(ids) > 0, "expected tokens")
	for id in ids {
		testing.expect(t, id != 0, "unknown token in chinese encode")
	}
}

@(test)
encode_chat_template :: proc(t: ^testing.T) {
	tok: Tokenizer
	build_tokenizer(&tok)
	defer free_tokenizer(&tok)

	prompt := "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
	ids, err := encode(&tok, prompt)
	testing.expect(t, err == nil, "encode failed")

	for id in ids {
		if id == 0 {
			testing.expect(t, false, "token id 0 used for unknown token")
		}
	}
}
