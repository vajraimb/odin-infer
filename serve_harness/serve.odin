/* serve_harness/serve.odin — HTTP dashboard + JSON API for odin-infer.
 *
 * Loads a Qwen3.5 engine and serves:
 *   GET  /              HTML dashboard with chat UI + live metrics
 *   GET  /api/info      JSON: model info, device, memory, env
 *   GET  /api/state     JSON: current position, last request stats
 *   POST /api/chat      JSON: {message} → {response, tok/s, ttft, pos}
 *
 * Minimal HTTP/1.0 server inlined (not RFC-compliant: Connection: close,
 * no keep-alive, no chunked encoding, no TLS). Designed for local dev and
 * profiling, not production.
 *
 * Run (from repo root):
 *   odin run serve_harness/serve.odin -file -- \
 *     /path/to/ornith.gguf --port 9748 --metal \
 *     -collection:ggml=. -collection:infer=. \
 *     -collection:qwen3_5=. -collection:qwen3_5_tokenizer=. \
 *     -collection:sampler=. -collection:tokenizer=.
 *
 * Then open http://127.0.0.1:9748 in a browser.
 */

package serve

import "core:fmt"
import "core:net"
import "core:os"
import "core:strconv"
import "core:strings"
import "core:time"
import q35 "qwen3_5:qwen3_5"
import tok35 "qwen3_5_tokenizer:qwen3_5_tokenizer"
import sampler "sampler:sampler"

EOS :: 248046 // <|im_end|> for Ornith

// ====================== HTTP server ======================

Request :: struct {
	method:  string,
	path:    string,
	body:    string,
	headers: map[string]string,
}

free_request :: proc(r: ^Request) {
	delete(r.headers)
}

Server_Handler :: proc(req: ^Request, resp: ^Response)

Response :: struct {
	status:  int,
	headers: map[string]string,
	body:    string,
}

start_server :: proc(port: u16, handler: Server_Handler) -> bool {
	addr := net.Endpoint{
		address = net.IP4_Address{127, 0, 0, 1},
		port    = int(port),
	}
	sock, err := net.listen_tcp(addr, 16)
	if err != nil || sock == 0 {
		fmt.eprintf("serve: listen_tcp on port %d failed (in use? err=%v)\n", port, err)
		return false
	}
	defer net.close(sock)
	fmt.printfln("serve: listening on http://127.0.0.1:%d", port)

	for {
		client, _, aerr := net.accept_tcp(sock)
		if aerr != nil || client == 0 {
			fmt.eprintf("serve: accept err %v\n", aerr)
			continue
		}
		handle_client(client, handler)
		net.close(client)
	}
}

handle_client :: proc(client: net.TCP_Socket, handler: Server_Handler) {
	buf: [65536]u8
	total := 0
	header_end := -1
	body_len_needed := 0  // expected body size from Content-Length
	for total < len(buf) {
		// If we already have headers + full body, stop reading.
		if header_end >= 0 && total >= header_end + body_len_needed { break }
		n, rerr := net.recv_tcp(client, buf[total:])
		if n <= 0 || rerr != nil { break }
		total += n
		if header_end < 0 {
			// Look for end of headers (\r\n\r\n)
			for i in 0 ..< total - 3 {
				if buf[i] == '\r' && buf[i+1] == '\n' && buf[i+2] == '\r' && buf[i+3] == '\n' {
					header_end = i + 4
					break
				}
			}
			if header_end >= 0 {
				// Parse Content-Length from headers so we know how much body to wait for.
				hdr_str := string(buf[:header_end])
				cl_idx := strings.index(hdr_str, "Content-Length:")
				if cl_idx >= 0 {
					v_start := cl_idx + len("Content-Length:")
					v_end_rel := strings.index(hdr_str[v_start:], "\r\n")
					if v_end_rel > 0 {
						v_str := strings.trim_space(hdr_str[v_start:v_start + v_end_rel])
						body_len_needed, _ = strconv_parse_int(v_str)
					}
				}
			}
		}
	}
	if header_end < 0 {
		write_simple(client, 400, "text/plain", "no header end")
		return
	}

	req_str := string(buf[:header_end])
	lines := strings.split(req_str, "\r\n", context.allocator)
	defer delete(lines)
	if len(lines) < 1 { return }
	parts := strings.split(lines[0], " ", context.allocator)
	defer delete(parts)
	if len(parts) != 3 {
		write_simple(client, 400, "text/plain", "bad request line")
		return
	}
	req: Request
	req.method = parts[0]
	req.path = parts[1]
	req.headers = make(map[string]string)
	for i in 1 ..< len(lines) {
		colon := strings.index(lines[i], ":")
		if colon < 0 { continue }
		key := strings.trim_space(lines[i][:colon])
		val := strings.trim_space(lines[i][colon+1:])
		req.headers[key] = val
	}
	if body_len_needed > 0 && header_end + body_len_needed <= total {
		req.body = string(buf[header_end:header_end + body_len_needed])
	}
	defer free_request_safe(&req)

	resp: Response
	resp.status = 200
	resp.headers = make(map[string]string)
	defer delete_resp_headers(&resp)
	handler(&req, &resp)

	write_full_response(client, &resp)
}

free_request_safe :: proc(r: ^Request) {
	delete(r.headers)
}

delete_resp_headers :: proc(resp: ^Response) {
	delete(resp.headers)
}

strconv_parse_int :: proc(s: string) -> (int, bool) {
	if len(s) == 0 { return 0, false }
	v: int = 0
	for c in s {
		if c < '0' || c > '9' { return 0, false }
		v = v * 10 + int(c - '0')
	}
	return v, true
}

write_simple :: proc(client: net.TCP_Socket, status: int, mime, body: string) {
	resp: Response
	resp.status = status
	resp.headers = make(map[string]string)
	defer delete(resp.headers)
	resp.headers["Content-Type"] = mime
	resp.body = body
	write_full_response(client, &resp)
}

write_full_response :: proc(client: net.TCP_Socket, resp: ^Response) {
	reason := "OK"
	switch resp.status {
	case 200: reason = "OK"
	case 201: reason = "Created"
	case 400: reason = "Bad Request"
	case 404: reason = "Not Found"
	case 405: reason = "Method Not Allowed"
	case 500: reason = "Internal Server Error"
	case:
	}
	out: strings.Builder
	strings.builder_init(&out, context.allocator)
	defer strings.builder_destroy(&out)
	strings.write_string(&out, fmt.tprintf("HTTP/1.0 %d %s\r\n", resp.status, reason))
	ct, ok := resp.headers["Content-Type"]
	if !ok { ct = "text/plain" }
	strings.write_string(&out, fmt.tprintf("Content-Type: %s\r\n", ct))
	strings.write_string(&out, fmt.tprintf("Content-Length: %d\r\n", len(resp.body)))
	strings.write_string(&out, "Connection: close\r\n")
	strings.write_string(&out, "Access-Control-Allow-Origin: *\r\n")
	strings.write_string(&out, "\r\n")
	strings.write_string(&out, resp.body)
	s := strings.clone(strings.to_string(out))
	if len(s) > 0 {
		bytes := transmute([]byte)s
		n, serr := net.send_tcp(client, bytes)
	}
	delete(s)
}

// ====================== shared state ======================

Engine_State :: struct {
	engine:     q35.Engine,
	tok:        tok35.Tokenizer,
	tb:         tok35.Token_Buffer,
	samp:       sampler.Sampler,
	pos:        int,
	last_tps:   f64,
	last_ttft:  i64,
	last_gen:   int,
	model_path: string,
}

g_state: Engine_State

// ====================== HTML (single-page dashboard) ======================

INDEX_HTML := #load ("./index.html")

// ====================== request handlers ======================

handle_request :: proc(req: ^Request, resp: ^Response) {
	switch {
	case req.method == "GET" && req.path == "/":
		resp.headers["Content-Type"] = "text/html; charset=utf-8"
		resp.body = string(INDEX_HTML)

	case req.method == "GET" && req.path == "/api/info":
		resp.headers["Content-Type"] = "application/json"
		resp.body = json_info()

	case req.method == "GET" && req.path == "/api/state":
		resp.headers["Content-Type"] = "application/json"
		resp.body = json_state()

	case req.method == "POST" && req.path == "/api/chat":
		resp.headers["Content-Type"] = "application/json"
		resp.body = handle_chat(req.body)

	case req.method == "POST" && (req.path == "/v1/chat/completions" || req.path == "/chat/completions"):
		resp.headers["Content-Type"] = "application/json"
		resp.body = handle_openai_chat(req.body)

	case req.method == "GET" && req.path == "/v1/models":
		resp.headers["Content-Type"] = "application/json"
		resp.body = openai_models_list()

	case req.method == "OPTIONS":
		resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
		resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
		resp.status = 204

	case:
		resp.status = 404
		resp.headers["Content-Type"] = "application/json"
		resp.body = `{"error": "not found"}`
	}
}

// ====================== JSON builders ======================

json_escape :: proc(s: string, out: ^strings.Builder) {
	strings.write_byte(out, byte('"'))
	// Iterate BYTES not runes — Chinese/emoji chars are multi-byte UTF-8 and
	// must be preserved verbatim. Only ASCII control chars and JSON specials
	// need escaping.
	for i in 0 ..< len(s) {
		c := s[i]
		switch c {
		case byte('"'):  strings.write_string(out, "\\\"")
		case byte('\\'): strings.write_string(out, "\\\\")
		case byte('\n'): strings.write_string(out, "\\n")
		case byte('\r'): strings.write_string(out, "\\r")
		case byte('\t'): strings.write_string(out, "\\t")
		case:
			if c < 0x20 {
				strings.write_string(out, fmt.tprintf("\\u%04x", c))
			} else {
				strings.write_byte(out, c)
			}
		}
	}
	strings.write_byte(out, byte('"'))
}

json_info :: proc() -> string {
	cfg := q35.engine_config(&g_state.engine)
	out: strings.Builder
	strings.builder_init(&out, context.temp_allocator)
	defer strings.builder_destroy(&out)
	strings.write_byte(&out, byte('{'))
	strings.write_string(&out, fmt.tprintf( "\"model\":"))
	json_escape(g_state.model_path, &out)
	strings.write_string(&out, fmt.tprintf( ",\"vocab\":%d,", cfg.vocab_size))
	strings.write_string(&out, fmt.tprintf( "\"dim\":%d,", cfg.dim))
	strings.write_string(&out, fmt.tprintf( "\"hidden_dim\":%d,", cfg.hidden_dim))
	strings.write_string(&out, fmt.tprintf( "\"n_layers\":%d,", cfg.n_layers))
	strings.write_string(&out, fmt.tprintf( "\"n_full\":%d,", cfg.n_full))
	strings.write_string(&out, fmt.tprintf( "\"n_linear\":%d,", cfg.n_linear))
	strings.write_string(&out, fmt.tprintf( "\"n_heads\":%d,", cfg.n_heads))
	strings.write_string(&out, fmt.tprintf( "\"n_kv_heads\":%d,", cfg.n_kv_heads))
	strings.write_string(&out, fmt.tprintf( "\"head_dim\":%d,", cfg.head_dim))
	strings.write_string(&out, fmt.tprintf( "\"seq_len\":%d,", cfg.seq_len))
	strings.write_string(&out, fmt.tprintf( "\"max_seq\":%d,", cfg.max_seq))
	strings.write_string(&out, fmt.tprintf("\"metal\":%v", g_state.engine.metal_ready))
	strings.write_byte(&out, byte('}'))
	return strings.to_string(out)
}

json_state :: proc() -> string {
	out: strings.Builder
	strings.builder_init(&out, context.temp_allocator)
	defer strings.builder_destroy(&out)
	strings.write_byte(&out, byte('{'))
	strings.write_string(&out, fmt.tprintf( "\"pos\":%d,", g_state.pos))
	strings.write_string(&out, fmt.tprintf( "\"last_tps\":%.2f,", g_state.last_tps))
	strings.write_string(&out, fmt.tprintf( "\"last_ttft_ms\":%d,", g_state.last_ttft))
	strings.write_string(&out, fmt.tprintf( "\"last_generated\":%d", g_state.last_gen))
	strings.write_byte(&out, byte('}'))
	return strings.to_string(out)
}

handle_chat :: proc(body: string) -> string {
	msg := extract_json_field(body, "message")
	if len(msg) == 0 {
		return `{"error": "missing 'message' field"}`
	}

	rendered := fmt.tprintf("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n", msg)
	encoded, err := tok35.encode(&g_state.tok, rendered)
	if err != nil { return `{"error": "encode failed"}` }

	if g_state.pos == 0 {
		if len(encoded) > 1 {
			_ = q35.engine_forward_batch(&g_state.engine, encoded[:len(encoded)-1], 0)
			g_state.pos = len(encoded) - 1
		}
	} else {
		if len(encoded) > 0 {
			_ = q35.engine_forward_batch(&g_state.engine, encoded, g_state.pos)
			g_state.pos += len(encoded)
		}
	}

	out: strings.Builder
	strings.builder_init(&out, context.allocator)
	defer strings.builder_destroy(&out)

	t_start := time_to_ms()
	t_first := t_start
	max_tokens := 256
	gen := 0
	next: int = 0

	if g_state.pos > 0 && len(encoded) > 0 {
		last := encoded[len(encoded) - 1]
		logits := q35.engine_forward(&g_state.engine, last, g_state.pos - 1)
		next = sampler.sample(&g_state.samp, logits)
	} else {
		return `{"error": "empty prompt"}`
	}

	for gen < max_tokens {
		if next == EOS { break }
		decoded := tok35.decode_token_id(&g_state.tok, next)
		strings.write_string(&out, decoded)
		delete(decoded)
		if gen == 0 { t_first = time_to_ms() }
		gen += 1
		g_state.pos += 1
		if g_state.pos >= q35.engine_config(&g_state.engine).seq_len { break }
		logits := q35.engine_forward(&g_state.engine, next, g_state.pos - 1)
		next = sampler.sample(&g_state.samp, logits)
	}

	ttft := t_first - t_start
	t_total := time_to_ms() - t_start
	tps := gen > 0 && t_total > 0 ? f64(gen) * 1000.0 / f64(t_total) : 0.0
	g_state.last_ttft = ttft
	g_state.last_tps = tps
	g_state.last_gen = gen

	resp_text := strings.to_string(out)
	// NOTE: resp_text is a view into `out`'s buffer (not a fresh allocation),
	// so we must NOT delete(resp_text) — builder_destroy(&out) below frees the
	// buffer. We only use resp_text inside this proc, before that defer fires.

	// Build response into a heap-owned string. The builder uses context.allocator
	// (NOT temp), and we copy out the final string so the caller doesn't hold a
	// dangling pointer after builder_destroy.
	rb: strings.Builder
	strings.builder_init(&rb, context.allocator)
	defer strings.builder_destroy(&rb)
	strings.write_string(&rb, "{")
	strings.write_string(&rb, "\"response\":")
	json_escape(resp_text, &rb)
	strings.write_string(&rb, fmt.tprintf(",\"tok_per_s\":%.2f", tps))
	strings.write_string(&rb, fmt.tprintf(",\"ttft_ms\":%d", ttft))
	strings.write_string(&rb, fmt.tprintf(",\"generated\":%d", gen))
	strings.write_string(&rb, fmt.tprintf(",\"pos\":%d", g_state.pos))
	strings.write_string(&rb, "}")
	// Clone to fresh allocation that survives builder_destroy
	final := strings.clone(strings.to_string(rb))
	return final
}

extract_json_field :: proc(json, field: string) -> string {
	key := fmt.tprintf("\"%s\":", field)
	idx := strings.index(json, key)
	if idx < 0 { return "" }
	i := idx + len(key)
	for i < len(json) && (json[i] == ' ' || json[i] == '\t') { i += 1 }
	if i >= len(json) || json[i] != '"' { return "" }
	i += 1
	out: strings.Builder
	strings.builder_init(&out, context.temp_allocator)
	defer strings.builder_destroy(&out)
	for i < len(json) {
		c := json[i]
		if c == '\\' && i + 1 < len(json) {
			i += 1
			switch json[i] {
			case 'n':  strings.write_byte(&out, byte('\n'))
			case 't':  strings.write_byte(&out, byte('\t'))
			case 'r':  strings.write_byte(&out, byte('\r'))
			case '"':  strings.write_byte(&out, byte('"'))
			case '\\': strings.write_byte(&out, byte('\\'))
			case '/':  strings.write_byte(&out, byte('/'))
			case:      strings.write_byte(&out, json[i])
			}
		} else if c == '"' {
			break
		} else {
			strings.write_byte(&out, c)
		}
		i += 1
	}
	return strings.clone(strings.to_string(out))
}

// ====================== OpenAI-compatible API ======================

// strings.index starting from byte offset `from`.
index_from :: proc(s, sub: string, from: int) -> int {
	if from >= len(s) { return -1 }
	rel := strings.index(s[from:], sub)
	if rel < 0 { return -1 }
	return from + rel
}

// Extract all "content":"..." strings that are inside "role":"..." objects.
// Very lenient: scans the messages array, extracts (role, content) pairs in
// order. Ignores everything else (tool_calls, function_call, name, etc).
openai_extract_messages :: proc(json: string) -> []Message_Pair {
	out: [dynamic]Message_Pair
	out = make([dynamic]Message_Pair, 0, 8)
	defer delete(out)
	// We scan for `"role":"X"` then look for the next `"content":"..."`.
	i := 0
	for i < len(json) - 8 {
		// find next "role"
		role_idx := index_from(json, `"role":"`, i)
		if role_idx < 0 { break }
		r_start := role_idx + len(`"role":"`)
		r_end := index_from(json, `"`, r_start)
		if r_end < 0 { break }
		role := json[r_start:r_end]
		// find next "content"
		c_idx := index_from(json, `"content":"`, r_end)
		if c_idx < 0 { break }
		c_start := c_idx + len(`"content":"`)
		// walk to matching closing quote, honoring escapes
		j := c_start
		content_buf: strings.Builder
		strings.builder_init(&content_buf, context.temp_allocator)
		for j < len(json) {
			ch := json[j]
			if ch == '\\' && j + 1 < len(json) {
				j += 1
				switch json[j] {
				case 'n':  strings.write_byte(&content_buf, byte('\n'))
				case 't':  strings.write_byte(&content_buf, byte('\t'))
				case 'r':  strings.write_byte(&content_buf, byte('\r'))
				case '"':  strings.write_byte(&content_buf, byte('"'))
				case '\\': strings.write_byte(&content_buf, byte('\\'))
				case '/':  strings.write_byte(&content_buf, byte('/'))
				case:      strings.write_byte(&content_buf, json[j])
				}
			} else if ch == '"' {
				break
			} else {
				strings.write_byte(&content_buf, ch)
			}
			j += 1
		}
		append(&out, Message_Pair{
			role = role,
			content = strings.clone(strings.to_string(content_buf)),
		})
		strings.builder_destroy(&content_buf)
		i = j + 1
	}
	// Copy dynamic to plain slice (caller owns; entries' strings are heap-allocated)
	result := make([]Message_Pair, len(out))
	for k in 0 ..< len(out) { result[k] = out[k] }
	return result
}

Message_Pair :: struct {
	role:    string,
	content: string,
}

free_message_pairs :: proc(msgs: []Message_Pair) {
	for m in msgs {
		delete(m.content)
	}
	delete(msgs)
}


openai_models_list :: proc() -> string {
	out: strings.Builder
	strings.builder_init(&out, context.temp_allocator)
	defer strings.builder_destroy(&out)
	strings.write_string(&out, `{"object":"list","data":[{"id":"odin-infer","object":"model","created":0,"owned_by":"local"}]}`)
	return strings.clone(strings.to_string(out))
}

handle_openai_chat :: proc(body: string) -> string {
	// Extract messages array
	msgs := openai_extract_messages(body)
	defer free_message_pairs(msgs)
	if len(msgs) == 0 {
		return `{"error":{"message":"no messages found","type":"invalid_request_error"}}`
	}

	// Optional: parse max_tokens (default 256), temperature (default 0.6)
	max_tokens := parse_int_field(body, "max_tokens", 256)
	temperature := parse_f32_field(body, "temperature", 0.6)
	_ = temperature  // currently unused; could set on sampler

	// Render conversation as Qwen chat template (stateless per request)
	prompt_buf: strings.Builder
	strings.builder_init(&prompt_buf, context.allocator)
	defer strings.builder_destroy(&prompt_buf)
	for m in msgs {
		// role normalization: tool/function -> user
		role := m.role
		if role != "system" && role != "user" && role != "assistant" {
			role = "user"
		}
		strings.write_string(&prompt_buf, fmt.tprintf("<|im_start|>%s\n%s<|im_end|>\n", role, m.content))
	}
	strings.write_string(&prompt_buf, "<|im_start|>assistant\n<think>\n\n</think>\n")

	prompt := strings.clone(strings.to_string(prompt_buf))
	defer delete(prompt)

	// RESET engine state for stateless OpenAI semantics
	q35.engine_reset_state(&g_state.engine)
	g_state.pos = 0

	encoded, err := tok35.encode(&g_state.tok, prompt)
	if err != nil {
		return `{"error":{"message":"encode failed","type":"internal_error"}}`
	}

	// Prefill all but last token
	if len(encoded) > 1 {
		_ = q35.engine_forward_batch(&g_state.engine, encoded[:len(encoded)-1], 0)
		g_state.pos = len(encoded) - 1
	}

	// Generate
	gen_buf: strings.Builder
	strings.builder_init(&gen_buf, context.allocator)
	defer strings.builder_destroy(&gen_buf)

	t_start := time_to_ms()
	t_first := t_start
	gen := 0
	next: int = 0

	if g_state.pos > 0 && len(encoded) > 0 {
		last := encoded[len(encoded) - 1]
		logits := q35.engine_forward(&g_state.engine, last, g_state.pos - 1)
		next = sampler.sample(&g_state.samp, logits)
	} else {
		return `{"error":{"message":"empty prompt","type":"internal_error"}}`
	}

	for gen < max_tokens {
		if next == EOS { break }
		decoded := tok35.decode_token_id(&g_state.tok, next)
		strings.write_string(&gen_buf, decoded)
		delete(decoded)
		if gen == 0 { t_first = time_to_ms() }
		gen += 1
		g_state.pos += 1
		if g_state.pos >= q35.engine_config(&g_state.engine).seq_len { break }
		logits := q35.engine_forward(&g_state.engine, next, g_state.pos - 1)
		next = sampler.sample(&g_state.samp, logits)
	}

	ttft := t_first - t_start
	t_total := time_to_ms() - t_start
	tps := gen > 0 && t_total > 0 ? f64(gen) * 1000.0 / f64(t_total) : 0.0
	g_state.last_ttft = ttft
	g_state.last_tps = tps
	g_state.last_gen = gen

	response_text := strings.to_string(gen_buf)

	// Build OpenAI-format response
	out: strings.Builder
	strings.builder_init(&out, context.allocator)
	defer strings.builder_destroy(&out)
	strings.write_string(&out, "{")
	strings.write_string(&out, fmt.tprintf(`"id":"chatcmpl-%d",`, t_start))
	strings.write_string(&out, `"object":"chat.completion",`)
	strings.write_string(&out, fmt.tprintf(`"created":%d,`, t_start / 1000))
	strings.write_string(&out, `"model":"odin-infer",`)
	strings.write_string(&out, `"choices":[{`)
	strings.write_string(&out, `"index":0,`)
	strings.write_string(&out, `"message":{"role":"assistant","content":`)
	json_escape(response_text, &out)
	strings.write_string(&out, "},")
	strings.write_string(&out, fmt.tprintf(`"finish_reason":"%s"`, next == EOS ? "stop" : "length"))
	strings.write_string(&out, "}],")
	strings.write_string(&out, `"usage":{`)
	strings.write_string(&out, fmt.tprintf(`"prompt_tokens":%d,`, len(encoded)))
	strings.write_string(&out, fmt.tprintf(`"completion_tokens":%d,`, gen))
	strings.write_string(&out, fmt.tprintf(`"total_tokens":%d`, len(encoded) + gen))
	strings.write_string(&out, "}}")
	final := strings.clone(strings.to_string(out))
	return final
}

parse_int_field :: proc(json, field: string, def: int) -> int {
	key := fmt.tprintf("\"%s\":", field)
	idx := strings.index(json, key)
	if idx < 0 { return def }
	i := idx + len(key)
	for i < len(json) && (json[i] == ' ' || json[i] == '\t') { i += 1 }
	v: int = 0
	got := false
	for i < len(json) && json[i] >= '0' && json[i] <= '9' {
		v = v * 10 + int(json[i] - '0')
		got = true
		i += 1
	}
	return got ? v : def
}

parse_f32_field :: proc(json, field: string, def: f32) -> f32 {
	key := fmt.tprintf("\"%s\":", field)
	idx := strings.index(json, key)
	if idx < 0 { return def }
	i := idx + len(key)
	for i < len(json) && (json[i] == ' ' || json[i] == '\t') { i += 1 }
	// scan a float
	start := i
	for i < len(json) && ((json[i] >= '0' && json[i] <= '9') || json[i] == '.' || json[i] == '-' || json[i] == '+' || json[i] == 'e' || json[i] == 'E') {
		i += 1
	}
	if start == i { return def }
	v, ok := strconv.parse_f64(json[start:i])
	if !ok { return def }
	return f32(v)
}

// ====================== entry ======================

main :: proc() {
	args := os.args
	if len(args) < 2 {
		fmt.eprintln("usage: serve <model.gguf> [--port N] [--metal] [--max-ctx N]")
		os.exit(1)
	}
	model_path := args[1]
	port: u16 = 9748
	use_metal := false
	max_ctx := 4096

	i := 2
	for i < len(args) {
		a := args[i]
		if a == "--port" && i + 1 < len(args) {
			v, _ := strconv_parse_int(args[i+1])
			port = u16(v)
			i += 2
		} else if a == "--metal" {
			use_metal = true
			i += 1
		} else if a == "--max-ctx" && i + 1 < len(args) {
			max_ctx, _ = strconv_parse_int(args[i+1])
			i += 2
		} else {
			i += 1
		}
	}

	g_state.model_path = model_path
	fmt.println("serve: loading engine…")
	e, ok := q35.engine_load(model_path, q35.Engine_Opts{
		max_ctx = max_ctx,
		use_metal = use_metal,
		num_threads = 8,
	})
	if !ok {
		fmt.eprintln("serve: engine_load failed")
		os.exit(1)
	}
	g_state.engine = e
	defer q35.engine_destroy(&g_state.engine)
	defer q35.destroy_matmul_pool()

	tok35.build_tokenizer(&g_state.tok)
	defer tok35.free_tokenizer(&g_state.tok)
	tok35.build_token_buffer(&g_state.tb)
	defer tok35.free_token_buffer(&g_state.tb)

	cfg := q35.engine_config(&g_state.engine)
	sampler.build_sampler(&g_state.samp, int(cfg.vocab_size), 0.6, 0.95, 0)
	defer sampler.free_sampler(&g_state.samp)

	fmt.printfln("serve: ready (vocab=%d, seq_len=%d, metal=%v)",
		cfg.vocab_size, cfg.seq_len, g_state.engine.metal_ready)

	if !start_server(port, handle_request) {
		os.exit(1)
	}
}

time_to_ms :: proc() -> i64 {
	return time.to_unix_nanoseconds(time.now()) / 1_000_000
}
