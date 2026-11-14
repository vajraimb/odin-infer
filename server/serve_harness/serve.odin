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
 *     -collection:sampler=. -collection:tokenizer=. \
 *     -collection:cuda=.
 *
 * Then open http://127.0.0.1:9748 in a browser.
 */

package serve

import "base:runtime"
import "core:fmt"
import "core:mem/virtual"
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

Server_Handler :: proc(client: net.TCP_Socket, req: ^Request, resp: ^Response) -> bool
// Returns true if the handler already wrote the response to the socket
// (e.g., streaming SSE) and the caller should NOT call write_full_response.

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
	// Retry with backoff: after a kill -9, lingering TIME_WAIT connections can
	// make bind() fail with Address_In_Use for up to ~60s (no SO_REUSEADDR in
	// core:net's listen_tcp). Most failures clear within 5-10s.
	sock: net.TCP_Socket
	err: net.Network_Error
	for attempt in 0 ..< 12 {
		sock, err = net.listen_tcp(addr, 16)
		if err == nil && sock != 0 {
			break
		}
		if attempt < 11 {
			time.sleep(5 * time.Second)
		}
	}
	if err != nil || sock == 0 {
		fmt.eprintf("serve: listen_tcp on port %d failed (in use? err=%v)\n", port, err)
		return false
	}
	defer net.close(sock)
	fmt.eprintfln("serve: listening on http://127.0.0.1:%d", port)

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
	t0 := time_to_ms()
	// Per-request arena: swap context.allocator for the request's duration and
	// release everything in one shot on exit — no per-allocation delete pairs,
	// and no dangling-pointer risk for request-scoped data. Requests run
	// serially on this thread (single accept loop) and the matmul pool's task
	// alloc/free both happen on this thread, so the unsynchronized bump
	// allocator is safe. Existing delete() calls on arena memory are harmless
	// no-ops and are kept for clarity.
	prev_alloc := context.allocator
	context.allocator = virtual.arena_allocator(&g_req_arena)
	defer {
		context.allocator = prev_alloc
		virtual.arena_free_all(&g_req_arena)
		// The default temp arena is never reset anywhere else in this process —
		// without this, every fmt.tprintf / temp-builder allocation in a request
		// accumulates forever (the "temp arena is never reset" note in
		// qwen3_5/kv_persist.odin documents one symptom of exactly that).
		free_all(context.temp_allocator)
	}

	// Arena-backed read buffer — Codex prompts can be 100KB+ (tools spec +
	// history). Pages stay reserved across requests after the first request.
	buf := make([]u8, 2_000_000, context.allocator)
	defer delete(buf)
	total := 0
	header_end := -1
	body_len_needed := 0  // expected body size from Content-Length
	chunked_request := false
	for total < len(buf) {
		// If we already have headers + full body, stop reading.
		if header_end >= 0 {
			if chunked_request {
				// check for end-of-chunks marker (0\r\n\r\n)
				if total >= header_end + 5 {
					// look for "0\r\n\r\n" in the body part
					body := buf[header_end:total]
					if _last_5_eq(body, "\r\n0\r\n\r\n") || _has_terminator(body) {
						break
					}
				}
			} else if total >= header_end + body_len_needed {
				break
			}
		}
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
				hdr_str := string(buf[:header_end])
				// Detect chunked transfer encoding
				if strings.contains(hdr_str, "Transfer-Encoding: chunked") ||
				   strings.contains(hdr_str, "transfer-encoding: chunked") {
					chunked_request = true
				}
				// Parse Content-Length if not chunked
				if !chunked_request {
					cl_idx := strings.index(hdr_str, "Content-Length:")
					if cl_idx < 0 { cl_idx = strings.index(hdr_str, "content-length:") }
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
	}
	if header_end < 0 {
		write_simple(client, 400, "text/plain", "no header end")
		return
	}

	req_str := string(buf[:header_end])
	t_read := time_to_ms()
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

	// Extract body
	if chunked_request {
		// Decode chunked body from buf[header_end:total]
		body := decode_chunked(buf[header_end:total])
		req.body = body
		defer delete(body)
	} else if body_len_needed > 0 && header_end + body_len_needed <= total {
		req.body = string(buf[header_end:header_end + body_len_needed])
	}
	defer free_request_safe(&req)


	resp: Response
	resp.status = 200
	resp.headers = make(map[string]string)
	defer delete_resp_headers(&resp)
	streamed := handler(client, &req, &resp)
	t1 := time_to_ms()
	if !streamed {
		write_full_response(client, &resp)
	}
	t2 := time_to_ms()
	fmt.eprintfln("[perf] client: read={}ms parse+handler={}ms write={}ms total={}ms", t_read-t0, t1-t_read, t2-t1, t2-t0)
}

// Look for chunked-body terminator "0\r\n\r\n" anywhere in body
_has_terminator :: proc(body: []u8) -> bool {
	if len(body) < 5 { return false }
	for i in 0 ..< len(body) - 4 {
		if body[i] == '\r' && body[i+1] == '\n' && body[i+2] == '0' && body[i+3] == '\r' && body[i+4] == '\n' {
			return true
		}
	}
	return false
}

_last_5_eq :: proc(body: []u8, suffix: string) -> bool {
	if len(body) < len(suffix) { return false }
	for i in 0 ..< len(suffix) {
		if body[len(body) - len(suffix) + i] != suffix[i] { return false }
	}
	return true
}

// Decode HTTP/1.1 chunked transfer encoding from raw bytes.
decode_chunked :: proc(raw: []u8) -> string {
	out: strings.Builder
	strings.builder_init(&out, context.allocator)
	i := 0
	for i < len(raw) {
		// read chunk size line (hex until \r\n)
		line_end := i
		for line_end + 1 < len(raw) && !(raw[line_end] == '\r' && raw[line_end+1] == '\n') {
			line_end += 1
		}
		size_str := string(raw[i:line_end])
		// strip any chunk extensions (;...)
		semi := strings.index(size_str, ";")
		if semi >= 0 { size_str = size_str[:semi] }
		size_str = strings.trim_space(size_str)
		chunk_size, ok := parse_hex(size_str)
		if !ok { break }
		i = line_end + 2  // skip \r\n
		if chunk_size == 0 { break }
		if i + chunk_size > len(raw) { break }
		strings.write_bytes(&out, raw[i:i + chunk_size])
		i += chunk_size + 2  // skip data + \r\n
	}
	return strings.to_string(out)
}

parse_hex :: proc(s: string) -> (int, bool) {
	v: int = 0
	got := false
	for c in s {
		switch {
		case c >= '0' && c <= '9':
			v = v * 16 + int(c - '0')
			got = true
		case c >= 'a' && c <= 'f':
			v = v * 16 + int(c - 'a') + 10
			got = true
		case c >= 'A' && c <= 'F':
			v = v * 16 + int(c - 'A') + 10
			got = true
		case:
			return 0, false
		}
	}
	return v, got
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
	// Prefix-cache state for stateless protocols (Ollama/OpenAI). Stores the
	// last fully-prefilled token sequence so multi-turn requests that extend
	// the previous conversation can skip re-prefilling the shared prefix.
	last_tokens: [dynamic]int,
	// System prompt prefix cache: snapshot of full engine state after prefilling
	// the system prompt + tools portion. Restored on subsequent requests to skip
	// re-prefilling ~1000-1500 fixed tokens. O(1) restore (~6ms memcpy).
	prefix_kv_k: []u8,
	prefix_kv_v: []u8,
	prefix_conv: []u8,
	prefix_rec:  []u8,
	prefix_tokens: [dynamic]int,  // token IDs that were prefilled before snapshot
	prefix_pos:  int,             // position at snapshot time
	// Hit-rate counters (Ollama handler outcomes). Exposed via /api/state so
	// prefix-cache effectiveness can be quantified from real traffic instead
	// of assumed. See docs/PREFIX_SNAPSHOT_PLAN.md Step 0a.
	pc_snapshot_hits:    int, // restored full-state snapshot at system boundary
	pc_continuation_hits: int, // extended last_tokens in place (multi-turn)
	pc_misses:           int, // full re-prefill from pos 0
}

g_state: Engine_State

// QDUMP_DIVERGE=1: on a full-prefill MISS, decode and print the token window
// around the divergence point vs last_tokens / prefix_tokens. Diagnostic for
// "why did both caches miss" — pinpoints volatile prompt segments.
g_dump_diverge: bool

// Concatenate the decoded text of a token-id window (diagnostics only).
// Uses the current allocator — inside the request handler that is the
// per-request arena, freed at request end.
detok_window :: proc(ids: []int) -> string {
	b := strings.builder_make(context.allocator)
	for id in ids {
		s := tok35.decode_token_id(&g_state.tok, id)
		strings.write_string(&b, s)
		delete(s, context.allocator)
	}
	return strings.to_string(b)
}

// Per-request arena (see handle_client): every allocation scoped to one HTTP
// request — the 2MB read buffer, header parsing, JSON building, per-token
// decode strings — bumps this pointer and is released in a single free_all
// when the request ends. The arena keeps its reserved pages across requests,
// so the read buffer stops paying mmap+page-fault cost per request. Anything
// that must OUTLIVE the request (prefix-cache snapshots, token histories, the
// tool-token cache, KV-persist staging) allocates on g_heap explicitly.
g_heap:      runtime.Allocator
g_req_arena: virtual.Arena

// ====================== HTML (single-page dashboard) ======================

INDEX_HTML := #load ("./index.html")

// ====================== request handlers ======================

handle_request :: proc(client: net.TCP_Socket, req: ^Request, resp: ^Response) -> bool {
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
		// Auto-detect: Ollama-style (has "messages" array) vs dashboard (has "message" string)
		if strings.contains(req.body, `"messages":`) {
			write_ndjson_headers(client)
			handle_ollama_chat_stream(client, req.body)
			return true
		}
		resp.headers["Content-Type"] = "application/json"
		resp.body = handle_chat(req.body)

	case req.method == "POST" && (req.path == "/v1/chat/completions" || req.path == "/chat/completions"):
		resp.headers["Content-Type"] = "application/json"
		resp.body = handle_openai_chat(req.body)

	case req.method == "POST" && req.path == "/v1/responses":
		// SSE streaming — write directly to socket, signal caller to skip
		write_sse_headers(client)
		handle_openai_responses_stream(client, req.body)
		return true

	case req.method == "POST" && req.path == "/api/chat-ollama":
		// deprecated — /api/chat auto-detects now. Kept for backward compat.
		write_ndjson_headers(client)
		handle_ollama_chat_stream(client, req.body)
		return true

	case req.method == "GET" && req.path == "/api/tags":
		resp.headers["Content-Type"] = "application/json"
		resp.body = `{"models":[{"name":"ornith","model":"ornith","size":0,"digest":"","modified":"","details":{"family":"qwen3_5","parameter_size":"9B","quantization_level":"Q4_K_M"}}]}`

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
	return false
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

// ====================== UTF-8 stream reassembly ======================
//
// Byte-level BPE tokens can split a multi-byte UTF-8 character across tokens
// (e.g. ☀ = U+2600 is E2 98 80, and the tokenizer may emit it as two tokens
// [E2 98] + [80]). decode_token_id returns RAW BYTES, so streaming one token
// per NDJSON/SSE line can put a truncated UTF-8 sequence on the wire — that
// line is then invalid UTF-8 AND invalid JSON (JSON text must be Unicode).
// Strict clients reject the whole line and fall back to printing raw bytes,
// which is what showed up as garbled characters in pie-odin.
//
// Fix: hold any incomplete trailing sequence in a small per-request buffer and
// prepend it to the next token's bytes, so every line on the wire is complete,
// valid UTF-8.

// Length of the longest prefix of s consisting of complete UTF-8 sequences.
// Bytes past the cut are a truncated multi-byte character to be held back.
utf8_complete_prefix_len :: proc(s: string) -> int {
	i := 0
	good := 0
	for i < len(s) {
		c := s[i]
		n := 1
		if c >= 0xF0 { n = 4 } else if c >= 0xE0 { n = 3 } else if c >= 0xC2 { n = 2 }
		// 0x80-0xC1 = stray continuation/overlong lead: treat as a single
		// pass-through byte so a malformed sequence can't stall the stream.
		if i + n > len(s) { break }
		ok := true
		for k in 1 ..< n {
			if s[i + k] & 0xC0 != 0x80 { ok = false; break }
		}
		if !ok {
			// Malformed lead — emit it alone rather than wedging the stream.
			i += 1
			good = i
			continue
		}
		i += n
		good = i
	}
	return good
}

// Append newly decoded token bytes to `pending` and return the longest prefix
// safe to emit now (complete UTF-8 only), heap-owned on the context allocator.
// The incomplete tail (0-3 bytes) stays in `pending` for the next call.
utf8_stream_take :: proc(pending: ^[dynamic]u8, s: string) -> string {
	append(pending, s)
	n := utf8_complete_prefix_len(string(pending[:]))
	if n == 0 { return "" }
	out := strings.clone(string(pending[:n]))
	remaining := len(pending) - n
	if remaining > 0 {
		copy(pending[:remaining], pending[n:])
	}
	resize(pending, remaining)
	return out
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
	strings.write_string(&out, fmt.tprintf("\"metal\":%v,", g_state.engine.metal_ready))
	strings.write_string(&out, fmt.tprintf("\"backend\":\"%s\"", q35.gpu_backend_name(g_state.engine.metal_ready)))
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
	strings.write_string(&out, fmt.tprintf( "\"last_generated\":%d,", g_state.last_gen))
	strings.write_string(&out, fmt.tprintf( "\"pc_snapshot_hits\":%d,", g_state.pc_snapshot_hits))
	strings.write_string(&out, fmt.tprintf( "\"pc_continuation_hits\":%d,", g_state.pc_continuation_hits))
	strings.write_string(&out, fmt.tprintf( "\"pc_misses\":%d", g_state.pc_misses))
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
		// Leave the last token for the per-token kickoff forward below;
		// forwarding it here too would feed it to the recurrent state twice.
		if len(encoded) > 1 {
			_ = q35.engine_forward_batch(&g_state.engine, encoded[:len(encoded)-1], g_state.pos)
			g_state.pos += len(encoded) - 1
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
		logits := q35.engine_forward(&g_state.engine, last, g_state.pos)
		next = sampler.sample(&g_state.samp, logits)
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
		logits := q35.engine_forward(&g_state.engine, next, g_state.pos)
		next = sampler.sample(&g_state.samp, logits)
	}

	save_session_state()

	ttft := t_first - t_start
	t_total := time_to_ms() - t_first  // decode only (excludes prefill)
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

// Parse 4 hex digits at s[i:i+4]; returns -1 when out of range or non-hex.
parse_hex4 :: proc(s: string, i: int) -> int {
	if i + 4 > len(s) { return -1 }
	v := 0
	for k in 0 ..< 4 {
		c := s[i + k]
		d := -1
		switch {
		case c >= '0' && c <= '9': d = int(c - '0')
		case c >= 'a' && c <= 'f': d = int(c - 'a') + 10
		case c >= 'A' && c <= 'F': d = int(c - 'A') + 10
		}
		if d < 0 { return -1 }
		v = v * 16 + d
	}
	return v
}

// Decode a JSON \uXXXX escape (s[i] == 'u') as UTF-8 into `out`, handling
// surrogate pairs. Standard JSON clients escape non-ASCII (Python json.dumps
// does it by default), so without this the model is fed literal "u4f60" text
// and starts mirroring escape-speak. Returns how many extra input bytes were
// consumed beyond the 'u' itself (4 for BMP, 10 for a surrogate pair, 0 when
// invalid — a literal 'u' is emitted then).
write_json_unicode_escape :: proc(out: ^strings.Builder, s: string, i: int) -> int {
	cp := parse_hex4(s, i + 1)
	if cp < 0 {
		strings.write_byte(out, 'u')
		return 0
	}
	consumed := 4
	if cp >= 0xD800 && cp <= 0xDBFF &&
	   i + 6 < len(s) && s[i + 5] == '\\' && s[i + 6] == 'u' {
		if lo := parse_hex4(s, i + 7); lo >= 0xDC00 && lo <= 0xDFFF {
			cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)
			consumed = 10
		}
	}
	strings.write_rune(out, rune(cp))
	return consumed
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
			case 'u':  i += write_json_unicode_escape(&out, json, i)
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

// Decode common JSON Unicode escape sequences (\uXXXX) back to UTF-8.
// Some models (esp. small quantized ones primed with JSON-heavy tool prompts)
// generate literal \uXXXX text instead of actual Unicode characters.
decode_unicode_escapes :: proc(s: string) -> string {
	if !strings.contains(s, "\\u") { return s }
	out: strings.Builder
	strings.builder_init(&out, context.temp_allocator)
	defer strings.builder_destroy(&out)
	i := 0
	for i < len(s) {
		if i + 5 < len(s) && s[i] == '\\' && s[i+1] == 'u' {
			// Parse 4 hex digits
			cp: u32 = 0
			valid := true
			for j in 0 ..< 4 {
				c := s[i + 2 + j]
				switch {
				case c >= '0' && c <= '9': cp = cp * 16 + u32(c - '0')
				case c >= 'a' && c <= 'f': cp = cp * 16 + u32(c - 'a') + 10
				case c >= 'A' && c <= 'F': cp = cp * 16 + u32(c - 'A') + 10
				case: valid = false
				}
			}
			if valid && cp > 0 {
				// Encode codepoint as UTF-8
				if cp < 0x80 {
					strings.write_byte(&out, byte(cp))
				} else if cp < 0x800 {
					strings.write_byte(&out, byte(0xC0 | (cp >> 6)))
					strings.write_byte(&out, byte(0x80 | (cp & 0x3F)))
				} else {
					strings.write_byte(&out, byte(0xE0 | (cp >> 12)))
					strings.write_byte(&out, byte(0x80 | ((cp >> 6) & 0x3F)))
					strings.write_byte(&out, byte(0x80 | (cp & 0x3F)))
				}
				i += 6
				continue
			}
		}
		strings.write_byte(&out, s[i])
		i += 1
	}
	return strings.clone(strings.to_string(out))
}

// ====================== N-gram speculative decoding ======================

NGRAM_N :: 3    // context length for n-gram lookup
SPEC_K :: 4    // max draft tokens per speculation step

Spec_State :: struct {
	ngram:    map[u64]u32,      // hash(N tokens) → predicted next token
	history:  [dynamic]int,     // rolling token history for n-gram extraction
	hits:     int,              // accepted speculations (for stats)
	misses:   int,              // rejected speculations
}

spec_init :: proc() -> Spec_State {
	s: Spec_State
	s.ngram = make(map[u64]u32, 8192)
	s.history = make([dynamic]int, 0, 4096)
	return s
}

spec_free :: proc(s: ^Spec_State) {
	delete(s.ngram)
	delete(s.history)
}

// Hash the last N tokens from history starting at given index.
ngram_hash_at :: proc(s: ^Spec_State, idx: int) -> u64 {
	if idx + NGRAM_N > len(s.history) { return 0 }
	h: u64 = 14695981039346656037
	for i in 0 ..< NGRAM_N {
		h ~= u64(s.history[idx + i])
		h *= 1099511628211
	}
	return h
}

// Record a token: append to history, update n-gram map.
spec_record :: proc(s: ^Spec_State, token: int) {
	if len(s.history) >= NGRAM_N {
		h := ngram_hash_at(s, len(s.history) - NGRAM_N)
		s.ngram[h] = u32(token)
	}
	append(&s.history, token)
}

// Seed from prompt tokens.
spec_seed :: proc(s: ^Spec_State, tokens: []int) {
	for t in tokens { spec_record(s, t) }
}

// Propose up to K draft tokens based on n-gram lookup.
// Uses the last NGRAM_N tokens from history as context.
spec_propose :: proc(s: ^Spec_State, k: int) -> []int {
	out: [dynamic]int
	out = make([dynamic]int, 0, k)
	defer delete(out)
	if len(s.history) < NGRAM_N { return out[:] }
	// Build a temporary extended history for chained lookup
	temp: [dynamic]int
	temp = make([dynamic]int, 0, NGRAM_N + k)
	defer delete(temp)
	// Start with last NGRAM_N tokens
	start := len(s.history) - NGRAM_N
	for i in 0 ..< NGRAM_N { append(&temp, s.history[start + i]) }
	// Chain lookup
	for _ in 0 ..< k {
		// Hash last NGRAM_N tokens of temp
		h: u64 = 14695981039346656037
		base := len(temp) - NGRAM_N
		for i in 0 ..< NGRAM_N {
			h ~= u64(temp[base + i])
			h *= 1099511628211
		}
		if tok, ok := s.ngram[h]; ok {
			append(&out, int(tok))
			append(&temp, int(tok))
		} else {
			break
		}
	}
	result := make([]int, len(out))
	for i in 0 ..< len(out) { result[i] = out[i] }
	return result
}

// Remove last n tokens from history (rollback on rejection).
spec_rollback :: proc(s: ^Spec_State, n: int) {
	for _ in 0 ..< n {
		if len(s.history) > 0 { _ = pop(&s.history) }
	}
}

// ====================== end N-gram speculation ======================

// ====================== Ollama-compatible API (pie-odin's odinfer provider) ======================

// Extract "tools" array from request body and format as Qwen3.5 tool calling prompt section.
// Returns the text to inject right after the system message.
extract_tools_section :: proc(body: string) -> string {
	// Find "tools" key, then scan to the [ that starts the array
	tools_key_idx := strings.index(body, `"tools"`)
	if tools_key_idx < 0 { return "" }
	// Skip past key, whitespace, colon, whitespace to find [
	i := tools_key_idx + len(`"tools"`)
	for i < len(body) && (body[i] == ' ' || body[i] == '\t' || body[i] == '\n' || body[i] == ':') { i += 1 }
	if i >= len(body) || body[i] != '[' { return "" }
	tools_start := i
	// Walk to matching ] (naive depth count)
	depth := 1
	i = tools_start + 1
	for i < len(body) && depth > 0 {
		if body[i] == '[' { depth += 1 }
		else if body[i] == ']' { depth -= 1 }
		i += 1
	}
	if depth != 0 { return "" }
	tools_json := body[tools_start:i]

	out: strings.Builder
	strings.builder_init(&out, context.temp_allocator)
	defer strings.builder_destroy(&out)
	strings.write_string(&out, "\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n")
	// Extract each function definition from the tools array
	// Format: each tool is {"type":"function","function":{"name":"...","description":"...","parameters":{...}}}
	scan_i := 0
	for scan_i < len(tools_json) {
		fn_idx := index_from(tools_json, `"function":`, scan_i)
		if fn_idx < 0 { break }
		// Find the { that starts the function object
		obj_start := fn_idx + len(`"function":`)
		for obj_start < len(tools_json) && tools_json[obj_start] == ' ' { obj_start += 1 }
		if obj_start >= len(tools_json) || tools_json[obj_start] != '{' { scan_i = fn_idx + 1; continue }
		// Walk to matching }
		depth_fn := 1
		obj_end := obj_start + 1
		for obj_end < len(tools_json) && depth_fn > 0 {
			if tools_json[obj_end] == '{' { depth_fn += 1 }
			else if tools_json[obj_end] == '}' { depth_fn -= 1 }
			obj_end += 1
		}
		fn_json := tools_json[obj_start:obj_end]  // {name, description, parameters}
		strings.write_string(&out, fn_json)
		strings.write_string(&out, "\n")
		scan_i = obj_end
	}
	// Match the model's native chat template (tokenizer.chat_template in the
	// GGUF): <function=name><parameter=key>value</parameter></function> nested
	// in <tool_call>. Instructing a different format makes the model improvise
	// (`memory(script: ...)`), which then can't be parsed back.
	strings.write_string(&out, `
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the user's question normally.
</IMPORTANT>
`)
	return strings.clone(strings.to_string(out))
}

// Parse model output for <tool_call>...</tool_call> blocks. Two payload forms
// are accepted:
//   1. JSON object {"name":"...","arguments":{...}} (returned as-is)
//   2. The model's native trained form:
//      <function=name>\n<parameter=key>\nvalue\n</parameter>\n</function>
//      which is converted to the same JSON shape.
// Returns a slice of JSON strings, each being the function call object
// like {"name":"...","arguments":{...}}
parse_tool_calls :: proc(text: string) -> []string {
	out: [dynamic]string
	out = make([dynamic]string, 0, 4)
	defer delete(out)
	i := 0
	for i < len(text) {
		out_len_before := len(out)
		tc_start := index_from(text, "<tool_call>", i)
		if tc_start < 0 { break }
		tc_start += len("<tool_call>")
		// Skip whitespace
		for tc_start < len(text) && (text[tc_start] == '\n' || text[tc_start] == ' ' || text[tc_start] == '\r') { tc_start += 1 }
		tc_end := index_from(text, "</tool_call>", tc_start)
		// Truncation salvage: <tool_call> opened but never closed (the model
		// hit max_tokens / context mid-call). Take the payload up to the end
		// of text instead of dropping the whole call — a native call is often
		// still parseable from its prefix.
		truncated := false
		if tc_end < 0 {
			tc_end = len(text)
			truncated = true
		}
		payload := strings.trim_space(text[tc_start:tc_end])
		if len(payload) > 0 {
			if payload[0] == '{' {
				// JSON form. Repair common 9B malformations (trailing comma,
				// missing close brace) before forwarding; if it's too broken
				// to salvage, drop it with a log instead of forwarding garbage
				// that the client's JSON parser will choke on.
				if repaired, ok := repair_tool_call_json(payload); ok {
					append(&out, repaired)
				} else {
					fmt.eprintfln("tool_call: dropped malformed JSON payload (len %d): %s", len(payload), payload[:min(len(payload), 120)])
				}
			} else if strings.contains(payload, "<function=") {
				if json_str, ok := native_tool_call_to_json(payload); ok {
					append(&out, json_str)
				} else {
					fmt.eprintfln("tool_call: native payload failed to convert (len %d): %s", len(payload), payload[:min(len(payload), 120)])
				}
			} else if truncated {
				fmt.eprintfln("tool_call: truncated <tool_call> with no <function=> (len %d): %s", len(payload), payload[:min(len(payload), 120)])
			}
		}
		if truncated {
			// A call was salvaged from a generation that never closed
			// </tool_call> — flag it so a truncated/partial argument is
			// traceable in the logs rather than looking like a clean call.
			if len(out) > out_len_before {
				fmt.eprintfln("tool_call: salvaged from truncated generation (no </tool_call>) — last call's args may be incomplete")
			}
			break  // consumed to end of text
		}
		i = tc_end + len("</tool_call>")
	}
	result := make([]string, len(out))
	for k in 0 ..< len(out) { result[k] = out[k] }
	return result
}

// Convert a native <function=name><parameter=key>value</parameter>...</function>
// payload into {"name":"name","arguments":{"key":"value",...}}. All values stay
// strings; the tool's own params parser interprets them.
native_tool_call_to_json :: proc(payload: string) -> (string, bool) {
	fs := strings.index(payload, "<function=")
	if fs < 0 { return "", false }
	name_start := fs + len("<function=")
	name_end := index_from(payload, ">", name_start)
	if name_end < 0 { return "", false }
	name := strings.trim_space(payload[name_start:name_end])
	if len(name) == 0 { return "", false }

	out: strings.Builder
	strings.builder_init(&out, context.temp_allocator)
	defer strings.builder_destroy(&out)
	strings.write_string(&out, `{"name":`)
	strings.write_string(&out, json_quote(name))
	strings.write_string(&out, `,"arguments":{`)

	n_params := 0
	pos := name_end
	for {
		ps := index_from(payload, "<parameter=", pos)
		if ps < 0 { break }
		key_start := ps + len("<parameter=")
		key_end := index_from(payload, ">", key_start)
		if key_end < 0 { break }
		key := strings.trim_space(payload[key_start:key_end])
		val_start := key_end + 1
		val_end := index_from(payload, "</parameter>", val_start)
		// Truncation salvage: if the model hit max_tokens before closing
		// </parameter>, take the value up to the end of the payload instead
		// of dropping the whole call. A trailing newline strip still applies.
		truncated_param := false
		if val_end < 0 {
			val_end = len(payload)
			truncated_param = true
		}
		value := payload[val_start:val_end]
		// Template puts the value on its own line — strip one leading and one
		// trailing newline, keep interior formatting intact.
		if len(value) > 0 && value[0] == '\n' { value = value[1:] }
		if len(value) > 0 && value[len(value) - 1] == '\n' { value = value[:len(value) - 1] }
		if n_params > 0 { strings.write_string(&out, ",") }
		strings.write_string(&out, json_quote(key))
		strings.write_string(&out, ":")
		strings.write_string(&out, json_quote(value))
		n_params += 1
		if truncated_param { break }
		pos = val_end + len("</parameter>")
	}
	strings.write_string(&out, "}}")
	// Zero-param salvage: the model named a function but emitted no parseable
	// parameters (forgot them, or truncated before the first value). Return
	// empty arguments so the tool gets CALLED and can report a missing-arg
	// error, instead of the call vanishing silently and stalling the agent.
	if n_params == 0 {
		fmt.eprintfln("tool_call: \"%s\" emitted with no parseable parameters — salvaging with empty args", name)
	}
	return strings.clone(strings.to_string(out)), true
}

// Repair common 9B-model JSON malformations in a {...} tool-call payload:
//   - trailing comma before a closing brace (the most frequent error)
//   - missing close braces from truncation (braces unbalanced, depth > 0)
// Drops (returns false) on structurally fatal cases: negative brace depth
// (more closes than opens) or an unterminated string (can't be salvaged
// safely — forwarding it would only move the parse failure to the client).
// serve.odin has no JSON parser available, so this is a single-pass
// structural heuristic, not a full validate.
repair_tool_call_json :: proc(payload: string) -> (string, bool) {
	if len(payload) == 0 || payload[0] != '{' { return "", false }
	out: [dynamic]u8
	out = make([dynamic]u8, 0, len(payload) + 4)
	defer delete(out)
	depth := 0
	in_str := false
	esc := false
	for i in 0 ..< len(payload) {
		c := payload[i]
		if in_str {
			append(&out, c)
			if esc { esc = false }
			else if c == '\\' { esc = true }
			else if c == '"' { in_str = false }
			continue
		}
		switch c {
		case '"': in_str = true
		case '{': depth += 1
		case '}':
			// Strip a trailing comma (with any whitespace) before this brace.
			for len(out) > 0 {
				last := out[len(out) - 1]
				if last == ' ' || last == '\t' || last == '\n' || last == '\r' { pop(&out) }
				else if last == ',' { pop(&out); break }
				else { break }
			}
			depth -= 1
		case:
		}
		append(&out, c)
	}
	if in_str { return "", false }  // unterminated string — unsafe to salvage
	// Strip a trailing comma left dangling at end of input, then re-balance.
	for len(out) > 0 {
		last := out[len(out) - 1]
		if last == ' ' || last == '\t' || last == '\n' || last == '\r' || last == ',' { pop(&out) }
		else { break }
	}
	if depth < 0 { return "", false }
	for _ in 0 ..< depth { append(&out, '}') }
	return strings.clone(string(out[:])), true
}

// Deterministic, model-free regression for parse_tool_calls + its helpers.
// Run with: ./serve.bin --self-test  (or `odin run serve_harness/serve.odin
// -file -- --self-test ...`). Locks in the hardening: truncation salvage
// (no </tool_call>), truncated-parameter salvage, zero-param salvage,
// malformed-JSON repair, multi-call, prose-before-call. Inner helper procs
// take all data as params and do NOT capture outer locals (Odin has no
// closures) — they only log + return pass/fail, aggregated by the caller.
run_parse_self_test :: proc() -> bool {
	eq_str :: proc(label, got, want: string) -> bool {
		if got == want do return true
		fmt.eprintfln("self-test FAIL: %s\n  want: %s\n  got:  %s", label, want, got)
		return false
	}
	has_sub :: proc(label: string, parsed: []string, needle: string) -> bool {
		for s in parsed {
			if strings.contains(s, needle) do return true
		}
		fmt.eprintfln("self-test FAIL: %s — parsed %d call(s), none contained \"%s\"", label, len(parsed), needle)
		return false
	}
	is_count :: proc(label: string, got, want: int) -> bool {
		if got == want do return true
		fmt.eprintfln("self-test FAIL: %s — want count %d, got %d", label, want, got)
		return false
	}

	pass, total := 0, 0

	// 1. Happy native form (the format ornith is trained on).
	r := parse_tool_calls("<tool_call>\n<function=weather>\n<parameter=city>\nsf\n</parameter>\n</function>\n</tool_call>")
	total += 1; if is_count("happy-native count", len(r), 1) do pass += 1
	total += 1; if has_sub("happy-native name", r, `"name":"weather"`) do pass += 1
	total += 1; if has_sub("happy-native arg value", r, `"city":"sf"`) do pass += 1

	// 2. Happy JSON form (inside a <tool_call> wrapper, like the native form).
	r = parse_tool_calls(`<tool_call>{"name":"weather","arguments":{"city":"sf"}}</tool_call>`)
	total += 1; if is_count("happy-json count", len(r), 1) do pass += 1
	total += 1; if has_sub("happy-json name", r, `"name":"weather"`) do pass += 1

	// 3. Truncation: <tool_call> with NO closing tag — must salvage, not drop.
	r = parse_tool_calls("<tool_call>\n<function=weather>\n<parameter=city>\nsf\n</parameter>\n</function>")
	total += 1; if is_count("truncated-no-close count", len(r), 1) do pass += 1
	total += 1; if has_sub("truncated-no-close name", r, `"name":"weather"`) do pass += 1

	// 4. Truncated parameter: <parameter> opened, never closed.
	r = parse_tool_calls("<tool_call><function=weather><parameter=city>san fran</tool_call>")
	total += 1; if is_count("truncated-param count", len(r), 1) do pass += 1
	total += 1; if has_sub("truncated-param salvaged value", r, `"city":"san fran"`) do pass += 1

	// 5. Zero parameters: function named, no params — salvage as empty args.
	r = parse_tool_calls("<tool_call><function=list_files></function></tool_call>")
	total += 1; if is_count("zero-param count", len(r), 1) do pass += 1
	total += 1; if eq_str("zero-param empty args", r[0], `{"name":"list_files","arguments":{}}`) do pass += 1

	// 6. Prose/reasoning before the call — still parsed.
	r = parse_tool_calls("Let me check that.\n<tool_call><function=weather><parameter=city>sf</parameter></function></tool_call>")
	total += 1; if is_count("prose-before count", len(r), 1) do pass += 1
	total += 1; if has_sub("prose-before name", r, `"name":"weather"`) do pass += 1

	// 7. Multiple tool calls in one generation.
	r = parse_tool_calls("<tool_call><function=a><parameter=x>1</parameter></function></tool_call>\n<tool_call><function=b><parameter=y>2</parameter></function></tool_call>")
	total += 1; if is_count("multi count", len(r), 2) do pass += 1

	// 8. Trailing-comma JSON — repair and keep.
	r = parse_tool_calls(`<tool_call>{"name":"weather","arguments":{"city":"sf",}}</tool_call>`)
	total += 1; if is_count("trailing-comma count", len(r), 1) do pass += 1
	total += 1; if has_sub("trailing-comma value", r, `"city":"sf"`) do pass += 1

	// 9. Truncated JSON object (closed strings, missing braces) — re-balance.
	r = parse_tool_calls(`<tool_call>{"name":"read","arguments":{"path":"/tmp",</tool_call>`)
	total += 1; if is_count("truncated-json salvage count", len(r), 1) do pass += 1
	total += 1; if has_sub("truncated-json name", r, `"name":"read"`) do pass += 1

	// 10. Truncated mid-string-value — can't be salvaged safely; drop it.
	r = parse_tool_calls(`<tool_call>{"name":"read","arguments":{"path":"/tmp</tool_call>`)
	total += 1; if is_count("mid-string-trunc dropped", len(r), 0) do pass += 1

	// 11. No tool call at all — empty result.
	r = parse_tool_calls("just a normal answer, no tools here")
	total += 1; if is_count("no-call count", len(r), 0) do pass += 1

	// ---- tool-call grammar fixtures (model-free state machine checks) ----
	feed_bytes :: proc(g: ^sampler.Tool_Call_Grammar, s: string) -> bool {
		for i in 0 ..< len(s) {
			if !sampler.tcg_advance_byte(g, s[i]) { return false }
		}
		return true
	}
	expect_true :: proc(label: string, v: bool) -> bool {
		if v do return true
		fmt.eprintfln("self-test FAIL: %s", label)
		return false
	}

	table: sampler.TCG_Table
	mem := sampler.tcg_table_add_tool(&table, "memory")
	sampler.tcg_tool_add_param(mem, "script")
	sampler.tcg_table_add_tool(&table, "list_directory") // no params → any key allowed
	sampler.tcg_table_add_tool(&table, "search")
	sampler.tcg_table_add_tool(&table, "search_web")

	// G1. Fresh grammar: EOS allowed in prose state.
	g: sampler.Tool_Call_Grammar
	sampler.tcg_init(&g, &table)
	total += 1; if expect_true("grammar: fresh is completable", sampler.tcg_is_completable(&g)) do pass += 1

	// G2. Prose + well-formed call accepted end-to-end; EOS allowed again after.
	sampler.tcg_init(&g, &table)
	ok := feed_bytes(&g, "好的，先记下来。\n<tool_call>\n<function=memory>\n<parameter=script>\nset m [recall user]\n</parameter>\n</function>\n</tool_call>")
	total += 1; if expect_true("grammar: accepts well-formed call", ok) do pass += 1
	total += 1; if expect_true("grammar: completable again after call", sampler.tcg_is_completable(&g)) do pass += 1

	// G3. Wrong function name rejected.
	sampler.tcg_init(&g, &table)
	ok = feed_bytes(&g, "<tool_call>\n<function=memX")
	total += 1; if expect_true("grammar: rejects bad func name", !ok) do pass += 1

	// G4. Wrong parameter key rejected (schema known).
	sampler.tcg_init(&g, &table)
	ok = feed_bytes(&g, "<tool_call>\n<function=memory>\n<parameter=wrong>")
	total += 1; if expect_true("grammar: rejects bad param key", !ok) do pass += 1

	// G5. Unknown-schema tool: any key allowed.
	sampler.tcg_init(&g, &table)
	ok = feed_bytes(&g, "<tool_call>\n<function=list_directory>\n<parameter=path>\n/tmp\n</parameter>\n</function>\n</tool_call>")
	total += 1; if expect_true("grammar: any key allowed when schema unknown", ok) do pass += 1

	// G6. Mid-call: EOS not allowed.
	sampler.tcg_init(&g, &table)
	_ = feed_bytes(&g, "<tool_call>\n<function=mem")
	total += 1; if expect_true("grammar: mid-call not completable", !sampler.tcg_is_completable(&g)) do pass += 1

	// G7. Prefix-name tools: both "search" and "search_web" committable.
	sampler.tcg_init(&g, &table)
	ok_a := feed_bytes(&g, "<tool_call>\n<function=search>\n</function>\n</tool_call>")
	sampler.tcg_init(&g, &table)
	ok_b := feed_bytes(&g, "<tool_call>\n<function=search_web>\n</function>\n</tool_call>")
	total += 1; if expect_true("grammar: prefix-name commit", ok_a && ok_b) do pass += 1

	// G8. Value length cap forces termination.
	sampler.tcg_init(&g, &table)
	_ = feed_bytes(&g, "<tool_call>\n<function=memory>\n<parameter=script>\n")
	capped := true
	for _ in 0 ..< sampler.TCG_MAX_VALUE {
		if !sampler.tcg_advance_byte(&g, 'a') { capped = false; break }
	}
	over := sampler.tcg_advance_byte(&g, 'b') // must be rejected past the cap
	fin := feed_bytes(&g, "\n</parameter>\n</function>\n</tool_call>")
	total += 1; if expect_true("grammar: value cap forces terminator", capped && !over && fin) do pass += 1

	// G9. Multiple calls in one generation.
	sampler.tcg_init(&g, &table)
	ok = feed_bytes(&g, "<tool_call>\n<function=search>\n</function>\n</tool_call>\n\n<tool_call>\n<function=memory>\n<parameter=script>\nx\n</parameter>\n</function>\n</tool_call>")
	total += 1; if expect_true("grammar: multiple calls in one generation", ok) do pass += 1

	fmt.eprintfln("self-test: %d/%d cases passed", pass, total)
	return pass == total
}

// ====================== end tool calling helpers ======================

// ====================== tool-call grammar constraint (VISION.md P0) ======================

g_tool_token_cache_ready: bool

decode_token_for_cache :: proc(id: int, allocator: runtime.Allocator) -> string {
	return tok35.decode_token_id(&g_state.tok, id, allocator)
}

// Build the token-id → bytes cache once (one-time cost over the full vocab);
// shared by the tool-call grammar mask on every structured token.
ensure_tool_token_cache :: proc() {
	if g_tool_token_cache_ready do return
	if g_state.samp.token_bytes == nil {
		t0 := time_to_ms()
		// The cache is built lazily inside a request but lives for the whole
		// process — allocate it on g_heap, not the per-request arena.
		prev := context.allocator
		context.allocator = g_heap
		g_state.samp.token_bytes = sampler.build_token_cache(g_state.samp.vocab_size, decode_token_for_cache)
		context.allocator = prev
		fmt.eprintfln("ollama: tool-grammar token cache built (%d tokens, %dms)",
			g_state.samp.vocab_size, time_to_ms() - t0)
	}
	g_tool_token_cache_ready = true
}

// Scan a JSON string starting at or after `start` (skips to the opening quote).
// Returns the inner text (escapes kept raw — tool names/keys are identifiers),
// the index past the closing quote, and success.
tcg_scan_json_string :: proc(text: string, start: int) -> (string, int, bool) {
	i := start
	for i < len(text) && text[i] != '"' { i += 1 }
	if i >= len(text) { return "", i, false }
	j := i + 1
	for j < len(text) {
		if text[j] == '\\' { j += 2; continue }
		if text[j] == '"' { return text[i+1:j], j+1, true }
		j += 1
	}
	return "", j, false
}

// Extract tool names + parameter keys from the request's "tools" array into a
// grammar table. Heuristic depth-1 scan of parameters.properties; when no
// properties are found the tool degrades to "any parameter key allowed".
// Returned strings alias `body`, which outlives the request handler.
extract_tool_specs :: proc(body: string, table: ^sampler.TCG_Table) {
	table.n_tools = 0
	search := 0
	for table.n_tools < sampler.TCG_MAX_TOOLS {
		rel := strings.index(body[search:], `"function"`)
		if rel < 0 { break }
		fi := search + rel
		// Find the { that opens the function object
		obj_start := -1
		for j := fi + len(`"function"`); j < len(body); j += 1 {
			c := body[j]
			if c == '{' { obj_start = j; break }
			if c == '}' || c == ']' { break }
		}
		if obj_start < 0 { break }
		// Walk to the matching close brace (string-aware)
		depth := 1
		j := obj_start + 1
		in_str := false
		for j < len(body) && depth > 0 {
			c := body[j]
			if in_str {
				if c == '\\' { j += 1 }
				else if c == '"' { in_str = false }
			} else {
				if c == '"' { in_str = true }
				else if c == '{' { depth += 1 }
				else if c == '}' { depth -= 1 }
			}
			j += 1
		}
		region := body[obj_start:j]
		search = j
		// name
		name := ""
		if ni := strings.index(region, `"name"`); ni >= 0 {
			s, _, ok := tcg_scan_json_string(region, ni + len(`"name"`))
			if ok { name = s }
		}
		if name == "" { continue }
		spec := sampler.tcg_table_add_tool(table, name)
		if spec == nil { break }
		// parameters.properties: depth-1 keys followed by ':'
		pi := strings.index(region, `"properties"`)
		if pi < 0 { continue }
		ps := -1
		for k := pi + len(`"properties"`); k < len(region); k += 1 {
			if region[k] == '{' { ps = k; break }
			if region[k] == '}' { break }
		}
		if ps < 0 { continue }
		d := 1
		k := ps + 1
		for k < len(region) && d > 0 {
			c := region[k]
			if c == '"' {
				s, e, ok := tcg_scan_json_string(region, k)
				if !ok { break }
				if d == 1 {
					m := e
					for m < len(region) && (region[m] == ' ' || region[m] == '\t' || region[m] == '\n' || region[m] == '\r') { m += 1 }
					if m < len(region) && region[m] == ':' {
						sampler.tcg_tool_add_param(spec, s)
					}
				}
				k = e
				continue
			}
			if c == '{' || c == '[' { d += 1 }
			else if c == '}' || c == ']' { d -= 1 }
			k += 1
		}
	}
}

// ====================== end tool-call grammar constraint ======================

// ====================== KV persistence across restarts (VISION.md P0) ======================
//
// Two rolling artifacts under g_kv_dir (default .kv-cache/, KV_PERSIST_DIR
// overrides, KV_PERSIST=0 disables):
//   prefix.oikv/.tokens  — full engine state at the system-prompt boundary,
//                          saved when the in-memory prefix snapshot is taken
//   session.oikv/.tokens — full engine state after each completed request,
//                          with the exact forwarded token list
// On startup both are restored: the prefix snapshot goes back into the
// in-memory cache (first request after a restart is a PREFIX CACHE HIT
// instead of a ~30s prefill) and the session state restores continuation
// bookkeeping. Corruption is refused via the kv file's CRC + fingerprint.

TOKENS_FILE_MAGIC :: u32(0x4F49544B) // 'OITK'

g_kv_dir: string // empty = persistence disabled

kv_persist_path :: proc(name: string) -> string {
	return fmt.tprintf("%s/%s", g_kv_dir, name)
}

write_tokens_file :: proc(path: string, tokens: []int) -> bool {
	put32 :: proc(buf: []u8, off: int, v: u32) {
		buf[off+0] = u8(v & 0xff);       buf[off+1] = u8((v >> 8) & 0xff)
		buf[off+2] = u8((v >> 16) & 0xff); buf[off+3] = u8((v >> 24) & 0xff)
	}
	buf := make([]u8, 8 + 4*len(tokens)) // heap: delete() pairs with context.allocator
	defer delete(buf)
	put32(buf, 0, TOKENS_FILE_MAGIC)
	put32(buf, 4, u32(len(tokens)))
	for t, i in tokens { put32(buf, 8 + 4*i, u32(t)) }
	f, err := os.open(path, os.O_CREATE | os.O_WRONLY | os.O_TRUNC)
	if err != os.ERROR_NONE { return false }
	defer os.close(f)
	n, werr := os.write(f, buf)
	return werr == os.ERROR_NONE && n == len(buf)
}

read_tokens_file :: proc(path: string) -> ([]int, bool) {
	get32 :: proc(buf: []u8, off: int) -> u32 {
		return u32(buf[off]) | u32(buf[off+1]) << 8 | u32(buf[off+2]) << 16 | u32(buf[off+3]) << 24
	}
	data, err := os.read_entire_file_from_path(path, context.allocator) // heap, freed below
	if err != os.ERROR_NONE || len(data) < 8 { return nil, false }
	defer delete(data)
	if get32(data, 0) != TOKENS_FILE_MAGIC { return nil, false }
	count := int(get32(data, 4))
	if len(data) != 8 + 4*count { return nil, false }
	out := make([]int, count) // long-lived (process lifetime), intentionally not freed
	for i in 0 ..< count { out[i] = int(get32(data, 8 + 4*i)) }
	return out, true
}

tokens_equal :: proc(a: []int, b: [dynamic]int) -> bool {
	if len(a) != len(b) { return false }
	for t, i in a { if t != b[i] { return false } }
	return true
}

// Called right after the in-memory prefix snapshot is taken (engine state sits
// exactly at prefix_pos there). Skips the ~140MB write when the cached system
// prefix is unchanged.
save_prefix_snapshot :: proc() {
	if len(g_kv_dir) == 0 || len(g_state.prefix_tokens) == 0 { return }
	// Called mid-request; the CRC re-read pulls the whole KV file (50-200MB)
	// through context.allocator — keep that on g_heap so the per-request
	// arena's reserved high-water mark isn't inflated by file-sized buffers.
	prev := context.allocator
	context.allocator = g_heap
	defer context.allocator = prev
	tokens_path := kv_persist_path("prefix.tokens")
	if old, ok := read_tokens_file(tokens_path); ok && tokens_equal(old, g_state.prefix_tokens) {
		return
	}
	tmp := kv_persist_path("prefix.oikv.tmp")
	if !q35.engine_save_kv(&g_state.engine, tmp, g_state.prefix_pos) {
		fmt.eprintln("kv-persist: prefix save failed")
		return
	}
	_ = os.rename(tmp, kv_persist_path("prefix.oikv"))
	if !write_tokens_file(tokens_path, g_state.prefix_tokens[:]) {
		fmt.eprintln("kv-persist: prefix tokens write failed")
		return
	}
	fmt.eprintfln("kv-persist: saved prefix snapshot (%d tokens)", g_state.prefix_pos)
}

// Called after a request completes (response already streamed). The engine
// coverage invariant is len(last_tokens) forwarded tokens.
save_session_state :: proc() {
	if len(g_kv_dir) == 0 || len(g_state.last_tokens) == 0 { return }
	// See save_prefix_snapshot: file-sized staging stays on g_heap.
	prev := context.allocator
	context.allocator = g_heap
	defer context.allocator = prev
	n_valid := len(g_state.last_tokens)
	tmp := kv_persist_path("session.oikv.tmp")
	if !q35.engine_save_kv(&g_state.engine, tmp, n_valid) {
		fmt.eprintln("kv-persist: session save failed")
		return
	}
	_ = os.rename(tmp, kv_persist_path("session.oikv"))
	if !write_tokens_file(kv_persist_path("session.tokens"), g_state.last_tokens[:]) {
		fmt.eprintln("kv-persist: session tokens write failed")
		return
	}
	fmt.eprintfln("kv-persist: saved session state (%d tokens)", n_valid)
}

// Called once at startup, after engine + tokenizer are ready. Failures degrade
// to a normal cold start (CRC/fingerprint/seq_len mismatches refuse the load).
restore_persisted_state :: proc() {
	if len(g_kv_dir) == 0 { return }

	// 1) System-prefix snapshot → back into the in-memory prefix cache.
	prefix_loaded := 0
	if pt, ok := read_tokens_file(kv_persist_path("prefix.tokens")); ok {
		n, lok := q35.engine_load_kv(&g_state.engine, kv_persist_path("prefix.oikv"))
		if lok && n == len(pt) {
			g_state.prefix_kv_k, g_state.prefix_kv_v,
				g_state.prefix_conv, g_state.prefix_rec =
				q35.engine_save_full_state(&g_state.engine)
			g_state.prefix_pos = n
			clear(&g_state.prefix_tokens)
			append(&g_state.prefix_tokens, ..pt)
			prefix_loaded = n
			fmt.eprintfln("kv-persist: restored prefix snapshot (%d tokens)", n)
		} else if lok {
			fmt.eprintfln("kv-persist: prefix tokens/state mismatch (%d vs %d) — ignoring", n, len(pt))
		}
	}

	// 2) Session continuation state → engine + bookkeeping.
	if st, ok := read_tokens_file(kv_persist_path("session.tokens")); ok {
		n, lok := q35.engine_load_kv(&g_state.engine, kv_persist_path("session.oikv"))
		if lok && n == len(st) {
			clear(&g_state.last_tokens)
			append(&g_state.last_tokens, ..st)
			g_state.pos = n - 1 // engine coverage n, pos = last forwarded position
			fmt.eprintfln("kv-persist: restored session state (%d tokens)", n)
			return
		}
		if lok {
			fmt.eprintfln("kv-persist: session tokens/state mismatch (%d vs %d) — ignoring", n, len(st))
		}
	}

	// No usable session — but if the prefix load above left the engine sitting
	// at the system boundary, keep that as the starting point.
	if prefix_loaded > 0 {
		clear(&g_state.last_tokens)
		append(&g_state.last_tokens, ..g_state.prefix_tokens[:])
		g_state.pos = prefix_loaded - 1
	} else if len(g_state.last_tokens) > 0 || g_state.pos != 0 {
		q35.engine_reset_state(&g_state.engine)
		g_state.pos = 0
	}
}

// ====================== end KV persistence ======================

// Pie-odin sends: POST /api/chat {"model":"ornith","messages":[...],"stream":true,"options":{...}}
// Expects NDJSON response: one JSON object per line, each {"message":{"role":"assistant","content":"delta"},"done":false}
// Final line: {"message":{"role":"assistant","content":""},"done":true,"done_reason":"stop",...}
handle_ollama_chat_stream :: proc(client: net.TCP_Socket, body: string) {
	// Extract messages from Ollama-format request
	msgs := openai_extract_messages(body)  // works for {"role":"X","content":"Y"} too
	defer free_message_pairs(msgs)
	if len(msgs) == 0 {
		ndjson_write(client, `{"error":"no messages"}`)
		ndjson_end(client)
		return
	}

	max_tokens := parse_int_field(body, "num_predict", 256)
	if max_tokens == 0 { max_tokens = 256 }
	// Honor Ollama options.temperature (previously parsed only on the OpenAI
	// path and ignored there). temperature=0 selects greedy argmax in the
	// sampler — required by the prefix-cache bit-identical regression test.
	g_state.samp.temperature = parse_f32_field(body, "temperature", 0.6)
	has_tools := strings.contains(body, `"tools"`) && strings.contains(body, `"function"`)

	// Syntax-constrained tool-call generation (P0): when the request carries
	// tools, mask sampling so the model cannot break the native <tool_call>
	// format. The serve-side parser stays as a fallback. TOOLCALL_GRAMMAR=0 disables.
	tcg_table: sampler.TCG_Table
	tcg: sampler.Tool_Call_Grammar
	if has_tools && os.get_env("TOOLCALL_GRAMMAR", context.temp_allocator) != "0" {
		extract_tool_specs(body, &tcg_table)
		if tcg_table.n_tools > 0 {
			ensure_tool_token_cache()
			sampler.tcg_init(&tcg, &tcg_table)
			g_state.samp.tool_grammar = &tcg
			defer g_state.samp.tool_grammar = nil
			fmt.eprintfln("ollama: tool-grammar ON (%d tools)", tcg_table.n_tools)
		}
	}

	// Render Qwen chat template
	prompt_buf: strings.Builder
	strings.builder_init(&prompt_buf, context.allocator)
	defer strings.builder_destroy(&prompt_buf)
	for m in msgs {
		role := m.role
		if role == "tool" {
			// Tool result → plain user turn. Probe verdict: this model does NOT
			// know the <tool_response> wrapper (loops with it, in either a tool
			// or user role); a plain "工具执行结果: ..." user message is the
			// only rendering that makes it consume the result and answer.
			strings.write_string(&prompt_buf, "<|im_start|>user\n工具执行结果: ")
			strings.write_string(&prompt_buf, m.content)
			strings.write_string(&prompt_buf, "<|im_end|>\n")
			continue
		}
		if role != "system" && role != "user" && role != "assistant" {
			role = "user"
		}
		if role == "system" && has_tools {
			// Tools section goes INSIDE the system block, matching the model's
			// native chat template (and letting the prefix snapshot cache it).
			strings.write_string(&prompt_buf, fmt.tprintf("<|im_start|>system\n%s%s<|im_end|>\n",
				m.content, extract_tools_section(body)))
			continue
		}
		if role == "assistant" {
			// Render past assistant turns with the same empty think stub the
			// generation prompt uses (line below: "<|im_start|>assistant\n
			// <think>\n\n</think>\n"). Without it, last_tokens (prompt incl.
			// stub + generated ids) and any follow-up request diverge right
			// after "assistant\n", so the continuation path could NEVER hit
			// in chat — every multi-turn request fell back to the system
			// snapshot and re-prefilled the whole conversation. Keeping the
			// stub in history makes the token stream structurally identical
			// between full prefill and continuation, preserving the
			// bit-identical invariant.
			strings.write_string(&prompt_buf, "<|im_start|>assistant\n<think>\n\n</think>\n")
			strings.write_string(&prompt_buf, m.content)
			// Re-render tool calls in JSON form inside <tool_call>. Dropping
			// them (the old behavior) hid the model's own calls from history
			// — it re-called the same tool forever; and the NATIVE form
			// (byte-identical to generation) still loops, per template probe.
			if len(m.tool_calls_raw) > 0 {
				render_json_tool_calls(&prompt_buf, m.tool_calls_raw)
			}
			strings.write_string(&prompt_buf, "<|im_end|>\n")
			continue
		}
		strings.write_string(&prompt_buf, fmt.tprintf("<|im_start|>%s\n%s<|im_end|>\n", role, m.content))
	}
	strings.write_string(&prompt_buf, "<|im_start|>assistant\n<think>\n\n</think>\n")

	prompt := strings.clone(strings.to_string(prompt_buf))
	defer delete(prompt)

	encoded, err := tok35.encode(&g_state.tok, prompt)
	if err != nil {
		ndjson_write(client, `{"error":"encode failed"}`)
		ndjson_end(client)
		return
	}

	// Prefix-cache: if new tokens start with the previous tokens, only
	// prefill the new tail. Else reset and prefill from scratch.
	common := 0
	max_common := min(len(encoded), len(g_state.last_tokens))
	for common < max_common && encoded[common] == g_state.last_tokens[common] {
		common += 1
	}

	// Locate the system-prompt boundary (first <|im_end|> + following "\n") so a
	// state snapshot can be taken exactly there. The linear-attention conv and
	// recurrent states evolve with position and are NOT rewindable, so a snapshot
	// is only sound when the engine has forwarded exactly prefix_pos tokens at
	// save time — saving later and resuming earlier replays the suffix tokens on
	// top of an already-advanced recurrent state and corrupts generation.
	snap_pos := 0
	for i in 3 ..< len(encoded) - 1 {
		if encoded[i] == EOS { snap_pos = i + 2; break }
	}
	snap_ok := snap_pos >= 5 && snap_pos < len(encoded)

	t_prefill_start := time_to_ms()
	t_reset_ms: i64 = 0

	// Does the incoming prompt start with the cached system-prefix tokens?
	// (Direct compare with cached prefix tokens, not common — common compares
	// against last request's FULL token set which may have diverged.)
	prefix_matches := false
	if len(g_state.prefix_tokens) > 0 && len(encoded) > len(g_state.prefix_tokens) {
		prefix_matches = true
		for i in 0 ..< len(g_state.prefix_tokens) {
			if encoded[i] != g_state.prefix_tokens[i] { prefix_matches = false; break }
		}
	}

	// Continuation — sound only when the engine state sits exactly at the
	// divergence point. g_state.pos is the INDEX of the last forwarded token,
	// i.e. (tokens forwarded so far) - 1, so the invariant is pos == common-1
	// (an earlier pos == common check was off by one and could never fire).
	// Otherwise the non-rewindable recurrent state would absorb the tail
	// tokens a second time.
	can_continue := common == len(g_state.last_tokens) && common > 0 &&
		common < len(encoded) && g_state.pos == common - 1

	// Diagnostics: dump the divergence window whenever continuation was
	// plausible (shared prefix with last_tokens) but failed — regardless of
	// which fallback path (snapshot / full prefill) ends up serving.
	if g_dump_diverge && !can_continue && common > 0 && common < len(g_state.last_tokens) {
		lo := max(0, common - 6)
		hi_new := min(len(encoded), common + 10)
		hi_old := min(len(g_state.last_tokens), common + 10)
		fmt.eprintfln("DIVERGE vs last_tokens @%d/%d (pos=%d):", common, len(g_state.last_tokens), g_state.pos)
		fmt.eprintfln("  new: %q", detok_window(encoded[lo:hi_new]))
		fmt.eprintfln("  old: %q", detok_window(g_state.last_tokens[lo:hi_old]))
	}

	// Prefer continuation over the system-boundary snapshot whenever it reuses
	// at least as many tokens — extending the live state skips re-prefilling
	// the entire conversation history, not just the system prompt. (The
	// snapshot check used to run first, so continuation was unreachable
	// whenever a snapshot existed.)
	if can_continue && (!prefix_matches || common >= g_state.prefix_pos) {
		new_tokens := encoded[common:len(encoded) - 1]
		if len(new_tokens) > 0 {
			_ = q35.engine_forward_batch(&g_state.engine, new_tokens, common)
		}
		g_state.pos = len(encoded) - 1
		g_state.pc_continuation_hits += 1
		fmt.eprintfln("ollama: continuation hit (%d/%d tokens reused), prefilling only %d new",
			common, len(encoded), len(new_tokens))
	} else if prefix_matches {
		// Restore full state from snapshot — instantly positions us at prefix_pos
		q35.engine_restore_full_state(&g_state.engine,
			g_state.prefix_kv_k, g_state.prefix_kv_v,
			g_state.prefix_conv, g_state.prefix_rec)
		g_state.pos = g_state.prefix_pos
		// Now prefill only the tokens AFTER the cached prefix
		suffix_tokens := encoded[g_state.prefix_pos:len(encoded) - 1]  // leave last for per-token fwd
		if len(suffix_tokens) > 0 {
			_ = q35.engine_forward_batch(&g_state.engine, suffix_tokens, g_state.prefix_pos)
		}
		g_state.pos = len(encoded) - 1
		g_state.pc_snapshot_hits += 1
		fmt.eprintfln("ollama: PREFIX CACHE HIT (restored %d tokens, prefilling %d new)",
			g_state.prefix_pos, len(suffix_tokens))
	} else {
			// Diagnostics: why did both caches miss? Dump the token window
			// around the divergence vs last_tokens and vs prefix_tokens.
			if g_dump_diverge {
				if common > 0 && common < min(len(encoded), len(g_state.last_tokens)) {
					lo := max(0, common - 6)
					hi_new := min(len(encoded), common + 10)
					hi_old := min(len(g_state.last_tokens), common + 10)
					fmt.eprintfln("DIVERGE vs last_tokens @%d/%d:", common, len(g_state.last_tokens))
					fmt.eprintfln("  new: %q", detok_window(encoded[lo:hi_new]))
					fmt.eprintfln("  old: %q", detok_window(g_state.last_tokens[lo:hi_old]))
				}
				if len(g_state.prefix_tokens) > 0 {
					d := 0
					max_d := min(len(encoded), len(g_state.prefix_tokens))
					for d < max_d && encoded[d] == g_state.prefix_tokens[d] { d += 1 }
					if d < len(g_state.prefix_tokens) {
						lo := max(0, d - 6)
						hi_new := min(len(encoded), d + 10)
						hi_old := min(len(g_state.prefix_tokens), d + 10)
						fmt.eprintfln("DIVERGE vs prefix_tokens @%d/%d:", d, len(g_state.prefix_tokens))
						fmt.eprintfln("  new: %q", detok_window(encoded[lo:hi_new]))
						fmt.eprintfln("  old: %q", detok_window(g_state.prefix_tokens[lo:hi_old]))
					}
				}
			}
			// Reset and full prefill. When the system boundary is known, prefill
			// up to it FIRST and snapshot the state at exactly that position,
			// then prefill the rest.
			t_r0 := time_to_ms()
			q35.engine_reset_state(&g_state.engine)
			t_reset_ms = time_to_ms() - t_r0
			g_state.pos = 0

			prefill_from := 0
			if snap_ok {
				_ = q35.engine_forward_batch(&g_state.engine, encoded[:snap_pos], 0)
				g_state.pos = snap_pos
				prefill_from = snap_pos

				// Free old snapshot, take the new one at the boundary. The
				// snapshot buffers (~140MB) must outlive this request — allocate
				// and free them on g_heap, never the per-request arena.
				{
					prev := context.allocator
					context.allocator = g_heap
					if len(g_state.prefix_kv_k) > 0 {
						delete(g_state.prefix_kv_k); delete(g_state.prefix_kv_v)
						delete(g_state.prefix_conv); delete(g_state.prefix_rec)
					}
					g_state.prefix_kv_k, g_state.prefix_kv_v,
						g_state.prefix_conv, g_state.prefix_rec =
						q35.engine_save_full_state(&g_state.engine)
					context.allocator = prev
				}
				g_state.prefix_pos = snap_pos
				clear(&g_state.prefix_tokens)
				for i in 0 ..< snap_pos { append(&g_state.prefix_tokens, encoded[i]) }
				fmt.eprintfln("ollama: prefix snapshot at system boundary pos=%d (%d tokens cached)",
					snap_pos, snap_pos)
				save_prefix_snapshot()
			}

			if len(encoded) - 1 > prefill_from {
				_ = q35.engine_forward_batch(&g_state.engine, encoded[prefill_from:len(encoded) - 1], prefill_from)
			}
			g_state.pos = len(encoded) - 1
			g_state.pc_misses += 1
			fmt.eprintfln("ollama: full prefill (%d tokens, common=%d)", len(encoded), common)
		}
	t_prefill_ms := time_to_ms() - t_prefill_start

	// Cache this turn's tokens for next time
	clear(&g_state.last_tokens)
	append(&g_state.last_tokens, ..encoded)

	gen := 0
	next: int = 0
	t_gen_start := time_to_ms()

	// When tools are present, buffer full output for tool_call parsing (no streaming).
	// Otherwise stream normally.
	gen_buf: strings.Builder
	strings.builder_init(&gen_buf, context.allocator)
	defer strings.builder_destroy(&gen_buf)

	// BPE tokens can split a multi-byte UTF-8 character (e.g. ☀ as [E2 98]+[80]).
	// Hold an incomplete trailing sequence here and prepend it to the next
	// token's bytes so every streamed NDJSON line is valid UTF-8 / valid JSON.
	utf8_pending: [dynamic]u8
	defer delete(utf8_pending)

	if g_state.pos > 0 && len(encoded) > 0 {
		last := encoded[len(encoded) - 1]
		logits := q35.engine_forward(&g_state.engine, last, g_state.pos)
		next = sampler.sample(&g_state.samp, logits)
	}

	finish_reason := "stop"
	for gen < max_tokens {
		if next == EOS { finish_reason = "stop"; break }
		decoded := tok35.decode_token_id(&g_state.tok, next)
		if len(decoded) > 0 {
			strings.write_string(&gen_buf, decoded)
			// Stream immediately (per-token NDJSON), emitting only complete
			// UTF-8 sequences — see utf8_stream_take.
			emit := utf8_stream_take(&utf8_pending, decoded)
			if len(emit) > 0 {
				line := concat_json(
					`{"message":{"role":"assistant","content":`, json_quote(emit), `},"done":false}`)
				ndjson_write(client, line)
				delete(emit)
			}
		}
		delete(decoded)
		gen += 1
		g_state.pos += 1
		if g_state.pos >= q35.engine_config(&g_state.engine).seq_len {
			finish_reason = "length"
			break
		}
		logits := q35.engine_forward(&g_state.engine, next, g_state.pos)
		// Track forwarded tokens so the next request can use the continuation path.
		append(&g_state.last_tokens, next)
		next = sampler.sample(&g_state.samp, logits)
	}

	// Flush any dangling partial sequence. Shouldn't happen (generated text
	// ends at a character boundary), but never strand bytes.
	if len(utf8_pending) > 0 {
		line := concat_json(
			`{"message":{"role":"assistant","content":`, json_quote(string(utf8_pending[:])), `},"done":false}`)
		ndjson_write(client, line)
	}

	// Decode the full accumulated text for tool-call parsing (not for streaming —
	// streamed tokens were already emitted raw above).
	full_text := decode_unicode_escapes(strings.to_string(gen_buf))

	// When tools are present, parse for <tool_call> blocks
	if has_tools {
		tool_calls_json := parse_tool_calls(full_text)
		if len(tool_calls_json) > 0 {
			// Emit each tool_call as separate NDJSON line
			for idx in 0 ..< len(tool_calls_json) {
				tc := tool_calls_json[idx]
				line := concat_json(
					`{"message":{"role":"assistant","content":"","tool_calls":[{"index":`, fmt.tprintf("%d", idx), `,"id":"call_`, fmt.tprintf("%d", idx), `","function":`, tc, `}]},"done":false}`)
				ndjson_write(client, line)
			}
			finish_reason = "tool_calls"
			fmt.eprintfln("ollama: %d tool calls detected", len(tool_calls_json))
		} else {
			// Tools were offered but nothing parseable came back. Distinguish a
			// real parse failure (model emitted <tool_call>/<function=> but it
			// was malformed/truncated) from the model simply choosing to answer
			// in prose. Either way, log a trimmed snippet so the failure isn't
			// a black box — this is the signal that the tool-prompt format
			// drifted or that max_tokens truncated a call.
			snippet_end := min(len(full_text), 160)
			if strings.contains(full_text, "<tool_call>") || strings.contains(full_text, "<function=") {
				fmt.eprintfln("ollama: PARSE FAILURE — model emitted a tool-call tag but none parsed (gen=%d, finish=%s): %s",
					gen, finish_reason, full_text[:snippet_end])
			} else {
				fmt.eprintfln("ollama: tools offered, model answered in prose (gen=%d, finish=%s): %s",
					gen, finish_reason, full_text[:snippet_end])
			}
		}
	} else {
		// No tools — text was already streamed per-token above. Nothing to emit here.
	}

	// Final done line with usage stats
	final_line := concat_json(
		`{"message":{"role":"assistant","content":""},"done":true,"done_reason":"`, finish_reason, `","input_tokens":`, fmt.tprintf("%d", len(encoded)), `,"output_tokens":`, fmt.tprintf("%d", gen), `}`)
	ndjson_write(client, final_line)
	ndjson_end(client)
	t_gen_ms := time_to_ms() - t_gen_start
	tps := gen > 0 && t_gen_ms > 0 ? f64(gen) * 1000.0 / f64(t_gen_ms) : 0.0
	fmt.eprintfln("ollama: %d in / %d out | reset=%dms prefill=%dms gen=%dms (%.1f tok/s)",
		len(encoded), gen, t_reset_ms, t_prefill_ms, t_gen_ms, tps)
	save_session_state()
}

// ====================== OpenAI Responses API (streaming, for Codex) ======================

// Write HTTP/1.1 headers for SSE streaming response (chunked encoding).
write_sse_headers :: proc(client: net.TCP_Socket) {
	hdr := "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n"
	_, _ = net.send_tcp(client, transmute([]byte)hdr)
}

// Write HTTP/1.1 headers for NDJSON streaming (Ollama-compatible).
write_ndjson_headers :: proc(client: net.TCP_Socket) {
	hdr := "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nTransfer-Encoding: chunked\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n"
	_, _ = net.send_tcp(client, transmute([]byte)hdr)
}

// Write one NDJSON line as a single HTTP chunk.
ndjson_write :: proc(client: net.TCP_Socket, json_line: string) {
	// Each NDJSON line: <json>\n  wrapped in chunked encoding
	body := fmt.tprintf("%s\n", json_line)
	chunk := fmt.tprintf("%x\r\n%s\r\n", len(body), body)
	_, _ = net.send_tcp(client, transmute([]byte)chunk)
}

ndjson_end :: proc(client: net.TCP_Socket) {
	end_marker := "0\r\n\r\n"
	_, _ = net.send_tcp(client, transmute([]byte)end_marker)
}

// Write one SSE event as a single HTTP/1.1 chunk.
sse_write :: proc(client: net.TCP_Socket, event_name, data: string) {
	// SSE event format uses CRLF line endings.
	body := fmt.tprintf("event: %s\r\ndata: %s\r\n\r\n", event_name, data)
	// Wrap in chunked encoding: <hex_size>\r\n<data>\r\n
	chunk := fmt.tprintf("%x\r\n%s\r\n", len(body), body)
	_, _ = net.send_tcp(client, transmute([]byte)chunk)
}

// End the chunked stream (write terminating 0-length chunk).
sse_end :: proc(client: net.TCP_Socket) {
	end_marker := "0\r\n\r\n"
	_, _ = net.send_tcp(client, transmute([]byte)end_marker)
}

handle_openai_responses_stream :: proc(client: net.TCP_Socket, body: string) {
	input_str := extract_responses_input(body)
	max_tokens := parse_int_field(body, "max_output_tokens", 256)

	resp_id := fmt.tprintf("resp_%d", time_to_ms())

	if len(input_str) == 0 {
		sse_write(client, "error", `{"error":"no input"}`)
		return
	}

	// RESET engine state per request
	q35.engine_reset_state(&g_state.engine)
	g_state.pos = 0

	rendered := fmt.tprintf("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n", input_str)
	encoded, err := tok35.encode(&g_state.tok, rendered)
	if err != nil {
		sse_write(client, "error", `{"error":"encode failed"}`)
		return
	}
	if len(encoded) > 1 {
		_ = q35.engine_forward_batch(&g_state.engine, encoded[:len(encoded)-1], 0)
		g_state.pos = len(encoded) - 1
	}

	// Opening events — note: each data MUST include "type" matching event name
	sse_write(client, "response.created",
		concat_json(
			`{"type":"response.created","response":{`,
			`"id":"`, resp_id, `","object":"response","status":"in_progress","model":"odinfer"}}`))
	sse_write(client, "response.output_item.added",
		concat_json(
			`{"type":"response.output_item.added","output_index":0,"item":`,
			`{"id":"msg_0","type":"message","role":"assistant","status":"in_progress","content":[]}}`))
	sse_write(client, "response.content_part.added",
		concat_json(
			`{"type":"response.content_part.added","output_index":0,"content_index":0,"part":`,
			`{"type":"output_text","text":"","annotations":[]}}`))

	// Generate tokens, streaming deltas as they arrive
	gen := 0
	next: int = 0
	full_text: strings.Builder
	strings.builder_init(&full_text, context.allocator)
	defer strings.builder_destroy(&full_text)

	// Hold incomplete trailing UTF-8 across tokens so every SSE delta is
	// valid UTF-8 / valid JSON (BPE can split a multi-byte char — see
	// utf8_stream_take).
	utf8_pending: [dynamic]u8
	defer delete(utf8_pending)

	if g_state.pos > 0 && len(encoded) > 0 {
		last := encoded[len(encoded) - 1]
		logits := q35.engine_forward(&g_state.engine, last, g_state.pos)
		next = sampler.sample(&g_state.samp, logits)
	}

	for gen < max_tokens {
		if next == EOS { break }
		decoded := tok35.decode_token_id(&g_state.tok, next)
		if len(decoded) > 0 {
			strings.write_string(&full_text, decoded)
			// emit delta event with type field (complete UTF-8 only)
			emit := utf8_stream_take(&utf8_pending, decoded)
			if len(emit) > 0 {
				delta_json := concat_json(
					`{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":`,
					json_quote(emit), `}`)
				sse_write(client, "response.output_text.delta", delta_json)
				delete(emit)
			}
		}
		delete(decoded)
		gen += 1
		g_state.pos += 1
		if g_state.pos >= q35.engine_config(&g_state.engine).seq_len { break }
		logits := q35.engine_forward(&g_state.engine, next, g_state.pos)
		next = sampler.sample(&g_state.samp, logits)
	}

	// Flush any dangling partial sequence before the closing events.
	if len(utf8_pending) > 0 {
		delta_json := concat_json(
			`{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":`,
			json_quote(string(utf8_pending[:])), `}`)
		sse_write(client, "response.output_text.delta", delta_json)
	}

	final_text := strings.to_string(full_text)

	// Closing events
	sse_write(client, "response.output_text.done",
		concat_json(`{"type":"response.output_text.done","output_index":0,"content_index":0,"text":`, json_quote(final_text), `}`))
	sse_write(client, "response.content_part.done",
		concat_json(`{"type":"response.content_part.done","output_index":0,"content_index":0,"part":{"type":"output_text","text":`, json_quote(final_text), `}}`))
	sse_write(client, "response.output_item.done",
		concat_json(`{"type":"response.output_item.done","output_index":0,"item":{"id":"msg_0","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":`, json_quote(final_text), `,"annotations":[]}]}}`))
	sse_write(client, "response.completed",
		concat_json(
			`{"type":"response.completed","response":{"id":"`, resp_id, `","object":"response","status":"completed","model":"odinfer","output":[{"id":"msg_0","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":`, json_quote(final_text), `,"annotations":[]}]}],"usage":{"input_tokens":`, fmt.tprintf("%d", len(encoded)), `,"output_tokens":`, fmt.tprintf("%d", gen), `,"total_tokens":`, fmt.tprintf("%d", len(encoded) + gen), `}}}`))

	sse_end(client)
	fmt.eprintfln("responses: %d in / %d out", len(encoded), gen)
}

// Format a single SSE event.
sse_event :: proc(event_name, data: string) -> string {
	return fmt.tprintf("event: %s\ndata: %s\n\n", event_name, data)
}

// Quote a string as JSON (just surrounding quotes — assumes already escaped).
json_quote :: proc(s: string) -> string {
	out: strings.Builder
	strings.builder_init(&out, context.temp_allocator)
	defer strings.builder_destroy(&out)
	json_escape(s, &out)
	return strings.clone(strings.to_string(out))
}

// Concatenate strings into one heap-owned string.
concat_json :: proc(parts: ..string) -> string {
	out: strings.Builder
	strings.builder_init(&out, context.temp_allocator)
	defer strings.builder_destroy(&out)
	for p in parts {
		strings.write_string(&out, p)
	}
	return strings.clone(strings.to_string(out))
}

// Extract "input" from Responses API request. Input can be:
//   "input": "string"
//   "input": [{"role":"user","content":"string"}]
//   "input": [{"role":"user","content":[{"type":"input_text","text":"string"}]}]
extract_responses_input :: proc(body: string) -> string {
	// Find "input":
	idx := strings.index(body, `"input":`)
	if idx < 0 { return "" }
	i := idx + len(`"input":`)
	// skip ws
	for i < len(body) && (body[i] == ' ' || body[i] == '\t') { i += 1 }
	if i >= len(body) { return "" }
	if body[i] == '"' {
		// string form
		return extract_json_field(body, "input")
	}
	if body[i] == '[' {
		// array form — extract last user message text
		// Look for the last "text":"..." or "content":"..." near end
		// Strategy: find all "text":"..." occurrences, take the last one
		last_text := ""
		scan_i := i
		for scan_i < len(body) - 8 {
			t_idx := strings.index(body[scan_i:], `"text":"`)
			if t_idx < 0 { break }
			real_idx := scan_i + t_idx + len(`"text":"`)
			// walk to closing quote
			end := real_idx
			for end < len(body) {
				if body[end] == '\\' { end += 2; continue }
				if body[end] == '"' { break }
				end += 1
			}
			if end > real_idx {
				// decode escapes
				last_text = decode_json_string(body[real_idx:end])
			}
			scan_i = end + 1
		}
		return last_text
	}
	return ""
}

decode_json_string :: proc(s: string) -> string {
	out: strings.Builder
	strings.builder_init(&out, context.temp_allocator)
	defer strings.builder_destroy(&out)
	i := 0
	for i < len(s) {
		c := s[i]
		if c == '\\' && i + 1 < len(s) {
			i += 1
			switch s[i] {
			case 'n':  strings.write_byte(&out, byte('\n'))
			case 't':  strings.write_byte(&out, byte('\t'))
			case 'r':  strings.write_byte(&out, byte('\r'))
			case '"':  strings.write_byte(&out, byte('"'))
			case '\\': strings.write_byte(&out, byte('\\'))
			case '/':  strings.write_byte(&out, byte('/'))
			case 'u':  i += write_json_unicode_escape(&out, s, i)
			case:      strings.write_byte(&out, s[i])
			}
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
	// Scan for "role" then next "content" — handles whitespace after colons
	// (standard JSON: "role": "user" vs compact: "role":"user")
	i := 0
	for i < len(json) - 6 {
		role_idx := index_from(json, `"role"`, i)
		if role_idx < 0 { break }
		r_val_start := role_idx + len(`"role"`)
		for r_val_start < len(json) && (json[r_val_start] == ' ' || json[r_val_start] == '\t' || json[r_val_start] == ':') { r_val_start += 1 }
		if r_val_start >= len(json) || json[r_val_start] != '"' { i = role_idx + 1; continue }
		r_val_start += 1
		r_end := index_from(json, `"`, r_val_start)
		if r_end < 0 { break }
		role := json[r_val_start:r_end]
		c_idx := index_from(json, `"content"`, r_end)
		if c_idx < 0 { break }
		c_val_start := c_idx + len(`"content"`)
		for c_val_start < len(json) && (json[c_val_start] == ' ' || json[c_val_start] == '\t' || json[c_val_start] == ':') { c_val_start += 1 }
		if c_val_start >= len(json) || json[c_val_start] != '"' { i = c_idx + 1; continue }
		c_val_start += 1
		j := c_val_start
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
				case 'u':  j += write_json_unicode_escape(&content_buf, json, j)
				case:      strings.write_byte(&content_buf, json[j])
				}
			} else if ch == '"' {
				break
			} else {
				strings.write_byte(&content_buf, ch)
			}
			j += 1
		}
		// Capture a "tool_calls" array belonging to THIS message (between the
		// end of its content and the next message's "role" key).
		tc_raw := ""
		{
			scan_end := index_from(json, `"role"`, j)
			if scan_end < 0 { scan_end = len(json) }
			tc_idx := index_from(json, `"tool_calls"`, j)
			if tc_idx >= 0 && tc_idx < scan_end {
				arr_start := tc_idx + len(`"tool_calls"`)
				for arr_start < scan_end && (json[arr_start] == ' ' || json[arr_start] == '\t' ||
					json[arr_start] == ':' || json[arr_start] == '\n') { arr_start += 1 }
				if arr_start < scan_end && json[arr_start] == '[' {
					if arr_end, ok := json_balanced_end(json, arr_start); ok {
						tc_raw = strings.clone(json[arr_start:arr_end])
					}
				}
			}
		}
		append(&out, Message_Pair{
			role = role,
			content = strings.clone(strings.to_string(content_buf)),
			tool_calls_raw = tc_raw,
		})
		strings.builder_destroy(&content_buf)
		i = j + 1
	}
	result := make([]Message_Pair, len(out))
	for k in 0 ..< len(out) { result[k] = out[k] }
	return result
}

Message_Pair :: struct {
	role:    string,
	content: string,
	// Raw JSON text of the message's "tool_calls" array (verbatim substring of
	// the request body), empty when absent. Used to re-render assistant tool
	// calls into history in the native <tool_call> format.
	tool_calls_raw: string,
}

free_message_pairs :: proc(msgs: []Message_Pair) {
	for m in msgs {
		delete(m.content)
		delete(m.tool_calls_raw)
	}
	delete(msgs)
}

// Return the index just past the closing bracket/brace matching s[start]
// (which must be '[' or '{'), honoring strings and escapes. ok=false on
// unbalanced input.
json_balanced_end :: proc(s: string, start: int) -> (int, bool) {
	depth := 0
	in_str := false
	esc := false
	for i in start ..< len(s) {
		ch := s[i]
		if in_str {
			if esc { esc = false; continue }
			if ch == '\\' { esc = true; continue }
			if ch == '"' { in_str = false }
			continue
		}
		switch ch {
		case '"': in_str = true
		case '[', '{': depth += 1
		case ']', '}':
			depth -= 1
			if depth == 0 { return i + 1, true }
		}
	}
	return 0, false
}

// Extract the raw (balanced) value text for "key" in a flat JSON object.
// Works for object/array/scalar values. Returns "" when absent.
json_raw_value :: proc(obj: string, key: string) -> (string, bool) {
	needle := fmt.tprintf("\"%s\"", key)
	defer delete(needle)
	i := 0
	for {
		k := index_from(obj, needle, i)
		if k < 0 { return "", false }
		// Ensure this is a key position (followed by ':').
		j := k + len(needle)
		for j < len(obj) && (obj[j] == ' ' || obj[j] == '\t' || obj[j] == '\n') { j += 1 }
		if j < len(obj) && obj[j] == ':' {
			j += 1
			for j < len(obj) && (obj[j] == ' ' || obj[j] == '\t' || obj[j] == '\n') { j += 1 }
			if j >= len(obj) { return "", false }
			if obj[j] == '"' {
				// String value: find closing quote honoring escapes.
				p := j + 1
				for p < len(obj) {
					if obj[p] == '\\' { p += 2; continue }
					if obj[p] == '"' { return obj[j:p+1], true }
					p += 1
				}
				return "", false
			}
			if obj[j] == '{' || obj[j] == '[' {
				end, ok := json_balanced_end(obj, j)
				if !ok { return "", false }
				return obj[j:end], true
			}
			// Scalar: read until , } ] or whitespace.
			p := j
			for p < len(obj) && obj[p] != ',' && obj[p] != '}' && obj[p] != ']' &&
				obj[p] != ' ' && obj[p] != '\n' && obj[p] != '\t' { p += 1 }
			return obj[j:p], true
		}
		i = k + len(needle)
	}
}

// Unescape a JSON string literal (with surrounding quotes) into b.
json_unescape_into :: proc(b: ^strings.Builder, lit: string) {
	if len(lit) < 2 { return }
	j := 1
	for j < len(lit) - 1 {
		ch := lit[j]
		if ch == '\\' && j + 1 < len(lit) - 1 {
			j += 1
			switch lit[j] {
			case 'n':  strings.write_byte(b, byte('\n'))
			case 't':  strings.write_byte(b, byte('\t'))
			case 'r':  strings.write_byte(b, byte('\r'))
			case '"':  strings.write_byte(b, byte('"'))
			case '\\': strings.write_byte(b, byte('\\'))
			case '/':  strings.write_byte(b, byte('/'))
			case 'u':  j += write_json_unicode_escape(b, lit, j)
			case:      strings.write_byte(b, lit[j])
			}
		} else {
			strings.write_byte(b, ch)
		}
		j += 1
	}
}

// Re-render one request-side tool_call element ({"id":..,"function":{...}})
// for history. Empirically (probe: 4 template variants x greedy decode) this
// model only recognizes a COMPLETED tool call when history shows the JSON
// form — the native <function=...> form (what the model actually generates
// under the tool-call grammar) pattern-matches as "about to call" and the
// model re-calls the same tool forever. JSON form + a plain user-role result
// breaks the loop. The function object is emitted verbatim (byte-stable).
render_json_tool_calls :: proc(b: ^strings.Builder, raw_array: string) {
	i := strings.index(raw_array, "[")
	if i < 0 { return }
	i += 1
	for i < len(raw_array) {
		for i < len(raw_array) && raw_array[i] != '{' && raw_array[i] != ']' { i += 1 }
		if i >= len(raw_array) || raw_array[i] == ']' { break }
		elem_end, ok := json_balanced_end(raw_array, i)
		if !ok { break }
		elem := raw_array[i:elem_end]
		i = elem_end

		fn_raw, fok := json_raw_value(elem, "function")
		if !fok { continue }
		strings.write_string(b, "<tool_call>\n")
		strings.write_string(b, fn_raw)
		strings.write_string(b, "\n</tool_call>")
	}
}


openai_models_list :: proc() -> string {
	out: strings.Builder
	strings.builder_init(&out, context.temp_allocator)
	defer strings.builder_destroy(&out)
	strings.write_string(&out, `{"object":"list","data":[{"id":"odin-infer","object":"model","created":0,"owned_by":"local"}]}`)
	return strings.clone(strings.to_string(out))
}

handle_openai_chat :: proc(body: string) -> string {
	t0 := time_to_ms()
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
	t1 := time_to_ms()

	encoded, err := tok35.encode(&g_state.tok, prompt)
	if err != nil {
		return `{"error":{"message":"encode failed","type":"internal_error"}}`
	}
	t3 := time_to_ms()

	// ---- prefix cache (same three-branch logic as the Ollama path) ----
	common := 0
	max_common := min(len(encoded), len(g_state.last_tokens))
	for common < max_common && encoded[common] == g_state.last_tokens[common] {
		common += 1
	}
	snap_pos := 0
	for i in 3 ..< len(encoded) - 1 {
		if encoded[i] == EOS { snap_pos = i + 2; break }
	}
	snap_ok := snap_pos >= 5 && snap_pos < len(encoded)

	prefix_matches := false
	if len(g_state.prefix_tokens) > 0 && len(encoded) > len(g_state.prefix_tokens) {
		prefix_matches = true
		for i in 0 ..< len(g_state.prefix_tokens) {
			if encoded[i] != g_state.prefix_tokens[i] { prefix_matches = false; break }
		}
	}
	can_continue := common == len(g_state.last_tokens) && common > 0 &&
		common < len(encoded) && g_state.pos == common - 1

	t_reset_ms: i64 = 0
	if can_continue && (!prefix_matches || common >= g_state.prefix_pos) {
		new_tokens := encoded[common:len(encoded) - 1]
		if len(new_tokens) > 0 {
			_ = q35.engine_forward_batch(&g_state.engine, new_tokens, common)
		}
		g_state.pos = len(encoded) - 1
		g_state.pc_continuation_hits += 1
	} else if prefix_matches {
		q35.engine_restore_full_state(&g_state.engine,
			g_state.prefix_kv_k, g_state.prefix_kv_v,
			g_state.prefix_conv, g_state.prefix_rec)
		g_state.pos = g_state.prefix_pos
		suffix_tokens := encoded[g_state.prefix_pos:len(encoded) - 1]
		if len(suffix_tokens) > 0 {
			_ = q35.engine_forward_batch(&g_state.engine, suffix_tokens, g_state.prefix_pos)
		}
		g_state.pos = len(encoded) - 1
		g_state.pc_snapshot_hits += 1
	} else {
		t_r0 := time_to_ms()
		q35.engine_reset_state(&g_state.engine)
		t_reset_ms = time_to_ms() - t_r0
		g_state.pos = 0
		if snap_ok {
			_ = q35.engine_forward_batch(&g_state.engine, encoded[:snap_pos], 0)
			g_state.pos = snap_pos
			{
				prev := context.allocator
				context.allocator = g_heap
				if len(g_state.prefix_kv_k) > 0 {
					delete(g_state.prefix_kv_k); delete(g_state.prefix_kv_v)
					delete(g_state.prefix_conv); delete(g_state.prefix_rec)
				}
				g_state.prefix_kv_k, g_state.prefix_kv_v,
					g_state.prefix_conv, g_state.prefix_rec =
					q35.engine_save_full_state(&g_state.engine)
				context.allocator = prev
			}
			clear(&g_state.prefix_tokens)
			append(&g_state.prefix_tokens, ..encoded[:snap_pos])
			g_state.prefix_pos = snap_pos
		}
		if snap_pos < len(encoded) - 1 {
			_ = q35.engine_forward_batch(&g_state.engine, encoded[snap_pos:len(encoded) - 1], g_state.pos)
		}
		g_state.pos = len(encoded) - 1
	}
	t2 := time_to_ms()
	t4 := t2

	// refresh the continuation cache with this request's full token sequence
	// (generated tokens are appended inside the loop below)
	clear(&g_state.last_tokens)
	append(&g_state.last_tokens, ..encoded)

	// Generate
	gen_buf: strings.Builder
	strings.builder_init(&gen_buf, context.allocator)
	defer strings.builder_destroy(&gen_buf)

	t_start := time_to_ms()
	t_first := t_start
	gen := 0
	fwd_cpu_ms: i64 = 0
	next: int = 0

	if g_state.pos > 0 && len(encoded) > 0 {
		last := encoded[len(encoded) - 1]
		t_fwd0 := time_to_ms()
		logits := q35.engine_forward(&g_state.engine, last, g_state.pos)
		fwd_cpu_ms += time_to_ms() - t_fwd0
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
		append(&g_state.last_tokens, next)
		if gen >= max_tokens { break }
		if g_state.pos >= q35.engine_config(&g_state.engine).seq_len { break }
		t_fwd0 := time_to_ms()
		logits := q35.engine_forward(&g_state.engine, next, g_state.pos)
		fwd_cpu_ms += time_to_ms() - t_fwd0
		next = sampler.sample(&g_state.samp, logits)
	}

	save_session_state()

	ttft := t_first - t_start
	t_total := time_to_ms() - t_first  // decode only (excludes prefill)
	tps := gen > 0 && t_total > 0 ? f64(gen) * 1000.0 / f64(t_total) : 0.0
	g_state.last_ttft = ttft
	g_state.last_tps = tps
	g_state.last_gen = gen

	response_text := strings.to_string(gen_buf)
	t5 := time_to_ms()

	// Build OpenAI-format response
	out: strings.Builder
	strings.builder_init(&out, context.allocator)
	defer strings.builder_destroy(&out)
	strings.write_string(&out, "{")
	strings.write_string(&out, fmt.tprintf(`"id":"chatcmpl-%d",`, t_start))
	strings.write_string(&out, `"object":"chat.completion",`)
	strings.write_string(&out, fmt.tprintf(`"created":%d,`, t_start / 1000))
	strings.write_string(&out, `"model":"odinfer",`)
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
	fmt.eprintfln("[perf] openai_chat: parse/render={}ms reset={}ms encode={}ms prefill={}ms decode={}ms (ttft={}ms, tps={:.2f}, fwd_cpu={}ms) build_resp={}ms total={}ms (gen={})",
		t1-t0, t2-t1, t3-t2, t4-t3, t_total, ttft, tps, fwd_cpu_ms, time_to_ms()-t5, time_to_ms()-t0, gen)
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

// ====================== MCP (Model Context Protocol, stdio transport) ======================
// JSON-RPC 2.0 over newline-delimited stdio. Exposes the local engine to MCP
// clients (Claude Code, Cursor, pie): tools odin.chat + odin.models.
// Protocol details: https://modelcontextprotocol.io — stdout is reserved for
// protocol messages; all diagnostics go to stderr.

mcp_read_line :: proc(scratch: ^[dynamic]u8) -> (string, bool) {
	clear(scratch)
	b: [1]u8
	for {
		n, err := os.read(os.stdin, b[:])
		if n > 0 {
			if b[0] == '\n' {
				return string(scratch[:]), true
			}
			append(scratch, b[0])
			continue
		}
		if len(scratch) > 0 {
			return string(scratch[:]), true
		}
		_ = err
		return "", false
	}
}

// Raw text of the top-level "id" JSON value (number or string), or "" when the
// message is a notification. Skips any "id" occurrence that is not followed by ':'.
mcp_id_raw :: proc(line: string) -> string {
	i := 0
	for i < len(line) {
		idx := index_from(line, `"id"`, i)
		if idx < 0 { return "" }
		j := idx + 4
		for j < len(line) && (line[j] == ' ' || line[j] == '	') { j += 1 }
		if j < len(line) && line[j] == ':' {
			j += 1
			for j < len(line) && (line[j] == ' ' || line[j] == '	') { j += 1 }
			start := j
			in_str := false
			for j < len(line) {
				c := line[j]
				if c == '"' { in_str = !in_str; j += 1; continue }
				if !in_str && (c == ',' || c == '}') { break }
				j += 1
			}
			return strings.trim_space(line[start:j])
		}
		i = idx + 1
	}
	return ""
}

mcp_envelope_result :: proc(id, result_json: string) -> string {
	return fmt.tprintf(`{{"jsonrpc":"2.0","id":%s,"result":%s}}`, id, result_json)
}

mcp_envelope_error :: proc(id: string, code: int, msg: string) -> string {
	b: strings.Builder
	strings.builder_init(&b, context.allocator)
	defer strings.builder_destroy(&b)
	strings.write_string(&b, fmt.tprintf(`{{"jsonrpc":"2.0","id":%s,"error":{{"code":%d,"message":`, id, code))
	json_escape(msg, &b)
	strings.write_string(&b, `}}`)
	return strings.clone(strings.to_string(b))
}

mcp_tool_text_result :: proc(id, text: string) -> string {
	b: strings.Builder
	strings.builder_init(&b, context.allocator)
	defer strings.builder_destroy(&b)
	strings.write_string(&b, fmt.tprintf(`{{"jsonrpc":"2.0","id":%s,"result":{{"content":[{{"type":"text","text":`, id))
	json_escape(text, &b)
	strings.write_string(&b, `}]}}`)
	return strings.clone(strings.to_string(b))
}

mcp_handle :: proc(line: string) -> (string, bool) {
	method := extract_json_field(line, "method")
	id := mcp_id_raw(line)
	respond := len(id) > 0
	if len(method) == 0 {
		return mcp_envelope_error(id, -32700, "Parse error"), respond
	}
	switch method {
	case "initialize":
		return fmt.tprintf(`{{"jsonrpc":"2.0","id":%s,"result":{{"protocolVersion":"2024-11-05","capabilities":{{"tools":{{}}}},"serverInfo":{{"name":"odin-infer","version":"0.2.0","title":"odin-infer local Qwen engine"}}}}}}`, id), respond
	case "ping":
		return mcp_envelope_result(id, `{}`), respond
	case "notifications/initialized", "notifications/cancelled":
		return "", false
	case "tools/list":
		return fmt.tprintf(`{{"jsonrpc":"2.0","id":%s,"result":{{"tools":[{{"name":"odin.chat","description":"Run the local Qwen3/Qwen3.5 model (odin-infer) on a chat request and return the generated reply.","inputSchema":{{"type":"object","properties":{{"messages":{{"type":"array","description":"Chat messages in OpenAI format (role/content)."}},"max_tokens":{{"type":"integer","description":"Maximum tokens to generate.","default":256}},"temperature":{{"type":"number","description":"Sampling temperature (0 = greedy).","default":0.6}}}},"required":["messages"]}}}},{{"name":"odin.models","description":"List the loaded model and engine status.","inputSchema":{{"type":"object","properties":{{}}}}}}]}}}}`, id), respond
	case "tools/call":
		name := extract_json_field(line, "name")
		switch name {
		case "odin.models":
			cfg := q35.engine_config(&g_state.engine)
			info := fmt.tprintf(
				"model: %s\narchitecture: qwen3_5\nvocab: %d\nseq_len: %d\ndim: %d\nmetal: %v\nbackend: %s",
				g_state.model_path, cfg.vocab_size, cfg.seq_len, cfg.dim, g_state.engine.metal_ready,
				q35.gpu_backend_name(g_state.engine.metal_ready),
			)
			return mcp_tool_text_result(id, info), respond
		case "odin.chat":
			// Reuse the OpenAI handler: it scans the whole JSON for
			// messages/max_tokens/temperature, which live in params.arguments.
			resp_json := handle_openai_chat(line)
			if strings.contains(resp_json, "invalid_request_error") {
				return mcp_tool_text_result(id, strings.concatenate({"error: ", extract_json_field(resp_json, "message")}, context.temp_allocator)), respond
			}
			text := extract_json_field(resp_json, "content")
			return mcp_tool_text_result(id, text), respond
		case:
			return mcp_tool_text_result(id, fmt.tprintf("error: unknown tool '%s'", name)), respond
		}
	case:
		return mcp_envelope_error(id, -32601, "Method not found"), respond
	}
	return "", false
}

mcp_loop :: proc() {
	fmt.eprintln("mcp: stdio loop ready (stdout is reserved for protocol)")
	scratch := make([dynamic]u8, 0, 65536, g_heap)
	for {
		line, ok := mcp_read_line(&scratch)
		if !ok { break }
		line = strings.trim_space(line)
		if len(line) == 0 { continue }
		resp, respond := mcp_handle(line)
		if respond && len(resp) > 0 {
			fmt.println(resp)
		}
	}
	fmt.eprintln("mcp: stdin closed, exiting")
}

// ====================== entry ======================

main :: proc() {
	// Capture the process heap before handle_client ever swaps
	// context.allocator to the per-request arena; g_heap is the escape hatch
	// for allocations that outlive a request.
	g_heap = context.allocator
	if virtual.arena_init_growing(&g_req_arena) != nil {
		fmt.eprintln("serve: request arena init failed")
		os.exit(1)
	}
	args := os.args
	// --self-test: exercise parse_tool_calls + repair_tool_call_json over a
	// fixture table WITHOUT loading the model. Exits 0 on all-pass, 1 on any
	// failure. Fast (no model load), model-free, deterministic.
	for a in args {
		if a == "--self-test" {
			if run_parse_self_test() {
				os.exit(0)
			}
			os.exit(1)
		}
	}
	if len(args) < 2 {
		fmt.eprintln("usage: serve <model.gguf> [--port N] [--metal|--cuda|--vulkan] [--max-ctx N]")
		fmt.eprintln("       serve --self-test  (parse regression, no model needed)")
		os.exit(1)
	}
	model_path := args[1]
	port: u16 = 9748
	use_metal := false
	use_mcp := false
	max_ctx := 4096

	i := 2
	for i < len(args) {
		a := args[i]
		if a == "--port" && i + 1 < len(args) {
			v, _ := strconv_parse_int(args[i+1])
			port = u16(v)
			i += 2
		} else if a == "--metal" || a == "--cuda" || a == "--vulkan" {
			use_metal = true
			i += 1
		} else if a == "--max-ctx" && i + 1 < len(args) {
			max_ctx, _ = strconv_parse_int(args[i+1])
			i += 2
		} else if a == "--mcp" {
			use_mcp = true
			i += 1
		} else {
			i += 1
		}
	}

	g_state.model_path = model_path
	// Cross-request token histories: bind to the process heap now so appends
	// made while a request's arena is installed as context.allocator never
	// allocate from the arena (a [dynamic] array keeps the allocator captured
	// by its initial make, and clear() preserves it).
	g_state.last_tokens = make([dynamic]int, 0, 4096, g_heap)
	g_state.prefix_tokens = make([dynamic]int, 0, 4096, g_heap)
	fmt.eprintln("serve: loading engine…")
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
	g_state.samp.eos_token = EOS
	defer sampler.free_sampler(&g_state.samp)

	// KV persistence across restarts: resume the last session + system-prefix
	// snapshot. KV_PERSIST=0 disables; KV_PERSIST_DIR overrides .kv-cache/.
	if os.get_env("KV_PERSIST", context.temp_allocator) == "0" {
		g_kv_dir = ""
	} else {
		dir := os.get_env("KV_PERSIST_DIR", context.temp_allocator)
		if dir == "" { dir = ".kv-cache" }
		g_kv_dir = strings.clone(dir)
		_ = os.make_directory(g_kv_dir)
	}
	restore_persisted_state()

	// QDUMP_DIVERGE=1: dump divergence windows on cache-miss full prefills.
	g_dump_diverge = os.get_env("QDUMP_DIVERGE", context.temp_allocator) == "1"

	fmt.eprintfln("serve: ready (vocab=%d, seq_len=%d, metal=%v, backend=%s)",
		cfg.vocab_size, cfg.seq_len, g_state.engine.metal_ready, q35.gpu_backend_name(g_state.engine.metal_ready))

	if use_mcp {
		mcp_loop()
		return
	}

	if !start_server(port, handle_request) {
		os.exit(1)
	}
}

time_to_ms :: proc() -> i64 {
	return time.to_unix_nanoseconds(time.now()) / 1_000_000
}
