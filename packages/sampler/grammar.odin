/* sampler/grammar.odin — built-in JSON grammar for constrained decoding.
 *
 * Forces the model's output to be valid JSON (RFC 8259 subset). Activated via
 * Sampler.grammar field; sampler.sample() applies the mask before sampling.
 *
 * Approach: single enum State + stack of "return" states for nested values.
 * For each candidate token the state machine is advanced on the token's
 * bytes; if any byte is invalid in context, the token is masked out
 * (logit → -inf).
 *
 * Simplifications vs strict JSON:
 *   - \uXXXX not strictly validated (any escape \X allowed)
 *   - numbers allow leading zeros
 *   - whitespace allowed: space, tab, cr, lf
 *
 * Not a general grammar engine — only JSON. Inspired by colibri's
 * GRAMMAR=file.gbnf but inlined for the JSON-only common case.
 */

package sampler

import "core:fmt"
import "core:math"
import "base:runtime"

// ---------- state machine ----------

State :: enum {
	Value,            // expecting start of any value
	Object_KeyOrEnd,  // after '{' + optional ws, expecting '"' or '}'
	Object_Colon,     // after key + optional ws, expecting ':'
	Object_ValStart,  // after ':' + optional ws, expecting value
	Object_Next,      // after value in object, expecting ',' or '}'
	Array_ValOrEnd,   // after '[' + optional ws, expecting value or ']'
	Array_Next,        // after value in array, expecting ',' or ']'
	String_Body,      // inside a string
	String_Esc,       // after '\' inside a string
	Keyword,          // accumulating true/false/null (sub-state in jpos)
	Number_AfterSign, // after '-'
	Number_Int,       // in integer digits
	Number_Frac,      // after '.' + first frac digit, in fraction
	Number_AfterDot,  // directly after '.', expects digit
	Number_Exp,       // after 'e' or 'E', expects digit or +/-
	Number_ExpSign,   // after 'e+' or 'e-', expects digit
	Number_ExpDig,    // in exponent digits
	Done,             // root value complete; only ws allowed
}

// Return frame: state to transition to when current value completes.
JSON_Grammar :: struct {
	state: State,
	stack: [64]State,   // fixed-size return-after-value stack (nesting depth)
	stack_len: int,
	jpos: int,          // keyword match position (1..len)
	kw_id: int,         // 1=true, 2=false, 3=null
}

KW_TRUE ::  "true"
KW_FALSE :: "false"
KW_NULL ::  "null"

// ---------- lifecycle ----------

init_json :: proc() -> JSON_Grammar {
	g: JSON_Grammar
	g.state = .Value
	g.stack[0] = .Done
	g.stack_len = 1
	return g
}

free_json :: proc(g: ^JSON_Grammar) {
	// nothing to free (fixed-size stack)
}

reset_json :: proc(g: ^JSON_Grammar) {
	g.state = .Value
	g.stack[0] = .Done
	g.stack_len = 1
	g.jpos = 0
	g.kw_id = 0
}

is_complete :: proc(g: ^JSON_Grammar) -> bool {
	return g.state == .Done
}

// Returns true if end-of-input would be valid here (i.e., the grammar is in
// a state where a complete JSON has been produced). Numbers don't have an
// explicit terminator — they end at the next non-digit byte — so we treat
// the in-digit number states as completable when at the root.
is_completable :: proc(g: ^JSON_Grammar) -> bool {
	if g.state == .Done { return true }
	if g.state == .Number_Int || g.state == .Number_Frac || g.state == .Number_ExpDig {
		// "Number could end here" — but only counts as complete if we're at
		// the root context (stack has only Done left).
		return g.stack_len == 1 && g.stack[0] == .Done
	}
	return false
}

// Push a return state onto the stack. Returns false on overflow (depth > 64).
push_ :: proc(g: ^JSON_Grammar, s: State) -> bool {
	if g.stack_len >= 64 { return false }
	g.stack[g.stack_len] = s
	g.stack_len += 1
	return true
}

// Pop the top of the stack.
pop_ :: proc(g: ^JSON_Grammar) -> State {
	if g.stack_len == 0 { return .Done }
	g.stack_len -= 1
	return g.stack[g.stack_len]
}

// ---------- byte-level state advance ----------

ws :: proc(b: u8) -> bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r'
}

digit :: proc(b: u8) -> bool { return b >= '0' && b <= '9' }

value_start :: proc(b: u8) -> bool {
	return b == '{' || b == '[' || b == '"' || b == '-' ||
		digit(b) || b == 't' || b == 'f' || b == 'n'
}

// Start parsing a fresh value. Caller has already pushed the return state.
// Returns false if b cannot start a value.
start_value :: proc(g: ^JSON_Grammar, b: u8) -> bool {
	switch {
	case b == '{':
		g.state = .Object_KeyOrEnd
	case b == '[':
		g.state = .Array_ValOrEnd
	case b == '"':
		g.state = .String_Body
	case b == '-':
		g.state = .Number_AfterSign
	case digit(b):
		g.state = .Number_Int
	case b == 't':
		g.kw_id = 1; g.jpos = 1; g.state = .Keyword
	case b == 'f':
		g.kw_id = 2; g.jpos = 1; g.state = .Keyword
	case b == 'n':
		g.kw_id = 3; g.jpos = 1; g.state = .Keyword
	case:
		return false
	}
	return true
}

// Called when the current value has just completed. Pops the stack to find
// the resume state.
complete_value :: proc(g: ^JSON_Grammar) {
	if g.stack_len == 0 {
		g.state = .Done
		return
	}
	g.state = pop_(g)
}

advance_byte :: proc(g: ^JSON_Grammar, b: u8) -> bool {
	switch g.state {
	case .Value:
		if ws(b) { return true } // leading ws
		if !start_value(g, b) { return false }
		return true

	case .Object_KeyOrEnd:
		if ws(b) { return true }
		if b == '}' {
			complete_value(g)
			return true
		}
		if b == '"' {
			g.state = .String_Body
			// push a marker so string close returns to colon state, not value pop
			_ = push_(g, .Object_Colon)
			return true
		}
		return false

	case .Object_Colon:
		if ws(b) { return true }
		if b == ':' { g.state = .Object_ValStart; return true }
		return false

	case .Object_ValStart:
		if ws(b) { return true }
		if !value_start(b) { return false }
		_ = push_(g, .Object_Next)
		g.state = .Value
		return start_value(g, b)

	case .Object_Next:
		if ws(b) { return true }
		if b == ',' { g.state = .Object_KeyOrEnd; return true }
		if b == '}' { complete_value(g); return true }
		return false

	case .Array_ValOrEnd:
		if ws(b) { return true }
		if b == ']' { complete_value(g); return true }
		if !value_start(b) { return false }
		_ = push_(g, .Array_Next)
		g.state = .Value
		return start_value(g, b)

	case .Array_Next:
		if ws(b) { return true }
		if b == ',' { g.state = .Array_ValOrEnd; return true }
		if b == ']' { complete_value(g); return true }
		return false

	case .String_Body:
		if b == '"' {
			// string closes — pop stack to find resume state
			g.state = pop_(g)
			return true
		}
		if b == '\\' { g.state = .String_Esc; return true }
		if b < 0x20 { return false } // control chars must be escaped
		return true // any other byte OK (including UTF-8 continuation)

	case .String_Esc:
		switch b {
		case '"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u':
			g.state = .String_Body
			return true
		case:
			return false
		}

	case .Keyword:
		kw := g.kw_id == 1 ? KW_TRUE : g.kw_id == 2 ? KW_FALSE : KW_NULL
		if g.jpos >= len(kw) { return false }
		if b != kw[g.jpos] { return false }
		g.jpos += 1
		if g.jpos == len(kw) {
			complete_value(g)
		}
		return true

	case .Number_AfterSign:
		if !digit(b) { return false }
		g.state = .Number_Int
		return true

	case .Number_Int:
		if digit(b) { return true }
		if b == '.' { g.state = .Number_AfterDot; return true }
		if b == 'e' || b == 'E' { g.state = .Number_Exp; return true }
		// number ended — process this byte in the parent context
		complete_value(g)
		return advance_byte(g, b)

	case .Number_AfterDot:
		if !digit(b) { return false }
		g.state = .Number_Frac
		return true

	case .Number_Frac:
		if digit(b) { return true }
		if b == 'e' || b == 'E' { g.state = .Number_Exp; return true }
		complete_value(g)
		return advance_byte(g, b)

	case .Number_Exp:
		if digit(b) { g.state = .Number_ExpDig; return true }
		if b == '+' || b == '-' { g.state = .Number_ExpSign; return true }
		return false

	case .Number_ExpSign:
		if !digit(b) { return false }
		g.state = .Number_ExpDig
		return true

	case .Number_ExpDig:
		if digit(b) { return true }
		complete_value(g)
		return advance_byte(g, b)

	case .Done:
		return ws(b)
	}
	return false
}

// Advance on a multi-byte buffer; returns false on first invalid byte.
advance_bytes :: proc(g: ^JSON_Grammar, bytes: []u8) -> bool {
	for b in bytes {
		if !advance_byte(g, b) do return false
	}
	return true
}

// ---------- token mask ----------

// Compute mask: out[i] = true if token i is a valid continuation.
// Does NOT advance grammar state — caller advances after sampling.
compute_token_mask :: proc(
	g: ^JSON_Grammar,
	vocab_size: int,
	token_bytes: [][]u8,
	out: []bool,
) {
	// Snapshot the whole struct once; restore by copy after each probe.
	// Fixed-size stack makes this trivial and cheap.
	saved := g^
	for i in 0 ..< vocab_size {
		bytes := token_bytes[i]
		if len(bytes) == 0 {
			out[i] = false
			continue
		}
		ok := advance_bytes(g, bytes)
		out[i] = ok
		g^ = saved  // restore state + stack + counters
	}
}

// Build token-bytes cache. The callback returns the decoded string for token
// id; we transmute to []u8 and store. Memory ownership: the caller must
// ensure the returned strings outlive the cache (use a long-lived allocator
// in the decode callback).
build_token_cache :: proc(
	vocab_size: int,
	decode: proc(id: int, allocator: runtime.Allocator) -> string,
) -> [][]u8 {
	out := make([][]u8, vocab_size)
	for i in 0 ..< vocab_size {
		s := decode(i, context.allocator)
		out[i] = transmute([]u8)s
	}
	return out
}

free_token_cache :: proc(cache: [][]u8, vocab_size: int) {
	// Each entry's bytes are owned by the allocator used in build_token_cache.
	// Free them too if they were heap-allocated.
	for i in 0 ..< vocab_size {
		if len(cache[i]) > 0 {
			delete(cache[i])
		}
	}
	delete(cache)
}

// ---------- apply mask to logits ----------

apply_mask :: proc(logits: []f32, mask: []bool) {
	if len(logits) != len(mask) {
		fmt.eprintf("grammar: logits/mask size mismatch (%d vs %d)\n",
			len(logits), len(mask))
		return
	}
	for i in 0 ..< len(logits) {
		if !mask[i] {
			logits[i] = -math.F32_MAX
		}
	}
}
