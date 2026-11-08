/* Syntax-constrained generation for native <tool_call> blocks (VISION.md P0).

   When a request carries tools, sampling is masked so the model physically
   cannot emit malformed tool-call syntax. The grammar tracks the native
   format byte-by-byte:

     <tool_call>
     <function=name>
     <parameter=key>
     value (free bytes)
     </parameter>
     ...more parameters...
     </function>
     </tool_call>

   Free prose is allowed before/after/between calls (Outside state); only the
   structural tokens are constrained. Function names and parameter keys are
   constrained to the request's tool schema (TCG_Table). Parameter values are
   free-form, with a hard cap (TCG_MAX_VALUE) after which the terminator is
   forced.

   The mask probe copies the small mutable state (~40 bytes) and replays each
   token's bytes against it — same technique as grammar.odin. The static tool
   table is shared by pointer, never copied.
*/

package sampler

TCG_MAX_TOOLS  :: 24
TCG_MAX_PARAMS :: 24
TCG_MAX_VALUE  :: 8192

TCG_LIT_TOOL_OPEN  :: "<tool_call>"
TCG_LIT_FUNCTION   :: "<function="
TCG_LIT_PARAM_OPEN :: "\n<parameter="
TCG_LIT_FUNC_CLOSE :: "\n</function>"
TCG_LIT_PARAM_TERM :: "\n</parameter>"
TCG_LIT_TOOL_CLOSE :: "\n</tool_call>"

TCG_State :: enum {
	Outside,     // free prose; tracking TCG_LIT_TOOL_OPEN prefix
	Expect_Func, // ws* then TCG_LIT_FUNCTION
	Func_Name,   // tool name via candidate bitmask; '>' commits
	Between,     // TCG_LIT_PARAM_OPEN or TCG_LIT_FUNC_CLOSE
	Param_Key,   // parameter key via candidate bitmask; '>' commits
	After_Key,   // single '\n' (stray ws tolerated)
	Param_Value, // free bytes; tracking TCG_LIT_PARAM_TERM; forced past TCG_MAX_VALUE
	After_Func,  // TCG_LIT_TOOL_CLOSE
}

TCG_Tool_Spec :: struct {
	name:     string,
	params:   [TCG_MAX_PARAMS]string,
	n_params: int,
}

TCG_Table :: struct {
	tools:   [TCG_MAX_TOOLS]TCG_Tool_Spec,
	n_tools: int,
}

Tool_Call_Grammar :: struct {
	table:     ^TCG_Table, // static for the request; probes share it by pointer
	state:     TCG_State,
	pos:       int,        // progress in current literal / terminator / name
	alt:       int,        // Between: -1 undecided, 0 = param-open, 1 = func-close
	cand:      u32,        // candidate bitmask (Func_Name: tools, Param_Key: params)
	tool_idx:  int,
	value_len: int,
	enabled:   bool,
}

// ---- table construction (called by the serve layer) ----

tcg_table_add_tool :: proc(t: ^TCG_Table, name: string) -> ^TCG_Tool_Spec {
	if t.n_tools >= TCG_MAX_TOOLS do return nil
	spec := &t.tools[t.n_tools]
	spec.name = name
	spec.n_params = 0
	t.n_tools += 1
	return spec
}

tcg_tool_add_param :: proc(s: ^TCG_Tool_Spec, key: string) {
	if s.n_params >= TCG_MAX_PARAMS do return
	s.params[s.n_params] = key
	s.n_params += 1
}

// ---- lifecycle ----

tcg_init :: proc(g: ^Tool_Call_Grammar, table: ^TCG_Table) {
	g.table = table
	g.state = .Outside
	g.pos = 0
	g.alt = -1
	g.cand = 0
	g.tool_idx = -1
	g.value_len = 0
	g.enabled = true
}

tcg_reset :: proc(g: ^Tool_Call_Grammar) {
	tbl := g.table
	tcg_init(g, tbl)
}

// EOS is only legal in free prose that isn't halfway through "<tool_call>".
tcg_is_completable :: proc(g: ^Tool_Call_Grammar) -> bool {
	return g.state == .Outside && g.pos == 0
}

// ---- byte-level state machine ----

@(private)
tcg_is_ws :: proc(b: u8) -> bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r'
}

@(private)
tcg_all_mask :: proc(n: int) -> u32 {
	if n <= 0 do return 0
	return (u32(1) << u32(n)) - 1
}

tcg_advance_byte :: proc(g: ^Tool_Call_Grammar, b: u8) -> bool {
	switch g.state {
	case .Outside:
		lit := TCG_LIT_TOOL_OPEN
		if b == lit[g.pos] {
			g.pos += 1
			if g.pos == len(lit) {
				g.state = .Expect_Func
				g.pos = 0
			}
		} else {
			g.pos = b == lit[0] ? 1 : 0
		}
		return true

	case .Expect_Func:
		lit := TCG_LIT_FUNCTION
		if g.pos == 0 && tcg_is_ws(b) do return true
		if b != lit[g.pos] do return false
		g.pos += 1
		if g.pos == len(lit) {
			g.state = .Func_Name
			g.pos = 0
			g.cand = tcg_all_mask(g.table.n_tools)
		}
		return true

	case .Func_Name:
		if b == '>' {
			// Commit only when a candidate matched its full length — handles
			// prefix names ("search" vs "search_web") correctly.
			for i in 0 ..< g.table.n_tools {
				if g.cand & (u32(1) << u32(i)) != 0 && len(g.table.tools[i].name) == g.pos {
					g.tool_idx = i
					g.state = .Between
					g.pos = 0
					g.alt = -1
					return true
				}
			}
			return false
		}
		survivors := false
		for i in 0 ..< g.table.n_tools {
			bit := u32(1) << u32(i)
			if g.cand & bit == 0 do continue
			name := g.table.tools[i].name
			if g.pos < len(name) && name[g.pos] == b {
				survivors = true
			} else {
				g.cand &= ~bit
			}
		}
		if !survivors do return false
		g.pos += 1
		return true

	case .Between:
		lit_open := TCG_LIT_PARAM_OPEN
		lit_close := TCG_LIT_FUNC_CLOSE
		m0: bool = g.alt != 1 && b == lit_open[g.pos]
		m1: bool = g.alt != 0 && b == lit_close[g.pos]
		if !m0 && !m1 {
			// Tolerate stray spaces/tabs before the next block ('\n' always
			// matches both literals' first byte, so it never lands here).
			if g.pos == 0 && tcg_is_ws(b) do return true
			return false
		}
		if m0 && m1 {
			g.pos += 1 // shared prefix, stay undecided
			return true
		}
		g.alt = m0 ? 0 : 1
		g.pos += 1
		lit_len := m0 ? len(lit_open) : len(lit_close)
		if g.pos == lit_len {
			if m0 {
				g.state = .Param_Key
				g.pos = 0
				g.cand = tcg_all_mask(g.table.tools[g.tool_idx].n_params)
			} else {
				g.state = .After_Func
				g.pos = 0
			}
		}
		return true

	case .Param_Key:
		spec := &g.table.tools[g.tool_idx]
		if spec.n_params == 0 {
			// Schema keys unknown — accept any key until '>'.
			if b == '>' {
				g.state = .After_Key
				g.pos = 0
			}
			return true
		}
		if b == '>' {
			for i in 0 ..< spec.n_params {
				if g.cand & (u32(1) << u32(i)) != 0 && len(spec.params[i]) == g.pos {
					g.state = .After_Key
					g.pos = 0
					return true
				}
			}
			return false
		}
		survivors := false
		for i in 0 ..< spec.n_params {
			bit := u32(1) << u32(i)
			if g.cand & bit == 0 do continue
			key := spec.params[i]
			if g.pos < len(key) && key[g.pos] == b {
				survivors = true
			} else {
				g.cand &= ~bit
			}
		}
		if !survivors do return false
		g.pos += 1
		return true

	case .After_Key:
		if b == '\n' {
			g.state = .Param_Value
			g.pos = 0
			g.value_len = 0
			return true
		}
		return tcg_is_ws(b)

	case .Param_Value:
		term := TCG_LIT_PARAM_TERM
		if g.value_len >= TCG_MAX_VALUE {
			// Forced termination: only the terminator may follow.
			if b != term[g.pos] do return false
			g.pos += 1
			if g.pos == len(term) {
				g.state = .Between
				g.pos = 0
				g.alt = -1
			}
			return true
		}
		if b == term[g.pos] {
			g.pos += 1
			if g.pos == len(term) {
				g.state = .Between
				g.pos = 0
				g.alt = -1
			}
		} else {
			g.value_len += 1
			g.pos = b == term[0] ? 1 : 0
		}
		return true

	case .After_Func:
		lit := TCG_LIT_TOOL_CLOSE
		if g.pos == 0 && b != lit[0] && tcg_is_ws(b) do return true
		if b != lit[g.pos] do return false
		g.pos += 1
		if g.pos == len(lit) {
			g.state = .Outside // supports multiple calls in one generation
			g.pos = 0
		}
		return true
	}
	return false
}

tcg_advance_token :: proc(g: ^Tool_Call_Grammar, bytes: []u8) -> bool {
	for b in bytes {
		if !tcg_advance_byte(g, b) do return false
	}
	return true
}

// States that never reject a byte — skip the whole-vocab probe entirely.
@(private)
tcg_accepts_everything :: proc(g: ^Tool_Call_Grammar) -> bool {
	#partial switch g.state {
	case .Outside:
		return true
	case .Param_Value:
		return g.value_len < TCG_MAX_VALUE
	case .Param_Key:
		return g.tool_idx >= 0 && g.table.tools[g.tool_idx].n_params == 0
	}
	return false
}

// Probe every token's decoded bytes against a copy of the state; out[i] is
// true when token i keeps the syntax valid. EOS is handled by the sampler.
tcg_compute_token_mask :: proc(g: ^Tool_Call_Grammar, vocab_size: int, token_bytes: [][]u8, out: []bool) {
	if tcg_accepts_everything(g) {
		for i in 0 ..< vocab_size { out[i] = true }
		return
	}
	snapshot := g^
	for i in 0 ..< vocab_size {
		tb := token_bytes[i]
		if len(tb) == 0 {
			out[i] = true
			continue
		}
		g^ = snapshot
		ok := true
		for b in tb {
			if !tcg_advance_byte(g, b) {
				ok = false
				break
			}
		}
		out[i] = ok
	}
	g^ = snapshot
}
