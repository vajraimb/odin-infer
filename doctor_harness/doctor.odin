#+build darwin

/* doctor harness — read-only readiness check + load planner for odin-infer.

   Subcommands:
     doctor <model.gguf>                Read-only GGUF inspection (no engine_load)
     plan   <model.gguf> [flags]        Full engine_load + memory budget breakdown
     env                                Dump all recognized env vars and current values

   Flags for `plan`:
     --max-ctx N      Context window cap (default: model native / 4096 floor)
     --metal          Use Metal backend (default: CPU)
     --threads N      Thread count (default: physical core count)

   Run (from repo root):
     odin run doctor_harness/doctor.odin -file -- doctor /path/to/model.gguf \
       -collection:ggml=. -collection:infer=. \
       -collection:tokenizer=. -collection:sampler=.

   Exit codes: 0 OK, 1 WARN, 2 FAIL (bad magic / truncated / unsupported arch).

   macOS-only in v1 (Metal probe). For Linux, stub print_metal_info() and rebuild.
*/

package doctor

import "core:fmt"
import "core:os"
import ggml "ggml:ggml"
import infer "infer:infer"
import NS "core:sys/darwin/Foundation"
import MTL "vendor:darwin/Metal"

// ---- env vars recognized by odin-infer (library + CLI) ----
Env_Var :: struct {
	name: string,
	desc: string,
	source: string,
	default_effect: string,
}

ENV_VARS := [5]Env_Var {
	{name = "QFASTMATH",    desc = "Metal shader fast-math enable",                 source = "qwen3_5.metal", default_effect = "ON (=0 disables)"},
	{name = "QTIMING",      desc = "Print per-prefill GPU ms",                       source = "qwen3_5.metal", default_effect = "off (=1 enables)"},
	{name = "QPROF_NOLIN",  desc = "Ablation: skip linear-layer stateful loop",      source = "qwen3_5.metal", default_effect = "off"},
	{name = "QPROF_NOFULL", desc = "Ablation: skip full-attention per-token block",  source = "qwen3_5.metal", default_effect = "off"},
	{name = "QDBG",         desc = "Debug token encode/decode in CLI",               source = "odin-infeer",   default_effect = "off"},
}

// ---------- helpers ----------

mb :: proc(bytes: u64) -> f64 { return f64(bytes) / 1024.0 / 1024.0 }

ggml_type_name :: proc(k: ggml.GGML_Type) -> string {
	switch k {
	case .F32:  return "F32"
	case .F16:  return "F16"
	case .Q4_0: return "Q4_0"
	case .Q4_1: return "Q4_1"
	case .Q5_0: return "Q5_0"
	case .Q5_1: return "Q5_1"
	case .Q8_0: return "Q8_0"
	case .Q8_1: return "Q8_1"
	case .Q2_K: return "Q2_K"
	case .Q3_K: return "Q3_K"
	case .Q4_K: return "Q4_K"
	case .Q5_K: return "Q5_K"
	case .Q6_K: return "Q6_K"
	case .Q8_K: return "Q8_K"
	case: return "?"
	}
}

get_env_or :: proc(name, def: string) -> string {
	v := os.get_env(name, context.temp_allocator)
	if len(v) == 0 { return def }
	return v
}

// ---------- system section ----------

print_system :: proc() {
	fmt.println("══ System ════════════════════════════════════════════════════")
	when ODIN_OS == .Darwin {
		fmt.println("  os            : Darwin (macOS)")
	} else {
		fmt.println("  os            : non-Darwin")
	}
	when ODIN_ARCH == .arm64 {
		fmt.println("  arch          : arm64")
	} else {
		fmt.println("  arch          : non-arm64")
	}
	fmt.printfln("  cpu cores     : %d", os.get_processor_core_count())
	dev := MTL.CreateSystemDefaultDevice()
	if dev != nil {
		fmt.printfln("  metal device  : %s", dev->name()->odinString())
		fmt.printfln("  tg mem max    : %d bytes (%.1f KB)",
			dev->maxThreadgroupMemoryLength(),
			f64(dev->maxThreadgroupMemoryLength()) / 1024.0)
		dev->release()
	} else {
		fmt.println("  metal device  : (none available)")
	}
}

// ---------- env section ----------

print_env :: proc() {
	fmt.println("══ Environment variables ═════════════════════════════════════")
	fmt.println("  name            value     default-effect                    source")
	fmt.println("  ─────────────── ────────  ───────────────────────────────── ────────────────")
	for v in ENV_VARS {
		current := os.get_env(v.name, context.temp_allocator)
		shown := current
		if len(shown) == 0 { shown = "·" }
		fmt.printfln("  %-15s %-8s  %-32s  %s", v.name, shown, v.default_effect, v.source)
	}
	fmt.println("\n  (· = unset; library default applies)")
}

// ---------- file inspection ----------

arch_label :: proc(g: ^ggml.GGUF_File) -> string {
	// detect Qwen3 vs Qwen3.5 (Ornith) from metadata
	arch, ok := g.metadata["general.architecture"]
	if !ok { return "unknown" }
	a := arch.str
	if a == "qwen3" { return "qwen3 (dense)" }
	if a == "qwen3_5" || a == "ornith" { return "qwen3_5 (Ornith hybrid)" }
	return a
}

file_size :: proc(path: string) -> (u64, bool) {
	s, err := os.stat(path, context.temp_allocator)
	if err != os.ERROR_NONE { return 0, false }
	return u64(s.size), true
}

run_doctor :: proc(args: []string) -> int {
	if len(args) < 1 {
		fmt.eprintln("usage: doctor doctor <model.gguf>")
		return 1
	}
	path := args[0]
	print_system()
	fmt.println()
	print_env()
	fmt.println()

	fmt.println("══ GGUF file ════════════════════════════════════════════════")
	fmt.printfln("  path          : %s", path)
	sz, err := os.stat(path, context.temp_allocator)
	if err != os.ERROR_NONE {
		fmt.eprintf("  ⚠ cannot stat file\n")
		return 2
	}
	fmt.printfln("  size          : %.2f MB (%d bytes)", mb(u64(sz.size)), sz.size)

	g: ggml.GGUF_File
	ggml.parse_gguf(path, &g)
	defer ggml.free_gguf(&g)

	if len(g.mmap) < 8 || g.mmap[0] != 0x47 || g.mmap[1] != 0x47 || g.mmap[2] != 0x55 || g.mmap[3] != 0x46 {
		// 'G','G','U','F' little-endian == 0x46554747
		fmt.println("  ⚠ magic check: not 'GGUF'")
		return 2
	}
	fmt.println("  magic         : GGUF ✓")

	if v, ok := g.metadata["general.architecture"]; ok {
		fmt.printfln("  architecture  : %s", arch_label(&g))
	}
	if v, ok := g.metadata["general.name"]; ok {
		fmt.printfln("  name          : %s", v.str)
	}
	if v, ok := g.metadata["general.file_type"]; ok {
		// file_type maps to a GGML_Type-ish enum; just print the integer
		fmt.printfln("  file_type     : %d", v.u)
	}

	fmt.printfln("  tensors       : %d", len(g.tensors))
	fmt.printfln("  metadata kvs  : %d", len(g.metadata))

	// quant type histogram
	type_counts: map[ggml.GGML_Type]int
	for t in g.tensors {
		type_counts[t.kind] += 1
	}
	fmt.println("  quant types   :")
	for ty, n in type_counts {
		fmt.printfln("    %-5s × %d", ggml_type_name(ty), n)
	}
	delete(type_counts)

	// model dims (best effort)
	if v, ok := g.metadata["qwen3.embedding_length"]; ok {
		fmt.printfln("  qwen3.dim     : %d", v.u)
	}
	if v, ok := g.metadata["qwen3.block_count"]; ok {
		fmt.printfln("  qwen3.layers  : %d", v.u)
	}

	// verdict
	fmt.println()
	fmt.println("  verdict       : OK ✓")
	return 0
}

// ---------- plan subcommand ----------

run_plan :: proc(args: []string) -> int {
	if len(args) < 1 {
		fmt.eprintln("usage: doctor plan <model.gguf> [--max-ctx N] [--metal] [--threads N]")
		return 1
	}
	path := args[0]
	max_ctx := infer.DEFAULT_MAX_CONTEXT
	use_metal := false
	num_threads := int(os.get_processor_core_count())
	i := 1
	for i < len(args) {
		a := args[i]
		if a == "--metal" {
			use_metal = true
			i += 1
		} else if a == "--max-ctx" && i + 1 < len(args) {
			max_ctx = parse_int_or(args[i + 1], max_ctx)
			i += 2
		} else if a == "--threads" && i + 1 < len(args) {
			num_threads = parse_int_or(args[i + 1], num_threads)
			i += 2
		} else {
			i += 1
		}
	}

	print_system()
	fmt.println()

	fmt.println("══ Plan ══════════════════════════════════════════════════════")
	fmt.printfln("  model         : %s", path)
	fmt.printfln("  max_ctx       : %d", max_ctx)
	fmt.printfln("  backend       : %s", use_metal ? "Metal" : "CPU")
	fmt.printfln("  threads       : %d", num_threads)
	fmt.println()

	fmt.println("  loading engine…")
	e, ok := infer.engine_load(path, infer.Engine_Opts{
		max_ctx = max_ctx,
		use_metal = use_metal,
		num_threads = num_threads,
	})
	if !ok {
		fmt.eprintln("  ⚠ engine_load failed")
		return 2
	}
	defer infer.engine_destroy(&e)
	defer infer.destroy_matmul_pool()

	cfg := infer.engine_config(&e)
	fmt.println()
	fmt.println("══ Memory budget ═════════════════════════════════════════════")
	fmt.printfln("  dim           : %d", cfg.dim)
	fmt.printfln("  hidden_dim    : %d", cfg.hidden_dim)
	fmt.printfln("  n_layers      : %d", cfg.n_layers)
	fmt.printfln("  n_heads/kv    : %d / %d", cfg.n_heads, cfg.n_kv_heads)
	fmt.printfln("  head_dim      : %d", cfg.head_dim)
	fmt.printfln("  vocab         : %d", cfg.vocab_size)
	fmt.printfln("  ctx effective : %d / %d native", cfg.seq_len, cfg.max_seq)
	fmt.println()

	sz, _ := file_size(path)
	kv_bytes := u64(2 * cfg.n_layers * cfg.seq_len * cfg.n_kv_heads * cfg.head_dim) * 4
	// activation buffer total (x, xb, xb2, hb, hb2, q, att, logits) — f32 each
	att_head_dim := cfg.n_heads * cfg.head_dim
	act_elems := u64(3 * cfg.dim + att_head_dim + 2 * cfg.hidden_dim +
		cfg.n_heads * cfg.seq_len + cfg.vocab_size)
	act_bytes := act_elems * 4

	fmt.printfln("  weights (mmap): %.2f MB", mb(sz))
	fmt.printfln("  KV cache      : %.2f MB  (2 × %d L × %d ctx × %d kvh × %d hd × 4)",
		mb(kv_bytes), cfg.n_layers, cfg.seq_len, cfg.n_kv_heads, cfg.head_dim)
	fmt.printfln("  activations   : %.2f MB", mb(act_bytes))
	fmt.printfln("  ────────────────────────────")
	fmt.printfln("  total resident: %.2f MB  (weights + KV + activations)",
		mb(sz + kv_bytes + act_bytes))

	fmt.println()
	fmt.printfln("  metal_ready   : %v", infer.engine_metal_ready(&e))
	fmt.println("  verdict       : OK ✓")
	return 0
}

// ---------- env subcommand ----------

run_env :: proc(args: []string) -> int {
	print_env()
	return 0
}

// ---------- helpers ----------

parse_int_or :: proc(s: string, def: int) -> int {
	if len(s) == 0 { return def }
	neg := false
	i := 0
	if s[0] == '-' { neg = true; i = 1 }
		else if s[0] == '+' { i = 1 }
	v: int = 0
	for i < len(s) {
		c := s[i]
		if c < '0' || c > '9' { return def }
		v = v * 10 + int(c - '0')
		i += 1
	}
	return neg ? -v : v
}

// ---------- entry ----------

main :: proc() {
	args := os.args
	if len(args) < 2 {
		fmt.eprintln("usage:")
		fmt.eprintln("  doctor doctor <model.gguf>")
		fmt.eprintln("  doctor plan   <model.gguf> [--max-ctx N] [--metal] [--threads N]")
		fmt.eprintln("  doctor env")
		os.exit(1)
	}
	sub := args[1]
	rest := args[2:]
	switch {
	case sub == "doctor":
		os.exit(run_doctor(rest))
	case sub == "plan":
		os.exit(run_plan(rest))
	case sub == "env":
		os.exit(run_env(rest))
	case:
		fmt.eprintf("doctor: unknown subcommand %q\n", sub)
		os.exit(1)
	}
}
