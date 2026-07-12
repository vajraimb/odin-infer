/* Qwen3.5 model: config + hybrid weights resolved from a parsed GGUF file. */

package qwen3_5

import ggml "ggml:ggml"

import "core:fmt"
import "core:os"

DEFAULT_MAX_CONTEXT :: 4096

// llama.cpp bakes Qwen3.5RMSNorm weights (1+w) into the GGUF directly (verified
// empirically: output_norm.weight ~2.2, attn_norm ~1.1 — effective multipliers,
// not zero-init offsets). So all call sites use the standard `weight * x_normed`
// form and we do NOT add 1 here.
RMSNORM_BAKE_PLUS_ONE :: false

Layer_Type :: enum { Full_Attention, Linear_Attention }

Config :: struct {
	dim:               int, // hidden_size
	hidden_dim:        int, // intermediate_size
	n_layers:          int,
	n_heads:           int, // num_attention_heads (full attn layers)
	n_kv_heads:        int, // num_key_value_heads (full attn layers)
	head_dim:          int, // 256
	vocab_size:        int,
	seq_len:           int, // effective (capped) context
	max_seq:           int,
	rms_eps:           f32,
	rope_theta:        f32,
	partial_rotary:    f32, // 0.25 -> rotary_dim = head_dim * partial_rotary
	rotary_dim:        int, // head_dim * partial_rotary  (=64)
	// linear attention
	lin_n_k_heads:     int, // 16
	lin_n_v_heads:     int, // 32
	lin_head_k_dim:    int, // 128
	lin_head_v_dim:    int, // 128
	lin_conv_kernel:   int, // 4
	// hybrid layout
	full_attn_interval:int, // full attn every Nth layer (index (N-1), (2N-1), ...)
	layer_types:       []Layer_Type,
	n_full:            int,
	n_linear:          int,
}

// A weight handle pointing into the mmap, with its quant kind.
Tensor :: struct {
	kind: ggml.GGML_Type,
	data: []u8,
}

Full_Attn_Weights :: struct {
	q_norm: []f32, // [head_dim]
	k_norm: []f32, // [head_dim]
	wq:     Tensor, // [n_heads*head_dim*2, dim]  (q and gate interleaved per head)
	wk:     Tensor, // [n_kv_heads*head_dim, dim]
	wv:     Tensor, // [n_kv_heads*head_dim, dim]
	wo:     Tensor, // [dim, n_heads*head_dim]
}

Linear_Attn_Weights :: struct {
	in_qkv:  Tensor, // [key_dim*2 + value_dim, dim]
	in_z:    Tensor, // [value_dim, dim]
	in_b:    Tensor, // [num_v_heads, dim]
	in_a:    Tensor, // [num_v_heads, dim]
	conv:    Tensor, // [conv_dim, kernel] depthwise causal conv1d (F32 in GGUF; bound from mmap for Metal)
	dt_bias: []f32,  // [num_v_heads]
	a_decay: []f32,  // [num_v_heads] -- ssm_a on disk = -exp(A_log), used directly
	norm_w:  []f32,  // [head_v_dim]  (RMSNormGated, ones-init -> standard rmsnorm)
	out:     Tensor, // [dim, value_dim]
}

Layer_Weights :: struct {
	layer_type: Layer_Type,
	attn_norm:  []f32, // [dim]   input_layernorm
	ffn_norm:   []f32, // [dim]   post_attention_layernorm
	full:       Full_Attn_Weights,
	linear:     Linear_Attn_Weights,
	w1:         Tensor, // [hidden_dim, dim]  ffn gate
	w2:         Tensor, // [dim, hidden_dim]  ffn down
	w3:         Tensor, // [hidden_dim, dim]  ffn up
}

Transformer_Weights :: struct {
	token_embedding: Tensor,
	output:          Tensor,
	output_norm:     []f32, // baked
	layers:          []Layer_Weights,
}

Run_State :: struct {
	// shared
	x:   []f32, // [dim]
	xb:  []f32, // [dim]
	xb2: []f32, // [dim]
	hb:  []f32, // [hidden_dim]
	hb2: []f32, // [hidden_dim]
	logits: []f32, // [vocab_size]
	// full attention temporaries
	qproj: []f32, // [n_heads*head_dim*2]
	q:     []f32, // [n_heads*head_dim]
	xb3:   []f32, // [n_heads*head_dim]
	att:   []f32, // [n_heads * seq_len]
	// linear attention temporaries
	qkv_raw:   []f32, // [conv_dim]
	qkv_out:   []f32, // [conv_dim]
	z_vec:     []f32, // [value_dim]
	b_vec:     []f32, // [num_v_heads]
	a_vec:     []f32, // [num_v_heads]
	q_lin:     []f32, // [num_k_heads * head_k_dim]
	k_lin:     []f32, // [num_k_heads * head_k_dim]
	lin_out:   []f32, // [value_dim]
	delta_scr: []f32, // [head_v_dim]
	// caches
	key_cache:        []f32, // [n_full, seq_len, kv_dim]
	value_cache:      []f32, // [n_full, seq_len, kv_dim]
	conv_states:      []f32, // [n_linear, conv_dim, kernel-1]
	recurrent_states: []f32, // [n_linear, num_v_heads, head_k_dim, head_v_dim]
}

Transformer :: struct {
	config:     Config,
	weights:    Transformer_Weights,
	state:      Run_State,
	gguf:       ggml.GGUF_File,
	inv_freq:   []f32, // [rotary_dim/2]
	full_slot:  []int, // [n_layers] global layer idx -> full attn KV slot, or -1
	lin_slot:   []int, // [n_layers] global layer idx -> linear state slot, or -1
}

tensor_as_f32 :: proc(t: ^ggml.GGUF_Tensor) -> []f32 {
	if t.kind != .F32 {
		fmt.eprintf("expected F32 tensor for %s, got %v\n", t.name, t.kind)
		os.exit(1)
	}
	return (cast([^]f32)raw_data(t.data))[:len(t.data) / 4]
}

// Copy an F32 tensor into a mutable buffer; if `bake_plus_one`, add 1.0 to each
// element (for Qwen3.5RMSNorm tensors which are zero-initialised and applied as
// (1 + w) * x_normed).
tensor_as_f32_copy :: proc(t: ^ggml.GGUF_Tensor, bake_plus_one: bool) -> []f32 {
	if t.kind != .F32 {
		fmt.eprintf("expected F32 tensor for %s, got %v\n", t.name, t.kind)
		os.exit(1)
	}
	n := len(t.data) / 4
	src := (cast([^]f32)raw_data(t.data))[:n]
	out := make([]f32, n)
	if bake_plus_one {
		for i in 0 ..< n {
			out[i] = src[i] + 1.0
		}
	} else {
		copy(out, src)
	}
	return out
}

require_tensor :: proc(g: ^ggml.GGUF_File, name: string) -> ^ggml.GGUF_Tensor {
	t, ok := ggml.gguf_get_tensor(g, name)
	if !ok {
		fmt.eprintf("missing required tensor: %s\n", name)
		os.exit(1)
	}
	if !ggml.is_supported_quant(t.kind) && t.kind != .F16 {
		fmt.eprintf(
			"tensor %s has unsupported quant type %v\n",
			name,
			t.kind,
		)
		os.exit(1)
	}
	return t
}

find_tensor_any :: proc(g: ^ggml.GGUF_File, names: []string) -> (^ggml.GGUF_Tensor, bool) {
	for name in names {
		if t, ok := ggml.gguf_get_tensor(g, name); ok {
			return t, true
		}
	}
	return nil, false
}

require_tensor_any :: proc(g: ^ggml.GGUF_File, names: []string) -> ^ggml.GGUF_Tensor {
	t, ok := find_tensor_any(g, names)
	if !ok {
		fmt.eprintf("missing required tensor, tried: %v\n", names)
		os.exit(1)
	}
	if !ggml.is_supported_quant(t.kind) && t.kind != .F16 {
		fmt.eprintf("tensor %s has unsupported quant type %v\n", t.name, t.kind)
		os.exit(1)
	}
	return t
}

// Per-layer lookup with two suffix spellings (e.g. gdn_* / attn_*). Uses two
// separate stack buffers so the two name slices don't alias each other.
require_layer_tensor :: proc(g: ^ggml.GGUF_File, l: int, s0, s1: string) -> ^ggml.GGUF_Tensor {
	b0: [64]u8
	b1: [64]u8
	n0 := fmt.bprintf(b0[:], "blk.%d.%s", l, s0)
	n1 := fmt.bprintf(b1[:], "blk.%d.%s", l, s1)
	return require_tensor_any(g, {n0, n1})
}

tensor_handle :: proc(t: ^ggml.GGUF_Tensor) -> Tensor {
	return Tensor{kind = t.kind, data = t.data}
}

// Dequantize an entire tensor (any supported GGML type) into a flat f32 buffer.
tensor_dequant_to_f32 :: proc(t: ^ggml.GGUF_Tensor) -> []f32 {
	ne := 1
	for d in t.dims {
		ne *= int(d)
	}
	out := make([]f32, ne)
	ggml.dequant_row(t.kind, t.data, ne, out)
	return out
}

load_config :: proc(t: ^Transformer) {
	g := &t.gguf
	c := &t.config

	arch, _ := ggml.gguf_meta_str(g, "general.architecture")
	if arch != "qwen35" && arch != "qwen3_5" && arch != "qwen3.5" && arch != "qwen3_5_text" {
		fmt.eprintf("warning: architecture is '%s', expected 'qwen35'\n", arch)
	}

	// Read a u64 metadata key; fall back to the Ornith-1.0-9B default if absent.
	// llama.cpp uses the `qwen35.*` prefix (no underscore); try `qwen3_5.*` too.
	get_u :: proc(g: ^ggml.GGUF_File, key: string, fallback: int) -> int {
		if v, ok := ggml.gguf_meta_u64(g, key); ok {
			return int(v)
		}
		return fallback
	}
	get_f :: proc(g: ^ggml.GGUF_File, key: string, fallback: f32) -> f32 {
		if v, ok := ggml.gguf_meta_f32(g, key); ok {
			return v
		}
		return fallback
	}
	gu :: proc(g: ^ggml.GGUF_File, k0, k1: string, fb: int) -> int {
		v, ok := ggml.gguf_meta_u64(g, k0)
		if !ok do v, ok = ggml.gguf_meta_u64(g, k1)
		if !ok do return fb
		return int(v)
	}

	c.dim        = gu(g, "qwen35.embedding_length", "qwen3_5.embedding_length", 4096)
	c.hidden_dim = gu(g, "qwen35.feed_forward_length", "qwen3_5.feed_forward_length", 12288)
	c.n_layers   = gu(g, "qwen35.block_count", "qwen3_5.block_count", 32)
	c.n_heads    = gu(g, "qwen35.attention.head_count", "qwen3_5.attention.head_count", 16)
	c.n_kv_heads = gu(g, "qwen35.attention.head_count_kv", "qwen3_5.attention.head_count_kv", 4)
	c.head_dim   = gu(g, "qwen35.attention.key_length", "qwen3_5.attention.key_length", 256)
	c.max_seq    = gu(g, "qwen35.context_length", "qwen3_5.context_length", 262144)

	// Linear-attention (SSM) dims. llama.cpp names: ssm.group_count (key heads),
	// ssm.time_step_rank (value heads), ssm.state_size (head_k_dim),
	// ssm.inner_size (value_dim = num_v_heads * head_v_dim), ssm.conv_kernel.
	c.lin_n_k_heads   = gu(g, "qwen35.ssm.group_count", "qwen3_5.linear_num_key_heads", 16)
	c.lin_n_v_heads   = gu(g, "qwen35.ssm.time_step_rank", "qwen3_5.linear_num_value_heads", 32)
	c.lin_head_k_dim  = gu(g, "qwen35.ssm.state_size", "qwen3_5.linear_key_head_dim", 128)
	c.lin_conv_kernel = gu(g, "qwen35.ssm.conv_kernel", "qwen3_5.linear_conv_kernel_dim", 4)
	lin_inner := gu(g, "qwen35.ssm.inner_size", "qwen3_5.linear_inner_size", c.lin_n_v_heads * 128)
	c.lin_head_v_dim = lin_inner / max(c.lin_n_v_heads, 1)

	c.full_attn_interval = gu(g, "qwen35.full_attention_interval", "qwen3_5.full_attention_interval", 4)
	c.rope_theta = get_f(g, "qwen35.rope.freq_base", 10_000_000.0)
	c.rms_eps    = get_f(g, "qwen35.attention.layer_norm_rms_epsilon", 1e-6)
	// rope.dimension_count is the partial-rotary width directly (= head_dim * partial_rotary = 64).
	rot_dim_meta := gu(g, "qwen35.rope.dimension_count", "qwen3_5.rope.dimension_count", 0)
	if rot_dim_meta > 0 {
		c.rotary_dim = rot_dim_meta
		c.partial_rotary = f32(c.rotary_dim) / f32(max(c.head_dim, 1))
	} else {
		c.partial_rotary = get_f(g, "qwen35.partial_rotary_factor", 0.25)
		c.rotary_dim = int(f32(c.head_dim) * c.partial_rotary)
	}

	// Build layer_types from the interval: full attention at indices
	// (interval-1), (2*interval-1), ...   Matches config.layer_types for
	// Ornith-1.0-9B (full at 3,7,11,...,31).
	c.layer_types = make([]Layer_Type, c.n_layers)
	c.n_full = 0
	c.n_linear = 0
	for i in 0 ..< c.n_layers {
		if c.full_attn_interval > 0 && (i + 1) % c.full_attn_interval == 0 {
			c.layer_types[i] = .Full_Attention
			c.n_full += 1
		} else {
			c.layer_types[i] = .Linear_Attention
			c.n_linear += 1
		}
	}

	emb := require_tensor(g, "token_embd.weight")
	c.vocab_size = int(emb.dims[len(emb.dims) - 1])
}

memory_map_weights :: proc(t: ^Transformer) {
	g := &t.gguf
	w := &t.weights
	c := &t.config
	// Norm weights are F32 in the GGUF with (1+w) already baked by llama.cpp, so
	// read them directly as mmap slices (no copy) -- this also lets Metal bind
	// them via the mmap offset (woff). The RMSNORM_BAKE_PLUS_ONE flag is unused.

	w.token_embedding = tensor_handle(require_tensor(g, "token_embd.weight"))
	w.output_norm = tensor_as_f32(require_tensor(g, "output_norm.weight"))

	if out, ok := ggml.gguf_get_tensor(g, "output.weight"); ok {
		if !ggml.is_supported_quant(out.kind) && out.kind != .F16 {
			fmt.eprintf("output.weight has unsupported quant type %v\n", out.kind)
			os.exit(1)
		}
		w.output = tensor_handle(out)
	} else {
		w.output = w.token_embedding // tied
	}

	w.layers = make([]Layer_Weights, c.n_layers)
	buf: [64]u8
	for l in 0 ..< c.n_layers {
		lw := &w.layers[l]
		lw.layer_type = c.layer_types[l]

		norm_name :: proc(buf: []u8, l: int, suffix: string) -> string {
			return fmt.bprintf(buf, "blk.%d.%s", l, suffix)
		}

		lw.attn_norm = tensor_as_f32(require_tensor(g, norm_name(buf[:], l, "attn_norm.weight")))
		lw.ffn_norm = tensor_as_f32(require_tensor(g, norm_name(buf[:], l, "post_attention_norm.weight")))
		lw.w1 = tensor_handle(require_tensor(g, norm_name(buf[:], l, "ffn_gate.weight")))
		lw.w2 = tensor_handle(require_tensor(g, norm_name(buf[:], l, "ffn_down.weight")))
		lw.w3 = tensor_handle(require_tensor(g, norm_name(buf[:], l, "ffn_up.weight")))

		switch lw.layer_type {
		case .Full_Attention:
			lw.full.q_norm = tensor_as_f32(require_tensor(g, norm_name(buf[:], l, "attn_q_norm.weight")))
			lw.full.k_norm = tensor_as_f32(require_tensor(g, norm_name(buf[:], l, "attn_k_norm.weight")))
			lw.full.wq = tensor_handle(require_tensor(g, norm_name(buf[:], l, "attn_q.weight")))
			lw.full.wk = tensor_handle(require_tensor(g, norm_name(buf[:], l, "attn_k.weight")))
			lw.full.wv = tensor_handle(require_tensor(g, norm_name(buf[:], l, "attn_v.weight")))
			lw.full.wo = tensor_handle(require_tensor(g, norm_name(buf[:], l, "attn_output.weight")))
		case .Linear_Attention:
			la := &lw.linear
			la.in_qkv = tensor_handle(require_layer_tensor(g, l, "attn_qkv.weight", "gdn_in_qkv.weight"))
			la.in_z = tensor_handle(require_layer_tensor(g, l, "attn_gate.weight", "gdn_in_z.weight"))
			la.in_b = tensor_handle(require_layer_tensor(g, l, "ssm_beta.weight", "gdn_in_b.weight"))
			la.in_a = tensor_handle(require_layer_tensor(g, l, "ssm_alpha.weight", "gdn_in_a.weight"))
			la.conv = tensor_handle(require_layer_tensor(g, l, "ssm_conv1d.weight", "gdn_conv.weight"))
			la.dt_bias = tensor_as_f32(require_layer_tensor(g, l, "ssm_dt.bias", "gdn_dt_bias.weight"))
			la.a_decay = tensor_as_f32(require_layer_tensor(g, l, "ssm_a", "gdn_a_log.weight"))
			la.norm_w = tensor_as_f32(
				require_layer_tensor(g, l, "ssm_norm.weight", "gdn_norm.weight"),
			)
			la.out = tensor_handle(require_layer_tensor(g, l, "ssm_out.weight", "gdn_out.weight"))
		}
	}
}

malloc_run_state :: proc(s: ^Run_State, p: Config) {
	kv_dim := p.n_kv_heads * p.head_dim
	conv_dim := p.lin_n_k_heads * p.lin_head_k_dim * 2 + p.lin_n_v_heads * p.lin_head_v_dim
	value_dim := p.lin_n_v_heads * p.lin_head_v_dim

	s.x = make([]f32, p.dim)
	s.xb = make([]f32, p.dim)
	s.xb2 = make([]f32, p.dim)
	s.hb = make([]f32, p.hidden_dim)
	s.hb2 = make([]f32, p.hidden_dim)
	s.logits = make([]f32, p.vocab_size)

	s.qproj = make([]f32, p.n_heads * p.head_dim * 2)
	s.q = make([]f32, p.n_heads * p.head_dim)
	s.xb3 = make([]f32, p.n_heads * p.head_dim)
	s.att = make([]f32, p.n_heads * p.seq_len)

	s.qkv_raw = make([]f32, conv_dim)
	s.qkv_out = make([]f32, conv_dim)
	s.z_vec = make([]f32, value_dim)
	s.b_vec = make([]f32, p.lin_n_v_heads)
	s.a_vec = make([]f32, p.lin_n_v_heads)
	s.q_lin = make([]f32, p.lin_n_k_heads * p.lin_head_k_dim)
	s.k_lin = make([]f32, p.lin_n_k_heads * p.lin_head_k_dim)
	s.lin_out = make([]f32, value_dim)
	s.delta_scr = make([]f32, p.lin_head_v_dim)

	if p.n_full > 0 {
		s.key_cache = make([]f32, p.n_full * p.seq_len * kv_dim)
		s.value_cache = make([]f32, p.n_full * p.seq_len * kv_dim)
	}
	if p.n_linear > 0 {
		s.conv_states = make([]f32, p.n_linear * conv_dim * (p.lin_conv_kernel - 1))
		s.recurrent_states = make([]f32, p.n_linear * p.lin_n_v_heads * p.lin_head_k_dim * p.lin_head_v_dim)
	}
}

free_run_state :: proc(s: ^Run_State) {
	// Guarded: when Metal is active, Run_State was never allocated (all nil).
	if s.x != nil { delete(s.x) }; if s.xb != nil { delete(s.xb) }; if s.xb2 != nil { delete(s.xb2) }
	if s.hb != nil { delete(s.hb) }; if s.hb2 != nil { delete(s.hb2) }; if s.logits != nil { delete(s.logits) }
	if s.qproj != nil { delete(s.qproj) }; if s.q != nil { delete(s.q) }; if s.xb3 != nil { delete(s.xb3) }; if s.att != nil { delete(s.att) }
	if s.qkv_raw != nil { delete(s.qkv_raw) }; if s.qkv_out != nil { delete(s.qkv_out) }
	if s.z_vec != nil { delete(s.z_vec) }; if s.b_vec != nil { delete(s.b_vec) }; if s.a_vec != nil { delete(s.a_vec) }
	if s.q_lin != nil { delete(s.q_lin) }; if s.k_lin != nil { delete(s.k_lin) }
	if s.lin_out != nil { delete(s.lin_out) }; if s.delta_scr != nil { delete(s.delta_scr) }
	if s.key_cache != nil {
		delete(s.key_cache); delete(s.value_cache)
	}
	if s.conv_states != nil {
		delete(s.conv_states); delete(s.recurrent_states)
	}
}

build_transformer :: proc(t: ^Transformer, checkpoint_path: string, max_ctx: int, skip_cpu_state: bool = false) {
	ggml.parse_gguf(checkpoint_path, &t.gguf)
	load_config(t)

	t.config.seq_len = min(t.config.max_seq, max_ctx)
	if t.config.seq_len <= 0 {
		t.config.seq_len = max_ctx
	}

	// layer index -> per-type slot lookup tables
	t.full_slot = make([]int, t.config.n_layers)
	t.lin_slot = make([]int, t.config.n_layers)
	fi, li := 0, 0
	for i in 0 ..< t.config.n_layers {
		switch t.config.layer_types[i] {
		case .Full_Attention:
			t.full_slot[i] = fi; fi += 1
			t.lin_slot[i] = -1
		case .Linear_Attention:
			t.full_slot[i] = -1
			t.lin_slot[i] = li; li += 1
		}
	}

	memory_map_weights(t)
	// When Metal is active, forward_gpu uses Metal buffers exclusively — the CPU
	// Run_State (KV cache, conv/recurrent states, activations) is never read.
	// Skip its allocation to save up to GBs of dead memory at large -c.
	if !skip_cpu_state {
		malloc_run_state(&t.state, t.config)
	}

	t.inv_freq = make([]f32, t.config.rotary_dim / 2)
	inv_freq_default(t.config.rope_theta, t.config.rotary_dim, t.inv_freq)

	conv_dim := t.config.lin_n_k_heads * t.config.lin_head_k_dim * 2 + t.config.lin_n_v_heads * t.config.lin_head_v_dim
	value_dim := t.config.lin_n_v_heads * t.config.lin_head_v_dim
	kv_bytes := 2 * t.config.n_full * t.config.seq_len * t.config.n_kv_heads * t.config.head_dim * 4
	lin_bytes :=
		t.config.n_linear * conv_dim * (t.config.lin_conv_kernel - 1) * 4 +
		t.config.n_linear * t.config.lin_n_v_heads * t.config.lin_head_k_dim * t.config.lin_head_v_dim * 4
	fmt.printf(
		"qwen3_5: dim=%d layers=%d (full=%d linear=%d) heads=%d/%d head_dim=%d hidden=%d vocab=%d ctx=%d/%d  rotary=%d  conv_dim=%d value_dim=%d  KV=%.0f MB  lin=%.0f MB\n",
		t.config.dim,
		t.config.n_layers,
		t.config.n_full,
		t.config.n_linear,
		t.config.n_heads,
		t.config.n_kv_heads,
		t.config.head_dim,
		t.config.hidden_dim,
		t.config.vocab_size,
		t.config.seq_len,
		t.config.max_seq,
		t.config.rotary_dim,
		conv_dim,
		value_dim,
		f64(kv_bytes) / (1024 * 1024),
		f64(lin_bytes) / (1024 * 1024),
	)
}

free_transformer :: proc(t: ^Transformer) {
	free_run_state(&t.state)
	// norm weights, conv, dt_bias, a_decay are all mmap slices (tensor_as_f32 /
	// tensor_handle); they are freed by free_gguf, not here.
	delete(t.weights.layers)
	delete(t.config.layer_types)
	delete(t.full_slot)
	delete(t.lin_slot)
	delete(t.inv_freq)
	ggml.free_gguf(&t.gguf)
}
