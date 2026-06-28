/* Qwen3 model: config + weights resolved from a parsed GGUF file */

package infer

import ggml "ggml:ggml"

import "core:fmt"
import "core:os"

DEFAULT_MAX_CONTEXT :: 4096 // cap KV cache; override with -c

Config :: struct {
	dim:        int,
	hidden_dim: int,
	n_layers:   int,
	n_heads:    int,
	n_kv_heads: int,
	vocab_size: int,
	seq_len:    int, // effective (capped) context length
	max_seq:    int, // model's native context length
	head_dim:   int,
	rope_freq:  f32,
	rms_eps:    f32,
}

// A weight handle pointing into the mmap, with its quant kind.
Tensor :: struct {
	kind: ggml.GGML_Type,
	data: []u8,
}

Layer_Weights :: struct {
	attn_norm: []f32,
	ffn_norm:  []f32,
	q_norm:    []f32,
	k_norm:    []f32,
	wq:        Tensor,
	wk:        Tensor,
	wv:        Tensor,
	wo:        Tensor,
	w1:        Tensor, // ffn gate
	w2:        Tensor, // ffn down
	w3:        Tensor, // ffn up
}

Transformer_Weights :: struct {
	token_embedding: Tensor,
	output:          Tensor,
	output_norm:     []f32,
	layers:          []Layer_Weights,
}

Run_State :: struct {
	x:           []f32,
	xb:          []f32,
	xb2:         []f32,
	xb3:         []f32,
	hb:          []f32,
	hb2:         []f32,
	q:           []f32,
	att:         []f32,
	logits:      []f32,
	key_cache:   []f32,
	value_cache: []f32,
}

Transformer :: struct {
	config:  Config,
	weights: Transformer_Weights,
	state:   Run_State,
	gguf:    ggml.GGUF_File,
}

tensor_as_f32 :: proc(t: ^ggml.GGUF_Tensor) -> []f32 {
	if t.kind != .F32 {
		fmt.eprintf("expected F32 tensor for %s, got %v\n", t.name, t.kind)
		os.exit(1)
	}
	return (cast([^]f32)raw_data(t.data))[:len(t.data) / 4]
}

require_tensor :: proc(g: ^ggml.GGUF_File, name: string) -> ^ggml.GGUF_Tensor {
	t, ok := ggml.gguf_get_tensor(g, name)
	if !ok {
		fmt.eprintf("missing required tensor: %s\n", name)
		os.exit(1)
	}
	if !ggml.is_supported_quant(t.kind) {
		fmt.eprintf(
			"tensor %s has unsupported quant type %v (supported: F32,F16,Q8_0,Q4_0,Q4_K,Q6_K)\n",
			name,
			t.kind,
		)
		os.exit(1)
	}
	return t
}

load_config :: proc(t: ^Transformer) {
	g := &t.gguf
	c := &t.config

	arch, _ := ggml.gguf_meta_str(g, "general.architecture")
	if arch != "qwen3" {
		fmt.eprintf("warning: architecture is '%s', expected 'qwen3'\n", arch)
	}

	get :: proc(g: ^ggml.GGUF_File, key: string) -> int {
		v, ok := ggml.gguf_meta_u64(g, key)
		if !ok {
			fmt.eprintf("missing metadata key: %s\n", key)
			os.exit(1)
		}
		return int(v)
	}

	c.dim = get(g, "qwen3.embedding_length")
	c.hidden_dim = get(g, "qwen3.feed_forward_length")
	c.n_layers = get(g, "qwen3.block_count")
	c.n_heads = get(g, "qwen3.attention.head_count")
	c.n_kv_heads = get(g, "qwen3.attention.head_count_kv")
	c.max_seq = get(g, "qwen3.context_length")
	c.head_dim = get(g, "qwen3.attention.key_length")

	if v, ok := ggml.gguf_meta_f32(g, "qwen3.rope.freq_base"); ok {
		c.rope_freq = v
	} else {
		c.rope_freq = 1_000_000.0
	}
	if v, ok := ggml.gguf_meta_f32(g, "qwen3.attention.layer_norm_rms_epsilon"); ok {
		c.rms_eps = v
	} else {
		c.rms_eps = 1e-6
	}

	// vocab size from the embedding tensor's outer dimension
	emb := require_tensor(g, "token_embd.weight")
	c.vocab_size = int(emb.dims[len(emb.dims) - 1])
}

memory_map_weights :: proc(t: ^Transformer) {
	g := &t.gguf
	w := &t.weights
	c := &t.config

	w.token_embedding = tensor_handle(require_tensor(g, "token_embd.weight"))
	w.output_norm = tensor_as_f32(require_tensor(g, "output_norm.weight"))

	// Some models tie the classifier to the embedding table.
	if out, ok := ggml.gguf_get_tensor(g, "output.weight"); ok {
		if !ggml.is_supported_quant(out.kind) {
			fmt.eprintf("output.weight has unsupported quant type %v\n", out.kind)
			os.exit(1)
		}
		w.output = tensor_handle(out)
	} else {
		w.output = w.token_embedding
	}

	w.layers = make([]Layer_Weights, c.n_layers)
	buf: [64]u8
	for l in 0 ..< c.n_layers {
		lw := &w.layers[l]
		name :: proc(buf: []u8, l: int, suffix: string) -> string {
			return fmt.bprintf(buf, "blk.%d.%s", l, suffix)
		}
		lw.attn_norm = tensor_as_f32(require_tensor(g, name(buf[:], l, "attn_norm.weight")))
		lw.ffn_norm = tensor_as_f32(require_tensor(g, name(buf[:], l, "ffn_norm.weight")))
		lw.q_norm = tensor_as_f32(require_tensor(g, name(buf[:], l, "attn_q_norm.weight")))
		lw.k_norm = tensor_as_f32(require_tensor(g, name(buf[:], l, "attn_k_norm.weight")))
		lw.wq = tensor_handle(require_tensor(g, name(buf[:], l, "attn_q.weight")))
		lw.wk = tensor_handle(require_tensor(g, name(buf[:], l, "attn_k.weight")))
		lw.wv = tensor_handle(require_tensor(g, name(buf[:], l, "attn_v.weight")))
		lw.wo = tensor_handle(require_tensor(g, name(buf[:], l, "attn_output.weight")))
		lw.w1 = tensor_handle(require_tensor(g, name(buf[:], l, "ffn_gate.weight")))
		lw.w2 = tensor_handle(require_tensor(g, name(buf[:], l, "ffn_down.weight")))
		lw.w3 = tensor_handle(require_tensor(g, name(buf[:], l, "ffn_up.weight")))
	}
}

tensor_handle :: proc(t: ^ggml.GGUF_Tensor) -> Tensor {
	return Tensor{kind = t.kind, data = t.data}
}

malloc_run_state :: proc(s: ^Run_State, p: Config) {
	att_head_dim := p.n_heads * p.head_dim
	kv_dim := p.n_kv_heads * p.head_dim

	s.x = make([]f32, p.dim)
	s.xb = make([]f32, p.dim)
	s.xb2 = make([]f32, p.dim)
	s.xb3 = make([]f32, att_head_dim)
	s.hb = make([]f32, p.hidden_dim)
	s.hb2 = make([]f32, p.hidden_dim)
	s.q = make([]f32, att_head_dim)
	s.att = make([]f32, p.n_heads * p.seq_len)
	s.logits = make([]f32, p.vocab_size)
	s.key_cache = make([]f32, p.n_layers * p.seq_len * kv_dim)
	s.value_cache = make([]f32, p.n_layers * p.seq_len * kv_dim)
}

free_run_state :: proc(s: ^Run_State) {
	delete(s.x)
	delete(s.xb)
	delete(s.xb2)
	delete(s.xb3)
	delete(s.hb)
	delete(s.hb2)
	delete(s.q)
	delete(s.att)
	delete(s.logits)
	delete(s.key_cache)
	delete(s.value_cache)
}

build_transformer :: proc(t: ^Transformer, checkpoint_path: string, max_ctx: int) {
	ggml.parse_gguf(checkpoint_path, &t.gguf)
	load_config(t)

	t.config.seq_len = min(t.config.max_seq, max_ctx)
	if t.config.seq_len <= 0 {
		t.config.seq_len = max_ctx
	}

	memory_map_weights(t)
	malloc_run_state(&t.state, t.config)

	kv_bytes := 2 * t.config.n_layers * t.config.seq_len * t.config.n_kv_heads * t.config.head_dim * 4
	fmt.printf(
		"model: dim=%d layers=%d heads=%d/%d hidden=%d vocab=%d ctx=%d/%d  weights=%s  KV cache=%.0f MB\n",
		t.config.dim,
		t.config.n_layers,
		t.config.n_heads,
		t.config.n_kv_heads,
		t.config.hidden_dim,
		t.config.vocab_size,
		t.config.seq_len,
		t.config.max_seq,
		weight_type_label(&t.gguf),
		f64(kv_bytes) / (1024 * 1024),
	)
}

weight_type_label :: proc(g: ^ggml.GGUF_File) -> string {
	if t, ok := ggml.gguf_get_tensor(g, "blk.0.ffn_down.weight"); ok {
		switch t.kind {
		case .F32:
			return "F32"
		case .F16:
			return "F16"
		case .Q4_0:
			return "Q4_0"
		case .Q4_1:
			return "Q4_1"
		case .Q5_0:
			return "Q5_0"
		case .Q5_1:
			return "Q5_1"
		case .Q8_0:
			return "Q8_0"
		case .Q4_K:
			return "Q4_K"
		case .Q6_K:
			return "Q6_K"
		case .Q8_1, .Q2_K, .Q3_K, .Q5_K, .Q8_K:
			return "mixed"
		}
	}
	return "?"
}

free_transformer :: proc(t: ^Transformer) {
	free_run_state(&t.state)
	delete(t.weights.layers)
	ggml.free_gguf(&t.gguf)
}
