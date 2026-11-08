/* Forward pass for Qwen3.5 hybrid (full + linear attention) transformer. */

package qwen3_5

import ggml "ggml:ggml"

import "core:fmt"
import "core:math"
import "core:os"

get_embedding_row :: proc(t: ^Tensor, token, dim: int, dst: []f32) {
	rb := ggml.row_byte_size(t.kind, dim)
	ggml.dequant_row(t.kind, t.data[token * rb:], dim, dst)
}

forward :: proc(transformer: ^Transformer, token: int, pos: int) -> []f32 {
	p := &transformer.config
	w := &transformer.weights
	s := &transformer.state

	if pos < 0 || pos >= p.seq_len {
		fmt.eprintf("forward: pos=%d out of range (seq_len=%d)\n", pos, p.seq_len)
		os.exit(1)
	}
	if token < 0 || token >= p.vocab_size {
		fmt.eprintf("forward: token=%d out of range (vocab=%d)\n", token, p.vocab_size)
		os.exit(1)
	}

	get_embedding_row(&w.token_embedding, token, p.dim, s.x)

	for l in 0 ..< p.n_layers {
		switch p.layer_types[l] {
		case .Full_Attention:
			forward_full_layer(transformer, l, pos)
		case .Linear_Attention:
			forward_linear_layer(transformer, l, pos)
		}
	}

	rmsnorm(s.x, s.x, w.output_norm, p.rms_eps)
	matmul_t(s.logits, s.x, &w.output, p.dim, p.vocab_size)
	return s.logits
}

forward_full_layer :: proc(t: ^Transformer, l: int, pos: int) {
	p := &t.config
	w := &t.weights
	s := &t.state
	lw := &w.layers[l]
	fa := &lw.full

	dim := p.dim
	head_dim := p.head_dim
	n_heads := p.n_heads
	n_kv_heads := p.n_kv_heads
	kv_mul := n_heads / n_kv_heads
	seq_len := p.seq_len
	kv_dim := n_kv_heads * head_dim
	eps := p.rms_eps
	att_scale := 1.0 / math.sqrt_f32(f32(head_dim))

	slot := t.full_slot[l]
	loff := slot * seq_len * kv_dim
	k_slice := s.key_cache[loff + pos * kv_dim:loff + (pos + 1) * kv_dim]
	v_slice := s.value_cache[loff + pos * kv_dim:loff + (pos + 1) * kv_dim]

	rmsnorm(s.xb, s.x, lw.attn_norm, eps)

	matmul_t(s.qproj, s.xb, &fa.wq, dim, n_heads * head_dim * 2)
	matmul_t(k_slice, s.xb, &fa.wk, dim, kv_dim)
	matmul_t(v_slice, s.xb, &fa.wv, dim, kv_dim)

	// q_proj output is [head0_q(256) head0_gate(256) head1_q(256) ...]; split q
	// out per head and RMSNorm it. gate stays in s.qproj for later.
	half2 := head_dim * 2
	for h in 0 ..< n_heads {
		qsrc := s.qproj[h * half2:h * half2 + head_dim]
		qdst := s.q[h * head_dim:(h + 1) * head_dim]
		rmsnorm(qdst, qsrc, fa.q_norm, eps)
	}
	for h in 0 ..< n_kv_heads {
		kh := k_slice[h * head_dim:(h + 1) * head_dim]
		rmsnorm(kh, kh, fa.k_norm, eps)
	}

	// partial-rotary MRoPE (text-only form) on q and k
	for h in 0 ..< n_heads {
		apply_mrope_text_head(s.q[h * head_dim:(h + 1) * head_dim], pos, t.inv_freq, p.rotary_dim)
	}
	for h in 0 ..< n_kv_heads {
		apply_mrope_text_head(k_slice[h * head_dim:(h + 1) * head_dim], pos, t.inv_freq, p.rotary_dim)
	}

	// multihead attention
	for h in 0 ..< n_heads {
		qh := s.q[h * head_dim:(h + 1) * head_dim]
		att := s.att[h * seq_len:h * seq_len + pos + 1]
		kh := h / kv_mul

		for tt in 0 ..= pos {
			krow := s.key_cache[loff + tt * kv_dim + kh * head_dim:loff + tt * kv_dim + kh * head_dim + head_dim]
			att[tt] = ggml.dot_f32(qh, krow, head_dim) * att_scale
		}
		softmax(att)

		xb3h := s.xb3[h * head_dim:(h + 1) * head_dim]
		for i in 0 ..< head_dim {
			xb3h[i] = 0
		}
		for tt in 0 ..= pos {
			vrow := s.value_cache[loff + tt * kv_dim + kh * head_dim:loff + tt * kv_dim + kh * head_dim + head_dim]
			a := att[tt]
			for i in 0 ..< head_dim {
				xb3h[i] += a * vrow[i]
			}
		}
	}

	// attention output gate: o_h *= sigmoid(gate_h), elementwise per head
	for h in 0 ..< n_heads {
		gate_off := h * half2 + head_dim
		xb3h := s.xb3[h * head_dim:(h + 1) * head_dim]
		for i in 0 ..< head_dim {
			xb3h[i] *= sigmoid_f32(s.qproj[gate_off + i])
		}
	}

	matmul_t(s.xb2, s.xb3, &fa.wo, n_heads * head_dim, dim)
	for i in 0 ..< dim {
		s.x[i] += s.xb2[i]
	}

	mlp_layer(t, l)
}

forward_linear_layer :: proc(t: ^Transformer, l: int, pos: int) {
	p := &t.config
	w := &t.weights
	s := &t.state
	lw := &w.layers[l]
	la := &lw.linear

	dim := p.dim
	eps := p.rms_eps
	key_dim := p.lin_n_k_heads * p.lin_head_k_dim
	value_dim := p.lin_n_v_heads * p.lin_head_v_dim
	conv_dim := key_dim * 2 + value_dim
	kernel := p.lin_conv_kernel
	head_k_dim := p.lin_head_k_dim
	head_v_dim := p.lin_head_v_dim
	n_vh := p.lin_n_v_heads
	n_kh := p.lin_n_k_heads
	scale := 1.0 / math.sqrt_f32(f32(head_k_dim))

	slot := t.lin_slot[l]
	conv_stride := conv_dim * (kernel - 1)
	conv_state := s.conv_states[slot * conv_stride:slot * conv_stride + conv_stride]
	state_stride := n_vh * head_k_dim * head_v_dim
	state_base := slot * state_stride

	rmsnorm(s.xb, s.x, lw.attn_norm, eps)

	matmul_t(s.qkv_raw, s.xb, &la.in_qkv, dim, conv_dim)
	matmul_t(s.z_vec, s.xb, &la.in_z, dim, value_dim)
	matmul_t(s.b_vec, s.xb, &la.in_b, dim, n_vh)
	matmul_t(s.a_vec, s.xb, &la.in_a, dim, n_vh)

	// depthwise causal conv1d single-token update; conv weight is F32 in the GGUF.
	conv_w := (cast([^]f32)raw_data(la.conv.data))[:conv_dim * kernel]
	conv1d_step(s.qkv_out, s.qkv_raw, conv_state, conv_w, conv_dim, kernel)

	q_raw := s.qkv_out[0:key_dim]
	k_raw := s.qkv_out[key_dim:key_dim * 2]
	v_vec := s.qkv_out[key_dim * 2:key_dim * 2 + value_dim]

	// l2norm q,k per k_head; scale q by 1/sqrt(head_k_dim)
	for kh in 0 ..< n_kh {
		qsrc := q_raw[kh * head_k_dim:kh * head_k_dim + head_k_dim]
		ksrc := k_raw[kh * head_k_dim:kh * head_k_dim + head_k_dim]
		qdst := s.q_lin[kh * head_k_dim:kh * head_k_dim + head_k_dim]
		kdst := s.k_lin[kh * head_k_dim:kh * head_k_dim + head_k_dim]
		l2norm_into(qdst, qsrc, head_k_dim, 1e-6)
		l2norm_into(kdst, ksrc, head_k_dim, 1e-6)
		for i in 0 ..< head_k_dim {
			qdst[i] *= scale
		}
	}

	// gated delta-rule recurrent step per value head. V-heads are stored in
	// ggml "tiled" order in the GGUF, so v-head vh maps to k-head (vh % n_kh).
	for vh in 0 ..< n_vh {
		kh := vh % n_kh
		beta := sigmoid_f32(s.b_vec[vh])
		// ssm_a (GGUF blk.N.ssm_a) stores -exp(A_log) precomputed at conversion;
		// use it directly, do NOT apply another -exp().
		g_val := la.a_decay[vh] * softplus_f32(s.a_vec[vh] + la.dt_bias[vh])
		g_decay := math.exp_f32(g_val)

		q_t := s.q_lin[kh * head_k_dim:kh * head_k_dim + head_k_dim]
		k_t := s.k_lin[kh * head_k_dim:kh * head_k_dim + head_k_dim]
		v_t := v_vec[vh * head_v_dim:vh * head_v_dim + head_v_dim]
		st_off := state_base + vh * head_k_dim * head_v_dim
		st := s.recurrent_states[st_off:st_off + head_k_dim * head_v_dim]
		out_h := s.lin_out[vh * head_v_dim:vh * head_v_dim + head_v_dim]
		delta_recurrent_step(out_h, st, q_t, k_t, v_t, s.delta_scr, beta, g_decay, head_k_dim, head_v_dim)
	}

	// RMSNormGated: standard rmsnorm (norm_w, ones-init) then elementwise silu(z)
	for vh in 0 ..< n_vh {
		oh := s.lin_out[vh * head_v_dim:vh * head_v_dim + head_v_dim]
		zh := s.z_vec[vh * head_v_dim:vh * head_v_dim + head_v_dim]
		rmsnorm(oh, oh, la.norm_w, eps)
		for i in 0 ..< head_v_dim {
			oh[i] *= silu_f32(zh[i])
		}
	}

	matmul_t(s.xb2, s.lin_out, &la.out, value_dim, dim)
	for i in 0 ..< dim {
		s.x[i] += s.xb2[i]
	}

	mlp_layer(t, l)
}

mlp_layer :: proc(t: ^Transformer, l: int) {
	p := &t.config
	w := &t.weights
	s := &t.state
	lw := &w.layers[l]

	rmsnorm(s.xb, s.x, lw.ffn_norm, p.rms_eps)
	matmul_t(s.hb, s.xb, &lw.w1, p.dim, p.hidden_dim)
	matmul_t(s.hb2, s.xb, &lw.w3, p.dim, p.hidden_dim)
	for i in 0 ..< p.hidden_dim {
		s.hb2[i] = silu_f32(s.hb[i]) * s.hb2[i]
	}
	matmul_t(s.xb, s.hb2, &lw.w2, p.hidden_dim, p.dim)
	for i in 0 ..< p.dim {
		s.x[i] += s.xb[i]
	}
}
