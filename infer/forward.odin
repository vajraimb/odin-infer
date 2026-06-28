/* Forward pass for Qwen3 transformer */

package infer

import ggml "ggml:ggml"

import "core:fmt"
import "core:math"
import "core:os"

// Copy/dequantize one row of the embedding table for `token` into dst[0:dim].
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

	dim := p.dim
	hidden_dim := p.hidden_dim
	head_dim := p.head_dim
	n_heads := p.n_heads
	n_kv_heads := p.n_kv_heads
	n_layers := p.n_layers
	seq_len := p.seq_len
	kv_dim := n_kv_heads * head_dim
	kv_mul := n_heads / n_kv_heads
	att_head_dim := n_heads * head_dim
	eps := p.rms_eps

	get_embedding_row(&w.token_embedding, token, dim, s.x)

	for l in 0 ..< n_layers {
		lw := &w.layers[l]
		loff := l * seq_len * kv_dim
		k_slice := s.key_cache[loff + pos * kv_dim:loff + (pos + 1) * kv_dim]
		v_slice := s.value_cache[loff + pos * kv_dim:loff + (pos + 1) * kv_dim]

		rmsnorm(s.xb, s.x, lw.attn_norm, eps)

		matmul_t(s.q, s.xb, &lw.wq, dim, att_head_dim)
		matmul_t(k_slice, s.xb, &lw.wk, dim, kv_dim)
		matmul_t(v_slice, s.xb, &lw.wv, dim, kv_dim)

		// RoPE + per-head Q/K RMSNorm
		for h in 0 ..< n_heads {
			q := s.q[h * head_dim:(h + 1) * head_dim]
			rmsnorm(q, q, lw.q_norm, eps)

			if h < n_kv_heads {
				k := k_slice[h * head_dim:(h + 1) * head_dim]
				rmsnorm(k, k, lw.k_norm, eps)
			}

			for i in 0 ..< head_dim / 2 {
				freq := 1.0 / math.pow_f32(p.rope_freq, f32(i) / f32(head_dim / 2))
				fcr := math.cos_f32(f32(pos) * freq)
				fci := math.sin_f32(f32(pos) * freq)

				x_q := q[i]
				y_q := q[i + head_dim / 2]
				q[i] = x_q * fcr - y_q * fci
				q[i + head_dim / 2] = x_q * fci + y_q * fcr

				if h < n_kv_heads {
					k := k_slice[h * head_dim:(h + 1) * head_dim]
					x_k := k[i]
					y_k := k[i + head_dim / 2]
					k[i] = x_k * fcr - y_k * fci
					k[i + head_dim / 2] = x_k * fci + y_k * fcr
				}
			}
		}

		// multihead attention
		for h in 0 ..< n_heads {
			q := s.q[h * head_dim:(h + 1) * head_dim]
			att := s.att[h * seq_len:h * seq_len + pos + 1]

			for t in 0 ..= pos {
				k := s.key_cache[loff + t * kv_dim + (h / kv_mul) * head_dim:]
				score: f32 = 0
				for i in 0 ..< head_dim {
					score += q[i] * k[i]
				}
				att[t] = score / math.sqrt_f32(f32(head_dim))
			}

			softmax(att)

			xb3 := s.xb3[h * head_dim:(h + 1) * head_dim]
			for i in 0 ..< head_dim {
				xb3[i] = 0
			}

			for t in 0 ..= pos {
				v := s.value_cache[loff + t * kv_dim + (h / kv_mul) * head_dim:]
				a := att[t]
				for i in 0 ..< head_dim {
					xb3[i] += a * v[i]
				}
			}
		}

		matmul_t(s.xb2, s.xb3, &lw.wo, att_head_dim, dim)

		for i in 0 ..< dim {
			s.x[i] += s.xb2[i]
		}

		rmsnorm(s.xb, s.x, lw.ffn_norm, eps)

		matmul_t(s.hb, s.xb, &lw.w1, dim, hidden_dim)
		matmul_t(s.hb2, s.xb, &lw.w3, dim, hidden_dim)

		for i in 0 ..< hidden_dim {
			val := s.hb[i]
			val *= 1.0 / (1.0 + math.exp_f32(-val))
			val *= s.hb2[i]
			s.hb2[i] = val
		}

		matmul_t(s.xb, s.hb2, &lw.w2, hidden_dim, dim)

		for i in 0 ..< dim {
			s.x[i] += s.xb[i]
		}
	}

	rmsnorm(s.x, s.x, w.output_norm, eps)
	matmul_t(s.logits, s.x, &w.output, dim, p.vocab_size)

	return s.logits
}
