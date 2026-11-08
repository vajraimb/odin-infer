#!/usr/bin/env python3
"""Generate a tiny synthetic qwen35moe GGUF (all-F32) plus a Python CPU
reference of the first two decode steps. Used to validate the Odin MoE loader
and CPU forward end-to-end (tools/moe_load_test.odin).

Layout mirrors the real unsloth Qwen3.5-35B-A3B GGUF (verified by parsing its
header): ffn_gate_inp [dim, n_expert], ffn_up_exps/gate_exps [dim, n_ff_exp,
n_expert], ffn_down_exps [n_ff_exp, dim, n_expert], ffn_*_shexp 2D, where
dims[0] is the fastest (ggml ne[0] = contraction dim).
"""
import json
import math
import struct
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "models" / "moe_test.gguf"
REF = Path(__file__).resolve().parent.parent / "models" / "moe_test_ref.json"

DIM = 32
VOCAB = 64
N_LAYER = 2
N_HEADS = 2
HEAD_DIM = 16
KV_HEADS = 2
ROTARY_DIM = 4  # partial rotary 0.25
N_EXPERT = 6
N_EXPERT_USED = 2
N_FF_EXP = 16
N_FF_SHEXP = 16
EPS = 1e-6
CTX = 64

rng = np.random.default_rng(42)


def f32a(*shape):
    return rng.uniform(-0.5, 0.5, size=shape).astype(np.float32)


def gguf_write(path, meta, tensors):
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<IQQ", 3, len(tensors), len(meta)))
        for k, (vt, v) in meta.items():
            kb = k.encode()
            f.write(struct.pack("<Q", len(kb)) + kb)
            f.write(struct.pack("<I", vt))
            if vt == 4:
                f.write(struct.pack("<I", v))
            elif vt == 6:
                f.write(struct.pack("<f", v))
            elif vt == 8:
                vb = v.encode()
                f.write(struct.pack("<Q", len(vb)) + vb)
            elif vt == 10:
                f.write(struct.pack("<Q", v))
        off = 0
        for name, arr in tensors:
            nb = len(name.encode())
            f.write(struct.pack("<Q", nb) + name.encode())
            # GGUF dims are in ggml ne order (ne0 fastest) = reverse of numpy shape
            f.write(struct.pack("<I", len(arr.shape)))
            f.write(struct.pack("<" + "Q" * len(arr.shape), *arr.shape[::-1]))
            f.write(struct.pack("<IQ", 0, off))  # F32, offset
            off += arr.nbytes
        pos = f.tell()
        pad = (-pos) % 32  # GGUF: data section starts at 32-byte alignment
        if pad:
            f.write(b"\x00" * pad)
        for name, arr in tensors:
            f.write(arr.tobytes())


def rms_norm(x, w, eps):
    return (w * (x / math.sqrt(np.mean(x * x) + eps))).astype(np.float32)


def silu(x):
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def sigmoid(x):
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def apply_rope(v, pos, inv_freq, rotary_dim):
    half = rotary_dim // 2
    ang = pos * inv_freq[:half]
    c, s = np.cos(ang), np.sin(ang)
    x, y = v[:half].copy(), v[half:rotary_dim].copy()
    v[:half] = x * c - y * s
    v[half:rotary_dim] = x * s + y * c


def moe_ffn(x, mw):
    """x: [dim]; returns [dim]. Mirrors llama.cpp qwen35moe build_moe_ffn."""
    logits = mw["gate_inp"] @ x
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    sel = np.argsort(-probs)[:N_EXPERT_USED]
    weights = probs[sel]
    s = weights.sum()
    if s < 6.103515625e-5:
        s = 6.103515625e-5
    weights = weights / s
    out = np.zeros(DIM, dtype=np.float32)
    for e, w in zip(sel, weights):
        up = mw["up_exps"][e] @ x        # [n_ff_exp, dim] @ [dim]
        gate = mw["gate_exps"][e] @ x
        h = silu(gate) * up
        out += w * (mw["down_exps"][e] @ h)  # [dim, n_ff_exp] @ [n_ff_exp]
    sh = silu(mw["sh_gate"] @ x) * (mw["sh_up"] @ x)
    g = sigmoid(mw["sh_gate_inp"] @ x)
    out += g * (mw["sh_down"] @ sh)
    return out.astype(np.float32)


def main():
    meta = {
        "general.architecture": (8, "qwen35moe"),
        "qwen35moe.block_count": (4, N_LAYER),
        "qwen35moe.embedding_length": (4, DIM),
        "qwen35moe.feed_forward_length": (4, 64),
        "qwen35moe.context_length": (4, CTX),
        "qwen35moe.attention.head_count": (4, N_HEADS),
        "qwen35moe.attention.head_count_kv": (4, KV_HEADS),
        "qwen35moe.attention.key_length": (4, HEAD_DIM),
        "qwen35moe.attention.value_length": (4, HEAD_DIM),
        "qwen35moe.attention.layer_norm_rms_epsilon": (6, EPS),
        "qwen35moe.rope.dimension_count": (4, ROTARY_DIM),
        "qwen35moe.rope.freq_base": (6, 10000.0),
        "qwen35moe.full_attention_interval": (4, 1),
        "qwen35moe.ssm.conv_kernel": (4, 4),
        "qwen35moe.ssm.group_count": (4, 4),
        "qwen35moe.ssm.time_step_rank": (4, 2),
        "qwen35moe.ssm.state_size": (4, 8),
        "qwen35moe.ssm.inner_size": (4, 16),
        "qwen35moe.expert_count": (4, N_EXPERT),
        "qwen35moe.expert_used_count": (4, N_EXPERT_USED),
        "qwen35moe.expert_feed_forward_length": (4, N_FF_EXP),
        "qwen35moe.expert_shared_feed_forward_length": (4, N_FF_SHEXP),
        "qwen35moe.expert_shared_count": (4, 1),
        "qwen35moe.expert_weights_scale": (6, 1.0),
        "qwen35moe.nextn_predict_layers": (4, 0),
    }

    tensors = []
    emb = f32a(VOCAB, DIM)  # numpy [token][dim]; GGUF dims recorded as reversed (ne0 = dim)
    tensors.append(("token_embd.weight", emb))
    out_norm = f32a(DIM)
    tensors.append(("output_norm.weight", out_norm))
    out_w = f32a(VOCAB, DIM)
    tensors.append(("output.weight", out_w))

    layers = []
    for l in range(N_LAYER):
        lw = {}
        lw["attn_norm"] = f32a(DIM)
        lw["ffn_norm"] = f32a(DIM)
        lw["q_norm"] = f32a(HEAD_DIM)
        lw["k_norm"] = f32a(HEAD_DIM)
        lw["wq"] = f32a(N_HEADS * HEAD_DIM * 2, DIM)
        lw["wk"] = f32a(KV_HEADS * HEAD_DIM, DIM)
        lw["wv"] = f32a(KV_HEADS * HEAD_DIM, DIM)
        lw["wo"] = f32a(DIM, N_HEADS * HEAD_DIM)
        lw["gate_inp"] = f32a(N_EXPERT, DIM)
        lw["up_exps"] = f32a(N_EXPERT, N_FF_EXP, DIM)
        lw["gate_exps"] = f32a(N_EXPERT, N_FF_EXP, DIM)
        lw["down_exps"] = f32a(N_EXPERT, DIM, N_FF_EXP)
        lw["sh_up"] = f32a(N_FF_SHEXP, DIM)
        lw["sh_gate"] = f32a(N_FF_SHEXP, DIM)
        lw["sh_down"] = f32a(DIM, N_FF_SHEXP)
        lw["sh_gate_inp"] = f32a(DIM)
        layers.append(lw)
        blk = f"blk.{l}."
        tensors.append((blk + "attn_norm.weight", lw["attn_norm"]))
        tensors.append((blk + "post_attention_norm.weight", lw["ffn_norm"]))
        tensors.append((blk + "attn_q_norm.weight", lw["q_norm"]))
        tensors.append((blk + "attn_k_norm.weight", lw["k_norm"]))
        tensors.append((blk + "attn_q.weight", lw["wq"]))
        tensors.append((blk + "attn_k.weight", lw["wk"]))
        tensors.append((blk + "attn_v.weight", lw["wv"]))
        tensors.append((blk + "attn_output.weight", lw["wo"]))
        tensors.append((blk + "ffn_gate_inp.weight", lw["gate_inp"]))
        tensors.append((blk + "ffn_up_exps.weight", lw["up_exps"]))
        tensors.append((blk + "ffn_gate_exps.weight", lw["gate_exps"]))
        tensors.append((blk + "ffn_down_exps.weight", lw["down_exps"]))
        tensors.append((blk + "ffn_up_shexp.weight", lw["sh_up"]))
        tensors.append((blk + "ffn_gate_shexp.weight", lw["sh_gate"]))
        tensors.append((blk + "ffn_down_shexp.weight", lw["sh_down"]))
        tensors.append((blk + "ffn_gate_inp_shexp.weight", lw["sh_gate_inp"]))

    gguf_write(OUT, meta, tensors)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")

    # ---- CPU reference: two decode steps ----
    inv_freq = 1.0 / (10000.0 ** (np.arange(ROTARY_DIM // 2) / (ROTARY_DIM // 2)))
    tokens = [7, 23]

    kv = [[] for _ in layers]
    ref_logits = []
    for pos, tok in enumerate(tokens):
        x = emb[tok].astype(np.float32)
        for l, lw in enumerate(layers):
            xb = rms_norm(x, lw["attn_norm"], EPS)
            qproj = lw["wq"] @ xb
            k = lw["wk"] @ xb
            v = lw["wv"] @ xb
            q = np.zeros(N_HEADS * HEAD_DIM, dtype=np.float32)
            for h in range(N_HEADS):
                qsrc = qproj[h * 2 * HEAD_DIM:h * 2 * HEAD_DIM + HEAD_DIM]
                q[h * HEAD_DIM:(h + 1) * HEAD_DIM] = rms_norm(qsrc, lw["q_norm"], EPS)
            for h in range(KV_HEADS):
                kh = k[h * HEAD_DIM:(h + 1) * HEAD_DIM]
                k[h * HEAD_DIM:(h + 1) * HEAD_DIM] = rms_norm(kh, lw["k_norm"], EPS)
            for h in range(N_HEADS):
                apply_rope(q[h * HEAD_DIM:(h + 1) * HEAD_DIM], pos, inv_freq, ROTARY_DIM)
            for h in range(KV_HEADS):
                apply_rope(k[h * HEAD_DIM:(h + 1) * HEAD_DIM], pos, inv_freq, ROTARY_DIM)
            kv[l].append((k.copy(), v.copy()))
            att_scale = 1.0 / math.sqrt(HEAD_DIM)
            xb3 = np.zeros(N_HEADS * HEAD_DIM, dtype=np.float32)
            for h in range(N_HEADS):
                qh = q[h * HEAD_DIM:(h + 1) * HEAD_DIM]
                kh = h // (N_HEADS // KV_HEADS)
                att = np.zeros(pos + 1, dtype=np.float32)
                for tt in range(pos + 1):
                    krow = kv[l][tt][0][kh * HEAD_DIM:(kh + 1) * HEAD_DIM]
                    att[tt] = float(qh @ krow) * att_scale
                att = np.exp(att - att.max())
                att /= att.sum()
                out_h = np.zeros(HEAD_DIM, dtype=np.float32)
                for tt in range(pos + 1):
                    out_h += att[tt] * kv[l][tt][1][kh * HEAD_DIM:(kh + 1) * HEAD_DIM]
                gate = qproj[h * 2 * HEAD_DIM + HEAD_DIM:(h + 1) * 2 * HEAD_DIM]
                xb3[h * HEAD_DIM:(h + 1) * HEAD_DIM] = out_h * sigmoid(gate)
            x = x + (lw["wo"] @ xb3)
            xb = rms_norm(x, lw["ffn_norm"], EPS)
            x = x + moe_ffn(xb, lw)
        ref_logits.append((out_w @ rms_norm(x, out_norm, EPS)).tolist())

    REF.write_text(json.dumps({"tokens": tokens, "logits": ref_logits}, separators=(",", ":")))
    print(f"wrote reference {REF}")


if __name__ == "__main__":
    main()
