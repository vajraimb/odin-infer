#!/usr/bin/env python3
"""Synthetic Qwen3.5 GGUF for the lavapipe Vulkan-vs-CPU greedy check.

Every contraction length is 256, so Q4_0/Q8_0 (block 32) and the K-quants
(block 256) are legal. Scales are small finite float16 values. The bytes are
not a real model; both backends dequantize the same payload.

Layer 0 is linear attention, layer 1 is full attention (interval 2).
"""
import struct
import sys
from pathlib import Path

import numpy as np

DIM = 256
HIDDEN = 256
VOCAB = 64
N_LAYER = 2
N_HEADS = 4
N_KV = 2
HEAD = 64
ROTARY = 16
N_K = 2
N_V = 2
HEAD_K = 128
HEAD_V = 128
KERNEL = 4
CTX = 32

KEY = N_K * HEAD_K
VALUE = N_V * HEAD_V
CONV = KEY * 2 + VALUE

# ggml type ids
F32, F16 = 0, 1
Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 = 2, 3, 6, 7, 8
Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K = 10, 11, 12, 13, 14, 15

BLOCK = {
    F32: (1, 4), F16: (1, 2),
    Q4_0: (32, 18), Q4_1: (32, 20), Q5_0: (32, 22), Q5_1: (32, 24), Q8_0: (32, 34),
    Q2_K: (256, 84), Q3_K: (256, 110), Q4_K: (256, 144), Q5_K: (256, 176),
    Q6_K: (256, 210), Q8_K: (256, 292),
}

D_OFF = {
    Q4_0: (0,), Q5_0: (0,), Q8_0: (0,),
    Q4_1: (0, 2), Q5_1: (0, 2),
    Q4_K: (0, 2), Q5_K: (0, 2),
    Q2_K: (80, 82), Q3_K: (108,), Q6_K: (208,),
}


def f16_bytes(x):
    return np.float16(x).tobytes()


def quant_matrix(rng, rows, cols, kind):
    elems, bl = BLOCK[kind]
    assert cols % elems == 0, (kind, cols)
    rb = (cols // elems) * bl
    raw = rng.integers(0, 256, size=(rows, rb), dtype=np.uint8)
    if kind == F32:
        vals = rng.uniform(-0.02, 0.02, size=(rows, cols)).astype(np.float32)
        return vals.tobytes()
    if kind == F16:
        vals = rng.uniform(-0.02, 0.02, size=(rows, cols)).astype(np.float16)
        return vals.tobytes()
    if kind == Q8_K:
        d = np.float32(0.02).tobytes()
        for r in range(rows):
            for o in range(0, rb, bl):
                raw[r, o:o + 4] = np.frombuffer(d, dtype=np.uint8)
        return raw.tobytes()
    scale = f16_bytes(0.02)
    sb = np.frombuffer(scale, dtype=np.uint8)
    for off in D_OFF[kind]:
        for r in range(rows):
            for o in range(0, rb, bl):
                raw[r, o + off:o + off + 2] = sb
    return raw.tobytes()


def write_gguf(path, meta, tensors):
    """tensors: list of (name, ggml_type, numpy_shape, bytes)."""
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
            else:
                raise SystemExit(f"bad meta type {vt}")
        off = 0
        for name, kind, shape, raw in tensors:
            nb = name.encode()
            f.write(struct.pack("<Q", len(nb)) + nb)
            f.write(struct.pack("<I", len(shape)))
            f.write(struct.pack("<" + "Q" * len(shape), *shape[::-1]))
            f.write(struct.pack("<IQ", kind, off))
            off += len(raw)
        pos = f.tell()
        pad = (-pos) % 32
        if pad:
            f.write(b"\x00" * pad)
        for _, _, _, raw in tensors:
            f.write(raw)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/odinfer_vulkan_tiny.gguf")
    rng = np.random.default_rng(7)
    meta = {
        "general.architecture": (8, "qwen35"),
        "qwen35.block_count": (4, N_LAYER),
        "qwen35.embedding_length": (4, DIM),
        "qwen35.feed_forward_length": (4, HIDDEN),
        "qwen35.context_length": (4, CTX),
        "qwen35.attention.head_count": (4, N_HEADS),
        "qwen35.attention.head_count_kv": (4, N_KV),
        "qwen35.attention.key_length": (4, HEAD),
        "qwen35.attention.value_length": (4, HEAD),
        "qwen35.attention.layer_norm_rms_epsilon": (6, 1e-6),
        "qwen35.rope.dimension_count": (4, ROTARY),
        "qwen35.rope.freq_base": (6, 10000.0),
        "qwen35.full_attention_interval": (4, 2),
        "qwen35.ssm.conv_kernel": (4, KERNEL),
        "qwen35.ssm.group_count": (4, N_K),
        "qwen35.ssm.time_step_rank": (4, N_V),
        "qwen35.ssm.state_size": (4, HEAD_K),
        "qwen35.ssm.inner_size": (4, VALUE),
        "qwen35.nextn_predict_layers": (4, 0),
    }
    tensors = []

    def q(name, rows, cols, kind):
        raw = quant_matrix(rng, rows, cols, kind)
        tensors.append((name, kind, (rows, cols), raw))

    def f32(name, arr):
        a = np.asarray(arr, dtype=np.float32)
        tensors.append((name, F32, tuple(a.shape), a.tobytes()))

    q("token_embd.weight", VOCAB, DIM, Q8_0)
    f32("output_norm.weight", np.ones(DIM, dtype=np.float32))
    q("output.weight", VOCAB, DIM, Q6_K)

    # layer 0 linear
    f32("blk.0.attn_norm.weight", np.ones(DIM, dtype=np.float32))
    f32("blk.0.post_attention_norm.weight", np.ones(DIM, dtype=np.float32))
    q("blk.0.attn_qkv.weight", CONV, DIM, Q4_K)
    q("blk.0.attn_gate.weight", VALUE, DIM, Q5_1)
    q("blk.0.ssm_beta.weight", N_V, DIM, Q4_0)
    q("blk.0.ssm_alpha.weight", N_V, DIM, Q8_0)
    conv = np.full((CONV, KERNEL), 0.01, dtype=np.float32)
    f32("blk.0.ssm_conv1d.weight", conv)
    f32("blk.0.ssm_dt.bias", np.zeros(N_V, dtype=np.float32))
    f32("blk.0.ssm_a", np.full(N_V, -0.05, dtype=np.float32))
    f32("blk.0.ssm_norm.weight", np.ones(HEAD_V, dtype=np.float32))
    q("blk.0.ssm_out.weight", DIM, VALUE, Q8_K)
    q("blk.0.ffn_gate.weight", HIDDEN, DIM, Q2_K)
    q("blk.0.ffn_up.weight", HIDDEN, DIM, Q3_K)
    q("blk.0.ffn_down.weight", DIM, HIDDEN, Q5_K)

    # layer 1 full
    f32("blk.1.attn_norm.weight", np.ones(DIM, dtype=np.float32))
    f32("blk.1.post_attention_norm.weight", np.ones(DIM, dtype=np.float32))
    f32("blk.1.attn_q_norm.weight", np.ones(HEAD, dtype=np.float32))
    f32("blk.1.attn_k_norm.weight", np.ones(HEAD, dtype=np.float32))
    q("blk.1.attn_q.weight", N_HEADS * HEAD * 2, DIM, Q4_K)
    q("blk.1.attn_k.weight", N_KV * HEAD, DIM, F16)
    q("blk.1.attn_v.weight", N_KV * HEAD, DIM, Q5_0)
    q("blk.1.attn_output.weight", DIM, N_HEADS * HEAD, Q4_1)
    q("blk.1.ffn_gate.weight", HIDDEN, DIM, Q4_K)
    q("blk.1.ffn_up.weight", HIDDEN, DIM, Q6_K)
    q("blk.1.ffn_down.weight", DIM, HIDDEN, Q8_0)

    write_gguf(out, meta, tensors)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
