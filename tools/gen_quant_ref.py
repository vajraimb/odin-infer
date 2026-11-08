#!/usr/bin/env python3
"""Reference dequant + fixture generator for packages/ggml/quant.odin.

Independently transcribes llama.cpp's dequantize_row_* functions (from the
downloaded master snapshot /tmp/ggml-quants.c) in numpy, using the codebook
tables from GGML_TABLE_BEGIN blocks in /tmp/ggml-common.h.

Modes:
  --emit-odin PATH   write Odin table constants (packages/ggml/tables.odin)
  default            write the binary fixture (tools/quant_fixture.bin)

The fixture contains, per quant kind: n (elements per row), row_bytes, and
N random rows of raw block bytes plus the expected f32 dequant outputs.
All arithmetic mirrors the C code operation-for-operation in float32, so the
expected values are bit-exact w.r.t. llama.cpp CPU dequant.
"""

import argparse
import os
import re
import struct
import sys

import numpy as np

HDR_PATH = "/tmp/ggml-common.h"

TABLE_SPECS = [
    ("kmask_iq2xs", "uint8_t"),
    ("ksigns_iq2xs", "uint8_t"),
    ("ksigns64", "uint64_t"),
    ("iq2xxs_grid", "uint64_t"),
    ("iq2xs_grid", "uint64_t"),
    ("iq2s_grid", "uint64_t"),
    ("iq3xxs_grid", "uint32_t"),
    ("iq3s_grid", "uint32_t"),
    ("kvalues_iq4nl", "int8_t"),
    ("iq1s_grid", "uint64_t"),
]

EXPECTED_COUNTS = {
    "kmask_iq2xs": 8,
    "ksigns_iq2xs": 128,
    "ksigns64": 128,
    "iq2xxs_grid": 256,
    "iq2xs_grid": 512,
    "iq2s_grid": 1024,
    "iq3xxs_grid": 256,
    "iq3s_grid": 512,
    "kvalues_iq4nl": 16,
    "iq1s_grid": 2048,  # NGRID_IQ1S
}


def parse_tables(path=HDR_PATH):
    text = open(path).read()
    tables = {}
    pat = re.compile(
        r"GGML_TABLE_BEGIN\(\s*(uint8_t|uint32_t|uint64_t|int8_t)\s*,\s*(\w+)\s*,\s*([\w]+)\s*\)\s*(.*?)\s*GGML_TABLE_END\(\)",
        re.S,
    )
    for m in pat.finditer(text):
        ctype, name, size_expr, body = m.group(1), m.group(2), m.group(3), m.group(4)
        vals = [int(v, 0) for v in re.findall(r"0x[0-9a-fA-F]+|-?\d+", body)]
        tables[name] = (ctype, vals)
    for name, want in EXPECTED_COUNTS.items():
        if name not in tables:
            raise SystemExit(f"table {name} not found in {path}")
        got = len(tables[name][1])
        if got != want:
            raise SystemExit(f"table {name}: expected {want} values, got {got}")
    return tables


ODIN_CTYPE = {"uint8_t": "u8", "uint32_t": "u32", "uint64_t": "u64", "int8_t": "i8"}


def emit_odin(tables, path):
    lines = []
    lines.append("// Codebook tables extracted from ggml-common.h (llama.cpp master)")
    lines.append("// by tools/gen_quant_ref.py --emit-odin. Entries are stored as the")
    lines.append("// C integer literals; the C dequant code reads them byte-wise in")
    lines.append("// little-endian order, i.e. byte j of entry v is (v >> (8*j)) & 0xff.")
    lines.append("")
    lines.append("package ggml")
    lines.append("")
    for name, ctype in TABLE_SPECS:
        vals = tables[name][1]
        otype = ODIN_CTYPE[ctype]
        lines.append("@(rodata)")
        lines.append(f"{name} := [{len(vals)}]{otype} {{")
        row = []
        for i, v in enumerate(vals):
            row.append(str(v))
            if len(row) == 8:
                lines.append("\t" + ", ".join(row) + ",")
                row = []
        if row:
            lines.append("\t" + ", ".join(row) + ",")
        lines.append("}")
        lines.append("")
    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text)
    print(f"wrote {path}")


# --------------------------------------------------------------------------
# float helpers (mirror GGML_FP16_TO_FP32 / FP32 little-endian reads)
# --------------------------------------------------------------------------

def f16(b0, b1):
    return np.float32(struct.unpack("<e", bytes([b0, b1]))[0])


def f32le(b):
    return np.float32(struct.unpack("<f", b)[0])


def u16le(b):
    return b[0] | (b[1] << 8)


def u32le(b):
    return b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)


def s8(v):
    return v - 256 if v >= 128 else v


def gbyte(grid, j):
    return (grid >> (8 * j)) & 0xFF


QK_K = 256
QK4_NL = 32
IQ1S_DELTA = np.float32(0.125)
F32 = np.float32


# --------------------------------------------------------------------------
# reference dequants (numpy float32, operation-for-operation like the C code)
# --------------------------------------------------------------------------

def deq_q2_k(src):
    nb = len(src) // 84
    y = np.zeros(nb * QK_K, np.float32)
    yi = 0
    for sb in range(nb):
        b = sb * 84
        # block_q2_K layout: scales[16], qs[64], d(2), dmin(2)
        d = f16(src[b + 80], src[b + 81])
        mn = f16(src[b + 82], src[b + 83])
        is_ = 0
        qoff = 0
        for _ in range(2):
            shift = 0
            for _ in range(4):
                sc = src[b + is_]; is_ += 1
                dl = F32(d * F32(sc & 0xF))
                ml = F32(mn * F32(sc >> 4))
                for l in range(16):
                    y[yi] = F32(dl * F32((src[b + 16 + qoff + l] >> shift) & 3)) - ml
                    yi += 1
                sc = src[b + is_]; is_ += 1
                dl = F32(d * F32(sc & 0xF))
                ml = F32(mn * F32(sc >> 4))
                for l in range(16):
                    y[yi] = F32(dl * F32((src[b + 16 + qoff + l + 16] >> shift) & 3)) - ml
                    yi += 1
                shift += 2
            qoff += 32
    return y


def deq_q3_k(src):
    nb = len(src) // 110
    y = np.zeros(nb * QK_K, np.float32)
    yi = 0
    KMASK1 = 0x03030303
    KMASK2 = 0x0F0F0F0F
    for sb in range(nb):
        b = sb * 110
        # block_q3_K layout: hmask[32], qs[64], scales[12], d(2)
        d = f16(src[b + 108], src[b + 109])
        q = src[b + 32:b + 96]
        hm = src[b:b + 32]
        sc = src[b + 96:b + 108]
        aux = [u32le(sc[0:4]), u32le(sc[4:8]), u32le(sc[8:12]), 0]
        tmp = aux[2]
        aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4)
        aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4)
        aux[0] = (aux[0] & KMASK2) | (((tmp >> 0) & KMASK1) << 4)
        aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4)
        is_ = 0
        qoff = 0
        m = 1
        for _ in range(2):
            shift = 0
            for _ in range(4):
                scv = s8((aux[is_ // 4] >> (8 * (is_ % 4))) & 0xFF); is_ += 1
                dl = F32(d * F32(scv - 32))
                for l in range(16):
                    qv = ((q[qoff + l] >> shift) & 3) - (0 if (hm[l] & m) else 4)
                    y[yi] = F32(dl * F32(qv)); yi += 1
                scv = s8((aux[is_ // 4] >> (8 * (is_ % 4))) & 0xFF); is_ += 1
                dl = F32(d * F32(scv - 32))
                for l in range(16):
                    qv = ((q[qoff + l + 16] >> shift) & 3) - (0 if (hm[l + 16] & m) else 4)
                    y[yi] = F32(dl * F32(qv)); yi += 1
                shift += 2
                m <<= 1
            qoff += 32
    return y


def get_scale_min_k4(j, q):
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


def deq_q5_k(src):
    nb = len(src) // 176
    y = np.zeros(nb * QK_K, np.float32)
    yi = 0
    for sb in range(nb):
        b = sb * 176
        d = f16(src[b], src[b + 1])
        mn = f16(src[b + 2], src[b + 3])
        scales = src[b + 4:b + 16]
        qh = src[b + 16:b + 48]
        ql = src[b + 48:b + 176]
        is_ = 0
        qloff = 0
        u1 = 1
        u2 = 2
        for _ in range(4):
            sc0, m0 = get_scale_min_k4(is_ + 0, scales)
            sc1, m1 = get_scale_min_k4(is_ + 1, scales)
            d1 = F32(d * F32(sc0)); m1v = F32(mn * F32(m0))
            d2 = F32(d * F32(sc1)); m2v = F32(mn * F32(m1))
            for l in range(32):
                v = (ql[qloff + l] & 0xF) + (16 if (qh[l] & u1) else 0)
                y[yi] = F32(d1 * F32(v)) - m1v; yi += 1
            for l in range(32):
                v = (ql[qloff + l] >> 4) + (16 if (qh[l] & u2) else 0)
                y[yi] = F32(d2 * F32(v)) - m2v; yi += 1
            qloff += 32
            is_ += 2
            u1 <<= 2
            u2 <<= 2
    return y


def deq_q8_k(src):
    nb = len(src) // 292
    y = np.zeros(nb * QK_K, np.float32)
    yi = 0
    for sb in range(nb):
        b = sb * 292
        d = f32le(src[b:b + 4])
        for j in range(QK_K):
            y[yi] = F32(d * F32(s8(src[b + 4 + j]))); yi += 1
    return y


def deq_iq2_xxs(src, t):
    nb = len(src) // 66
    y = np.zeros(nb * QK_K, np.float32)
    yi = 0
    g = t["iq2xxs_grid"]; ksigns = t["ksigns_iq2xs"]; kmask = t["kmask_iq2xs"]
    for sb in range(nb):
        b = sb * 66
        d = f16(src[b], src[b + 1])
        for ib32 in range(8):
            # qs is uint16_t[32]: qs + 4*ib32 = byte offset 8*ib32
            qb = b + 2 + 8 * ib32
            aux1 = u32le(src[qb + 4:qb + 8])
            db = F32(F32(d * F32(np.float32(0.5) + F32(aux1 >> 28))) * F32(0.25))
            for l in range(4):
                grid = g[src[qb + l]]
                signs = ksigns[(aux1 >> (7 * l)) & 127]
                for j in range(8):
                    v = F32(gbyte(grid, j))
                    s = F32(-1.0) if (signs & kmask[j]) else F32(1.0)
                    y[yi] = F32(F32(db * v) * s); yi += 1
    return y


def deq_iq2_xs(src, t):
    nb = len(src) // 74
    y = np.zeros(nb * QK_K, np.float32)
    yi = 0
    g = t["iq2xs_grid"]; ksigns = t["ksigns_iq2xs"]; kmask = t["kmask_iq2xs"]
    for sb in range(nb):
        b = sb * 74
        d = f16(src[b], src[b + 1])
        for ib32 in range(8):
            sc = src[b + 66 + ib32]
            db = [
                F32(F32(d * F32(np.float32(0.5) + F32(sc & 0xF))) * F32(0.25)),
                F32(F32(d * F32(np.float32(0.5) + F32(sc >> 4))) * F32(0.25)),
            ]
            for l in range(4):
                q = u16le(src[b + 2 + 2 * (4 * ib32 + l):b + 4 + 2 * (4 * ib32 + l)])
                grid = g[q & 511]
                signs = ksigns[q >> 9]
                dl = db[l // 2]
                for j in range(8):
                    v = F32(gbyte(grid, j))
                    s = F32(-1.0) if (signs & kmask[j]) else F32(1.0)
                    y[yi] = F32(F32(dl * v) * s); yi += 1
    return y


def deq_iq2_s(src, t):
    nb = len(src) // 82
    y = np.zeros(nb * QK_K, np.float32)
    yi = 0
    g = t["iq2s_grid"]; kmask = t["kmask_iq2xs"]
    for sb in range(nb):
        b = sb * 82
        d = f16(src[b], src[b + 1])
        qs = src[b + 2:b + 34]
        signs = src[b + 34:b + 66]
        qh = src[b + 66:b + 74]
        scales = src[b + 74:b + 82]
        qoff = 0
        signoff = 0
        for ib32 in range(8):
            sc = scales[ib32]
            db = [
                F32(F32(d * F32(np.float32(0.5) + F32(sc & 0xF))) * F32(0.25)),
                F32(F32(d * F32(np.float32(0.5) + F32(sc >> 4))) * F32(0.25)),
            ]
            for l in range(4):
                idx = qs[qoff + l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)
                grid = g[idx]
                sn = signs[signoff + l]
                for j in range(8):
                    v = F32(gbyte(grid, j))
                    s = F32(-1.0) if (sn & kmask[j]) else F32(1.0)
                    y[yi] = F32(F32(db[l // 2] * v) * s); yi += 1
            qoff += 4
            signoff += 4
    return y


def deq_iq3_xxs(src, t):
    nb = len(src) // 98
    y = np.zeros(nb * QK_K, np.float32)
    yi = 0
    g = t["iq3xxs_grid"]; ksigns = t["ksigns_iq2xs"]; kmask = t["kmask_iq2xs"]
    for sb in range(nb):
        b = sb * 98
        d = f16(src[b], src[b + 1])
        qs = src[b + 2:b + 66]
        sas = src[b + 66:b + 98]
        qoff = 0
        for ib32 in range(8):
            aux = u32le(sas[4 * ib32:4 * ib32 + 4])
            db = F32(F32(d * F32(np.float32(0.5) + F32(aux >> 28))) * F32(0.5))
            for l in range(4):
                signs = ksigns[(aux >> (7 * l)) & 127]
                g1 = g[qs[qoff + 2 * l]]
                g2 = g[qs[qoff + 2 * l + 1]]
                for j in range(4):
                    v1 = F32(gbyte(g1, j))
                    v2 = F32(gbyte(g2, j))
                    s1 = F32(-1.0) if (signs & kmask[j + 0]) else F32(1.0)
                    s2 = F32(-1.0) if (signs & kmask[j + 4]) else F32(1.0)
                    y[yi + j + 0] = F32(F32(db * v1) * s1)
                    y[yi + j + 4] = F32(F32(db * v2) * s2)
                yi += 8
            qoff += 8
    return y


def deq_iq3_s(src, t):
    nb = len(src) // 110
    y = np.zeros(nb * QK_K, np.float32)
    yi = 0
    g = t["iq3s_grid"]; kmask = t["kmask_iq2xs"]
    for sb in range(nb):
        b = sb * 110
        d = f16(src[b], src[b + 1])
        qs = src[b + 2:b + 66]
        qh = src[b + 66:b + 74]
        signs = src[b + 74:b + 106]
        scales = src[b + 106:b + 110]
        qoff = 0
        signoff = 0
        qhoff = 0
        for _ in range(4):  # ib32: 0..8 step 2; scales index = ib32/2 = qhoff/2
            db1 = F32(d * F32(1 + 2 * (scales[(qhoff // 2)] & 0xF)))
            db2 = F32(d * F32(1 + 2 * (scales[(qhoff // 2)] >> 4)))
            for l in range(4):
                idx1 = qs[qoff + 2 * l + 0] | ((qh[qhoff + 0] << (8 - 2 * l)) & 256)
                idx2 = qs[qoff + 2 * l + 1] | ((qh[qhoff + 0] << (7 - 2 * l)) & 256)
                g1 = g[idx1]
                g2 = g[idx2]
                sn = signs[signoff + l]
                for j in range(4):
                    s1 = F32(-1.0) if (sn & kmask[j + 0]) else F32(1.0)
                    s2 = F32(-1.0) if (sn & kmask[j + 4]) else F32(1.0)
                    y[yi + j + 0] = F32(F32(db1 * F32(gbyte(g1, j))) * s1)
                    y[yi + j + 4] = F32(F32(db1 * F32(gbyte(g2, j))) * s2)
                yi += 8
            qoff += 8
            signoff += 4
            for l in range(4):
                idx1 = qs[qoff + 2 * l + 0] | ((qh[qhoff + 1] << (8 - 2 * l)) & 256)
                idx2 = qs[qoff + 2 * l + 1] | ((qh[qhoff + 1] << (7 - 2 * l)) & 256)
                g1 = g[idx1]
                g2 = g[idx2]
                sn = signs[signoff + l]
                for j in range(4):
                    s1 = F32(-1.0) if (sn & kmask[j + 0]) else F32(1.0)
                    s2 = F32(-1.0) if (sn & kmask[j + 4]) else F32(1.0)
                    y[yi + j + 0] = F32(F32(db2 * F32(gbyte(g1, j))) * s1)
                    y[yi + j + 4] = F32(F32(db2 * F32(gbyte(g2, j))) * s2)
                yi += 8
            qhoff += 2
            qoff += 8
            signoff += 4
    return y


def deq_iq1_s(src, t):
    nb = len(src) // 50
    y = np.zeros(nb * QK_K, np.float32)
    yi = 0
    g = t["iq1s_grid"]
    for sb in range(nb):
        b = sb * 50
        d = f16(src[b], src[b + 1])
        qs = src[b + 2:b + 34]
        qoff = 0
        for ib in range(8):
            qhv = u16le(src[b + 34 + 2 * ib:b + 36 + 2 * ib])
            dl = F32(d * F32(2 * ((qhv >> 12) & 7) + 1))
            delta = F32(-0.125) if (qhv & 0x8000) else IQ1S_DELTA
            for l in range(4):
                idx = qs[qoff + l] | (((qhv >> (3 * l)) & 7) << 8)
                grid = g[idx]
                for j in range(8):
                    gv = F32(s8(gbyte(grid, j)))
                    y[yi] = F32(dl * F32(gv + delta)); yi += 1
            qoff += 4
    return y


def deq_iq1_m(src, t):
    nb = len(src) // 56
    y = np.zeros(nb * QK_K, np.float32)
    yi = 0
    g = t["iq1s_grid"]
    for sb in range(nb):
        b = sb * 56
        # block_iq1_m layout: qs[32], qh[16], scales[8]
        sc = [u16le(src[b + 48:b + 50]), u16le(src[b + 50:b + 52]),
              u16le(src[b + 52:b + 54]), u16le(src[b + 54:b + 56])]
        scale = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000)
        d = f16(scale & 0xFF, (scale >> 8) & 0xFF)
        qs = src[b:b + 32]
        qh = src[b + 32:b + 48]
        qoff = 0
        qhoff = 0
        for ib in range(8):
            dl1 = F32(d * F32(2 * ((sc[ib // 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1))
            dl2 = F32(d * F32(2 * ((sc[ib // 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1))
            idx = [
                qs[qoff + 0] | ((qh[qhoff + 0] << 8) & 0x700),
                qs[qoff + 1] | ((qh[qhoff + 0] << 4) & 0x700),
                qs[qoff + 2] | ((qh[qhoff + 1] << 8) & 0x700),
                qs[qoff + 3] | ((qh[qhoff + 1] << 4) & 0x700),
            ]
            delta = [
                F32(-0.125) if (qh[qhoff + 0] & 0x08) else IQ1S_DELTA,
                F32(-0.125) if (qh[qhoff + 0] & 0x80) else IQ1S_DELTA,
                F32(-0.125) if (qh[qhoff + 1] & 0x08) else IQ1S_DELTA,
                F32(-0.125) if (qh[qhoff + 1] & 0x80) else IQ1S_DELTA,
            ]
            for l in range(2):
                grid = g[idx[l]]
                for j in range(8):
                    gv = F32(s8(gbyte(grid, j)))
                    y[yi] = F32(dl1 * F32(gv + delta[l])); yi += 1
            for l in range(2, 4):
                grid = g[idx[l]]
                for j in range(8):
                    gv = F32(s8(gbyte(grid, j)))
                    y[yi] = F32(dl2 * F32(gv + delta[l])); yi += 1
            qoff += 4
            qhoff += 2
    return y


def deq_iq4_nl(src, t):
    nb = len(src) // 18
    y = np.zeros(nb * QK4_NL, np.float32)
    yi = 0
    kv = t["kvalues_iq4nl"]
    for sb in range(nb):
        b = sb * 18
        d = f16(src[b], src[b + 1])
        qs = src[b + 2:b + 18]
        for j in range(QK4_NL // 2):
            y[yi + j + 0] = F32(d * F32(kv[qs[j] & 0xF]))
            y[yi + j + QK4_NL // 2] = F32(d * F32(kv[qs[j] >> 4]))
        yi += QK4_NL
    return y


def deq_iq4_xs(src, t):
    nb = len(src) // 136
    y = np.zeros(nb * QK_K, np.float32)
    yi = 0
    kv = t["kvalues_iq4nl"]
    for sb in range(nb):
        b = sb * 136
        d = f16(src[b], src[b + 1])
        sh = u16le(src[b + 2:b + 4])
        sl = src[b + 4:b + 8]
        qs = src[b + 8:b + 136]
        qoff = 0
        for ib in range(8):
            ls = ((sl[ib // 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4)
            dl = F32(d * F32(ls - 32))
            for j in range(16):
                y[yi + j + 0] = F32(dl * F32(kv[qs[qoff + j] & 0xF]))
                y[yi + j + 16] = F32(dl * F32(kv[qs[qoff + j] >> 4]))
            yi += 32
            qoff += 16
    return y


# --------------------------------------------------------------------------
# existing kinds (spot checks in the Odin test)
# --------------------------------------------------------------------------

def deq_q4_k(src):
    nb = len(src) // 144
    y = np.zeros(nb * QK_K, np.float32)
    yi = 0
    for sb in range(nb):
        b = sb * 144
        d = f16(src[b], src[b + 1])
        mn = f16(src[b + 2], src[b + 3])
        scales = src[b + 4:b + 16]
        qs = src[b + 16:b + 144]
        is_ = 0
        qoff = 0
        for _ in range(4):
            sc0, m0 = get_scale_min_k4(is_ + 0, scales)
            sc1, m1 = get_scale_min_k4(is_ + 1, scales)
            d1 = F32(d * F32(sc0)); mm1 = F32(mn * F32(m0))
            d2 = F32(d * F32(sc1)); mm2 = F32(mn * F32(m1))
            for l in range(32):
                y[yi + l] = F32(d1 * F32(qs[qoff + l] & 0xF)) - mm1
            for l in range(32):
                y[yi + 32 + l] = F32(d2 * F32(qs[qoff + l] >> 4)) - mm2
            qoff += 32
            is_ += 2
            yi += 64
    return y


def deq_q6_k(src):
    nb = len(src) // 210
    y = np.zeros(nb * QK_K, np.float32)
    yi = 0
    for sb in range(nb):
        b = sb * 210
        ql = src[b:b + 128]
        qh = src[b + 128:b + 192]
        sc = src[b + 192:b + 208]
        d = f16(src[b + 208], src[b + 209])
        qloff = 0
        qhoff = 0
        scoff = 0
        for _ in range(2):
            for l in range(32):
                is_ = l // 16
                q1 = ((ql[qloff + l] & 0xF) | (((qh[qhoff + l] >> 0) & 3) << 4)) - 32
                q2 = ((ql[qloff + l + 32] & 0xF) | (((qh[qhoff + l] >> 2) & 3) << 4)) - 32
                q3 = ((ql[qloff + l] >> 4) | (((qh[qhoff + l] >> 4) & 3) << 4)) - 32
                q4 = ((ql[qloff + l + 32] >> 4) | (((qh[qhoff + l] >> 6) & 3) << 4)) - 32
                y[yi + l] = F32(F32(d * F32(s8(sc[scoff + is_ + 0]))) * F32(q1))
                y[yi + l + 32] = F32(F32(d * F32(s8(sc[scoff + is_ + 2]))) * F32(q2))
                y[yi + l + 64] = F32(F32(d * F32(s8(sc[scoff + is_ + 4]))) * F32(q3))
                y[yi + l + 96] = F32(F32(d * F32(s8(sc[scoff + is_ + 6]))) * F32(q4))
            qloff += 64
            qhoff += 32
            scoff += 8
            yi += 128
    return y


# --------------------------------------------------------------------------
# fixture
# --------------------------------------------------------------------------

KINDS = [
    # (ggml type value, name, block elems, block bytes, dequant fn)
    (10, "Q2_K", 256, 84, deq_q2_k),
    (11, "Q3_K", 256, 110, deq_q3_k),
    (13, "Q5_K", 256, 176, deq_q5_k),
    (15, "Q8_K", 256, 292, deq_q8_k),
    (16, "IQ2_XXS", 256, 66, deq_iq2_xxs),
    (17, "IQ2_XS", 256, 74, deq_iq2_xs),
    (18, "IQ3_XXS", 256, 98, deq_iq3_xxs),
    (19, "IQ1_S", 256, 50, deq_iq1_s),
    (20, "IQ4_NL", 32, 18, deq_iq4_nl),
    (21, "IQ3_S", 256, 110, deq_iq3_s),
    (22, "IQ2_S", 256, 82, deq_iq2_s),
    (23, "IQ4_XS", 256, 136, deq_iq4_xs),
    (29, "IQ1_M", 256, 56, deq_iq1_m),
    (12, "Q4_K", 256, 144, deq_q4_k),
    (14, "Q6_K", 256, 210, deq_q6_k),
]

N = 512
NROWS = 3


def sanitize(rng, kind, data, block_bytes):
    """Force every f16/f32 scale field in the random bytes to a finite value
    (NaN comparisons would spuriously fail). Dequant only reads scale fields;
    everything else is arbitrary bytes."""
    nblk = len(data) // block_bytes
    if kind in (10,):  # Q2_K: d, dmin at 80, 82
        for k in range(nblk):
            off = k * block_bytes
            for o in (80, 82):
                h = data[off + o] | (data[off + o + 1] << 8)
                h = (h & 0x83FF) | 0x3800
                data[off + o] = h & 0xFF
                data[off + o + 1] = (h >> 8) & 0xFF
    elif kind in (11,):  # Q3_K: d at 108
        for k in range(nblk):
            off = k * block_bytes
            h = data[off + 108] | (data[off + 109] << 8)
            h = (h & 0x83FF) | 0x3800
            data[off + 108] = h & 0xFF
            data[off + 109] = (h >> 8) & 0xFF
    elif kind in (16, 17, 18, 19, 20, 21, 22, 23):  # single d at 0
        for k in range(nblk):
            off = k * block_bytes
            h = data[off] | (data[off + 1] << 8)
            h = (h & 0x83FF) | 0x3800
            data[off] = h & 0xFF
            data[off + 1] = (h >> 8) & 0xFF
    elif kind in (12, 13):  # Q4_K, Q5_K: d, dmin at 0,2
        for k in range(nblk):
            off = k * block_bytes
            for o in (0, 2):
                h = data[off + o] | (data[off + o + 1] << 8)
                h = (h & 0x83FF) | 0x3800
                data[off + o] = h & 0xFF
                data[off + o + 1] = (h >> 8) & 0xFF
    elif kind == 14:  # Q6_K: d at 208,209
        for k in range(nblk):
            off = k * block_bytes
            h = data[off + 208] | (data[off + 209] << 8)
            h = (h & 0x83FF) | 0x3800
            data[off + 208] = h & 0xFF
            data[off + 209] = (h >> 8) & 0xFF
    elif kind == 15:  # Q8_K: f32 d at 0..3
        for k in range(nblk):
            off = k * block_bytes
            data[off + 1] &= 0x7F  # exponent bits 30..23 <= 0xFE → finite
    elif kind == 29:  # IQ1_M: f16 scale comes from sc[0] (bytes 48..49) bits 15..12 → clear bit 12
        for k in range(nblk):
            off = k * block_bytes
            data[off + 49] &= 0xEF
    return data


def make_fixture(path, seed=12345):
    tables = parse_tables()
    tvals = {k: v[1] for k, v in tables.items()}
    rng = np.random.default_rng(seed)
    with open(path, "wb") as f:
        f.write(b"ODQN")
        f.write(struct.pack("<II", 1, len(KINDS)))
        for kind, name, be, bb, fn in KINDS:
            row_bytes = (N // be) * bb
            rows = []
            for _ in range(NROWS):
                raw = rng.integers(0, 256, size=row_bytes, dtype=np.uint8).tobytes()
                b = bytearray(raw)
                sanitize(rng, kind, b, bb)
                rows.append(bytes(b))
            fn0 = fn
            if fn0 in (deq_iq2_xxs, deq_iq2_xs, deq_iq2_s, deq_iq3_xxs, deq_iq3_s, deq_iq1_s, deq_iq1_m, deq_iq4_nl, deq_iq4_xs):
                fn = lambda r: fn0(r, tvals)
            expected = [fn(r) for r in rows]
            f.write(struct.pack("<IIII", kind, N, row_bytes, NROWS))
            for r in rows:
                f.write(r)
            for e in expected:
                f.write(e.astype("<f4").tobytes())
    print(f"wrote {path} ({os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-odin", metavar="PATH", help="write Odin tables to PATH")
    args = ap.parse_args()
    tables = parse_tables()
    if args.emit_odin:
        emit_odin(tables, args.emit_odin)
    else:
        make_fixture(os.path.join(os.path.dirname(os.path.abspath(__file__)), "quant_fixture.bin"))
