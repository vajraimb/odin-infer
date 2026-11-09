#!/usr/bin/env python3
"""Print per-layer tensor quant kinds for an IQ GGUF (inspect blk.3's full-attn kinds)."""
import struct
import sys

GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "IQ1_M",
}

def main(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GGUF", "not a GGUF"
        version, n_tensors, n_meta = struct.unpack("<IQQ", f.read(20))
        for _ in range(n_meta):
            klen = struct.unpack("<Q", f.read(8))[0]
            key = f.read(klen).decode()
            vt = struct.unpack("<I", f.read(4))[0]
            if vt == 0:  # uint8
                f.read(1)
            elif vt == 1:  # int8
                f.read(1)
            elif vt == 2:  # uint16
                f.read(2)
            elif vt == 3:
                f.read(2)
            elif vt == 4:  # uint32
                f.read(4)
            elif vt == 5:
                f.read(4)
            elif vt == 6:  # float32
                f.read(4)
            elif vt == 7:  # bool
                f.read(1)
            elif vt == 8:  # string
                vlen = struct.unpack("<Q", f.read(8))[0]
                f.read(vlen)
            elif vt == 9:  # array
                at, alen = struct.unpack("<IQ", f.read(12))
                for _ in range(alen):
                    if at == 8:  # array of strings
                        vlen = struct.unpack("<Q", f.read(8))[0]
                        f.read(vlen)
                    else:
                        sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 4, 12: 4, 13: 8}.get(at, 1)
                        f.read(sz)
            elif vt == 10:  # uint64
                f.read(8)
            elif vt == 11:
                f.read(8)
            elif vt == 12:
                f.read(8)
            else:
                raise ValueError(f"unknown meta vt {vt}")
        tensors = []
        for i in range(n_tensors):
            nlen = struct.unpack("<Q", f.read(8))[0]
            name = f.read(nlen).decode()
            nd = struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
            gt, off = struct.unpack("<IQ", f.read(12))
            tensors.append((name, dims, gt))
    print(f"n_tensors={n_tensors}")
    # print all unique tensor name suffixes with kinds per layer range
    from collections import defaultdict
    by_suffix = defaultdict(dict)
    for name, dims, gt in tensors:
        # name like blk.3.attn_q.weight
        parts = name.split(".")
        if parts[0] == "blk" and len(parts) >= 3:
            lid = parts[1]
            suffix = ".".join(parts[2:])
            by_suffix[suffix][int(lid)] = GGML_TYPES.get(gt, f"?{gt}")
    for suffix in sorted(by_suffix):
        kinds = by_suffix[suffix]
        # summarize: map kind -> list of layers
        from collections import defaultdict as dd
        by_kind = dd(list)
        for lid in sorted(kinds):
            by_kind[kinds[lid]].append(lid)
        summary = ", ".join(f"{k}:{_range(v)}" for k, v in sorted(by_kind.items(), key=lambda kv: kv[1][0]))
        print(f"{suffix:35s} {summary}")

def _range(layers):
    out = []
    s = p = layers[0]
    for x in layers[1:] + [None]:
        if x is not None and x == p + 1:
            p = x
            continue
        out.append(str(s) if s == p else f"{s}-{p}")
        if x is not None:
            s = p = x
    return ",".join(out)

if __name__ == "__main__":
    main(sys.argv[1])
