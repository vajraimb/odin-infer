#!/usr/bin/env python3
"""Print GGUF metadata (config) for a model."""
import struct
import sys

def read_meta(path):
    with open(path, "rb") as f:
        f.read(4)
        version, n_tensors, n_meta = struct.unpack("<IQQ", f.read(20))
        meta = {}
        for _ in range(n_meta):
            klen = struct.unpack("<Q", f.read(8))[0]
            key = f.read(klen).decode()
            vt = struct.unpack("<I", f.read(4))[0]
            val = None
            if vt == 0: val = struct.unpack("<B", f.read(1))[0]
            elif vt == 1: val = struct.unpack("<b", f.read(1))[0]
            elif vt == 2: val = struct.unpack("<H", f.read(2))[0]
            elif vt == 3: val = struct.unpack("<h", f.read(2))[0]
            elif vt == 4: val = struct.unpack("<I", f.read(4))[0]
            elif vt == 5: val = struct.unpack("<i", f.read(4))[0]
            elif vt == 6: val = struct.unpack("<f", f.read(4))[0]
            elif vt == 7: val = struct.unpack("<B", f.read(1))[0] != 0
            elif vt == 8:
                vlen = struct.unpack("<Q", f.read(8))[0]
                val = f.read(vlen).decode()
            elif vt == 9:
                at, alen = struct.unpack("<IQ", f.read(12))
                if at == 8:
                    val = []
                    for _ in range(alen):
                        vlen = struct.unpack("<Q", f.read(8))[0]
                        val.append(f.read(vlen).decode())
                elif at == 0:
                    val = list(struct.unpack(f"<{alen}B", f.read(alen)))
                elif at == 4:
                    val = list(struct.unpack(f"<{alen}I", f.read(4 * alen)))
                elif at == 5:
                    val = list(struct.unpack(f"<{alen}i", f.read(4 * alen)))
                elif at == 6:
                    val = list(struct.unpack(f"<{alen}f", f.read(4 * alen)))
                else:
                    raise ValueError(f"unhandled array elem type {at}")
            elif vt == 10: val = struct.unpack("<Q", f.read(8))[0]
            elif vt == 11: val = struct.unpack("<q", f.read(8))[0]
            elif vt == 12: val = struct.unpack("<d", f.read(8))[0]
            meta[key] = val
        return meta

if __name__ == "__main__":
    meta = read_meta(sys.argv[1])
    keys = sys.argv[2].split(",") if len(sys.argv) > 2 else sorted(meta)
    for k in keys:
        v = meta.get(k)
        if v is None:
            print(f"{k} = <missing>")
        elif isinstance(v, list) and len(v) > 12:
            print(f"{k} = [{len(v)} items] {v[:6]}...")
        else:
            print(f"{k} = {v}")
