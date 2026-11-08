#!/usr/bin/env python3
"""One-shot export: HF tokenizer.json -> vocab.txt + merges.txt for the Odin
qwen3_5_tokenizer package. Output format matches the existing Qwen3 tokenizer
data: vocab.txt is one byte-level-encoded token per line (id == line number,
special tokens overlaid at their ids); merges.txt is `left right` per line in
rank order, no header.

Usage:  python3 export_vocab.py <tokenizer.json> <out_dir>
"""
import json
import os
import sys


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    src, out_dir = sys.argv[1], sys.argv[2]
    with open(src, "r", encoding="utf-8") as f:
        d = json.load(f)

    vocab = d["model"]["vocab"]            # {token: id}
    merges = d["model"]["merges"]          # [[left, right], ...]
    added = d.get("added_tokens", [])      # [{id, content, ...}, ...]

    max_id = max(vocab.values())
    max_added = max((t["id"] for t in added), default=-1)
    size = max(max_id, max_added) + 1
    arr = [""] * size
    for tok, i in vocab.items():
        arr[i] = tok
    for t in added:
        arr[t["id"]] = t["content"]

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "vocab.txt"), "w", encoding="utf-8") as f:
        for tok in arr:
            f.write(tok + "\n")
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for pair in merges:
            if len(pair) != 2:
                raise ValueError(f"unexpected merge entry: {pair!r}")
            f.write(f"{pair[0]} {pair[1]}\n")

    print(f"vocab.txt: {len(arr)} lines (ids 0..{size - 1})")
    print(f"merges.txt: {len(merges)} lines")


if __name__ == "__main__":
    main()
