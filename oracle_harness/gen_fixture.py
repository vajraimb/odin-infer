#!/usr/bin/env python3
"""Generate a token-exact oracle fixture for odin-infer.

Loads Qwen3-0.6B in fp16 via HF transformers, teacher-forwards a fixed set of
6 prompts (CN/EN x2 + code + math), and dumps logits to a raw .bin file that
oracle.odin can slice-cast directly.

Usage:
    python3 gen_fixture.py --model Qwen/Qwen3-0.6B --out fixtures/qwen3-0.6b.bin

Output format (little-endian):
    magic      : u32 = 0x4F524331  ('ORC1')
    vocab      : u32
    n_prompts  : u32
    per prompt:
        prompt_len : u32
        tokens     : [u32; prompt_len]
        logits     : [f16; vocab * prompt_len]   row-major [pos][vocab]
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAGIC = 0x4F524331  # 'O','R','C','1' little-endian

# Fixed prompt set: CN x2, EN x2, code, math. Each ~16-32 tokens after BPE.
PROMPTS = [
    "Hello, how are you",          # EN smalltalk
    "The capital of France is",    # EN factual
    "你好,你是谁",                  # CN smalltalk
    "今天天气真好,我想去",            # CN open-ended
    "def fibonacci(n):",           # code
    "2 + 2 =",                     # math
]


def teacher_force_logits(model, tok, prompt: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (tokens[u32; L], logits[f16; L x vocab]) for the prompt."""
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0]                       # [L]
    # We take logits AT each input position — i.e. run the full prompt and grab
    # model(ids).logits. odin-infer's engine_forward(e, token, pos) returns
    # logits AT position `pos` after consuming `token`, matching this exactly.
    with torch.no_grad():
        out = model(ids.unsqueeze(0))
    logits = out.logits[0].to(torch.float16).cpu().numpy()  # [L, vocab]
    tokens = ids.to(np.uint32).numpy()
    return tokens, logits


def write_fixture(out_path: Path, vocab: int, prompts_data):
    with out_path.open("wb") as f:
        f.write(struct.pack("<III", MAGIC, vocab, len(prompts_data)))
        for tokens, logits in prompts_data:
            assert logits.shape == (len(tokens), vocab), \
                f"logits shape {logits.shape} != ({len(tokens)}, {vocab})"
            f.write(struct.pack("<I", len(tokens)))
            tokens.astype(np.uint32).tofile(f)
            # row-major C-order, fp16 — Odin reads as []f16 then casts per row
            logits.astype(np.float16).tofile(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B",
                    help="HF model id (default: Qwen/Qwen3-0.6B)")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output .bin path")
    args = ap.parse_args()

    print(f"[gen_fixture] loading {args.model} (fp16, CPU)…", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.model)
    vocab = tok.vocab_size
    print(f"[gen_fixture] vocab_size = {vocab}")

    prompts_data = []
    for i, p in enumerate(PROMPTS):
        tokens, logits = teacher_force_logits(model, tok, p)
        prompts_data.append((tokens, logits))
        print(f"  [{i}] {p!r:40s} tokens={len(tokens):3d}  logits={logits.shape}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_fixture(args.out, vocab, prompts_data)
    size_mb = args.out.stat().st_size / 1024 / 1024
    print(f"[gen_fixture] wrote {args.out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
