# qwen3_5_tokenizer

BPE tokenizer for Qwen3.5 (Ornith-1.0-9B), sibling to the Qwen3 `tokenizer`
package. Same GPT-2 byte-level BPE scheme; differs only in:

- vocab (248,077 tokens: 248,044 BPE + 33 special, ids 248044..248076)
- merges (247,587 rules)
- the special-token set (33 Qwen3.5 control tokens)
- O(1) merge lookup via a composite-key hash map (linear scan over 247k rules
  is unusably slow)
- per-instance `unicode_to_byte` map (the Qwen3 package keeps it as a global,
  which races under parallel use)

The embedding table is padded to 248,320; ids 248077..248319 are unused by the
tokenizer (decode returns "" for them).

## Files

| File | Purpose |
|------|---------|
| `tokenizer.odin` | BPE encode/decode, `Tokenizer` struct, unit tests |
| `vocab.txt` | 248,077 byte-level tokens, one per line (id == line number) |
| `merges.txt` | 247,587 BPE merges, `left right` per line, in rank order |
| `tools/export_vocab.py` | one-shot HF `tokenizer.json` -> vocab.txt/merges.txt |
| `build.sh` | `odin test .` |

vocab.txt/merges.txt are baked into the binary at compile time via `#load`, so a
standalone binary works without the data files on disk.

## Regenerating the data

If a newer Ornith/Qwen3.5 `tokenizer.json` is published:

```sh
hf download deepreinforce-ai/Ornith-1.0-9B tokenizer.json --local-dir /tmp/t
python3 tools/export_vocab.py /tmp/t/tokenizer.json .
```

## Build / test

```sh
./build.sh
```

## Usage

```odin
import tok "qwen3_5_tokenizer:qwen3_5_tokenizer"

t: tok.Tokenizer
tok.build_tokenizer(&t)
defer tok.free_tokenizer(&t)

ids, _ := tok.encode(&t, "<|im_start|>user\nHello<|im_end|>\n")
s := tok.decode_token_id(&t, ids[0])
```

Build a consumer with the collection:

```sh
LIB=/path/to/odin-infer
odin build . -collection:qwen3_5_tokenizer=$LIB
```

## Special token ids (Ornith-1.0-9B)

| id | token |
|----|-------|
| 248044 | `<\|endoftext\|>` |
| 248045 | `<\|im_start\|>` |
| 248046 | `<\|im_end\|>` |
| 248053 | `<\|vision_start\|>` |
| 248054 | `<\|vision_end\|>` |
| 248056 | `<\|image_pad\|>` |
| 248068 | `<think>` |
| 248069 | `</think>` |
| 248058 | `<tool_call>` |
| 248059 | `</tool_call>` |

(EOS for chat is `<|im_end|>` = 248046.)
