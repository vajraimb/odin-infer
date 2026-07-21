# JSON grammar-constrained output

`odin-infer` can force the model's output to be valid JSON. Activated via the
`-J 1` flag on the CLI (Qwen3.5 only).

Inspired by colibri's `GRAMMAR=file.gbnf` but inlined for the JSON-only common
case — no grammar parser, no GBNF syntax to learn. Just JSON.

## How it works

A stack-based state machine tracks position within a JSON value as bytes are
generated. Before each sampling step:

1. For every token in the vocab, decode it to bytes and simulate the state
   machine on those bytes. If the bytes would violate the grammar, mask the
   token's logit to `-inf`.
2. Sample normally (greedy or top-p) from the masked logits.
3. Advance the real state machine by the chosen token's bytes.

The EOS token is allowed only when the grammar is in a "completable" state
(a complete JSON value has been produced).

The state machine supports the full JSON subset:
- Object, array, string, number, boolean, null
- Nested structures (stack-based, depth ≤ 64)
- Whitespace: space, tab, CR, LF
- String escapes: `\" \\ \/ \b \f \n \r \t \uXXXX` (lenient on `\u` hex)

Simplifications vs strict RFC 8259:
- Numbers allow leading zeros (`007` is accepted)
- `\u` escape doesn't strictly validate 4-hex-digit format

## Usage

```sh
./odin-infer-mac ornith-1.0-9b-Q4_K_M.gguf -g 1 -c 2048 -t 0.7 -J 1
```

Then prompt the model normally:

```
Q: Give me a JSON object with keys "name" and "age".
A: {
  "name": "Alice",
  "age": 30
}
```

Without `-J 1`, the same prompt might produce invalid JSON (missing colons,
trailing commas, etc.) on smaller or less-aligned models. With `-J 1`, the
output is always syntactically valid JSON.

## Performance

The token-bytes cache builds once at startup (~25-30 ms for 248k-token Qwen3.5
vocab). Per-token overhead during generation is one state-machine simulation
per vocab entry — ~5-10 ms on M3 for 248k tokens, negligible vs the GPU
forward pass.

## Caveats

- **Schema not enforced.** The grammar only checks syntactic validity. It
  cannot guarantee specific keys, value types, or ranges. Pair with
  prompt-level instructions for schema constraints.
- **Stuck model.** If the model is strongly biased against producing JSON
  (e.g., asked for prose with `-J 1`), the grammar will only allow
  whitespace tokens indefinitely. The prompt must invite JSON output.
- **Qwen3.5 only.** The dense Qwen3 path doesn't support `-J`. The flag is
  ignored with a warning.

## Implementation

- `sampler/grammar.odin` — state machine, mask computation, apply logic
- `sampler/sampler.odin` — `Sampler.grammar` field, mask applied in `sample()`
- `main.odin` — `-J` flag, token-bytes cache build

## Future work

- General GBNF parser (for non-JSON grammars)
- JSON Schema → grammar conversion
- Token-mask caching by state (states repeat across tokens)
- Cross-architecture support (Qwen3 dense)
