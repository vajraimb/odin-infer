# Postmortem: garbled `\uXXXX` output on multi-turn chat

Date: 2026-07-24 · Affected: `serve_harness/serve.odin` (Ollama `/api/chat`
endpoint, pie client) · Fixed in: same file, plus regression test
`serve_harness/test_prefix_cache.py`

## Symptom

With `serve.sh` + `pie.sh` (Ollama NDJSON API), turn 1 in Chinese was fine,
but turn 2 came back as literal unicode codepoint text:

```
> 你擅长做什么
u6211u64c5u957fu505au7528u6237u5bb6u7684u966au4f34u8005u3002...\u6bd4\u5982\u9605\u8bfb...
```

Decoding those codepoints spells a perfectly coherent answer
("我擅长做用户的陪伴者。我可以帮助用户处理各种任务，比如阅读…") — the model
"knew" what to say, it was just leaking it as escape text. That signature
points at the *input* the model saw, or at a badly degraded internal state —
not at the detokenizer.

## Red herrings ruled out first

- **`vocab exceeds MAX_VOCAB (248077)` log line** — benign. `vocab.txt` has
  exactly 248077 lines; the trailing newline produces one empty extra "line"
  that trips the guard. No tokens are truncated (ids 248077..248319 are padding
  in the embedding table).
- **Tokenizer data/code** — `qwen3_5_tokenizer/vocab.txt` matches the GGUF
  embedded vocab line-for-line (0 mismatches), and Chinese encode/decode
  roundtrips produce the exact GGUF ids (你=95933, 擅长=108768). Tokenizer was
  never the problem.

## Root cause 1: prefix-cache snapshot taken at the wrong position

qwen3_5 is a **hybrid** architecture: 8 full-attention layers (KV cache) +
24 linear-attention layers whose conv/recurrent state **evolves with position
and cannot be rewound** (`engine.odin` says so in a comment).

`handle_ollama_chat_stream` saved the "system prompt snapshot" *after*
prefilling the entire prompt (system + user1 + template, pos = n−1), but
labelled it with the system boundary position (e.g. 1206). On the next turn's
`PREFIX CACHE HIT`, prefill resumed at 1206 and replayed the suffix tokens —
but the linear layers had *already absorbed them once*. 24 of 32 layers ran on
doubly-advanced, corrupted recurrent state.

**Fix:** two-phase prefill. Prefill exactly up to the system boundary, snapshot
*at that moment* (conv/rec state then genuinely corresponds to `prefix_pos`),
then prefill the rest.

Related hardening in the same change:

- The continuation branch (`common == len(last_tokens)`) now also requires
  `g_state.pos == common` — the engine state must sit exactly at the divergence
  point, otherwise the recurrent state absorbs the tail twice.
- `last_tokens` now includes the generated (and actually forwarded) tokens so
  continuation is possible at all.
- Fixed a global off-by-one: the kickoff and generation loops called
  `engine_forward(token, pos - 1)`, making the last prompt token share a
  RoPE position / KV slot with the previous token. Now `pos`.
- Dashboard `handle_chat` re-forwarded the last token on continuation
  (batch + kickoff) — fed to the recurrent state twice. Fixed the same way.

## Root cause 2: JSON `\uXXXX` escapes never decoded (the likely main cause)

The regression test failed *on its first run* even after root cause 1 was
fixed — because it sends requests with Python's `json.dumps`, which escapes
all non-ASCII by default. The server's lenient JSON message parser handled
`\n \" \\ \r \t \/` but passed `\u4f60` through as the literal 6 characters
`u4f60`. Consequences:

- The model's *input* was escape-speak: every Chinese character arrived as
  ASCII codepoint text (and cost ~3× the tokens — a 111-token prompt became
  428).
- Models mirror their input. Turn 1 it answered robustly in Chinese anyway;
  turn 2 — with the Chinese history *also* escaped in the echo — it flipped
  into answering in codepoints. Exactly the reported symptom, including why
  only turn 2+ broke.

Any spec-compliant JSON client that escapes non-ASCII (Python `json.dumps`
default, some Go/JS stacks) triggered this. `curl` with raw UTF-8 did not —
which is why manual testing initially "couldn't reproduce" it.

**Fix:** `write_json_unicode_escape` (with surrogate-pair support) wired into
all three JSON string parsers: `openai_extract_messages`,
`extract_json_field`, `decode_json_string`.

## Why the model's "garbage" was so clean

Corrupted context rarely produces random tokens — it produces *fluent but
wrong* text, because the output distribution still concentrates on plausible
continuations. Both bugs left the model able to "see" roughly what was being
asked (codepoints are human-legible too, and these models have seen plenty of
JSON), so it answered coherently — just in escape notation. Clean-looking
garbage is still garbage; don't rule out state corruption because the output
is grammatical.

## Lessons / checklist for "works turn 1, breaks turn 2"

1. **Diff turn 1 vs turn 2 mechanics first.** What changed? (Here: prefix
   cache HIT vs full prefill, and history echo in the request body.)
2. **Log the token counts.** Turn 2's `428 in` vs the expected ~110 exposed
   the `\uXXXX` input bloat immediately — before any code reading.
3. **Reproduce with the real client's serialization.** `curl` with raw UTF-8
   hid root cause 2; `json.dumps` defaults exposed it. Test with
   `ensure_ascii=True` JSON, not just hand-typed requests.
4. **Hybrid/SSM models: snapshots are positional contracts.** KV cache can be
   truncated/overwritten freely; conv/recurrent state cannot. A snapshot is
   only valid for the exact token count that had been forwarded when it was
   taken. Save early, label honestly.
5. **"Position" and "count forwarded" must be the same number.** Most of the
   bugs here reduced to off-by-one drift between the two. Keep the invariant
   `pos == tokens forwarded so far` and pass `pos` (not `pos - 1`) to
   `engine_forward`.
6. **JSON parsers must handle `\uXXXX`.** It's not optional for
   interoperability — it's the default output of major JSON libraries.

## Regression coverage

```sh
python3 serve_harness/test_prefix_cache.py
```

Sends `ensure_ascii=True` JSON (covering root cause 2), runs a 3-turn Chinese
conversation, asserts `PREFIX CACHE HIT` on turns 2+ and that every reply is
clean CJK with no `\uXXXX` / bare `uXXXX` text (covering root cause 1).
Run it after any change to `serve.odin`'s prefill/cache/JSON logic.
