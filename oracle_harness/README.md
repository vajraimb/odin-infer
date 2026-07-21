# oracle_harness

Token-level end-to-end correctness test for `odin-infer`'s forward pass.

Validates `infer.engine_forward` output against a **HF `transformers` fp16 reference**
on a fixed set of 6 prompts (CN × 2, EN × 2, code, math). Reports per-prompt
top-1 agreement, max-abs and mean-abs logit diff, and top-5 overlap, plus an
aggregate `PASS` / `WARN` / `FAIL` verdict.

## Why statistical, not bit-exact

`odin-infer` consumes third-party GGUF quantization (Q4_K_M, Q5_K_M, …). HF fp16
and Q4_K_M logits cannot match bit-exactly. colibri gets 32/32 token-exact because
they own their int4 container — we don't. So we measure statistical agreement
instead, with thresholds calibrated empirically after the first end-to-end run.

## Files

| File | Purpose |
|------|---------|
| `oracle.odin`            | Odin harness: load fixture + engine, teacher-force, compare, report |
| `gen_fixture.py`         | Python: HF `transformers` Qwen3 fp16 → `.bin` (the real oracle) |
| `gen_self_fixture.odin`  | Odin: produce a fixture from odin-infer itself (plumbing smoke test) |
| `fixtures/`              | Generated `.bin` files (gitignored) |

## Binary format (little-endian)

```
magic      : u32 = 0x4F524331   ('ORC1')
vocab      : u32
n_prompts  : u32
per prompt:
  prompt_len : u32
  tokens     : [u32; prompt_len]
  logits     : [f16; vocab * prompt_len]   row-major [pos][vocab]
```

## Usage

### Run the comparison

```sh
# From odin-infer repo root
odin run oracle_harness/oracle.odin -file -- \
  run /path/to/Qwen3-0.6B-Q4_K_M.gguf oracle_harness/fixtures/qwen3-0.6b-q4km.bin \
  -collection:ggml=. -collection:infer=. \
  -collection:tokenizer=. -collection:sampler=. \
  -collection:qwen3_5=. -collection:qwen3_5_tokenizer=.
```

Append `--metal` to test the Metal backend instead of CPU.

Exit codes: `0` PASS, `1` WARN, `2` FAIL.

### Inspect a fixture

```sh
odin run oracle_harness/oracle.odin -file -- \
  dump oracle_harness/fixtures/qwen3-0.6b-q4km.bin
```

### Generate the real fixture (needs HF reference)

```sh
pip install torch transformers numpy
python3 oracle_harness/gen_fixture.py \
  --model Qwen/Qwen3-0.6B \
  --out   oracle_harness/fixtures/qwen3-0.6b-q4km.bin
```

### Generate a self-fixture (plumbing test, no torch)

```sh
odin run oracle_harness/gen_self_fixture.odin -file -- \
  /path/to/Qwen3-0.6B-Q4_K_M.gguf \
  oracle_harness/fixtures/self-qwen3-0.6b.bin \
  -collection:ggml=. -collection:infer=. \
  -collection:tokenizer=. -collection:sampler=.
```

Run `oracle run` against the self-fixture: should report ~100% top-1 match and
near-zero max-abs (limited by non-deterministic FP reduction across threads).
This validates the harness plumbing independent of any external oracle.

## Tolerance policy (v0)

Thresholds are encoded as compile-time constants in `oracle.odin`:

| Metric            | PASS    | WARN       | FAIL       |
|-------------------|---------|------------|------------|
| top-1 agreement   | ≥ 80%   | 65–80%     | < 65%      |
| max_abs_logit     | < 6.0   | 6.0–12.0   | > 12.0     |
| top-5 overlap     | ≥ 60%   | 30–60%     | < 30%      |

These are educated guesses; **calibrate after the first end-to-end run** by
inspecting the per-prompt table and tightening/loosening as real data warrants.

## Adding new models

1. Run `gen_fixture.py --model <HF-id> --out fixtures/<name>.bin`.
2. Run `oracle run <matching-GGUF> fixtures/<name>.bin`.
3. Adjust tolerance constants if the quantization type differs.

The harness is model-agnostic; only the fixture + GGUF pair matters.
