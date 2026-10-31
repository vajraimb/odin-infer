#!/usr/bin/env python3
r"""Restart-resume regression: KV state persistence across server restarts.

Why this test exists
--------------------
serve.odin keeps two in-memory accelerators: a system-prompt prefix snapshot
(skips re-prefilling ~1000-1500 fixed tokens) and last-request continuation
bookkeeping. Both died with the process — every restart meant a full ~30s
prefill. serve.odin now persists both through qwen3_5/kv_persist.odin:

  .kv-cache/prefix.oikv  + prefix.tokens   (state at system boundary)
  .kv-cache/session.oikv + session.tokens  (state after each request)

On startup they are restored (CRC + model fingerprint + seq_len verified),
so the first request after a restart is a PREFIX CACHE HIT, not a prefill.

What it does
------------
  1. builds serve.odin into the shared cached test binary,
  2. starts it with a throwaway KV_PERSIST_DIR, runs 2 turns (creates files),
  3. kills and restarts the server with the same dir,
  4. asserts both states were restored at startup, the first post-restart
     turn is a PREFIX CACHE HIT with a small prefill, and the reply is
     clean CJK (a corrupted f16<->f32 state round-trip would produce
     garbage, same symptom class as the old prefix-cache mispositioning bug),
  5. bit-identical check: both servers get the SAME first-turn prompt with
     temperature=0 (greedy argmax); the cold and warm replies must match
     byte-for-byte, proving the disk round-trip is numerically lossless.

Usage
-----
    python3 serve_harness/test_kv_persist.py

Env: MODEL (gguf path), PORT (9833). Exit code 0 on PASS, 1 on FAIL.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
TEST_BIN = os.path.join(SCRIPT_DIR, ".test-serve.bin")
SERVE_SRC = os.path.join(SCRIPT_DIR, "serve.odin")
KV_DIR = os.path.join(SCRIPT_DIR, ".test-kv-cache")

MODEL = os.environ.get(
    "MODEL", "/Users/vajra/Claw/odin-infeer/models/ornith-1.0-9b-Q4_K_M.gguf"
)
PORT = int(os.environ.get("PORT", "9833"))
BASE = f"http://127.0.0.1:{PORT}"
NUM_PREDICT = 48

SYSTEM = (
    "You are pie, a warm and helpful coding assistant. "
    "Always reply in the language the user uses."
)

CJK_RE = re.compile(r"[一-鿿]")
ESCAPED_RE = re.compile(r"\\u[0-9a-fA-F]{4}")
BARE_CODEPOINT_RE = re.compile(r"(?:u[0-9a-f]{4}){3,}")
PREFILL_RE = re.compile(r"prefill=(\d+)ms")


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def build_if_stale():
    srcs = [SERVE_SRC] + [
        os.path.join(ROOT, "sampler", f)
        for f in ("sampler.odin", "grammar.odin", "grammar_toolcall.odin")
    ]
    if os.path.exists(TEST_BIN) and all(
        os.path.getmtime(TEST_BIN) > os.path.getmtime(s) for s in srcs if os.path.exists(s)
    ):
        print("build: cached test binary is fresh")
        return
    print("build: compiling serve.odin (first run may take a minute)...")
    cmd = [
        "odin", "build", "serve_harness/serve.odin", "-file", "-o:speed",
        "-collection:ggml=.", "-collection:infer=.",
        "-collection:qwen3_5=.", "-collection:qwen3_5_tokenizer=.",
        "-collection:sampler=.", "-collection:tokenizer=.",
        f"-out:{TEST_BIN}",
    ]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        fail("odin build failed")


def start_server(tag):
    log_path = os.path.join(SCRIPT_DIR, f".test-kv-{tag}.log")
    log = open(log_path, "w")
    env = dict(os.environ)
    env["KV_PERSIST_DIR"] = KV_DIR
    proc = subprocess.Popen(
        [TEST_BIN, MODEL, "--metal", "--max-ctx", "2048", "--port", str(PORT)],
        stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env,
    )
    deadline = time.time() + 180
    while time.time() < deadline:
        if proc.poll() is not None:
            with open(log_path) as f:
                print(f.read())
            fail(f"server ({tag}) exited during startup")
        try:
            with open(log_path) as f:
                if "listening" in f.read():
                    print(f"server ({tag}): up on {BASE}")
                    return proc, log_path
        except FileNotFoundError:
            pass
        time.sleep(1)
    proc.kill()
    fail(f"server ({tag}) did not start within 180s")


def stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def chat(messages):
    payload = {
        "model": "ornith",
        "messages": messages,
        "stream": True,
        "options": {"num_predict": NUM_PREDICT, "temperature": 0},
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE}/api/chat", data=body, headers={"Content-Type": "application/json"}
    )
    parts = []
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            parts.append(d.get("message", {}).get("content", ""))
    return "".join(parts)


def check_clean_reply(label, reply):
    print(f"{label} reply: {reply[:100]}{'...' if len(reply) > 100 else ''}")
    if not reply.strip():
        fail(f"{label}: empty reply")
    if ESCAPED_RE.search(reply) or BARE_CODEPOINT_RE.search(reply):
        fail(f"{label}: unicode-escape garbage — persisted state round-trip corrupted?")
    if not CJK_RE.search(reply):
        fail(f"{label}: expected Chinese reply — persisted state round-trip corrupted?")


def main():
    if not os.path.exists(MODEL):
        fail(f"model not found: {MODEL} (set MODEL env var)")
    build_if_stale()
    shutil.rmtree(KV_DIR, ignore_errors=True)

    # ---- phase 1: cold start, 2 turns, files created ----
    proc, log1 = start_server("cold")
    try:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "你是哪个模型"}]
        cold_reply1 = chat(msgs)
        check_clean_reply("cold turn 1", cold_reply1)
        msgs.append({"role": "assistant", "content": "我是 pie。"})
        msgs.append({"role": "user", "content": "你擅长做什么"})
        check_clean_reply("cold turn 2", chat(msgs))
    finally:
        stop_server(proc)
    with open(log1) as f:
        log1_text = f.read()
    if "kv-persist: saved prefix snapshot" not in log1_text:
        fail("cold run: prefix snapshot was not persisted")
    if "kv-persist: saved session state" not in log1_text:
        fail("cold run: session state was not persisted")
    for name in ("prefix.oikv", "prefix.tokens", "session.oikv", "session.tokens"):
        p = os.path.join(KV_DIR, name)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            fail(f"cold run: expected non-empty {name} in {KV_DIR}")
    print("cold run: prefix + session persisted to disk")

    # ---- phase 2: restart, state restored, first turn is a cache hit ----
    proc, log2 = start_server("warm")
    try:
        with open(log2) as f:
            startup_log = f.read()
        if "kv-persist: restored prefix snapshot" not in startup_log:
            fail("warm start: prefix snapshot was not restored")
        if "kv-persist: restored session state" not in startup_log:
            fail("warm start: session state was not restored")
        print("warm start: prefix + session restored from disk")

        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "你是哪个模型"}]
        warm_reply1 = chat(msgs)
        check_clean_reply("warm turn 1", warm_reply1)
        if warm_reply1 != cold_reply1:
            fail("warm turn 1: reply differs from cold turn 1 — "
                 "disk round-trip is not bit-identical")
        print("warm turn 1: reply byte-identical to cold turn 1 "
              "(disk round-trip lossless)")
        with open(log2) as f:
            log2_text = f.read()
        if "PREFIX CACHE HIT" not in log2_text:
            fail("warm turn 1: expected PREFIX CACHE HIT from restored snapshot")
        m = PREFILL_RE.search(log2_text)
        prefill_ms = int(m.group(1)) if m else -1
        print(f"warm turn 1: prefix cache hit from disk, prefill={prefill_ms}ms")
        if prefill_ms < 0 or prefill_ms > 10000:
            fail(f"warm turn 1: prefill took {prefill_ms}ms — restored snapshot not effective?")
    finally:
        stop_server(proc)

    shutil.rmtree(KV_DIR, ignore_errors=True)
    print("PASS: KV state survives a server restart "
          "(prefix hit + bit-identical generation)")


if __name__ == "__main__":
    main()
