#!/usr/bin/env python3
r"""Regression test: prefix-cache snapshot fidelity must be BIT-IDENTICAL.

Why this test exists
--------------------
test_prefix_cache.py only asserts "no unicode-escape garbage" on cache hits.
That catches catastrophic state corruption (the historical snapshot-mispositioning
bug), but a subtler drift — e.g. conv state off by one position, recurrent state
restored from a stale slot — can still produce fluent-but-different text that
passes the garbage check while silently diverging from the no-cache baseline.

The strong property we actually rely on (docs/PREFIX_SNAPSHOT_PLAN.md Step 0b):
restoring the full-state snapshot and prefilling only the suffix must produce
exactly the same engine state as a full prefill from pos 0. Under greedy
decoding (temperature=0) that means: the same request issued twice — first a
MISS (full prefill + snapshot taken), then a HIT (snapshot restored + suffix
only) — must generate byte-identical replies.

What it does
------------
  1. builds serve.odin into a cached test binary (rebuilds only when stale),
  2. starts it with KV_PERSIST=0 (no disk restore — guaranteed cold state),
  3. sends the same request twice with temperature=0,
  4. asserts the server log shows MISS then PREFIX CACHE HIT,
  5. asserts the two replies are byte-identical,
  6. asserts /api/state hit-rate counters moved (pc_misses=1, pc_snapshot_hits=1),
  7. multi-turn phase: appends the reply + a new user message, asserts a
     CONTINUATION hit (whole history reused, only the new turn prefilled),
  8. restarts cold and re-sends the multi-turn request (full prefill),
     asserting the continuation-path reply is byte-identical to it.

Usage
-----
    python3 server/serve_harness/test_prefix_bitidentical.py

Env: MODEL (gguf path), PORT (9764). Exit code 0 on PASS, 1 on FAIL.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
TEST_BIN = os.path.join(SCRIPT_DIR, ".test-serve.bin")
SERVE_SRC = os.path.join(SCRIPT_DIR, "serve.odin")
LOG_PATH = os.path.join(SCRIPT_DIR, ".test-serve-bitidentical.log")

MODEL = os.environ.get(
    "MODEL", "/Users/vajra/Claw/odin-infeer/models/ornith-1.0-9b-Q4_K_M.gguf"
)
PORT = int(os.environ.get("PORT", "9764"))
BASE = f"http://127.0.0.1:{PORT}"

SYSTEM = (
    "You are pie, a warm and helpful coding assistant. "
    "Always reply in the language the user uses."
)
USER_MSG = "用三句话解释什么是递归神经网络"
NUM_PREDICT = 64


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def build_if_stale():
    if os.path.exists(TEST_BIN) and os.path.getmtime(TEST_BIN) > os.path.getmtime(SERVE_SRC):
        print("build: cached test binary is fresh")
        return
    print("build: compiling serve.odin (first run may take a minute)...")
    cmd = [
        "odin", "build", "server/serve_harness/serve.odin", "-file", "-o:speed",
        "-collection:ggml=packages", "-collection:infer=packages",
        "-collection:qwen3_5=packages", "-collection:qwen3_5_tokenizer=packages",
        "-collection:sampler=packages", "-collection:tokenizer=packages",
        f"-out:{TEST_BIN}",
    ]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        fail("odin build failed")


def start_server():
    env = dict(os.environ)
    env["KV_PERSIST"] = "0"  # no disk restore: first request must be a MISS
    log = open(LOG_PATH, "w")
    proc = subprocess.Popen(
        [TEST_BIN, MODEL, "--metal", "--max-ctx", "2048", "--port", str(PORT)],
        stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env,
    )
    deadline = time.time() + 180
    while time.time() < deadline:
        if proc.poll() is not None:
            with open(LOG_PATH) as f:
                print(f.read())
            fail("server exited during startup")
        try:
            with open(LOG_PATH) as f:
                if "listening" in f.read():
                    print(f"server: up on {BASE}")
                    return proc
        except FileNotFoundError:
            pass
        time.sleep(1)
    proc.kill()
    fail("server did not start within 180s")


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


def server_log():
    with open(LOG_PATH) as f:
        return f.read()


def api_state():
    with urllib.request.urlopen(f"{BASE}/api/state", timeout=10) as resp:
        return json.loads(resp.read().decode())


def main():
    if not os.path.exists(MODEL):
        fail(f"model not found: {MODEL} (set MODEL env var)")
    build_if_stale()
    proc = start_server()
    try:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_MSG},
        ]

        print("request 1 (expect MISS: full prefill + snapshot taken)...")
        reply1 = chat(messages)
        if not reply1.strip():
            fail("request 1: empty reply")
        log1 = server_log()
        if "full prefill" not in log1:
            fail("request 1: expected 'full prefill' in server log (cold MISS)")
        if "prefix snapshot at system boundary" not in log1:
            fail("request 1: expected snapshot taken at system boundary")

        print("request 2 (expect HIT: snapshot restored, suffix only)...")
        reply2 = chat(messages)
        if not reply2.strip():
            fail("request 2: empty reply")
        log2 = server_log()
        if "PREFIX CACHE HIT" not in log2:
            fail("request 2: expected 'PREFIX CACHE HIT' in server log")

        print(f"reply 1: {reply1[:100]}{'...' if len(reply1) > 100 else ''}")
        print(f"reply 2: {reply2[:100]}{'...' if len(reply2) > 100 else ''}")

        if reply1 != reply2:
            # Find first divergence for debugging.
            n = min(len(reply1), len(reply2))
            div = next((i for i in range(n) if reply1[i] != reply2[i]), n)
            fail(
                f"replies differ (first divergence at char {div}):\n"
                f"  r1[{div}:{div+40}]={reply1[div:div+40]!r}\n"
                f"  r2[{div}:{div+40}]={reply2[div:div+40]!r}\n"
                f"snapshot restore is NOT bit-identical to full prefill"
            )
        print("bit-identical: replies match exactly")

        st = api_state()
        print(f"/api/state counters: { {k: st.get(k) for k in ('pc_snapshot_hits', 'pc_continuation_hits', 'pc_misses')} }")
        if st.get("pc_misses", 0) < 1:
            fail("/api/state: pc_misses did not increment on the cold request")
        if st.get("pc_snapshot_hits", 0) < 1:
            fail("/api/state: pc_snapshot_hits did not increment on the repeat request")

        print("PASS phase 1: snapshot restore bit-identical + hit-rate counters live")

        # ---- phase 2: multi-turn continuation must reuse the whole history ----
        messages3 = messages + [
            {"role": "assistant", "content": reply2},
            {"role": "user", "content": "那它和前馈网络的区别呢"},
        ]
        print("request 3 (expect continuation hit: history reused, new turn only)...")
        reply3 = chat(messages3)
        if not reply3.strip():
            fail("request 3: empty reply")
        log3 = server_log()
        if "continuation hit" not in log3:
            fail("request 3: expected 'continuation hit' in server log "
                 "(whole-history reuse); got snapshot hit or full prefill instead")
        st = api_state()
        print(f"/api/state counters: { {k: st.get(k) for k in ('pc_snapshot_hits', 'pc_continuation_hits', 'pc_misses')} }")
        if st.get("pc_continuation_hits", 0) < 1:
            fail("/api/state: pc_continuation_hits did not increment on the follow-up turn")
    finally:
        proc.kill()

    # ---- phase 3: cold full prefill of the same multi-turn request ----
    # The continuation-path reply must be byte-identical to a from-scratch
    # prefill of the same messages (the live-state reuse changes nothing).
    proc = start_server()
    try:
        print("request 4 (cold restart, full prefill of the multi-turn request)...")
        reply3_cold = chat(messages3)
        if not reply3_cold.strip():
            fail("request 4: empty reply")
        log4 = server_log()
        if "full prefill" not in log4[log4.rfind("listening"):]:
            fail("request 4: expected 'full prefill' after cold restart")
        print(f"reply 3 (continuation): {reply3[:80]}{'...' if len(reply3) > 80 else ''}")
        print(f"reply 3 (cold prefill): {reply3_cold[:80]}{'...' if len(reply3_cold) > 80 else ''}")
        if reply3 != reply3_cold:
            n = min(len(reply3), len(reply3_cold))
            div = next((i for i in range(n) if reply3[i] != reply3_cold[i]), n)
            fail(
                f"continuation reply differs from cold prefill (first divergence at char {div}):\n"
                f"  cont[{div}:{div+40}]={reply3[div:div+40]!r}\n"
                f"  cold[{div}:{div+40}]={reply3_cold[div:div+40]!r}\n"
                f"continuation reuse is NOT bit-identical to full prefill"
            )
        print("PASS phase 2/3: continuation hit bit-identical to cold full prefill")
    finally:
        proc.kill()


if __name__ == "__main__":
    main()
