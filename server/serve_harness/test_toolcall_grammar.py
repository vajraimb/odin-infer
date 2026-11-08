#!/usr/bin/env python3
r"""A/B regression: syntax-constrained tool-call generation (VISION.md P0).

Why this test exists
--------------------
A 9B local model intermittently breaks its native tool-call syntax — wrong
format (`memory(script: ...)`), truncated blocks, bad parameter keys — and the
serve-side parser has to salvage or drop the call. serve.odin now constrains
sampling with a byte-level state machine (sampler/grammar_toolcall.odin):
when a request carries tools, any token that would break the native
<tool_call><function=name><parameter=key> format is masked out before
sampling. The parser stays as a fallback. TOOLCALL_GRAMMAR=0 disables.

What it does
------------
  1. builds serve.odin into the shared cached test binary,
  2. starts it twice — once default (grammar ON), once TOOLCALL_GRAMMAR=0,
  3. fires N memory-storage requests (diverse facts) at each, with the
     memory tool schema pie registers,
  4. asserts every ON-mode request yields a parseable memory() tool call,
     and reports the OFF-mode rate for comparison plus gen tok/s both sides.

Usage
-----
    python3 server/serve_harness/test_toolcall_grammar.py

Env: MODEL (gguf path), PORT_ON (9771), PORT_OFF (9772), N (5).
Exit code 0 on PASS (ON-mode 100% parseable), 1 on FAIL.
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
TEST_BIN = os.path.join(SCRIPT_DIR, ".test-serve.bin")
SERVE_SRC = os.path.join(SCRIPT_DIR, "serve.odin")

MODEL = os.environ.get(
    "MODEL", "/Users/vajra/Claw/odin-infeer/models/ornith-1.0-9b-Q4_K_M.gguf"
)
PORT_ON = int(os.environ.get("PORT_ON", "9771"))
PORT_OFF = int(os.environ.get("PORT_OFF", "9772"))
N = int(os.environ.get("N", "5"))
NUM_PREDICT = 128

SYSTEM = (
    "You are pie, a warm and helpful coding assistant with a long-term memory "
    "tool. When the user shares a fact worth remembering, store it with the "
    "memory tool, then reply briefly. Always reply in the user's language."
)
FACTS = ["苹果", "咖啡", "骑行", "村上春树", "蓝色", "火锅", "摄影", "游泳"]

MEMORY_TOOL = [{
    "type": "function",
    "function": {
        "name": "memory",
        "description": "Long-term memory via Tcl script. Commands: remember "
                       "<id> <subject> Fact key value, recall <handle>, "
                       "respond {reply}.",
        "parameters": {
            "type": "object",
            "properties": {"script": {"type": "string",
                                      "description": "a Tcl script using remember/recall/respond"}},
            "required": ["script"],
        },
    },
}]

TOKS_RE = re.compile(r"\(([\d.]+) tok/s\)")


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


def start_server(port, env_extra, tag):
    log_path = os.path.join(SCRIPT_DIR, f".test-grammar-{tag}.log")
    log = open(log_path, "w")
    env = dict(os.environ)
    env.update(env_extra)
    proc = subprocess.Popen(
        [TEST_BIN, MODEL, "--metal", "--max-ctx", "2048", "--port", str(port)],
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
                    print(f"server ({tag}): up on http://127.0.0.1:{port}")
                    return proc, log_path
        except FileNotFoundError:
            pass
        time.sleep(1)
    proc.kill()
    fail(f"server ({tag}) did not start within 180s")


def chat_once(port, fact):
    payload = {
        "model": "ornith",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"记住：我最喜欢的东西是{fact}"},
        ],
        "stream": True,
        "options": {"num_predict": NUM_PREDICT},
        "tools": MEMORY_TOOL,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/chat", data=body,
        headers={"Content-Type": "application/json"},
    )
    tool_calls = []
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            tool_calls.extend(d.get("message", {}).get("tool_calls", []) or [])
    return tool_calls


def run_mode(tag, port, env_extra):
    proc, log_path = start_server(port, env_extra, tag)
    ok = 0
    details = []
    try:
        for i in range(N):
            fact = FACTS[i % len(FACTS)]
            t0 = time.time()
            calls = chat_once(port, fact)
            dt = time.time() - t0
            good = (
                len(calls) == 1
                and calls[0].get("function", {}).get("name") == "memory"
                and calls[0].get("function", {}).get("arguments", {}).get("script")
            )
            ok += 1 if good else 0
            snippet = ""
            if calls:
                fn = calls[0].get("function", {})
                snippet = f"{fn.get('name')}(script={str(fn.get('arguments', {}).get('script'))[:60]}...)"
            details.append(f"  [{i+1}/{N}] fact={fact} {'OK ' if good else 'BAD'} ({dt:.1f}s) {snippet}")
        with open(log_path) as f:
            log_text = f.read()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    for line in details:
        print(line)
    toks = [float(m) for m in TOKS_RE.findall(log_text)]
    avg_toks = sum(toks) / len(toks) if toks else 0.0
    parse_failures = log_text.count("PARSE FAILURE")
    grammar_on = "tool-grammar ON" in log_text
    print(f"  [{tag}] parseable memory calls: {ok}/{N} | "
          f"server PARSE FAILUREs: {parse_failures} | "
          f"grammar engaged: {grammar_on} | avg gen: {avg_toks:.1f} tok/s")
    return ok, parse_failures, grammar_on


def main():
    if not os.path.exists(MODEL):
        fail(f"model not found: {MODEL} (set MODEL env var)")
    build_if_stale()

    print("== grammar ON (default) ==")
    on_ok, on_fail, on_engaged = run_mode("on", PORT_ON, {})
    print("== grammar OFF (TOOLCALL_GRAMMAR=0) ==")
    off_ok, off_fail, off_engaged = run_mode("off", PORT_OFF, {"TOOLCALL_GRAMMAR": "0"})

    print(f"\nsummary: ON {on_ok}/{N} parseable ({on_fail} server-side failures) | "
          f"OFF {off_ok}/{N} parseable ({off_fail} server-side failures)")
    if not on_engaged:
        fail("grammar was not engaged in ON mode (expected 'tool-grammar ON' in log)")
    if off_engaged:
        fail("grammar unexpectedly engaged in OFF mode (TOOLCALL_GRAMMAR=0 ignored?)")
    if on_ok < N:
        fail(f"grammar ON: only {on_ok}/{N} requests produced a parseable memory call")
    print("PASS: constrained generation produced parseable tool calls on every request")


if __name__ == "__main__":
    main()
