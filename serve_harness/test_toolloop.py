#!/usr/bin/env python3
r"""Regression: assistant tool_calls must round-trip through history.

Bug history (2026-10-31): serve.odin rendered assistant history turns with
only their text content — the tool_calls field was silently dropped. In an
agent loop the model therefore never saw that it had already called a tool,
and re-issued the same call forever (the "infinite tool-call loop" symptom:
each round a full prefill, nothing ever answered).

Fix: history renders the tool call as JSON inside <tool_call> and the tool
result as a plain "工具执行结果: ..." user turn. Both choices are empirical
(4+ template variants probed under greedy decode): this model treats the
native <function=...> form in history as "about to call" (loops), and does
not know the <tool_response> wrapper at all (loops in either role). JSON
form + plain user result is the only combination that reliably produces a
final answer.

What this test does
-------------------
  1. turn 1: user asks to run `echo hello` with a bash tool available;
     expects a bash tool_call,
  2. executes it for real,
  3. turn 2: assistant tool_call + tool result in history; expects a PROSE
     answer (no further tool call) served from a cache hit.

Usage
-----
    python3 serve_harness/test_toolloop.py

Env: MODEL (gguf path), PORT (9766). Exit code 0 on PASS, 1 on FAIL.
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
LOG_PATH = os.path.join(SCRIPT_DIR, ".test-toolloop.log")

MODEL = os.environ.get(
    "MODEL", "/Users/vajra/Claw/odin-infeer/models/ornith-1.0-9b-Q4_K_M.gguf"
)
PORT = int(os.environ.get("PORT", "9766"))
BASE = f"http://127.0.0.1:{PORT}"

TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"},
                                      "timeout": {"type": "integer"}},
                       "required": ["command"]},
    },
}]

SYSTEM = ("You are pie, a warm and helpful coding assistant. "
          "Always reply in the language the user uses.")


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
        return
    print("build: compiling serve.odin ...")
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


def start_server():
    env = dict(os.environ)
    env["KV_PERSIST"] = "0"
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
        if "listening" in open(LOG_PATH).read():
            return proc
        time.sleep(1)
    proc.kill()
    fail("server did not start within 180s")


def chat(messages):
    payload = {"model": "ornith", "messages": messages, "tools": TOOLS,
               "stream": True, "options": {"num_predict": 96, "temperature": 0}}
    req = urllib.request.Request(f"{BASE}/api/chat", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    content, tool_calls = [], []
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = d.get("message", {})
            content.append(m.get("content", ""))
            tool_calls.extend(m.get("tool_calls") or [])
    return "".join(content), tool_calls


def main():
    if not os.path.exists(MODEL):
        fail(f"model not found: {MODEL} (set MODEL env var)")
    build_if_stale()
    proc = start_server()
    try:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "请用 bash 工具执行 echo hello，不要解释"}]
        c1, tc1 = chat(msgs)
        if not tc1:
            fail("turn 1: expected a bash tool_call")
        fn = tc1[0]["function"]
        args = fn["arguments"]
        cmd = args.get("command", "")
        print(f"turn 1: tool call -> {cmd!r}")

        out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                             timeout=int(args.get("timeout", 10) or 10)).stdout.strip()
        print(f"turn 1: tool result -> {out!r}")

        # Clients store prose content and tool calls separately (Ollama
        # convention): strip the raw <tool_call> block from content.
        c1_clean = re.sub(r"<tool_call>.*?</tool_call>", "", c1, flags=re.S).strip()
        msgs2 = msgs + [
            {"role": "assistant", "content": c1_clean,
             "tool_calls": [{"function": {"name": fn["name"], "arguments": args}}]},
            {"role": "tool", "content": out or "(no output)"},
        ]
        c2, tc2 = chat(msgs2)
        print(f"turn 2: content={c2[:80]!r} tool_calls={len(tc2)}")

        with open(LOG_PATH) as f:
            log_text = f.read()
        if "PREFIX CACHE HIT" not in log_text and "continuation hit" not in log_text:
            fail("turn 2: expected a cache hit (snapshot or continuation)")
        if tc2:
            fail("turn 2 emitted ANOTHER tool call — loop not broken")
        if not c2.strip():
            fail("turn 2: empty prose answer")
        print("PASS: tool-call loop broken (cache hit + prose answer)")
    finally:
        proc.kill()


if __name__ == "__main__":
    main()
