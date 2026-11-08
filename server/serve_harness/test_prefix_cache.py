#!/usr/bin/env python3
r"""Regression test: multi-turn Chinese conversation through the Ollama endpoint
with system-prompt prefix caching.

Why this test exists
--------------------
Two distinct bugs produced the same symptom — the model emitting literal
unicode-escape text ("u6211u64c5...", "\\u6bd4\\u5982...") instead of Chinese
on turn 2+ of a conversation:

  1. Prefix-cache snapshot mispositioning. qwen3_5 is a hybrid architecture
     (8 full-attention + 24 linear-attention layers). The linear layers'
     conv/recurrent state evolves with position and is NOT rewindable.
     serve.odin used to save the prefix snapshot AFTER prefilling the whole
     prompt (user turn included) while labelling it as taken at the
     system-prompt boundary. On the next turn's PREFIX CACHE HIT, the suffix
     tokens were replayed on top of an already-advanced recurrent state,
     corrupting 24 of 32 layers.

  2. JSON \uXXXX escapes not decoded. The lenient JSON message parser handled
     \n \" \\ etc. but passed \uXXXX through as literal text. Standard JSON
     clients escape non-ASCII by default (Python json.dumps, Go encoding/json
     for some runes), so the model was fed literal "u4f60" text and started
     mirroring escape-speak in its output. This test intentionally sends
     ensure_ascii=True JSON (the default) to cover that path.

What it does
------------
  1. builds serve.odin into a cached test binary (rebuilds only when stale),
  2. starts it on a throwaway port with --metal,
  3. runs a 3-turn Chinese conversation via /api/chat (Ollama NDJSON),
     echoing assistant replies back into the history like pie does,
  4. asserts turn 2+ hit the PREFIX CACHE, and every reply is real CJK text
     with no unicode-escape garbage.

Usage
-----
    python3 server/serve_harness/test_prefix_cache.py

Env: MODEL (gguf path, defaults to the usual ornith location), PORT (9763).
Exit code 0 on PASS, 1 on FAIL.
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
LOG_PATH = os.path.join(SCRIPT_DIR, ".test-serve.log")

MODEL = os.environ.get(
    "MODEL", "/Users/vajra/Claw/odin-infeer/models/ornith-1.0-9b-Q4_K_M.gguf"
)
PORT = int(os.environ.get("PORT", "9763"))
BASE = f"http://127.0.0.1:{PORT}"

SYSTEM = (
    "You are pie, a warm and helpful coding assistant. "
    "Always reply in the language the user uses."
)
TURNS = [
    "你是哪个模型",
    "你擅长做什么",
    "用中文写一首关于秋天的四行短诗",
]

CJK_RE = re.compile(r"[一-鿿]")
ESCAPED_RE = re.compile(r"\\u[0-9a-fA-F]{4}")        # literal \uXXXX text
BARE_CODEPOINT_RE = re.compile(r"(?:u[0-9a-f]{4}){3,}")  # u6211u64c5u957f...
NUM_PREDICT = 48

# Minimal memory tool, mirroring what pie registers via /skill memory. Used to
# verify the model emits its NATIVE <tool_call><function=...> format (what it
# was trained on per its GGUF chat_template) and that serve parses it back.
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
    log = open(LOG_PATH, "w")
    proc = subprocess.Popen(
        [TEST_BIN, MODEL, "--metal", "--max-ctx", "2048", "--port", str(PORT)],
        stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
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


def chat(messages, tools=None):
    payload = {
        "model": "ornith",
        "messages": messages,
        "stream": True,
        "options": {"num_predict": NUM_PREDICT},
    }
    if tools:
        payload["tools"] = tools
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE}/api/chat", data=body, headers={"Content-Type": "application/json"}
    )
    parts = []
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
            msg = d.get("message", {})
            parts.append(msg.get("content", ""))
            tool_calls.extend(msg.get("tool_calls", []) or [])
    return "".join(parts), tool_calls


def server_log():
    with open(LOG_PATH) as f:
        return f.read()


def check_reply(turn_no, reply):
    print(f"turn {turn_no} reply: {reply[:120]}{'...' if len(reply) > 120 else ''}")
    if not reply.strip():
        fail(f"turn {turn_no}: empty reply")
    if ESCAPED_RE.search(reply):
        fail(f"turn {turn_no}: reply contains literal \\uXXXX escapes (prefix-cache state corruption?)")
    if BARE_CODEPOINT_RE.search(reply):
        fail(f"turn {turn_no}: reply contains bare uXXXX codepoint text (prefix-cache state corruption?)")
    if not CJK_RE.search(reply):
        fail(f"turn {turn_no}: expected Chinese reply, got none")


def main():
    if not os.path.exists(MODEL):
        fail(f"model not found: {MODEL} (set MODEL env var)")
    build_if_stale()
    proc = start_server()
    try:
        messages = [{"role": "system", "content": SYSTEM}]
        for i, user_msg in enumerate(TURNS, start=1):
            messages.append({"role": "user", "content": user_msg})
            reply, _ = chat(messages)
            check_reply(i, reply)
            messages.append({"role": "assistant", "content": reply})
            if i >= 2:
                # Either cache path is legitimate: a snapshot hit
                # ("PREFIX CACHE HIT") or a continuation hit (turn 2+ history
                # continues the previous request's tail). Since commit 88b2c42
                # fixed the continuation fast-path, turn 2 typically takes the
                # continuation route; both prove we skipped re-prefill.
                log = server_log()
                if "PREFIX CACHE HIT" not in log and "continuation hit" not in log:
                    fail(f"turn {i}: expected PREFIX CACHE HIT or continuation hit in server log")
                print(f"turn {i}: cache hit confirmed (snapshot or continuation)")

        # Turn 4 (with tools): the model must emit its NATIVE
        # <tool_call><function=memory><parameter=script>... form, and serve
        # must parse it back into a structured tool_calls NDJSON line.
        print("turn 4: requesting a memory tool call ...")
        tool_messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "记住：我最喜欢的水果是苹果"},
        ]
        _, tool_calls = chat(tool_messages, tools=MEMORY_TOOL)
        if not tool_calls:
            fail("turn 4: model produced no structured tool_call "
                 "(serve tool-prompt format drifted from the model's chat_template?)")
        fn = tool_calls[0].get("function", {})
        if fn.get("name") != "memory" or "script" not in (
            fn.get("arguments") if isinstance(fn.get("arguments"), dict) else {}
        ):
            fail(f"turn 4: unexpected tool_call shape: {tool_calls[0]}")
        if "tool calls detected" not in server_log():
            fail("turn 4: server did not log parsed tool calls")
        print(f"turn 4: native tool_call parsed OK -> {fn.get('name')}(script=...)")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("\nPASS: multi-turn Chinese conversation clean under prefix-cache hits")


if __name__ == "__main__":
    main()
