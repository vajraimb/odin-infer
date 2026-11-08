#!/usr/bin/env python3
"""MCP (stdio) integration test for odin-serve.

Builds a cached test binary (server/serve_harness/.test-serve.bin), spawns it
with --mcp, and runs a full MCP handshake: initialize -> tools/list ->
tools/call(odin.models) -> tools/call(odin.chat) -> unknown-method error.

Usage: python3 server/serve_harness/test_mcp.py   # MODEL=... PORT=... to override
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SERVE_SRC = os.path.join(ROOT, "server/serve_harness/serve.odin")
TEST_BIN = os.path.join(ROOT, "server/serve_harness/.test-serve.bin")
MODEL = os.environ.get(
    "MODEL",
    os.path.join(ROOT, "models/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf"),
)



def newest_src_mtime():
    newest = os.path.getmtime(SERVE_SRC)
    for root, _, files in os.walk(os.path.join(ROOT, "packages")):
        for f in files:
            if f.endswith(".odin"):
                newest = max(newest, os.path.getmtime(os.path.join(root, f)))
    return newest


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def build_binary():
    if os.path.exists(TEST_BIN) and os.path.getmtime(TEST_BIN) > newest_src_mtime():
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


def rpc(proc, method, params=None, msg_id=1):
    req = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        req["params"] = params
    proc.stdin.write(json.dumps(req) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        fail(f"no response for {method}")
    return json.loads(line)


def notify(proc, method):
    proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method}) + "\n")
    proc.stdin.flush()


def main():
    build_binary()
    print(f"spawn: {TEST_BIN} {MODEL} --mcp --metal --max-ctx 4096")
    proc = subprocess.Popen(
        [TEST_BIN, MODEL, "--mcp", "--metal", "--max-ctx", "4096"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=ROOT,
    )
    try:
        # initialize (block until the model is loaded: first response arrives after load)
        t0 = time.time()
        init = rpc(proc, "initialize",
                   {"protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "test_mcp", "version": "1.0"}})
        assert init["jsonrpc"] == "2.0" and init["id"] == 1, f"initialize: {init}"
        assert init["result"]["protocolVersion"] == "2024-11-05", f"protocol: {init}"
        assert "tools" in init["result"]["capabilities"], f"capabilities: {init}"
        print(f"initialize OK ({time.time()-t0:.1f}s incl. model load)")

        notify(proc, "notifications/initialized")

        # ping
        pong = rpc(proc, "ping", msg_id=2)
        assert pong["result"] == {}, f"ping: {pong}"
        print("ping OK")

        # tools/list
        tlist = rpc(proc, "tools/list", msg_id=3)
        names = sorted(t["name"] for t in tlist["result"]["tools"])
        assert names == ["odin.chat", "odin.models"], f"tools: {names}"
        print(f"tools/list OK: {names}")

        # tools/call odin.models
        models = rpc(proc, "tools/call",
                     {"name": "odin.models", "arguments": {}}, msg_id=4)
        text = models["result"]["content"][0]["text"]
        assert "model:" in text and "metal:" in text, f"models text: {text!r}"
        print(f"tools/call odin.models OK: {text.splitlines()[0]}")

        # tools/call odin.chat
        chat = rpc(proc, "tools/call",
                   {"name": "odin.chat",
                    "arguments": {"messages": [{"role": "user",
                                                "content": "What is 2+2? Answer in one word."}],
                                  "max_tokens": 8}}, msg_id=5)
        text = chat["result"]["content"][0]["text"]
        assert "4" in text, f"chat answer: {text!r}"
        print(f"tools/call odin.chat OK: {text!r}")

        # unknown method -> JSON-RPC error
        err = rpc(proc, "bogus/method", msg_id=6)
        assert err["error"]["code"] == -32601, f"error: {err}"
        print("unknown-method error OK (-32601)")

        # unknown tool -> text error, not protocol error
        bad = rpc(proc, "tools/call", {"name": "nope", "arguments": {}}, msg_id=7)
        assert "unknown tool" in bad["result"]["content"][0]["text"], f"bad tool: {bad}"
        print("unknown-tool OK")

        print("\nMCP handshake: ALL PASSED")
    finally:
        proc.kill()
        proc.wait(timeout=10)


if __name__ == "__main__":
    main()
