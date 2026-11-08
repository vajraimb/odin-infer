#!/usr/bin/env python3
"""First-run GEMV autotuning integration test.

Run 1 (fresh ODIN_TUNE_DIR): expects the engine to time the kernel candidates
and write the .tune cache. Run 2 (same dir): expects the cache to be loaded
without retuning. Both runs answer a chat request.

Usage: python3 server/serve_harness/test_tune.py   # MODEL=... to override
"""
import glob
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN = os.path.join(ROOT, "server/serve_harness/.test-serve.bin")
SRC = os.path.join(ROOT, "server/serve_harness/serve.odin")
MODEL = os.environ.get(
    "MODEL",
    os.path.join(ROOT, "models/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf"),
)
TUNE_DIR = tempfile.mkdtemp(prefix="odin-tune-test-")



def newest_src_mtime():
    newest = os.path.getmtime(SRC)
    for root, _, files in os.walk(os.path.join(ROOT, "packages")):
        for f in files:
            if f.endswith(".odin"):
                newest = max(newest, os.path.getmtime(os.path.join(root, f)))
    return newest


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def build_binary():
    if os.path.exists(BIN) and os.path.getmtime(BIN) > newest_src_mtime():
        return
    cmd = [
        "odin", "build", "server/serve_harness/serve.odin", "-file", "-o:speed",
        "-collection:ggml=packages", "-collection:infer=packages",
        "-collection:qwen3_5=packages", "-collection:qwen3_5_tokenizer=packages",
        "-collection:sampler=packages", "-collection:tokenizer=packages",
        f"-out:{BIN}",
    ]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        fail("odin build failed:\n" + r.stderr)


def run_once(expect):
    """Spawn the server in MCP mode, drive one chat call, return (stderr, answer)."""
    env = dict(os.environ)
    env["ODIN_TUNE_DIR"] = TUNE_DIR
    proc = subprocess.Popen(
        [BIN, MODEL, "--mcp", "--metal", "--max-ctx", "4096"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=ROOT, env=env,
    )
    answer = None
    try:
        req = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": "odin.chat",
                          "arguments": {"messages": [{"role": "user",
                                                      "content": "What is 2+2? Answer in one word."}],
                                        "max_tokens": 8}}}
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if line:
            answer = json.loads(line)["result"]["content"][0]["text"]
    finally:
        proc.stdin.close()
        proc.wait(timeout=600)
        stderr = proc.stderr.read() if proc.stderr else ""
    return stderr, answer


def main():
    build_binary()
    print(f"tune dir: {TUNE_DIR}")

    stderr1, answer1 = run_once(expect="tune")
    assert "4" in (answer1 or ""), f"run1 chat: {answer1!r}"
    print("run1 chat OK: '4'")
    if "tune: saved" not in stderr1:
        fail(f"run1 did not tune:\n{stderr1[-2000:]}")
    for line in stderr1.splitlines():
        if line.startswith("tune: Q"):
            print("  " + line)
    files = glob.glob(os.path.join(TUNE_DIR, "*.tune"))
    if len(files) != 1:
        fail(f"expected 1 cache file, got {files}")
    n_entries = len(open(files[0]).read().strip().splitlines())
    print(f"run1: cache written with {n_entries} entries")

    stderr2, answer2 = run_once(expect="load")
    assert "4" in (answer2 or ""), f"run2 chat: {answer2!r}"
    print("run2 chat OK: '4'")
    if "tune: loaded" not in stderr2:
        fail(f"run2 did not load cache:\n{stderr2[-2000:]}")
    if "tune: saved" in stderr2:
        fail("run2 retuned instead of loading the cache")
    print("run2: cache loaded, no retune")

    print("\nAUTOTUNE: ALL PASSED")


if __name__ == "__main__":
    main()
