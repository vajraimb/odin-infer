#!/usr/bin/env python3
"""A/B check for the bm=64 bk=64 MMQ adoption (engine follow-up of gemm_autotune).

Starts the OLD serve binary (pre-b64) and the NEW one in turn, sends the same
greedy (temperature=0) requests to both, and asserts byte-identical replies.
Perf signal: QTIMING=1 per-chunk GPU ms/tok lines from the server log
(median over chunks), immune to host-side wall-clock noise.

Usage: python3 serve_harness/ab_b64_check.py
Env: MODEL, ports fixed at 9765 (old) / 9766 (new).
"""
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
OLD_BIN = os.path.join(SCRIPT_DIR, ".test-serve-old.bin")
NEW_BIN = os.path.join(SCRIPT_DIR, ".test-serve.bin")
MODEL = os.environ.get(
    "MODEL", "/Users/vajra/Claw/odin-infeer/models/ornith-1.0-9b-Q4_K_M.gguf"
)

GREEDY_PROMPT = "用三句话解释什么是递归神经网络"
# long prefill probe: ~1500+ tokens of repeated Chinese text
LONG_PROMPT = "人工智能的发展历程充满了曲折与突破。" * 300
TIMING_RE = re.compile(r"\[timing\] forward_gpu_batch T=(\d+) GPU=([\d.]+) ms \(([\d.]+) ms/tok\)")


def port_of(bin_path):
    return 9765 if "old" in bin_path else 9766


def start(bin_path):
    port = port_of(bin_path)
    env = dict(os.environ)
    env["KV_PERSIST"] = "0"
    env["QTIMING"] = "1"
    log_path = os.path.join(SCRIPT_DIR, f".ab-{os.path.basename(bin_path)}.log")
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [bin_path, MODEL, "--metal", "--max-ctx", "8192", "--port", str(port)],
        stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env,
    )
    deadline = time.time() + 180
    while time.time() < deadline:
        if proc.poll() is not None:
            log.close()
            print(open(log_path).read())
            sys.exit(f"FAIL: {bin_path} exited during startup")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=1)
            log.close()
            return proc, log_path
        except Exception:
            time.sleep(1)
    proc.kill()
    sys.exit(f"FAIL: {bin_path} did not start")


def chat(port, prompt, num_predict):
    body = json.dumps({
        "model": "ornith",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {"num_predict": num_predict, "temperature": 0},
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/chat", data=body,
        headers={"Content-Type": "application/json"},
    )
    parts = []
    with urllib.request.urlopen(req, timeout=600) as resp:
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


def run(bin_path):
    proc, log_path = start(bin_path)
    try:
        reply = chat(port_of(bin_path), GREEDY_PROMPT, 64)
        chat(port_of(bin_path), LONG_PROMPT, 1)
    finally:
        proc.kill()
        proc.wait()
    with open(log_path) as f:
        lines = TIMING_RE.findall(f.read())
    # median ms/tok over full chunks (T=512; drop ragged tail chunks)
    ms_per_tok = [float(m) for t, g, m in lines if int(t) == 512]
    return reply, ms_per_tok


def main():
    if not (os.path.exists(OLD_BIN) and os.path.exists(NEW_BIN)):
        sys.exit("FAIL: need both .test-serve-old.bin and .test-serve.bin")
    old_reply, old_gpu = run(OLD_BIN)
    print(f"old: reply {len(old_reply)} chars, GPU ms/tok per chunk: {old_gpu}")
    old_med = statistics.median(old_gpu) if old_gpu else None
    new_reply, new_gpu = run(NEW_BIN)
    print(f"new: greedy reply {len(new_reply)} chars, GPU ms/tok per chunk: {new_gpu}")
    new_med = statistics.median(new_gpu) if new_gpu else None

    if old_reply != new_reply:
        print("--- old ---"); print(old_reply)
        print("--- new ---"); print(new_reply)
        sys.exit("FAIL: greedy replies differ (numeric drift vs pre-b64 kernel)")
    print("bit-identical vs pre-b64 kernel: PASS")
    if old_med and new_med:
        print(f"prefill GPU ms/tok (median): {old_med:.2f} -> {new_med:.2f} ({old_med / new_med:.3f}x)")
    else:
        print("WARN: no [timing] lines captured; GPU comparison unavailable")
    print("PASS")


if __name__ == "__main__":
    main()
