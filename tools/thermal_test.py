#!/usr/bin/env python3
"""Sustained-load thermal test for local inference engines.

Keeps the engine continuously generating for --duration seconds and samples
decode throughput every --interval seconds, producing a tps-over-time curve.
Passive-cooled machines (MacBook Air) throttle under sustained load; the
claim behind odin-infer's positioning is a FLATTER curve than llama.cpp.

Protocol (docs/THERMAL.md):
  - machine idle >= 20 min before each run, no other heavy processes
  - ONE engine resident at a time; same model file, same prompt, same params
  - run each engine twice and compare median curves

Usage:
  python3 tools/thermal_test.py --engine odin    [--duration 600] [--interval 30]
  python3 tools/thermal_test.py --engine ollama  --model hf.co/...:Q4_K_M
"""
import argparse
import json
import time
import urllib.request

ODIN = "http://127.0.0.1:9748"
OLLAMA = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "hf.co/mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-GGUF:Q4_K_M"
PROMPT = ("Explain the concept of recursion in programming in a few sentences, "
          "with a simple example, then discuss tail recursion and stack safety.")
MAX_TOKENS = 200


def post(url, body, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def odin_once():
    body = {"model": "odin-infer",
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MAX_TOKENS}
    t0 = time.time()
    post(ODIN + "/v1/chat/completions", body)
    wall = time.time() - t0
    st = json.loads(urllib.request.urlopen(ODIN + "/api/state", timeout=10).read())
    return st["last_tps"], wall


def ollama_once(model):
    body = {"model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "stream": False,
            "options": {"num_predict": MAX_TOKENS, "temperature": 0.6}}
    t0 = time.time()
    d = post(OLLAMA + "/api/chat", body)
    wall = time.time() - t0
    tps = d["eval_count"] / d["eval_duration"] * 1e9
    return tps, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["odin", "ollama"])
    ap.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    ap.add_argument("--duration", type=float, default=600)
    ap.add_argument("--interval", type=float, default=30)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    once = odin_once if args.engine == "odin" else (lambda: ollama_once(args.model))
    print(f"engine={args.engine} duration={args.duration}s interval={args.interval}s")
    print("t_s, wall_s, tps")

    samples = []  # (t, wall, tps)
    t0 = time.time()
    next_sample = args.interval
    out = open(args.out, "w") if args.out else None
    if out:
        out.write("t_s,wall_s,tps\n")

    # warmup request (also pays model load for ollama)
    once()
    while time.time() - t0 < args.duration:
        tps, wall = once()
        t = time.time() - t0
        if t >= next_sample:
            samples.append((t, wall, tps))
            print(f"{t:6.0f}, {wall:5.1f}, {tps:5.2f}")
            if out:
                out.write(f"{t:.0f},{wall:.1f},{tps:.2f}\n")
            next_sample += args.interval
    if out:
        out.close()

    if len(samples) >= 2:
        early = [s[2] for s in samples[:3]]
        late = [s[2] for s in samples[-3:]]
        peak = max(s[2] for s in samples)
        trough = min(s[2] for s in samples)
        droop = (1.0 - sum(late) / sum(early)) * 100.0
        print(f"\nsummary: early={sum(early)/len(early):.2f} tok/s "
              f"late={sum(late)/len(late):.2f} tok/s "
              f"peak={peak:.2f} trough={trough:.2f} droop={droop:.1f}%")


if __name__ == "__main__":
    main()
