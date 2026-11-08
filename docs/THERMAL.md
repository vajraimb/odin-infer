# Thermal test protocol (sustained load on MacBook Air)

Part of odin-infer's positioning is: **on a passively cooled Air, the
low-thread kernels sustain a flatter throughput curve than llama.cpp**. This
protocol verifies or falsifies that claim — until the data exists, the
"doesn't slow down" claim must NOT be advertised (POSITIONING.md §6).

[中文版 / Chinese version](THERMAL.zh.md)

## Protocol

1. **Cool down**: machine idle ≥ 20 min before each run; close other heavy
   processes (measured: pie deepseek agent + trading bots pulled BOTH engines
   down 2×).
2. **Solo residency**: two 9B engines co-resident on 16 GB thrash each other;
   run one at a time and unload it afterwards.
3. **Same model & params**: default heretic Q4_K_M, same prompt,
   max_tokens=200, same temperature. ollama model tag:
   `hf.co/mradermacher/...:Q4_K_M`.
4. **Two runs each**, compare median curves (single runs get polluted by
   background tasks).

## Run

```sh
# odin (start ./serve.sh first)
python3 tools/thermal_test.py --engine odin    --duration 600 --out odin_t1.csv
python3 tools/thermal_test.py --engine ollama --duration 600 --out llama_t1.csv
# ... cool down 20 min, swap order, run t2 ...
```

Output: `t_s, wall_s, tps` samples + summary (early/late/peak/trough/droop%).

## Verdict

Compare the two engines' **droop%** (mean of first 3 samples vs last 3) and
trough:

| Result | Conclusion |
|---|---|
| odin droop clearly smaller | positioning holds; write the measured data into the README |
| Roughly equal | reword positioning to neutral ("no worse") |
| odin droop larger | back to kernel power work (8-lane config is already optimal; profiler tools needed) |

> Observed on this M3 (a fanned Pro model): thermal throttling after ~2 h of
> sustained load (same config 13.4 → 4.7 tok/s), but no same-machine curve
> comparison against llama.cpp exists yet — that is exactly what this protocol
> is for.
