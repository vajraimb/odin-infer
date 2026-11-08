# Thermal test protocol (sustained load on MacBook Air) / 热测试协议（Air 持续负载对照）

Part of odin-infer's positioning is: **on a passively cooled Air, the
low-thread kernels sustain a flatter throughput curve than llama.cpp**. This
protocol verifies or falsifies that claim — until the data exists, the
"doesn't slow down" claim must NOT be advertised (POSITIONING.md §6).
odin-infer 定位的卖点之一是：被动散热的 Air 上，低线程内核的持续吞吐曲线比
llama.cpp 更平。本协议用于验证或证伪——在拿到数据前，不得对外宣称。

## Protocol / 协议

1. **Cool down**: machine idle ≥ 20 min before each run; close other heavy
   processes (measured: pie deepseek agent + trading bots pulled BOTH engines
   down 2×). 测试前机器空载 ≥ 20 分钟；关掉其他重负载进程。
2. **Solo residency**: two 9B engines co-resident on 16 GB thrash each other;
   run one at a time and unload it afterwards. 单引擎驻留：每次只跑一个引擎。
3. **Same model & params**: default heretic Q4_K_M, same prompt,
   max_tokens=200, same temperature. ollama model tag:
   `hf.co/mradermacher/...:Q4_K_M`. 同模型同参数（默认 heretic Q4_K_M、同一 prompt、max_tokens=200）。
4. **Two runs each**, compare median curves (single runs get polluted by
   background tasks). 各跑两次，取中位数曲线。

## Run / 执行

```sh
# odin (start ./serve.sh first) / odin（先 ./serve.sh 启动 HTTP 服务）
python3 tools/thermal_test.py --engine odin    --duration 600 --out odin_t1.csv
python3 tools/thermal_test.py --engine ollama --duration 600 --out llama_t1.csv
# ... cool down 20 min, swap order, run t2 ... / ... 冷却 20 分钟，交换顺序再跑一轮 t2 ...
```

Output: `t_s, wall_s, tps` samples + summary (early/late/peak/trough/droop%).
输出为 `t_s, wall_s, tps` 采样序列 + 汇总（early/late/peak/trough/droop%）。

## Verdict / 判定

Compare the two engines' **droop%** (mean of first 3 samples vs last 3) and
trough. 对比两引擎的 droop%（首 3 样本均值 vs 末 3 样本均值）与 trough：

| Result / 结果 | Conclusion / 结论 |
|---|---|
| odin droop clearly smaller / odin droop 明显更小 | positioning holds; write the measured data into the README / 定位成立，写入 README 实测数据 |
| Roughly equal / 两者接近 | reword positioning to neutral ("no worse") / 定位改为中性表述 |
| odin droop larger / odin droop 更大 | back to kernel power work (8-lane config is already optimal; profiler tools needed) / 回到内核功耗优化（需剖析工具） |

> Observed on this M3 (a fanned Pro model): thermal throttling after ~2 h of
> sustained load (same config 13.4 → 4.7 tok/s), but no same-machine curve
> comparison against llama.cpp exists yet — that is exactly what this protocol
> is for. 本会话在 M3 上已观察到持续负载 ~2 小时后的热降频，但未做过与
> llama.cpp 的同机对照曲线——这是本协议要补上的实证。
