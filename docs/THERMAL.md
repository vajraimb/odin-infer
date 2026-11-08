# 热测试协议（MacBook Air 持续负载对照）

odin-infer 定位的卖点之一是：**被动散热的 Air 上，低线程内核的持续吞吐曲线
比 llama.cpp 更平**。本协议用于验证或证伪这一点——在拿到数据前，不得在
宣传中把"不掉速"当作事实（见 POSITIONING.md 第六条）。

## 协议

1. **冷却**：测试前机器空载 ≥ 20 分钟（合盖休眠后唤醒即可加速）；
   关掉其他重负载进程（本会话实测：pie deepseek 智能体 + 交易机器人同时
   运行时，两个引擎的吞吐都被拉低 2×）。
2. **单引擎驻留**：16GB 机器同时驻留两个 9B 引擎会互换内存；每次只跑一个
   引擎，跑完卸载（ollama：`ollama stop`；odin：杀掉进程）。
3. **同模型同参数**：默认 heretic Q4_K_M、同一 prompt、max_tokens=200、
   温度一致。ollama 用 `hf.co/mradermacher/...:Q4_K_M`。
4. **各跑两次**，取中位数曲线（单次数据易被后台任务污染）。

## 执行

```sh
# odin（先 ./serve.sh 启动 HTTP 服务）
python3 tools/thermal_test.py --engine odin    --duration 600 --out odin_t1.csv
python3 tools/thermal_test.py --engine ollama --duration 600 --out llama_t1.csv
# ... 冷却 20 分钟，交换顺序再跑一轮 t2 ...
```

输出为 `t_s, wall_s, tps` 采样序列 + 汇总（early/late/peak/trough/droop%）。

## 判定

对比两引擎的 **droop%（首 3 样本均值 vs 末 3 样本均值）** 与 trough：

| 结果 | 结论 |
|---|---|
| odin droop 明显小于 llama.cpp | 定位成立，写入 README 实测数据 |
| 两者接近 | 定位改为中性表述（"不差于"） |
| odin droop 更大 | 需要回到内核功耗优化（低线程配置已是最优，需剖析工具） |

> 本会话在 M3（有风扇 Pro 机型）上已观察到持续负载 ~2 小时后的热降频
> （同配置 13.4 → 4.7 tok/s），但未做过与 llama.cpp 的同机对照曲线——这是
> 本协议要补上的实证。
