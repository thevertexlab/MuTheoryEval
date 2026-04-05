# MuTheoryEval — Claude Code 操作指南

## 跑 Benchmark 的标准流程

### 第一步：先估算再跑
```bash
python run.py --estimate --model gemini-3.1-flash-lite --benchmark all
```
确认费用和时间后再决定是否执行。

### 第二步：用 background task 跑（长任务必须）
超过 5 分钟的任务一律用 Bash tool 的 `run_in_background: true`，不要 blocking 等待。

```bash
cd /Users/dimpurr/Workflow/Code/academic/vertex/MuTheoryEval && \
  python run.py --model gemini-3.1-flash-lite --benchmark all 2>&1 | tee results/run_$(date +%Y%m%d_%H%M%S).log
```

### 第三步：定期检查进度
```bash
# 看最新 log 末尾
ls -t results/*.log | head -1 | xargs tail -20

# 看 partial 中间结果（每完成一个 bench 自动写入）
cat results/*_partial.json 2>/dev/null | python -m json.tool | grep -E '"benchmark"|"accuracy"'
```

### 断点续跑
中断后直接重跑同样命令，`run.py` 自动检测 `.checkpoint_*.json` 从断点恢复，不重复消耗 API。

### 查看历史结果
```bash
ls -lt results/*.json | grep -v partial | grep -v checkpoint
```

---

## Benchmark 状态

| Benchmark | 状态 | 数据集 |
|-----------|------|--------|
| `music_theory_bench` | ✅ 可跑 | HF: m-a-p/MusicTheoryBench |
| `ziqi_eval` | ✅ 可跑 | HF: MYTH-Lab/ZIQI-Eval（默认采样 500 题）|
| `abc_eval` | ❌ 未发布 | anonymous 链接已过期，无公开 repo |
| `ssmr_bench` | ❌ 未发布 | anonymous 链接已过期，无公开 repo |

---

## 费用参考（以 gemini-3.1-flash-lite 为基准）

| Benchmark | 题数 | 预估费用 | 预估时间 |
|-----------|------|---------|---------|
| music_theory_bench | 367 | ~$0.006 | ~3 min |
| ziqi_eval（采样） | 500 | ~$0.008 | ~4 min |
| **全量** | **867** | **~$0.014** | **~7 min** |

---

## 已知结果快照

| 模型 | MusicTheoryBench | ZIQI-Eval(500) | 加权分 |
|------|-----------------|----------------|--------|
| gemini-3.1-flash-lite | 66.8% | 待跑 | 待跑 |
