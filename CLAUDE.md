# MuTheoryEval — Claude Code Instructions

## Running Benchmarks

### 1. 先估算再跑
```bash
python run.py --estimate --model gemini-3.1-flash-lite --benchmark all
```
确认 cost / time 后再执行。

### 2. 用 background task 跑（必须）
长跑任务（>5min）一律用 Bash tool 的 `run_in_background: true`，不要直接 blocking 跑。

```bash
# 推荐命令格式
cd /Users/dimpurr/Workflow/Code/academic/vertex/MuTheoryEval && \
  python run.py --model gemini-3.1-flash-lite --benchmark all 2>&1 | tee results/run_$(date +%Y%m%d_%H%M%S).log
```

### 3. 检查进度
```bash
# 看最新 log
ls -t results/*.log | head -1 | xargs tail -20

# 看实时进度（partial 文件每完成一个 bench 就更新）
cat results/*_partial.json 2>/dev/null | python -m json.tool | grep -E '"benchmark"|"accuracy"'

# 或者直接 tail log
tail -f results/run_YYYYMMDD_HHMMSS.log
```

### 4. 断点续跑
如果任务中断，直接重跑同样命令。`run.py` 会自动检测 `.checkpoint_*.json` 文件从断点恢复。

### 5. 查看历史结果
```bash
ls -lt results/*.json | grep -v partial | grep -v checkpoint
```

## Benchmark 状态

- `music_theory_bench` ✅ 可跑 (HF: m-a-p/MusicTheoryBench)
- `ziqi_eval`          ✅ 可跑 (HF: MYTH-Lab/ZIQI-Eval，默认 500 题采样)
- `abc_eval`           ❌ UNRELEASED
- `ssmr_bench`         ❌ UNRELEASED

## 已知结果快照

| Model | MusicTheoryBench | ZIQI-Eval(500) | Weighted |
|-------|-----------------|----------------|----------|
| gemini-3.1-flash-lite | 66.8% | TBD | TBD |

## 费用参考（flash-lite 基准）

| Benchmark | 题数 | 预估费用 | 预估时间 |
|-----------|------|---------|---------|
| music_theory_bench | 367 | ~$0.006 | ~3min |
| ziqi_eval (500 sampled) | 500 | ~$0.008 | ~4min |
| **全量** | 867 | **~$0.014** | **~7min** |
