# MuTheoryEval — Claude Code Instructions

## Running Benchmarks

### Step 1: Estimate before running
```bash
python run.py --estimate --model gemini-3.1-flash-lite --benchmark all
```
Review cost and time before proceeding.

### Step 2: Always use background tasks for long runs (>5 min)
Use Bash tool with `run_in_background: true`. Never block-wait on long jobs.

```bash
cd /Users/dimpurr/Workflow/Code/academic/vertex/MuTheoryEval && \
  python run.py --model gemini-3.1-flash-lite --benchmark all 2>&1 | tee results/run_$(date +%Y%m%d_%H%M%S).log
```

### Step 3: Check progress periodically
```bash
# Tail the latest log
ls -t results/*.log | head -1 | xargs tail -20

# Check partial results (written after each benchmark completes)
cat results/*_partial.json 2>/dev/null | python -m json.tool | grep -E '"benchmark"|"accuracy"'
```

### Resume after interruption
Just re-run the same command. `run.py` auto-detects `.checkpoint_*.json` and resumes from where it stopped — no duplicate API calls.

### View past results
```bash
ls -lt results/*.json | grep -v partial | grep -v checkpoint
```

---

## Benchmark Status

| Benchmark | Status | Dataset |
|-----------|--------|---------|
| `music_theory_bench` | ✅ Ready | HF: m-a-p/MusicTheoryBench |
| `ziqi_eval` | ✅ Ready | HF: MYTH-Lab/ZIQI-Eval (default: 500-sample) |
| `abc_eval` | ❌ Unreleased | Anonymous review link expired, no public repo |
| `ssmr_bench` | ❌ Unreleased | Anonymous review link expired, no public repo |

---

## Cost Reference (gemini-3.1-flash-lite baseline)

| Benchmark | Questions | Est. Cost | Est. Time |
|-----------|-----------|-----------|-----------|
| music_theory_bench | 367 | ~$0.006 | ~3 min |
| ziqi_eval (sampled) | 500 | ~$0.008 | ~4 min |
| **All** | **867** | **~$0.014** | **~7 min** |

---

## Known Results

| Model | MusicTheoryBench | ZIQI-Eval (500) | Weighted |
|-------|-----------------|-----------------|---------|
| gemini-3.1-flash-lite | 66.8% | 81.0% | 73.9% |
