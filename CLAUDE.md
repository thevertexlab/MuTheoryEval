# MuTheoryEval — Claude Code Instructions

## Running Benchmarks

### Step 1: Estimate before running
```bash
# Default mode=lite (standardized samples, skip already-done cells)
python run.py --estimate --model gemini-3.1-flash-lite --benchmark all

# Including multimodal
python run.py --estimate --model gemini-3.1-flash-lite --benchmark all --multimodal

# Full dataset
python run.py --estimate --model gemini-3.1-flash-lite --benchmark all --mode full
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

### Step 4: Monitor for accuracy drops — intervene immediately

**During a run**, watch the rolling `acc=X%` in the log. If accuracy drops ≥15% from a stable value mid-run:
1. **Immediately check the JSONL** — bad cases are written in real-time:
   ```bash
   tail -20 results/{model}/{bench}_{mode}.errors.jsonl | python3 -c "
   import sys,json
   for l in sys.stdin:
       d=json.loads(l)
       print(f'[{d[\"subject\"]}] pred={d[\"pred\"]} ref={d[\"ref\"]}  {repr(d[\"raw\"][:60])}')"
   ```
2. Common causes: audio/image file missing → `FileNotFoundError`; format regression (pred='X' or ''); API change returning different response structure.
3. If format errors cluster → stop the run, diagnose prompt/extraction issue before wasting API budget.

### Step 5: Sanity-check JSONL after every completed run

Before committing results, always spot-check the errors JSONL:
```bash
python3 -c "
import json
from collections import defaultdict
from pathlib import Path
import sys

path = sys.argv[1]  # e.g. results/gemini-3.1-flash/cmi_bench_lite.errors.jsonl
lines = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
print(f'Total errors: {len(lines)}')

# Per-subject breakdown
by_subj = defaultdict(int)
for d in lines:
    by_subj[d.get('subject','?')] += 1
for k,v in sorted(by_subj.items()):
    print(f'  {k}: {v} errors')

# Sample 5 random errors
import random; random.seed(42)
print()
for d in random.sample(lines, min(5, len(lines))):
    print(f'[{d.get(\"subject\",\"?\")}] pred={d[\"pred\"]} ref={d[\"ref\"]}  raw={repr(d[\"raw\"][:60])}')
" results/{model}/{bench}_{mode}.errors.jsonl
```

What to look for:
- **All errors in one subject** → dataset/audio quality issue for that task
- **pred=X or pred=''** → format errors; check system prompt compatibility
- **pred reasonable but wrong** → genuine model difficulty (expected)
- **ref='' or ref='X'** → answer extraction bug in benchmark loader

### Resume after interruption
Just re-run the same command. `run.py` auto-detects `.checkpoint_*.json` and resumes from where it stopped — no duplicate API calls.

### View past results
```bash
ls -lt results/*.json | grep -v partial | grep -v checkpoint
```

---

## Multimodal Benchmarks (VLM / Audio-LM)

### Download required assets first

**WildScore** (score images):
```bash
python scripts/download_wildscore.py
# → data/wildscore/images/ (gitignored)
```

**CMI-Bench** (audio):
```bash
python scripts/download_cmibench.py
# → data/cmibench/ (gitignored, several GB)
```

**MuChoMusic** (audio inline): no download needed — HF streams audio automatically.

### Run multimodal benchmarks
```bash
# All multimodal (image + audio)
python run.py --model gemini-2.5-flash --benchmark multimodal

# Single benchmark
python run.py --model gemini-2.5-flash --benchmark wildscore

# Text + multimodal together
python run.py --model gemini-2.5-flash --benchmark all --multimodal
```

> Note: Multimodal benchmarks require a VLM/ALM (e.g. Gemini 1.5+, GPT-4V, Claude 3+).
> Text-only models (DeepSeek, Llama) will receive the prompt but no media — results will be meaningless.

---

## Benchmark Status

| Benchmark | Status | Modality | Dataset |
|-----------|--------|----------|---------|
| `music_theory_bench` | ✅ Ready | text | HF: m-a-p/MusicTheoryBench |
| `ziqi_eval` | ✅ Ready | text | HF: MYTH-Lab/ZIQI-Eval (default: 500-sample) |
| `wildscore` | ✅ Ready | image | HF: GM77/WildScore — run download script first |
| `muchomusic` | ✅ Ready | audio | HF: lmms-lab/muchomusic (inline, ~862 MB) |
| `cmi_bench` | ✅ Ready | audio | HF: nicolaus625/CMI-bench — selective shard download (~3GB), run download script first |
| `ssmr_bench` | ✅ Ready | abc | HF: Sylence/SSMR-Bench — 1600q ABC MCQ, 9 task types, lite=200q |
| `abc_eval` | ❌ Unreleased | abc | Anonymous review link expired, no public repo |
| `muse_bench` | 📋 Ref only | midi-text | arXiv:2026 Carone, ~84q, custom MIDI integer-list schema — too small for leaderboard |
| `musicxml_study` | 📋 Ref only | musicxml-text | arXiv:2503.22853, RCM exam questions not publicly released |

---

## Pricing Verification

`run.py` `MODEL_PRICING` is the cost estimate source. **Always cross-check against the LiteLLM pricing DB before adding a new model or reporting cost.**

### Canonical verification method

```python
# Pull the live LiteLLM pricing JSON (no install needed)
curl -s https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json \
  | python3 -c "
import json,sys
d=json.load(sys.stdin)
for key in ['gemini-3.1-pro-preview','claude-opus-4-6','claude-sonnet-4-6']:
    v=d.get(key,{})
    print(key, f\"in=\${v.get('input_cost_per_token',0)*1e6:.3f} out=\${v.get('output_cost_per_token',0)*1e6:.3f} /MTok\")
"
```

Or via the installed package (if version is recent enough):
```python
import litellm; cm = litellm.get_model_cost_map()
# keys are bare model IDs without provider prefix, e.g. "claude-opus-4-6"
```

### Key pricing facts (verified 2026-04)

| Model | Input $/MTok | Output $/MTok | Notes |
|-------|-------------|---------------|-------|
| gemini-3.1-flash-lite-preview | $0.25 | $1.50 | |
| gemini-3-flash-preview | $0.50 | $3.00 | thinking model — output rate applies to thinking tokens |
| gemini-3.1-pro-preview | $2.00 | $12.00 | thinking model — thinking tokens billed at output rate |
| claude-haiku-4-5 | $1.00 | $5.00 | |
| claude-sonnet-4-6 | $3.00 | $15.00 | |
| claude-opus-4-6 | $5.00 | $25.00 | **Not** $15/$75 — Opus 4.5/4.6 are cheaper than Opus 4.0/4.1 |
| gpt-5.4 | $2.50 | $15.00 | reasoning model |

### Thinking token cost impact

For **always-thinking models** (Gemini 3 series, deepseek-reasoner, o3), the `--estimate` output is a **lower bound** — it only counts text output tokens (3 tokens avg). Actual cost includes thinking tokens billed at the output rate:

```
real_cost ≈ estimated_cost + n_questions × avg_thinking_tokens × output_$/MTok / 1e6
```

Observed thinking tokens per question (lite benchmarks):
- `gemini-3-flash-preview` default: ~200–250 tok/q → +$0.5–0.6 per 872q run
- `gemini-3.1-pro-preview` (estimate): ~400–800 tok/q → +$4–8 per 872q run
- `claude-sonnet-4-6-xt8k` (budget=8000): actual usage varies, check `cell.usage.avg_thinking_tokens_per_q`

`run.py` captures actual token usage in `cell.usage` after each run — use that for accurate post-hoc cost analysis.

---

## Cost Reference (gemini-3.1-flash-lite baseline)

| Benchmark | Questions | Est. Cost | Est. Time |
|-----------|-----------|-----------|-----------|
| music_theory_bench | 372 | ~$0.006 | ~3 min |
| ziqi_eval (sampled) | 500 | ~$0.008 | ~4 min |
| muchomusic (sampled) | 200 | ~$0.003 | ~2 min |
| wildscore | varies | ~$0.003 | ~2 min |
| **Text all** | **872** | **~$0.014** | **~7 min** |

---

## Results

Results are stored per `(model, benchmark, mode)` cell:
```
results/{model}/{benchmark}_lite.json           ← accuracy + config + task_accuracy + error_counts
results/{model}/{benchmark}_lite.errors.jsonl   ← all wrong items, real-time streaming during run
results/{model}/{benchmark}_full.json
results/{model}/{benchmark}_full.errors.jsonl
```

Both files go into git. The JSONL is written incrementally during the run (real-time) and finalized at completion. It contains one JSON line per wrong item: `idx, subject, stem, pred, ref, score, raw, stop_reason, format_error`.

Generate leaderboard:
```bash
python scripts/gen_leaderboard.py
```

### Current leaderboard (lite mode)

| Model | MusicTheory | ZIQI-Eval | WildScore (image) | MuChoMusic (audio) |
|-------|-------------|-----------|-------------------|---------------------|
| gemini-3.1-flash-lite | 66.8% (367q) | 81.0% (500q) | 73.0% (100q) | 71.0% (200q) |

### Contributing results

Run any subset of cells and open a PR:
```bash
python run.py --model your-model --benchmark all --multimodal
# commit results/your-model/*.json
```
Existing cells are skipped automatically. Use `--force` to rerun.
