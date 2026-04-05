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
| `cmi_bench` | ⚠️ 55GB download | audio | HF: nicolaus625/CMI-bench — 551×100MB zip parts, skip unless needed |
| `abc_eval` | ❌ Unreleased | text | Anonymous review link expired, no public repo |
| `ssmr_bench` | ❌ Unreleased | text | Anonymous review link expired, no public repo |

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
results/{model}/{benchmark}_lite.json
results/{model}/{benchmark}_full.json
```

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
