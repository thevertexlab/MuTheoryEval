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

## Cost Reference (gemini-3.1-flash-lite baseline)

| Benchmark | Questions | Est. Cost | Est. Time |
|-----------|-----------|-----------|-----------|
| music_theory_bench | 372 | ~$0.006 | ~3 min |
| ziqi_eval (sampled) | 500 | ~$0.008 | ~4 min |
| muchomusic (sampled) | 200 | ~$0.003 | ~2 min |
| wildscore | varies | ~$0.003 | ~2 min |
| **Text all** | **872** | **~$0.014** | **~7 min** |

---

## Known Results

### Text benchmarks
| Model | MusicTheoryBench | ZIQI-Eval (500) | Weighted |
|-------|-----------------|-----------------|---------|
| gemini-3.1-flash-lite | 66.8% | 81.0% | 73.9% |

### Multimodal benchmarks
| Model | WildScore (100q, image) | MuChoMusic (200q, audio) |
|-------|------------------------|--------------------------|
| gemini-3.1-flash-lite | 73.0% | 71.0% |
