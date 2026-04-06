# MuTheoryEval

**A hub for evaluating LLM music theory knowledge.** Aggregates existing benchmarks, runs unified scoring across models, and tracks results on a live leaderboard.

Not a new benchmark — a runner that brings existing ones together with a weighted aggregate score.

**Live leaderboard**: [thevertexlab.github.io/MuTheoryEval](https://thevertexlab.github.io/MuTheoryEval)

---

## Benchmarks

### ✅ Text

| Name | Questions (lite) | Coverage | Weight |
|------|-----------------|----------|--------|
| [MusicTheoryBench](https://huggingface.co/datasets/m-a-p/MusicTheoryBench) | 372 | Knowledge + reasoning, ABC notation | 0.35 |
| [ZIQI-Eval](https://huggingface.co/datasets/MYTH-Lab/ZIQI-Eval) | 500 sampled | 10 categories, 56 subcategories | 0.35 |

### ✅ ABC Notation (Symbolic)

| Name | Questions (lite) | Coverage | Weight |
|------|-----------------|----------|--------|
| [SSMR-Bench](https://huggingface.co/datasets/Sylence/SSMR-Bench) | 200 sampled | Rhythm, chord, interval, scale — 9 task types | 1.0 |

### ✅ Image (VLM)

| Name | Questions (lite) | Coverage |
|------|-----------------|----------|
| [WildScore](https://arxiv.org/abs/2509.04744) | 100 | Score image reasoning, 5 categories — requires VLM |

### ✅ Audio (ALM)

| Name | Questions (lite) | Coverage |
|------|-----------------|----------|
| [MuChoMusic](https://arxiv.org/abs/2408.01337) | 200 sampled | Music perception + theory across genres — audio inline from HF |
| [CMI-Bench](https://arxiv.org/abs/2506.12285) | 99 | 14 MIR tasks (classification, captioning, regression…) — download script required |

### 📋 Reference Only (not in leaderboard)

| Name | Reason |
|------|--------|
| [ABC-Eval](https://arxiv.org/abs/2509.23350) | Dataset unreleased — anonymous review link expired |
| [MuseBench](https://arxiv.org/abs/2601.11968) | Dataset not yet released — multi-modal (text + score image + audio) |
| [MuseBench](https://arxiv.org/abs/2504.07721) | ~84q custom MIDI integer-list schema — too small for leaderboard |
| MusicXML study | RCM exam questions not publicly released |

---

## Results

**Live leaderboard** (auto-updated): [thevertexlab.github.io/MuTheoryEval](https://thevertexlab.github.io/MuTheoryEval)

Scores shown in lite mode (fixed-seed reproducible samples). Text aggregate = weighted score across MusicTheoryBench + ZIQI-Eval. ABC, Image, Audio columns shown separately.

---

## Quickstart

```bash
git clone https://github.com/thevertexlab/MuTheoryEval
cd MuTheoryEval
pip install -r requirements.txt
cp .env.example .env  # fill in your API keys

# Estimate cost before running
python run.py --estimate --model gemini-3.1-flash-lite --benchmark all

# Run one model on one benchmark
python run.py --model gpt-5.4-mini --benchmark music_theory_bench

# Run all text benchmarks
python run.py --model claude-sonnet-4-6 --benchmark all

# Run all including multimodal (image + audio)
python run.py --model gemini-3.1-flash --benchmark all --multimodal

# List available models / benchmarks
python run.py --list-models
python run.py --list-benchmarks
```

Results are written to `results/{model}/{benchmark}_lite.json` and auto-skipped on re-run. Use `--force` to rerun a completed cell.

---

## Supported Models

| Model key | Provider | Notes |
|-----------|----------|-------|
| `gpt-5.4` | OpenAI | flagship |
| `gpt-5.4-mini` | OpenAI | strong + fast |
| `gpt-5.4-nano` | OpenAI | cheapest |
| `o3` | OpenAI | reasoning |
| `claude-opus-4-6` | Anthropic | |
| `claude-sonnet-4-6` | Anthropic | |
| `claude-sonnet-4-6-xt8k` | Anthropic | extended thinking, budget=8k |
| `claude-haiku-4-5` | Anthropic | text/abc/image only, no audio |
| `gemini-3.1-pro` | Google | thinking model |
| `gemini-3.1-flash` | Google | thinking model |
| `gemini-3.1-flash-minimal` | Google | same model, minimal thinking |
| `gemini-3.1-flash-lite` | Google | thinking model |
| `gemini-2.5-pro` | Google | last stable GA |
| `gemini-2.5-flash` | Google | last stable GA |
| `deepseek-chat` | DeepSeek | V3.2 |
| `deepseek-reasoner` | DeepSeek | R1 thinking |
| `glm-5` | ZhipuAI | |
| `glm-5-thinking` | ZhipuAI | thinking model |
| `glm-z1-flash` | ZhipuAI | thinking model |
| `qwen3-max-thinking` | DeepInfra | thinking model |
| `qwen3.5-omni-plus` | Alibaba | multimodal, free preview |
| `qwen3.5-omni-flash` | Alibaba | multimodal, free preview |
| `llama-4-maverick` | DeepInfra | |
| `deepseek-r1` | DeepInfra | |

---

## Contributing

- **New model results**: run locally, commit `results/{model}/*.json` and regenerated `docs/data.json`, open a PR
- **New benchmark**: add a module to `benchmarks/` following the existing pattern, register in `benchmarks/__init__.py`

```bash
# Regenerate leaderboard after adding results
python scripts/gen_leaderboard.py
```

---

## Related Work

- [ChatMusician](https://arxiv.org/abs/2402.16153) — LLaMA2 fine-tuned on ABC notation; source of MusicTheoryBench
- [ZIQI-Eval paper](https://arxiv.org/abs/2406.15885) — "Massive Music Evaluation Benchmark", 16 models tested
- [ABC-Eval paper](https://arxiv.org/abs/2509.23350) — symbolic music understanding via ABC notation
- [SSMR-Bench paper](https://arxiv.org/abs/2509.04059) — programmatic sheet music reasoning, 9 task types
- [MuChoMusic](https://arxiv.org/abs/2408.01337) — multimodal audio-language benchmark (ISMIR 2024)
- [WildScore](https://arxiv.org/abs/2509.04744) — score image reasoning benchmark (EMNLP 2025)
- [MSU-Bench](https://openreview.net/pdf/6e87af4a985e84aec4ab4dd71171b7d7f3f30279.pdf) — musical score understanding, 4-level hierarchy
- [CMI-Bench](https://arxiv.org/abs/2506.12285) — comprehensive music instruction following, 14 MIR tasks
- [MARBLE](https://github.com/a43992899/MARBLE-Benchmark) — music audio representation benchmark
- [Music Flamingo](https://arxiv.org/abs/2511.10289) — audio-LM with 300K chain-of-thought music theory examples
- [AudioBench](https://arxiv.org/abs/2406.16020) — general audio LLM benchmark; music subset = MuChoMusic
- [Chorra](https://github.com/thevertexlab/papper) — the read-only DAW + AI music analysis tool that motivated this eval
