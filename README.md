# MuTheoryEval

**A hub for evaluating LLM music theory knowledge.** Aggregates existing benchmarks, runs unified scoring across models, and tracks results.

Not a new benchmark — a runner that brings existing ones together with a weighted aggregate score.

---

## Benchmarks

### ✅ Implemented

| Name | Questions | Coverage | Weight | Est. Cost (GPT-4o) |
|------|-----------|----------|--------|---------------------|
| [MusicTheoryBench](https://huggingface.co/datasets/m-a-p/MusicTheoryBench) | 372 | Knowledge + reasoning, ABC notation | 0.35 | ~$0.50 |
| [ZIQI-Eval](https://huggingface.co/datasets/MYTH-Lab/ZIQI-Eval) | 14,244 (500 sampled) | 10 categories, 56 subcategories | 0.35 | ~$0.30 |
| [ABC-Eval](https://arxiv.org/abs/2509.23350) | 1,086 | Symbolic music (ABC notation) understanding | 0.30 | ~$0.40 |

**Aggregate score** = weighted average across available benchmarks.

### 🔲 TBD — Text / Symbolic (no audio required)

| Name | Source | Questions | Coverage | Notes |
|------|--------|-----------|----------|-------|
| [SSMR-Bench](https://arxiv.org/abs/2509.04059) | arXiv 2025 | programmatic | Rhythm, chord, interval, scale — 9 templates, verifiable answers | Dataset release TBC |
| [MSU-Bench](https://openreview.net/pdf/6e87af4a985e84aec4ab4dd71171b7d7f3f30279.pdf) | OpenReview 2025 | 1,800 | 4-level: onset → notation → chord/harmony → texture/form | Score image input (VLM needed) |
| [WildScore](https://arxiv.org/abs/2509.04744) | EMNLP 2025 | — | 5 categories, 12 subcategories; real community questions | Score image input (VLM needed) |

### 🔲 TBD — Multimodal / Audio (requires audio input)

| Name | Source | Questions | Coverage | Notes |
|------|--------|-----------|----------|-------|
| [MuChoMusic](https://arxiv.org/abs/2408.01337) | ISMIR 2024 | 1,187 | Music perception + theory across genres | Requires audio playback; open-sourced |
| [CMI-Bench](https://arxiv.org/abs/2506.12285) | 2025 | — | 14 MIR tasks, 20 datasets (classification, captioning, regression…) | Audio-text LLMs only |
| [MARBLE](https://github.com/a43992899/MARBLE-Benchmark) | 2023 | — | Broad music representation evaluation | Audio model evaluation framework |

---

## Results

> Last updated: —  
> PRs with new model results welcome.

| Model | MusicTheoryBench | ZIQI-Eval (500) | Aggregate |
|-------|-----------------|-----------------|-----------|
| *Run `python run.py --model all --benchmark all` and submit a PR* | — | — | — |

---

## Quickstart

```bash
git clone https://github.com/thevertexlab/MuTheoryEval
cd MuTheoryEval
pip install -r requirements.txt
cp .env.example .env  # fill in your API keys

# Run one model on one benchmark
python run.py --model gpt-4o-mini --benchmark music_theory_bench

# Run multiple models
python run.py --model gpt-4o,claude-3-5-sonnet --benchmark music_theory_bench

# Run everything
python run.py --model all --benchmark all

# List available models / benchmarks
python run.py --list-models
python run.py --list-benchmarks
```

---

## Supported Models

| Model | Provider |
|-------|----------|
| gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo | OpenAI |
| claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus | Anthropic |
| gemini-2.0-flash, gemini-1.5-pro | Google |
| deepseek-chat, deepseek-reasoner | DeepSeek |
| llama-3.3-70b, qwen2.5-72b, mixtral-8x7b | DeepInfra |

---

## Contributing

- **New model results**: run locally, add JSON to `results/`, update the table in README, open a PR
- **New benchmark**: add a module to `benchmarks/` following the existing pattern, register in `benchmarks/__init__.py`

---

## Related Work

- [ChatMusician](https://arxiv.org/abs/2402.16153) — LLaMA2 fine-tuned on ABC notation; source of MusicTheoryBench
- [ZIQI-Eval paper](https://arxiv.org/abs/2406.15885) — "Massive Music Evaluation Benchmark", 16 models tested
- [ABC-Eval paper](https://arxiv.org/abs/2509.23350) — symbolic music understanding via ABC notation
- [MuChoMusic](https://arxiv.org/abs/2408.01337) — multimodal audio-language benchmark (ISMIR 2024)
- [WildScore](https://arxiv.org/abs/2509.04744) — score image reasoning benchmark (EMNLP 2025)
- [MSU-Bench](https://openreview.net/pdf/6e87af4a985e84aec4ab4dd71171b7d7f3f30279.pdf) — musical score understanding, 4-level hierarchy
- [SSMR-Bench](https://arxiv.org/abs/2509.04059) — programmatic sheet music reasoning problems
- [CMI-Bench](https://arxiv.org/abs/2506.12285) — comprehensive music instruction following, 14 MIR tasks
- [MARBLE](https://github.com/a43992899/MARBLE-Benchmark) — music audio representation benchmark
- [Music Flamingo](https://arxiv.org/abs/2511.10289) — audio-LM with 300K chain-of-thought music theory examples
- [Chorra](https://github.com/thevertexlab/papper) — the read-only DAW + AI music analysis tool that motivated this eval
