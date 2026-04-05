# MuTheoryEval

**A hub for evaluating LLM music theory knowledge.** Aggregates existing benchmarks, runs unified scoring across models, and tracks results.

Not a new benchmark — a runner that brings existing ones together with a weighted aggregate score.

---

## Benchmarks

### ✅ Implemented — Text / Symbolic

| Name | Questions | Coverage | Weight | Est. Cost (GPT-4.1) |
|------|-----------|----------|--------|----------------------|
| [MusicTheoryBench](https://huggingface.co/datasets/m-a-p/MusicTheoryBench) | 372 | Knowledge + reasoning, ABC notation | 0.35 | ~$0.50 |
| [ZIQI-Eval](https://huggingface.co/datasets/MYTH-Lab/ZIQI-Eval) | 14,244 (500 sampled) | 10 categories, 56 subcategories | 0.35 | ~$0.30 |
| [ABC-Eval](https://arxiv.org/abs/2509.23350) | 1,086 | Symbolic music (ABC notation) understanding | 0.30 | ~$0.40 ⚠️ dataset TBC |
| [SSMR-Bench](https://arxiv.org/abs/2509.04059) | 1,600 (textual) | Rhythm, chord, interval, scale — 9 templates | 0.20 | ~$1.20 ⚠️ dataset TBC |

**Aggregate score** = weighted average across available benchmarks.

### 🔲 TBD — VLM Only (score image input)

| Name | Source | Questions | Coverage | Notes |
|------|--------|-----------|----------|-------|
| [MSU-Bench](https://arxiv.org/abs/2511.20697) | OpenReview 2025 | 1,800 | 4-level: onset → notation → chord/harmony → texture/form | Requires VLM; best result: Claude+MEI 75% |
| [WildScore](https://arxiv.org/abs/2509.04744) | EMNLP 2025 | — | 5 categories, 12 subcategories; real community questions | Requires VLM |

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

| Model key | API model ID | Provider |
|-----------|-------------|----------|
| `gpt-5.4` | `gpt-5.4` | OpenAI — flagship |
| `gpt-5.4-mini` | `gpt-5.4-mini` | OpenAI — strong + fast |
| `gpt-5.4-nano` | `gpt-5.4-nano` | OpenAI — cheapest |
| `o3` | `o3` | OpenAI — reasoning (not deprecated, but succeeded by gpt-5.4) |
| `claude-opus-4-6` | `claude-opus-4-6` | Anthropic |
| `claude-sonnet-4-6` | `claude-sonnet-4-6` | Anthropic |
| `claude-haiku-4-5` | `claude-haiku-4-5-20251001` | Anthropic |
| `gemini-3.1-pro` | `gemini-3.1-pro-preview` | Google |
| `gemini-3.1-flash` | `gemini-3-flash-preview` | Google |
| `gemini-3.1-flash-lite` | `gemini-3.1-flash-lite-preview` | Google |
| `gemini-2.5-pro` | `gemini-2.5-pro` | Google (last stable GA) |
| `gemini-2.5-flash` | `gemini-2.5-flash` | Google (last stable GA) |
| `deepseek-chat` | `deepseek-chat` (= V3.2) | DeepSeek |
| `deepseek-reasoner` | `deepseek-reasoner` (= V3.2 thinking) | DeepSeek |
| `llama-4-maverick` | `meta-llama/Llama-4-Maverick-17B-128E-Instruct` | DeepInfra |
| `qwen3.5-72b` | `Qwen/Qwen3.5-72B-A10B` | DeepInfra |
| `deepseek-r1` | `deepseek-ai/DeepSeek-R1` | DeepInfra |

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
