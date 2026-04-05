"""
ZIQI-Eval (Massive Music Evaluation Benchmark)
Source: arXiv:2406.15885 — ACL Findings 2024
Dataset: huggingface.co/datasets/MYTH-Lab/ZIQI-Eval
GitHub: github.com/zcli-charlie/ZIQI-Eval

14,244 questions across 10 major categories / 56 subcategories:
  - Music performance, composition theory, world ethnic music
  - Pop music, Western music history, Chinese music history
  - Music theory fundamentals, ear training, instrumentation, music aesthetics

16 models evaluated in paper (best: GPT-4 at 63% comprehension F1)
Human baseline: music PhD students ~64.9%

Estimated cost per run: ~$8 (GPT-4o, full) / subsample 500q for ~$0.30
Estimated time: ~60 min full / ~5 min subsampled
Saturation risk: low (broad coverage, GPT-4 barely matches PhDs)
Weight in aggregate: 0.35
"""

METADATA = {
    "name": "ZIQI-Eval",
    "source": "arXiv:2406.15885",
    "hf_dataset": "MYTH-Lab/ZIQI-Eval",
    "n_questions": 14244,
    "subsets": ["comprehension", "generation"],
    "format": "multiple_choice_4",
    "cost_gpt4o_usd_full": 8.0,
    "cost_gpt4o_usd_sample500": 0.30,
    "default_sample": 500,
    "weight": 0.35,
}


def load(split="train", sample=500, seed=42):
    from datasets import load_dataset
    ds = load_dataset("MYTH-Lab/ZIQI-Eval", split=split)
    if sample and sample < len(ds):
        ds = ds.shuffle(seed=seed).select(range(sample))
    return ds


def score(predictions: list[str], references: list[str]) -> dict:
    correct = sum(p.strip().upper() == r.strip().upper() for p, r in zip(predictions, references))
    return {"accuracy": correct / len(references), "n": len(references)}
