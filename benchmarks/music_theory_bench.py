"""
MusicTheoryBench
Source: ChatMusician (ACL Findings 2024) — arXiv:2402.16153
Dataset: huggingface.co/datasets/m-a-p/MusicTheoryBench

372 multiple-choice questions (4-option):
  - 269 music knowledge (notes, chords, harmony, history, instrumentation)
  - 98  music reasoning (multi-step, ABC notation analysis)
  - 5   few-shot dev split

Estimated cost per run: ~$0.50 (GPT-4o) / ~$0.05 (GPT-3.5-turbo)
Estimated time: ~10 min (with concurrency)
Saturation risk: medium (best published: GPT-4 + CoT = 69.9% knowledge / 39.5% reasoning)
Weight in aggregate: 0.35
"""

METADATA = {
    "name": "MusicTheoryBench",
    "source": "arXiv:2402.16153",
    "hf_dataset": "m-a-p/MusicTheoryBench",
    "n_questions": 372,
    "subsets": ["knowledge", "reasoning"],
    "format": "multiple_choice_4",
    "has_abc_notation": True,
    "cost_gpt4o_usd": 0.50,
    "cost_gpt35_usd": 0.05,
    "weight": 0.35,
}


def load(split="test"):
    from datasets import load_dataset
    ds = load_dataset("m-a-p/MusicTheoryBench", split=split)
    return ds


def format_prompt(item):
    abc = f"\n\nABC notation:\n{item['abc_score']}" if item.get("abc_score") else ""
    opts = "\n".join(f"{k}. {v}" for k, v in item["options"].items())
    return (
        f"{item['instruction']}\n\n"
        f"{item['stem']}{abc}\n\n"
        f"{opts}"
    )


def score(predictions: list[str], references: list[str]) -> dict:
    correct = sum(p.strip().upper() == r.strip().upper() for p, r in zip(predictions, references))
    return {"accuracy": correct / len(references), "n": len(references)}
