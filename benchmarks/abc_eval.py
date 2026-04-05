"""
ABC-Eval
Source: arXiv:2509.23350 (2025)
Benchmarks LLM understanding of ABC notation (text-encoded symbolic music)

1,086 test samples across 10 sub-tasks:
  - Basic musical syntax comprehension
  - Complex sequence-level reasoning
  - Instruction following on ABC notation

Key finding: 7 SOTA LLMs all show "notable limitations" in symbolic music processing

Estimated cost per run: ~$0.40 (GPT-4o)
Estimated time: ~8 min
Saturation risk: low (even SOTA models struggle significantly)
Weight in aggregate: 0.30
"""

METADATA = {
    "name": "ABC-Eval",
    "source": "arXiv:2509.23350",
    "hf_dataset": None,  # check paper for dataset release status
    "n_questions": 1086,
    "subsets": ["syntax", "reasoning", "instruction_following"],
    "format": "mixed",
    "cost_gpt4o_usd": 0.40,
    "weight": 0.30,
}


def load():
    raise NotImplementedError(
        "ABC-Eval dataset release status unclear — check arXiv:2509.23350 for data access."
    )


def score(predictions: list[str], references: list[str]) -> dict:
    correct = sum(p.strip().upper() == r.strip().upper() for p, r in zip(predictions, references))
    return {"accuracy": correct / len(references), "n": len(references)}
