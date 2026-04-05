"""
ABC-Eval
Source: arXiv:2509.23350 (Jiahao Zhao et al., Kyoto University / RIKEN, 2025)

STATUS: UNRELEASED
  Anonymous review link (anonymous.4open.science/r/ABC-Eval-B622) has expired.
  No GitHub or HuggingFace release found as of 2026-04.
  Re-check: https://arxiv.org/abs/2509.23350 for future official release.

1,086 test samples across 10 sub-tasks:
  - Basic musical syntax comprehension
  - Complex sequence-level reasoning
  - Instruction following on ABC notation

Key finding: 7 SOTA LLMs all show "notable limitations" in symbolic music processing
Weight in aggregate: 0.30 (pending release)
"""

STATUS = "UNRELEASED"

METADATA = {
    "name": "ABC-Eval",
    "source": "arXiv:2509.23350",
    "hf_dataset": None,
    "status": STATUS,
    "n_questions": 1086,
    "subsets": ["syntax", "reasoning", "instruction_following"],
    "format": "mixed",
    "cost_gpt4o_usd": 0.40,
    "weight": 0.30,
}


def load():
    raise NotImplementedError(
        "ABC-Eval is UNRELEASED — anonymous review link expired, no public repo as of 2026-04. "
        "Check arXiv:2509.23350 for future release."
    )


def score(predictions: list[str], references: list[str]) -> dict:
    correct = sum(p.strip().upper() == r.strip().upper() for p, r in zip(predictions, references))
    return {"accuracy": correct / len(references), "n": len(references)}
