"""
WildScore — Benchmarking MLLMs in-the-Wild Symbolic Music Reasoning
Source: arXiv:2509.04744 — EMNLP 2025

Real-world music theory questions sourced from online communities,
paired with score images and LLM-generated MC answer options.

5 categories × 12 subcategories:
  Harmony, Rhythm, Form, Notation, Expression

INPUT: score images + question text
→ Requires a Vision-Language Model (VLM). Text-only LLMs cannot run this.

Unique feature: questions come from real community submissions
  (not synthetically generated) → higher ecological validity than MSU-Bench.

Dataset: huggingface.co — check arXiv:2509.04744 for slug.
Paper: aclanthology.org/2025.emnlp-main.853
"""

METADATA = {
    "name": "WildScore",
    "source": "arXiv:2509.04744",
    "hf_dataset": None,  # TBC — check paper
    "n_questions": None,  # not specified in available abstracts
    "subsets": ["harmony", "rhythm", "form", "notation", "expression"],
    "format": "multiple_choice_4",
    "modality": "image",  # score images — VLM required
    "requires_vlm": True,
    "weight": 0.0,  # excluded from text-only aggregate
}


def load():
    raise NotImplementedError(
        "WildScore requires score images and a VLM. "
        "Check arXiv:2509.04744 or ACL Anthology for dataset release. "
        "Text-only LLMs cannot run this benchmark."
    )


def score(predictions: list[str], references: list[str]) -> dict:
    correct = sum(p.strip().upper() == r.strip().upper() for p, r in zip(predictions, references))
    return {"accuracy": correct / len(references), "n": len(references)}
