"""
MSU-Bench — Musical Score Understanding Benchmark
Source: arXiv:2511.20697 / OpenReview 2025

1,800 human-curated generative QA pairs across 4 hierarchical levels:
  Level 1 — Onset Information   (note timing, beat positions)
  Level 2 — Notation & Note     (pitch, duration, accidentals)
  Level 3 — Chord & Harmony     (chord identification, Roman numerals)
  Level 4 — Texture & Form      (texture, structure, section labels)

INPUT: score images (PNG/SVG of sheet music)
→ Requires a Vision-Language Model (VLM). Text-only LLMs cannot run this.

Key result (from paper):
  - No context:  ChatGPT + MEI = 52% (best)
  - With context: Claude + MEI  = 75% (best)

Estimated cost per run: ~$3–5 (GPT-4.1 with vision, 1800 images)
Weight in aggregate: not included in text-only aggregate score
             included in VLM aggregate score (future)

Dataset: check arXiv:2511.20697 for release status.
"""

METADATA = {
    "name": "MSU-Bench",
    "source": "arXiv:2511.20697",
    "hf_dataset": None,  # TBC
    "n_questions": 1800,
    "subsets": ["onset", "notation", "chord_harmony", "texture_form"],
    "format": "generative_qa",
    "modality": "image",  # score images — VLM required
    "requires_vlm": True,
    "cost_gpt41vision_usd": 4.0,
    "weight": 0.0,  # excluded from text-only aggregate
}


def load():
    raise NotImplementedError(
        "MSU-Bench requires score images and a VLM. "
        "Dataset release TBC — check arXiv:2511.20697. "
        "Text-only LLMs cannot run this benchmark."
    )


def score(predictions: list[str], references: list[str]) -> dict:
    correct = sum(p.strip().upper() == r.strip().upper() for p, r in zip(predictions, references))
    return {"accuracy": correct / len(references), "n": len(references)}
