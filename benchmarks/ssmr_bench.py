"""
SSMR-Bench — Synthetic Sheet Music Reasoning Benchmark
Source: "Towards an AI Musician: Synthesizing Sheet Music Problems for Musical Reasoning"
        arXiv:2509.04059 (2025)

Programmatically generated verifiable QA pairs across 9 templates:
  Rhythm, Chord, Interval, Scale
  Both textual and visual (image) modalities — this module covers TEXTUAL only.

Dataset size:
  - Eval: 1,600 textual QA pairs + 1,600 visual QA pairs
  - Train: 8,000 pairs per modality

Generation approach: rule-based synthesis from music theory rules →
  verifiable ground truth, no human annotation needed.

Estimated cost per run (textual, 1600q): ~$1.20 (GPT-4.1) / ~$0.20 (gpt-4.1-mini)
Estimated time: ~15 min (with concurrency)
Saturation risk: medium-low (rule-based, systematically covers theory gaps)
Weight in aggregate: 0.20

Dataset availability: paper mentions code/data released; check arXiv:2509.04059 for link.
HF dataset: TBC — search huggingface.co for "SSMR" or "sheet music reasoning"
"""

METADATA = {
    "name": "SSMR-Bench (textual)",
    "source": "arXiv:2509.04059",
    "hf_dataset": None,  # TBC — check paper for release URL
    "n_questions": 1600,
    "subsets": ["rhythm", "chord", "interval", "scale"],
    "format": "multiple_choice_4",
    "modality": "text",  # textual ABC/lilypond representation
    "cost_gpt41_usd": 1.20,
    "cost_gpt41mini_usd": 0.20,
    "weight": 0.20,
}


def load():
    """
    Attempt to load SSMR-Bench textual split from HuggingFace.
    Dataset slug TBC — update once authors publish.
    """
    try:
        from datasets import load_dataset
        # TODO: replace with actual HF slug once published
        ds = load_dataset("SSMR-Bench/ssmr-bench", split="test")
        return [item for item in ds if item.get("modality") == "text"]
    except Exception as e:
        raise NotImplementedError(
            f"SSMR-Bench dataset not yet available or slug unknown. "
            f"Check arXiv:2509.04059 for the release URL. Original error: {e}"
        )


def format_prompt(item: dict) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in item["options"].items())
    notation = f"\n\nSheet music notation:\n{item['notation']}" if item.get("notation") else ""
    return (
        f"Music theory question ({item.get('category', 'general')}):\n\n"
        f"{item['question']}{notation}\n\n"
        f"{opts}"
    )


def score(predictions: list[str], references: list[str]) -> dict:
    correct = sum(p.strip().upper() == r.strip().upper() for p, r in zip(predictions, references))
    return {"accuracy": correct / len(references), "n": len(references)}
