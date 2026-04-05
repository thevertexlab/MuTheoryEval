"""
WildScore — Benchmarking MLLMs in-the-Wild Symbolic Music Reasoning
Source: arXiv:2509.04744 — EMNLP 2025
Dataset: huggingface.co/datasets/GM77/WildScore
Paper: aclanthology.org/2025.emnlp-main.853

Real-world music theory questions from online communities,
paired with music score images and MC answer options.

5 categories × 12 subcategories:
  Harmony, Rhythm, Form, Notation, Expression

Modality: image (score sheets) — requires VLM (e.g. Gemini, GPT-4V, Claude 3+)

Image download required before running:
  python scripts/download_wildscore.py
  → images saved to data/wildscore/images/ (gitignored)

Dataset columns: image (filename), question, final_options (JSON string),
                 truth_letter (correct answer)

Weight in aggregate: 0.15 (VLM-only; excluded from text-only weighted score)
"""

import json
from pathlib import Path

METADATA = {
    "name": "WildScore",
    "source": "arXiv:2509.04744",
    "hf_dataset": "GM77/WildScore",
    "n_questions": None,   # determined at load time
    "subsets": ["harmony", "rhythm", "form", "notation", "expression"],
    "format": "multiple_choice_4",
    "modality": "image",
    "requires_vlm": True,
    "data_dir": "data/wildscore",
    "weight": 0.15,
}

_DATA_DIR = Path(__file__).parent.parent / "data" / "wildscore"


def load(split="test", sample=None, seed=42):
    """Load WildScore. Raises FileNotFoundError if images not downloaded yet."""
    if not _DATA_DIR.exists():
        raise FileNotFoundError(
            f"WildScore image data not found at {_DATA_DIR}. "
            "Run: python scripts/download_wildscore.py"
        )

    from datasets import load_dataset
    ds = load_dataset("GM77/WildScore", split=split)

    if sample and sample < len(ds):
        ds = ds.shuffle(seed=seed).select(range(sample))
    return ds


def get_media(item: dict) -> list[dict]:
    """Return media list (image bytes) for a dataset item."""
    img_filename = item["image"]
    img_path = _DATA_DIR / "images" / img_filename
    if not img_path.exists():
        raise FileNotFoundError(
            f"Image not found: {img_path}. Run: python scripts/download_wildscore.py"
        )
    suffix = Path(img_filename).suffix.lower().lstrip(".")
    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "gif": "image/gif"}
    mime = mime_map.get(suffix, "image/jpeg")
    return [{"mime_type": mime, "data": img_path.read_bytes()}]


def format_prompt(item: dict) -> str:
    question = item["question"]
    try:
        options = json.loads(item["final_options"])
    except (json.JSONDecodeError, TypeError):
        options = item["final_options"]

    if isinstance(options, dict):
        opts = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    elif isinstance(options, list):
        letters = "ABCD"
        opts = "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))
    else:
        opts = str(options)

    return f"{question}\n\n{opts}"


def score(predictions: list[str], references: list[str]) -> dict:
    correct = sum(p.strip().upper() == r.strip().upper() for p, r in zip(predictions, references))
    return {"accuracy": correct / len(references), "n": len(references)}
