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

Lazy download: images are fetched on-demand from HF Hub and cached in
  data/wildscore/images/ (gitignored). No upfront full download needed.
  Requires HF_TOKEN in .env to avoid rate limits.

Dataset columns: image (filename), question, final_options (JSON string),
                 truth_letter (correct answer)

Weight in aggregate: 0.15 (VLM-only; excluded from text-only weighted score)
"""

import json
import os
from pathlib import Path

METADATA = {
    "name": "WildScore",
    "source": "arXiv:2509.04744",
    "hf_dataset": "GM77/WildScore",
    "n_questions": None,   # determined at load time
    "default_sample": 100,
    "subsets": ["harmony", "rhythm", "form", "notation", "expression"],
    "format": "multiple_choice_4",
    "modality": "image",
    "requires_vlm": True,
    "data_dir": "data/wildscore",
    "weight": 0.15,
}

_DATA_DIR = Path(__file__).parent.parent / "data" / "wildscore"
_HF_REPO = "GM77/WildScore"


def load(split="train", sample=100, seed=42):
    """Load WildScore metadata (CSV only — no images downloaded yet).

    Images are fetched lazily per-item in get_media().
    Requires HF_TOKEN in environment to avoid HF rate limits.
    """
    from datasets import load_dataset
    token = os.environ.get("HF_TOKEN")
    ds = load_dataset(_HF_REPO, "csv", split=split, token=token)
    if sample and sample < len(ds):
        ds = ds.shuffle(seed=seed).select(range(sample))
    return ds


def get_media(item: dict) -> list[dict]:
    """Lazily fetch image for this item from HF Hub; cache in data/wildscore/.

    The 'image' column contains a repo-relative path like 'images/abc06c.jpg'.
    We download it to data/wildscore/images/abc06c.jpg on first access.
    """
    hf_path = item["image"]          # e.g. "images/qo3i7b.jpg"
    img_path = _DATA_DIR / hf_path   # data/wildscore/images/qo3i7b.jpg

    if not img_path.exists():
        from huggingface_hub import hf_hub_download
        token = os.environ.get("HF_TOKEN")
        img_path.parent.mkdir(parents=True, exist_ok=True)
        cached = hf_hub_download(
            repo_id=_HF_REPO,
            filename=hf_path,          # exact repo path, no double-prefix
            repo_type="dataset",
            local_dir=str(_DATA_DIR),
            token=token,
        )
        img_path = Path(cached)

    suffix = img_path.suffix.lower().lstrip(".")
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


def get_answer(item: dict) -> str:
    return item["truth_letter"].strip().upper()


def score(predictions: list[str], references: list[str]) -> dict:
    correct = sum(p.strip().upper() == r.strip().upper() for p, r in zip(predictions, references))
    return {"accuracy": correct / len(references), "n": len(references)}
