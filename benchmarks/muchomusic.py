"""
MuChoMusic — Evaluating Music Understanding in Multimodal Audio-Language Models
Source: arXiv:2408.01337 — ISMIR 2024
Dataset: huggingface.co/datasets/lmms-lab/muchomusic
GitHub: github.com/mulab-mir/muchomusic

Multiple-choice QA with audio clips covering:
  Musical features, Emotion & mood, Contextual knowledge,
  Instrumentation, Music perception

443 questions (test set) — audio is inline in the HF dataset (~862 MB total)
No separate download needed; audio bytes extracted from Arrow table.

Modality: audio — requires an Audio-Language Model (e.g. Gemini 1.5+)
Text-only models cannot meaningfully answer these questions.

Dataset columns (actual):
  context  — Audio feature (192kHz), raw bytes extracted via Arrow table
  instruction — question text
  choices  — string like "(A) Reggae (B) Pop music (C) Latin rock (D) Ska"
  answer   — string like "(A) Reggae" (full option, not just letter)

Weight in aggregate: 0.15 (audio-only; excluded from text-only weighted score)
"""

METADATA = {
    "name": "MuChoMusic",
    "source": "arXiv:2408.01337",
    "hf_dataset": "lmms-lab/muchomusic",
    "n_questions": 443,
    "default_sample": 200,   # use a subset by default (~862 MB total)
    "subsets": ["musical_features", "emotion", "context", "instrumentation", "perception"],
    "format": "multiple_choice_4",
    "modality": "audio",
    "requires_alm": True,
    "weight": 0.15,
}


def load(split="test", sample=200, seed=42):
    """Load MuChoMusic. Returns a plain list[dict] with 'audio_bytes' pre-extracted.

    HF datasets >= 3.x requires torchcodec for Audio decoding (heavy PyTorch dep).
    We bypass it by reading raw bytes directly from the underlying Arrow table
    before any decode step, then returning a plain list of dicts.
    """
    import os
    from datasets import load_dataset
    token = os.environ.get("HF_TOKEN")
    ds = load_dataset("lmms-lab/muchomusic", split=split, token=token)
    if sample and sample < len(ds):
        ds = ds.shuffle(seed=seed).select(range(sample))

    # Read raw audio bytes from Arrow table — no torchcodec needed.
    # The audio column is named "context" (Audio feature at 192kHz).
    arrow_table = ds._data

    items = []
    for i in range(len(ds)):
        row = {}
        for col in arrow_table.column_names:
            if col == "context":
                val = arrow_table.column("context")[i].as_py()
                row["audio_bytes"] = val.get("bytes", b"") if isinstance(val, dict) else (val or b"")
            else:
                row[col] = arrow_table.column(col)[i].as_py()
        items.append(row)
    return items


def get_media(item: dict) -> list[dict]:
    """Return audio bytes as a MediaItem list for a dataset item.

    Expects 'audio_bytes' key (set by load()) containing raw encoded audio bytes.
    """
    raw = item.get("audio_bytes") or b""
    if not raw:
        raise ValueError("audio_bytes is empty — audio may not have loaded correctly")
    return [{"mime_type": "audio/mp3", "data": raw}]



def format_prompt(item: dict) -> str:
    # choices is already formatted: "(A) Reggae (B) Pop music (C) Latin rock (D) Ska"
    return f"{item['instruction']}\n\n{item['choices']}"


def get_answer(item: dict) -> str:
    """Extract letter A/B/C/D from answer string like '(A) Reggae'."""
    import re
    m = re.search(r'\(([A-D])\)', item["answer"])
    return m.group(1) if m else item["answer"].strip().upper()[:1]


def score(predictions: list[str], references: list[str]) -> dict:
    correct = sum(p.strip().upper() == r.strip().upper() for p, r in zip(predictions, references))
    return {"accuracy": correct / len(references), "n": len(references)}
