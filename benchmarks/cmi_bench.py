"""
CMI-Bench — A Comprehensive Benchmark for Evaluating Music Instruction-Following
Source: arXiv:2407.05830
Dataset: huggingface.co/datasets/nicolaus625/CMI-bench

Audio-based instruction-following evaluation across music tasks:
  Transcription, melody extraction, chord recognition, beat tracking,
  instrument classification, music captioning, QA, and more.

Audio storage: HF dataset uses multi-part zip files (Python zipfile can't read them).
Download requires the system `unzip` command:
  python scripts/download_cmibench.py
  → audio saved to data/cmibench/ (gitignored)

Note: CMI-Bench audio may also be accessible inline via the HF Audio feature
if the dataset is re-encoded as parquet. Check dataset card for updates.

Modality: audio — requires an Audio-Language Model (e.g. Gemini 1.5+)

Weight in aggregate: 0.15 (audio-only; excluded from text-only weighted score)
"""

import json
import os
from pathlib import Path

METADATA = {
    "name": "CMI-Bench",
    "source": "arXiv:2407.05830",
    "hf_dataset": "nicolaus625/CMI-bench",
    "n_questions": None,   # unknown without 55GB download
    "lite_n": 50,
    "lite_seed": 42,
    "format": "open_ended_mc",
    "modality": "audio",
    "requires_alm": True,
    "data_dir": "data/cmibench",
    "weight": 0.15,
}

_DATA_DIR = Path(__file__).parent.parent / "data" / "cmibench"


def load(split="test", sample=50, seed=42):
    """Load CMI-Bench. Requires audio downloaded via scripts/download_cmibench.py."""
    if not _DATA_DIR.exists():
        raise FileNotFoundError(
            f"CMI-Bench audio not found at {_DATA_DIR}.\n"
            "Audio is stored as multi-part zips on HF — download with:\n"
            "  python scripts/download_cmibench.py\n"
            "(requires system `unzip` command)"
        )
    from datasets import load_dataset
    token = os.environ.get("HF_TOKEN")
    ds = load_dataset("nicolaus625/CMI-bench", split=split, token=token)
    # Keep only items whose audio file was successfully extracted
    ds = ds.filter(lambda item: (_DATA_DIR / item["audio_path"]).exists())
    if len(ds) == 0:
        raise FileNotFoundError(
            f"No audio files found under {_DATA_DIR}. "
            "Run: python scripts/download_cmibench.py"
        )
    if sample and sample < len(ds):
        ds = ds.shuffle(seed=seed).select(range(sample))
    return ds


def get_media(item: dict) -> list[dict]:
    audio_path = _DATA_DIR / item["audio_path"]
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    suffix = audio_path.suffix.lower().lstrip(".")
    mime_map = {"mp3": "audio/mp3", "wav": "audio/wav", "flac": "audio/flac", "ogg": "audio/ogg"}
    mime = mime_map.get(suffix, "audio/mp3")
    return [{"mime_type": mime, "data": audio_path.read_bytes()}]


def format_prompt(item: dict) -> str:
    prompt = item.get("instruction") or item.get("question") or ""
    choices = item.get("choices") or item.get("options")
    if choices:
        letters = "ABCD"
        if isinstance(choices, str):
            try:
                choices = json.loads(choices)
            except json.JSONDecodeError:
                pass
        if isinstance(choices, list):
            opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
            prompt = f"{prompt}\n\n{opts}"
    return prompt


def score(predictions: list[str], references: list[str]) -> dict:
    correct = sum(p.strip().upper() == r.strip().upper() for p, r in zip(predictions, references))
    return {"accuracy": correct / len(references), "n": len(references)}
