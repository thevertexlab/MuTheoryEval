"""
CMI-Bench — A Comprehensive Benchmark for Evaluating Music Instruction-Following
Source: arXiv:2407.05830
Dataset: huggingface.co/datasets/nicolaus625/CMI-bench

Audio-based instruction-following evaluation across music tasks:
  Transcription, melody extraction, chord recognition, beat tracking,
  instrument classification, music captioning, QA, and more.

Audio is distributed as zip files on HuggingFace — download required:
  python scripts/download_cmibench.py
  → audio saved to data/cmibench/ (gitignored)

After download the dataset metadata (JSON) is loaded from the HF Hub,
while audio bytes are read from local data/cmibench/.

Modality: audio — requires an Audio-Language Model (e.g. Gemini 1.5+)

Weight in aggregate: 0.15 (audio-only; excluded from text-only weighted score)
"""

import json
from pathlib import Path

METADATA = {
    "name": "CMI-Bench",
    "source": "arXiv:2407.05830",
    "hf_dataset": "nicolaus625/CMI-bench",
    "n_questions": None,   # determined at load time
    "format": "open_ended_mc",
    "modality": "audio",
    "requires_alm": True,
    "data_dir": "data/cmibench",
    "weight": 0.15,
}

_DATA_DIR = Path(__file__).parent.parent / "data" / "cmibench"


def load(split="test", sample=None, seed=42):
    """Load CMI-Bench. Raises FileNotFoundError if audio not downloaded."""
    if not _DATA_DIR.exists():
        raise FileNotFoundError(
            f"CMI-Bench audio data not found at {_DATA_DIR}. "
            "Run: python scripts/download_cmibench.py"
        )
    from datasets import load_dataset
    ds = load_dataset("nicolaus625/CMI-bench", split=split)
    # Filter to items that have a corresponding local audio file
    ds = ds.filter(lambda item: (_DATA_DIR / item["audio_path"]).exists())
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
