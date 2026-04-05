"""
CMI-Bench — A Comprehensive Benchmark for Evaluating Music Instruction-Following
Source: arXiv:2407.05830
Dataset: huggingface.co/datasets/nicolaus625/CMI-bench

Covers 16 MIR tasks.  This implementation targets 4 classification tasks
with small fixed label pools, formatted as MULTI_SELECT (A-H lettered MCQ):

  Task         JSONL                         Labels  Test
  GTZAN        GTZAN/CMI_GTZAN.jsonl            10    290
  GS-key       GS-key/CMI_GS_key.jsonl          24   2406
  NSynth       NSynth/CMI_Nsynth_instrument.jsonl 10  4096
  VocalSet     VocalSet/CMI_VocalSet_tech.jsonl  10  1140

Audio path mapping:
  JSONL  audio_path[0]: "data/GTZAN/Data/..."
  ZIP    entry path:    "testdata/GTZAN/Data/..."
  Local: data/cmibench/audio/testdata/GTZAN/Data/...

Download audio:  python scripts/download_cmibench.py
Metadata clone:  python scripts/download_cmibench.py --meta-only

Modality: audio — requires an Audio-Language Model (e.g. Gemini 2.5+)
"""

import json
import random
from pathlib import Path

METADATA = {
    "name":              "CMI-Bench",
    "source":            "arXiv:2407.05830",
    "hf_dataset":        "nicolaus625/CMI-bench",
    "n_questions":       None,
    "lite_n":            100,
    "lite_seed":         42,
    "answer_format":     "MULTI_SELECT",
    "max_output_tokens": 16,   # single letter per item; allow short answer
    "modality":          "audio",
    "requires_alm":      True,
    "data_dir":          "data/cmibench",
    "weight":            0.15,
}

_REPO_ROOT = Path(__file__).parent.parent
_DATA_DIR  = _REPO_ROOT / "data" / "cmibench"
_META_DIR  = _DATA_DIR / "meta"
_AUDIO_DIR = _DATA_DIR / "audio"

# ── Task definitions ──────────────────────────────────────────────────────────

_TASK_CONFIGS = {
    "GTZAN": {
        "jsonl":    "GTZAN/CMI_GTZAN.jsonl",
        "labels":   ["blues", "classical", "country", "disco", "hiphop",
                     "jazz", "metal", "pop", "reggae", "rock"],
        "n_sample": 25,
        "question": "What is the genre of this music?",
    },
    "GS-key": {
        "jsonl":    "GS-key/CMI_GS_key.jsonl",
        "labels":   ["A major", "A minor", "Ab major", "Ab minor",
                     "B major", "B minor", "Bb major", "Bb minor",
                     "C major", "C minor", "D major", "D minor",
                     "Db major", "Db minor", "E major", "E minor",
                     "Eb major", "Eb minor", "F major", "F minor",
                     "G major", "G minor", "Gb major", "Gb minor"],
        "n_sample": 25,
        "question": "What is the musical key of this audio clip?",
    },
    "NSynth": {
        "jsonl":    "NSynth/CMI_Nsynth_instrument.jsonl",
        "labels":   ["bass", "brass", "flute", "guitar", "keyboard",
                     "mallet", "organ", "reed", "string", "vocal"],
        "n_sample": 25,
        "question": "What instrument family is playing in this audio?",
    },
    "VocalSet": {
        "jsonl":    "VocalSet/CMI_VocalSet_tech.jsonl",
        "labels":   ["belt", "breathy", "inhaled", "lip_trill", "spoken",
                     "straight", "trill", "trillo", "vibrato", "vocal_fry"],
        "n_sample": 25,
        "question": "What singing technique is being demonstrated in this audio?",
    },
}

_MIME_MAP = {
    "mp3": "audio/mp3", "wav": "audio/wav",
    "flac": "audio/flac", "ogg": "audio/ogg", "m4a": "audio/mp4",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _audio_local(audio_path_field) -> Path:
    """Resolve JSONL audio_path (list or str, prefixed 'data/') to local path."""
    rel = audio_path_field[0] if isinstance(audio_path_field, list) else audio_path_field
    if rel.startswith("data/"):
        rel = rel[len("data/"):]
    return _AUDIO_DIR / "testdata" / rel


def _audio_zip_path(audio_path_field) -> str:
    """Return the path as stored in the HF zip (testdata/ prefix)."""
    rel = audio_path_field[0] if isinstance(audio_path_field, list) else audio_path_field
    if rel.startswith("data/"):
        rel = rel[len("data/"):]
    return f"testdata/{rel}"


def _load_test_items(task_key: str) -> list[dict]:
    jsonl_path = _META_DIR / "data" / _TASK_CONFIGS[task_key]["jsonl"]
    if not jsonl_path.exists():
        return []
    items = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "test" in d.get("split", []):
                items.append(d)
    return items


def _build_mcq(item: dict, task_key: str, rng: random.Random) -> tuple[list[str], str]:
    """Return (options_list, correct_letter).

    Picks the correct label + up to 7 distractors, shuffled, → A-H.
    """
    correct = item["output"].strip() if isinstance(item["output"], str) else str(item["output"]).strip()
    pool = list(_TASK_CONFIGS[task_key]["labels"])

    distractors = [l for l in pool if l != correct]
    rng.shuffle(distractors)
    options = [correct] + distractors[:7]
    rng.shuffle(options)
    options = options[:8]

    letter = chr(65 + options.index(correct))
    return options, letter


# ── Public API ────────────────────────────────────────────────────────────────

def load(split="test", sample=100, seed=42) -> list[dict]:
    if not _META_DIR.exists():
        raise FileNotFoundError(
            f"CMI-Bench metadata not found at {_META_DIR}.\n"
            "Run: python scripts/download_cmibench.py --meta-only"
        )
    if not _AUDIO_DIR.exists():
        raise FileNotFoundError(
            f"CMI-Bench audio not found at {_AUDIO_DIR}.\n"
            "Run: python scripts/download_cmibench.py"
        )

    rng = random.Random(seed)
    items: list[dict] = []

    for task_key, cfg in _TASK_CONFIGS.items():
        raw_items = _load_test_items(task_key)
        rng.shuffle(raw_items)

        added = 0
        for raw in raw_items:
            if added >= cfg["n_sample"]:
                break
            audio_field = raw.get("audio_path")
            if not audio_field:
                continue
            audio_abs = _audio_local(audio_field)
            if not audio_abs.exists():
                continue

            options, correct_letter = _build_mcq(raw, task_key, rng)
            items.append({
                "task":           task_key,
                "audio_path":     str(audio_abs),
                "_options":       options,
                "_correct_letter": correct_letter,
                "_question":      cfg["question"],
            })
            added += 1

    if not items:
        raise FileNotFoundError(
            f"No CMI-Bench audio found under {_AUDIO_DIR}.\n"
            "Run: python scripts/download_cmibench.py"
        )

    if sample and sample < len(items):
        rng.shuffle(items)
        items = items[:sample]

    return items


def get_media(item: dict) -> list[dict]:
    audio_path = Path(item["audio_path"])
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    suffix = audio_path.suffix.lower().lstrip(".")
    mime = _MIME_MAP.get(suffix, "audio/mp3")
    return [{"mime_type": mime, "data": audio_path.read_bytes()}]


def format_prompt(item: dict) -> str:
    opts_str = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(item["_options"]))
    return f"{item['_question']}\n\n{opts_str}"


def get_answer(item: dict) -> str:
    return item["_correct_letter"]


def score(predictions: list[str], references: list[str]) -> dict:
    """Mean Jaccard (= exact match for single-letter answers)."""
    from benchmarks.answer_formats import ANSWER_FORMATS
    _jaccard = ANSWER_FORMATS["MULTI_SELECT"]["compare"]
    total = sum(_jaccard(p, r) for p, r in zip(predictions, references))
    n = len(references)
    return {
        "accuracy":     total / n if n else 0.0,
        "mean_jaccard": total / n if n else 0.0,
        "n":            n,
    }
