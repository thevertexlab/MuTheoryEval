"""
CMI-Bench — A Comprehensive Benchmark for Evaluating Music Instruction-Following
Source: arXiv:2407.05830
Dataset: huggingface.co/datasets/nicolaus625/CMI-bench

Covers 16 MIR tasks across four categories:
  Classification: key detection (24 labels), singing technique (10), pitch (106 MIDI),
                  GTZAN genre (10)
  Multi-label:    MTG genre (87), emotion (56), instrument (44), top50tags (50)
  Regression:     tempo, loudness, spectral features
  Transcription:  melody, chord, beat

This implementation covers the classification tasks (single and multi-label)
using the MULTI_SELECT answer format with A-H lettered options.

Audio storage: selective shard download via scripts/download_cmibench.py
  → audio saved to data/cmibench/audio/testdata/  (gitignored)
  → metadata (JSONL) at data/cmibench/meta/

Modality: audio — requires an Audio-Language Model (e.g. Gemini 2.5+)

Weight in aggregate: 0.15 (audio-only; excluded from text-only weighted score)
"""

import json
import random
from pathlib import Path

METADATA = {
    "name":          "CMI-Bench",
    "source":        "arXiv:2407.05830",
    "hf_dataset":    "nicolaus625/CMI-bench",
    "n_questions":   None,   # determined at load time from JSONL
    "lite_n":        100,
    "lite_seed":     42,
    "answer_format": "MULTI_SELECT",
    "max_output_tokens": 64,
    "modality":      "audio",
    "requires_alm":  True,
    "data_dir":      "data/cmibench",
    "weight":        0.15,
}

_REPO_ROOT  = Path(__file__).parent.parent
_DATA_DIR   = _REPO_ROOT / "data" / "cmibench"
_META_DIR   = _DATA_DIR / "meta"      # cloned from GitHub
_AUDIO_DIR  = _DATA_DIR / "audio"     # extracted shards land here

# ── Task definitions ──────────────────────────────────────────────────────────
# Only classification tasks (single and multi-label) are included.
# Each entry: task subfolder name, max 8 options per item, task type.
_TASKS = {
    # single-label (Jaccard of singleton sets = exact match)
    "key_detection":      {"type": "single"},
    "singing_technique":  {"type": "single"},
    "pitch":              {"type": "single"},
    "GTZAN":              {"type": "single"},
    # multi-label
    "MTGgenre":           {"type": "multi"},
    "emotion":            {"type": "multi"},
    "instrument":         {"type": "multi"},
    "top50tags":          {"type": "multi"},
}

_MIME_MAP = {
    "mp3": "audio/mp3", "wav": "audio/wav",
    "flac": "audio/flac", "ogg": "audio/ogg", "m4a": "audio/mp4",
}

# ── JSONL loader ──────────────────────────────────────────────────────────────

def _load_jsonl(task: str) -> list[dict]:
    """Load items from data/cmibench/meta/data/<task>/test.jsonl (GitHub clone)."""
    for candidate in [
        _META_DIR / "data" / task / "test.jsonl",
        _META_DIR / "data" / task / "test.json",
    ]:
        if candidate.exists():
            items = []
            with open(candidate) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))
            return items
    return []


def _audio_path(item_audio_path: str) -> Path:
    """Map JSONL audio_path (data/X/...) → local filesystem path.

    JSONL uses:  data/<task>/<filename>
    Zip extracts to: data/cmibench/audio/testdata/<task>/<filename>
    """
    # Strip leading "data/" if present
    rel = item_audio_path
    if rel.startswith("data/"):
        rel = rel[len("data/"):]
    return _AUDIO_DIR / "testdata" / rel


def _build_options(item: dict, rng: random.Random) -> tuple[list[str], list[str]]:
    """Build up to 8 A-H options for an item; return (options_list, correct_letters).

    For single-label: correct answer + up to 7 distractors from the task pool.
    For multi-label:  all correct labels as options (up to 8), rest are distractors.
    """
    # Normalise answers field
    raw_answers = item.get("answers") or item.get("answer") or item.get("label") or []
    if isinstance(raw_answers, str):
        try:
            raw_answers = json.loads(raw_answers)
        except Exception:
            raw_answers = [raw_answers]
    if not isinstance(raw_answers, list):
        raw_answers = [raw_answers]
    correct_labels = [str(a).strip() for a in raw_answers if str(a).strip()]

    # Get candidate pool from choices / options field, or use correct as pool
    pool = item.get("choices") or item.get("options") or []
    if isinstance(pool, str):
        try:
            pool = json.loads(pool)
        except Exception:
            pool = [pool]
    pool = list({str(c).strip() for c in pool if str(c).strip()})

    # Ensure correct labels are in the pool
    for lbl in correct_labels:
        if lbl not in pool:
            pool.append(lbl)

    # Limit: pick at most 8 options including all correct ones
    n_slots = 8
    correct_set = set(correct_labels)
    distractors = [c for c in pool if c not in correct_set]
    rng.shuffle(distractors)
    # Ensure we have enough room for all correct labels
    n_correct = min(len(correct_labels), n_slots)
    n_dist = min(len(distractors), n_slots - n_correct)
    selected_correct = correct_labels[:n_correct]
    selected = selected_correct + distractors[:n_dist]
    rng.shuffle(selected)
    selected = selected[:n_slots]

    letters = "ABCDEFGH"
    letter_map = {label: letters[i] for i, label in enumerate(selected)}
    correct_letters = sorted(letter_map[lbl] for lbl in selected_correct if lbl in letter_map)

    return selected, correct_letters


# ── Public API ────────────────────────────────────────────────────────────────

def load(split="test", sample=100, seed=42) -> list[dict]:
    """Load CMI-Bench classification items.

    Requires audio downloaded via scripts/download_cmibench.py
    and metadata cloned at data/cmibench/meta/.
    """
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

    for task_name in _TASKS:
        raw_items = _load_jsonl(task_name)
        for raw in raw_items:
            audio_rel = raw.get("audio_path") or raw.get("audio") or ""
            if not audio_rel:
                continue
            audio_abs = _audio_path(audio_rel)
            if not audio_abs.exists():
                continue   # only include items whose audio was downloaded

            options, correct_letters = _build_options(raw, rng)
            if not options or not correct_letters:
                continue

            items.append({
                "task":            task_name,
                "audio_path":      str(audio_abs),
                "_options":        options,
                "_correct_letters": correct_letters,
                "_raw":            raw,
            })

    if not items:
        raise FileNotFoundError(
            f"No CMI-Bench items found under {_AUDIO_DIR}.\n"
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
    """Build the MCQ prompt with A-H option letters."""
    options = item["_options"]
    raw = item["_raw"]

    task_question = raw.get("instruction") or raw.get("question") or ""
    if not task_question:
        task_question = f"Identify the {item['task'].replace('_', ' ')} of this audio clip."

    opts_str = "\n".join(
        f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)
    )
    return f"{task_question}\n\n{opts_str}"


def get_answer(item: dict) -> str:
    """Return sorted comma-joined correct letters, e.g. 'A' or 'A,C'."""
    return ",".join(item["_correct_letters"])


def score(predictions: list[str], references: list[str]) -> dict:
    """Mean Jaccard similarity across all items."""
    from benchmarks.answer_formats import ANSWER_FORMATS
    _jaccard = ANSWER_FORMATS["MULTI_SELECT"]["compare"]
    total = sum(_jaccard(p, r) for p, r in zip(predictions, references))
    return {
        "accuracy":     total / len(references) if references else 0.0,
        "mean_jaccard": total / len(references) if references else 0.0,
        "n":            len(references),
    }
