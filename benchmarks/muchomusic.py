"""
MuChoMusic — Evaluating Music Understanding in Multimodal Audio-Language Models
Source: arXiv:2408.01337 — ISMIR 2024
Dataset: huggingface.co/datasets/lmms-lab/muchomusic
GitHub: github.com/mulab-mir/muchomusic

Multiple-choice QA with audio clips covering:
  Musical features, Emotion & mood, Contextual knowledge,
  Instrumentation, Music perception

443 questions (test set) — audio is inline in the HF dataset (~862 MB total)
No separate download needed; streamed via HuggingFace datasets.

Modality: audio — requires an Audio-Language Model (e.g. Gemini 1.5+)
Text-only models cannot meaningfully answer these questions.

Dataset columns: audio (dict with array/sampling_rate or bytes),
                 instruction, choices (list[str]), answer (int index)

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
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/muchomusic", split=split)
    if sample and sample < len(ds):
        ds = ds.shuffle(seed=seed).select(range(sample))
    return ds


def get_media(item: dict) -> list[dict]:
    """Return audio bytes as a MediaItem list for a dataset item."""
    audio = item["audio"]

    if isinstance(audio, bytes):
        return [{"mime_type": "audio/mp3", "data": audio}]

    if isinstance(audio, dict):
        # HF Audio feature: may have 'bytes' key (encoded) or 'array'+'sampling_rate' (decoded)
        if audio.get("bytes"):
            raw = audio["bytes"]
            return [{"mime_type": "audio/mp3", "data": raw}]

        if audio.get("array") is not None:
            # Encode raw float32 PCM to WAV in-memory
            import io
            import struct
            import numpy as np
            arr = np.array(audio["array"], dtype=np.float32)
            sr = int(audio.get("sampling_rate", 22050))
            buf = io.BytesIO()
            _write_wav(buf, arr, sr)
            return [{"mime_type": "audio/wav", "data": buf.getvalue()}]

    raise ValueError(f"Unrecognized audio format: {type(audio)}")


def _write_wav(buf, arr, sr: int):
    """Write float32 PCM array to WAV bytes (16-bit, mono)."""
    import struct
    pcm = (arr * 32767).clip(-32768, 32767).astype("<i2").tobytes()
    n_bytes = len(pcm)
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + n_bytes))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", n_bytes))
    buf.write(pcm)


def format_prompt(item: dict) -> str:
    choices = item["choices"]
    letters = "ABCD"
    opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    return f"{item['instruction']}\n\n{opts}"


def get_answer(item: dict) -> str:
    """Return correct letter (A/B/C/D) from integer index."""
    return "ABCD"[int(item["answer"])]


def score(predictions: list[str], references: list[str]) -> dict:
    correct = sum(p.strip().upper() == r.strip().upper() for p, r in zip(predictions, references))
    return {"accuracy": correct / len(references), "n": len(references)}
