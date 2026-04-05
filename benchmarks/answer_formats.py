"""
Answer format definitions for MuTheoryEval.

Each format specifies the system prompt, max output tokens, answer extraction,
per-item comparison metric, and format-error detection.

Benchmarks declare their format via METADATA["answer_format"]. run.py reads
this and dispatches the right system prompt, extractor, and comparator.

Formats
-------
MCQ          — single letter A/B/C/D; exact match; accuracy
MULTI_SELECT — comma-separated letters A-H (select all that apply); Jaccard
CLOSED_SINGLE — single label from a known closed vocabulary; normalised exact match
CLOSED_MULTI  — comma-separated labels from a known closed vocabulary; Jaccard

Adding a new format: add an entry to ANSWER_FORMATS with the five required keys.
"""

from __future__ import annotations
import re


# ── Extraction helpers ────────────────────────────────────────────────────────

def _extract_mcq(raw: str) -> str:
    """Return single letter A/B/C/D, or 'X' on failure."""
    head = raw.strip()[:20]
    m = re.search(r'\b([A-D])\b', head)
    if m:
        return m.group(1).upper()
    m = re.search(r'\b([A-D])\b', raw.upper())
    return m.group(1) if m else "X"


def _extract_multi_select(raw: str) -> str:
    """Return sorted unique letters A-H found in response, comma-joined, or '' on failure.

    E.g. raw='A, C' → 'A,C'; raw='A and F and H' → 'A,F,H'; raw='nothing' → ''
    """
    letters = sorted(set(re.findall(r'\b([A-H])\b', raw.upper())))
    return ",".join(letters)


def _extract_closed_single(raw: str, *, labels: list[str] | None = None) -> str:
    """Find the first known label (case-insensitive substring) in response.

    When labels is None, falls back to whole stripped response (normalised).
    """
    normed = raw.strip().lower()
    if labels:
        # Prefer longer matches first to avoid partial shadowing
        for label in sorted(labels, key=len, reverse=True):
            if label.lower() in normed:
                return label.lower()
        return ""
    return normed


def _extract_closed_multi(raw: str, *, labels: list[str] | None = None) -> str:
    """Find all known labels in response; return sorted comma-joined string."""
    normed = raw.strip().lower()
    found: list[str] = []
    if labels:
        for label in labels:
            if label.lower() in normed:
                found.append(label.lower())
    return ",".join(sorted(set(found)))


# ── Comparison helpers ────────────────────────────────────────────────────────

def _compare_exact(pred: str, ref: str) -> float:
    """1.0 if equal (case-insensitive), else 0.0. 'X'/'' counts as wrong."""
    if pred in ("X", "") or ref in ("X", ""):
        return 0.0
    return 1.0 if pred.strip().upper() == ref.strip().upper() else 0.0


def _jaccard(pred: str, ref: str) -> float:
    """Jaccard similarity on comma-separated token sets.

    Empty pred → 0. Empty ref → 1 iff pred also empty, else 0.
    """
    set_p = set(t.strip() for t in pred.split(",") if t.strip()) if pred else set()
    set_r = set(t.strip() for t in ref.split(",")  if t.strip()) if ref else set()
    if not set_p and not set_r:
        return 1.0
    if not set_p or not set_r:
        return 0.0
    return len(set_p & set_r) / len(set_p | set_r)


# ── Format registry ───────────────────────────────────────────────────────────

ANSWER_FORMATS: dict[str, dict] = {
    "MCQ": {
        # !! Changing this string will invalidate all existing result hashes (451d3bf6). !!
        "system_prompt": (
            "You are a music theory expert. "
            "YOUR ENTIRE RESPONSE MUST BE EXACTLY ONE LETTER: A, B, C, or D. "
            "DO NOT write anything else. DO NOT explain. DO NOT show reasoning. ONLY the letter."
        ),
        "max_output_tokens": 16,
        "extract":        _extract_mcq,
        "compare":        _compare_exact,
        "is_format_error": lambda p: p == "X",
        "score_label":    "accuracy",
    },

    "MULTI_SELECT": {
        "system_prompt": (
            "You are a music expert. "
            "YOUR ENTIRE RESPONSE MUST BE the letter(s) of ALL correct options, "
            "separated by commas (e.g. 'A' or 'A, C, F'). "
            "DO NOT write anything else. DO NOT explain. ONLY the letter(s)."
        ),
        "max_output_tokens": 64,
        "extract":        _extract_multi_select,
        "compare":        _jaccard,
        "is_format_error": lambda p: p == "",
        "score_label":    "mean_jaccard",
    },

    "CLOSED_SINGLE": {
        "system_prompt": (
            "You are a music expert. "
            "YOUR ENTIRE RESPONSE MUST BE EXACTLY ONE LABEL from the provided list. "
            "DO NOT write anything else. ONLY the label."
        ),
        "max_output_tokens": 32,
        "extract":        _extract_closed_single,
        "compare":        _compare_exact,
        "is_format_error": lambda p: p == "",
        "score_label":    "accuracy",
    },

    "CLOSED_MULTI": {
        "system_prompt": (
            "You are a music expert. "
            "YOUR ENTIRE RESPONSE MUST BE ALL applicable labels from the provided list, "
            "separated by commas. DO NOT write anything else. ONLY the labels."
        ),
        "max_output_tokens": 128,
        "extract":        _extract_closed_multi,
        "compare":        _jaccard,
        "is_format_error": lambda p: p == "",
        "score_label":    "mean_jaccard",
    },
}

DEFAULT_FORMAT = "MCQ"


def get_format(bench) -> dict:
    """Return the format dict for a benchmark module or METADATA dict."""
    if isinstance(bench, dict):
        fmt_name = bench.get("answer_format", DEFAULT_FORMAT)
    else:
        fmt_name = getattr(bench, "METADATA", {}).get("answer_format", DEFAULT_FORMAT)
    return ANSWER_FORMATS.get(fmt_name, ANSWER_FORMATS[DEFAULT_FORMAT])


def get_format_name(bench) -> str:
    """Return the format name string."""
    if isinstance(bench, dict):
        return bench.get("answer_format", DEFAULT_FORMAT)
    return getattr(bench, "METADATA", {}).get("answer_format", DEFAULT_FORMAT)
