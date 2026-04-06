#!/usr/bin/env python3
"""
Generate leaderboard artifacts from result cells in results/.

Usage:
    python scripts/gen_leaderboard.py              # print markdown table to stdout
    python scripts/gen_leaderboard.py --update-readme   # update README.md leaderboard block
    python scripts/gen_leaderboard.py --update-data     # regenerate docs/data.json
    python scripts/gen_leaderboard.py --all             # both of the above

Called automatically by run.py after each cell is saved.
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DOCS_DIR = REPO_ROOT / "docs"

# ── Benchmark catalogue ────────────────────────────────────────────────────────
# Ordered for display; keys match results/{model}/{key}_*.json filenames.

BENCH_CATALOGUE = [
    {
        "key":       "music_theory_bench",
        "name":      "MusicTheoryBench",
        "short":     "MusicTheory",
        "modality":  "text",
        "source":    "arXiv:2402.16153",
        "hf":        "m-a-p/MusicTheoryBench",
        "weight":    0.35,
        "n_full":    372,
        "lite_n":    None,
    },
    {
        "key":       "ziqi_eval",
        "name":      "ZIQI-Eval",
        "short":     "ZIQI-Eval",
        "modality":  "text",
        "source":    "arXiv:2406.15885",
        "hf":        "MYTH-Lab/ZIQI-Eval",
        "weight":    0.35,
        "n_full":    14244,
        "lite_n":    500,
    },
    {
        "key":       "wildscore",
        "name":      "WildScore",
        "short":     "WildScore",
        "modality":  "image",
        "source":    "arXiv:2509.04744",
        "hf":        "GM77/WildScore",
        "weight":    0.15,
        "n_full":    807,
        "lite_n":    100,
    },
    {
        "key":       "muchomusic",
        "name":      "MuChoMusic",
        "short":     "MuChoMusic",
        "modality":  "audio",
        "source":    "arXiv:2408.01337",
        "hf":        "lmms-lab/muchomusic",
        "weight":    0.15,
        "n_full":    1187,
        "lite_n":    200,
    },
    {
        "key":       "cmi_bench",
        "name":      "CMI-Bench",
        "short":     "CMI-Bench",
        "modality":  "audio",
        "source":    "arXiv:2506.12285",
        "hf":        "nicolaus625/CMI-bench",
        "weight":    0.15,
        "n_full":    None,   # full dataset ~56k items across 16 tasks; not yet fully implemented
        "lite_n":    100,
    },
    {
        "key":       "ssmr_bench",
        "name":      "SSMR-Bench",
        "short":     "SSMR-Bench",
        "modality":  "abc",
        "source":    "arXiv:2509.04059",
        "hf":        "Sylence/SSMR-Bench",
        "weight":    1.0,
        "n_full":    1600,
        "lite_n":    200,
    },
]

# ── Model catalogue ────────────────────────────────────────────────────────────
# Inferred from model key prefix; used to annotate provider/capabilities in data.json.

# Capabilities reflect what our implementation actually passes to the API,
# not what the underlying model architecture theoretically supports.
# e.g. DeepInfra models skip media even if the model supports images.
_CAPABILITIES: list[tuple[str, list[str]]] = [
    ("gemini-",       ["text", "abc", "image", "audio"]),  # GeminiModel handles all modalities
    ("claude-",       ["text", "abc", "image"]),            # AnthropicModel: image yes, audio skipped
    ("gpt-",          ["text", "abc", "image"]),            # OpenAIModel: image yes, audio skipped
    ("o1",            ["text", "abc", "image"]),
    ("o3",            ["text", "abc", "image"]),
    ("o4",            ["text", "abc", "image"]),
    ("deepseek-",     ["text", "abc"]),
    ("llama-",        ["text", "abc"]),   # DeepInfraModel ignores media kwarg
    ("qwen3.5-omni",  ["text", "abc", "image", "audio"]),   # DashScope Omni: full multimodal
    ("qwen",          ["text", "abc"]),
    ("glm-",          ["text", "abc"]),                     # ZAIModel: text only
    ("mistral",       ["text", "abc"]),
]

def infer_capabilities(model_key: str) -> list[str]:
    for prefix, caps in _CAPABILITIES:
        if model_key.startswith(prefix):
            return caps
    return ["text", "abc"]

def infer_provider(model_key: str) -> str:
    if model_key.startswith("gpt-") or model_key.startswith("o1") or model_key.startswith("o3") or model_key.startswith("o4"):
        return "OpenAI"
    if model_key.startswith("claude-"):
        return "Anthropic"
    if model_key.startswith("gemini-"):
        return "Google"
    if model_key.startswith("deepseek-"):
        return "DeepSeek"
    if model_key.startswith("glm-"):
        return "ZhipuAI"
    if model_key.startswith("qwen3.5-omni"):
        return "Alibaba"
    if model_key.startswith(("llama-", "qwen", "mistral")):
        return "DeepInfra"
    return "Unknown"


# ── Thinking / reasoning model detection ──────────────────────────────────────
# "thinking" = model uses extended chain-of-thought reasoning at inference time,
# either always-on (Gemini 3, Z1, o-series, DeepSeek-R1) or explicitly enabled.
_THINKING_PREFIXES = ("gemini-3-", "gemini-3.1-")  # always-on: Gemini 3 series
_THINKING_KEYS = {
    # OpenAI reasoning (o-series)
    "o1", "o3", "o4",
    # DeepSeek
    "deepseek-reasoner",
    # Always-on open-weight
    "deepseek-r1", "qwen3-max-thinking",
}
_THINKING_SUBSTRINGS = ("-thinking", "glm-z1", "-xt")  # GLM-Z1, GLM-*-thinking, Claude-*-xt*

def infer_thinking(model_key: str) -> bool:
    if any(model_key.startswith(p) for p in _THINKING_PREFIXES):
        return True
    if model_key in _THINKING_KEYS:
        return True
    if any(sub in model_key for sub in _THINKING_SUBSTRINGS):
        return True
    return False


# ── Display names ──────────────────────────────────────────────────────────────
# For variants of a base model, show "base (variant)" instead of "base-variant"
# to avoid confusion with genuinely different models (e.g. flash-lite ≠ flash).
# Keys not listed here fall back to the model key itself.
_DISPLAY_NAMES: dict[str, str] = {
    "gemini-3.1-flash-minimal": "gemini-3.1-flash (minimal)",
    "claude-sonnet-4-6-xt8k":   "claude-sonnet-4-6 (thinking 8k)",
    "claude-opus-4-6-xt8k":     "claude-opus-4-6 (thinking 8k)",
    "glm-5-thinking":           "glm-5 (thinking)",
    "gemini-2.5-pro-thinking":  "gemini-2.5-pro (thinking)",
}

# Short note shown in model tooltip — explains variant suffix or special config.
# Especially useful when the result cell's config block doesn't capture the setting.
_MODEL_NOTES: dict[str, str] = {
    "gemini-3.1-flash-minimal": "Gemini 3 Flash with thinking_level=MINIMAL (explicit low-budget thinking).",
    "gemini-3.1-flash":         "Gemini 3 Flash — always-on thinking at model default level (HIGH).",
    "gemini-3.1-flash-lite":    "Gemini 3 Flash Lite — always-on thinking, lighter/faster variant.",
    "claude-sonnet-4-6-xt8k":   "Extended thinking enabled, budget=8000 tokens. Same base model as claude-sonnet-4-6.",
    "claude-opus-4-6-xt8k":     "Extended thinking enabled, budget=8000 tokens. Same base model as claude-opus-4-6.",
    "glm-5-thinking":           "GLM-5 with optional thinking mode enabled (Anthropic-compatible thinking blocks).",
    "glm-z1-flash":             "GLM-Z1-Flash — always-on reasoning model; thinking output stripped from response.",
}

def display_name(model_key: str) -> str:
    return _DISPLAY_NAMES.get(model_key, model_key)

def model_note(model_key: str) -> str | None:
    return _MODEL_NOTES.get(model_key)


# ── Cell loading ───────────────────────────────────────────────────────────────

def load_all_cells() -> list[dict]:
    """Scan results/ and return all valid cell dicts."""
    cells = []
    if not RESULTS_DIR.exists():
        return cells
    for model_dir in sorted(RESULTS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        for cell_file in sorted(model_dir.glob("*.json")):
            try:
                cell = json.loads(cell_file.read_text())
                # Ensure required keys
                if all(k in cell for k in ("model", "benchmark", "mode", "accuracy", "n")):
                    # Strip old samples block — bad cases live in .errors.jsonl now.
                    # Keeps data.json lean for old cells that still carry samples.
                    cell.pop("samples", None)
                    cells.append(cell)
            except Exception:
                pass
    return cells


# ── data.json ─────────────────────────────────────────────────────────────────

def build_data_json() -> dict:
    """Build the full data.json payload."""
    cells = load_all_cells()

    # Collect all known model keys from result files
    known_models = sorted({c["model"] for c in cells})

    models_meta = [
        {
            "key":          m,
            "display_name": display_name(m),
            "note":         model_note(m),
            "provider":     infer_provider(m),
            "capabilities": infer_capabilities(m),
            "thinking":     infer_thinking(m),
        }
        for m in known_models
    ]

    return {
        "updated": datetime.now(timezone.utc).isoformat(),
        "benchmarks": BENCH_CATALOGUE,
        "models": models_meta,
        "cells": cells,
    }


def write_data_json():
    """Regenerate docs/data.json from current results/."""
    DOCS_DIR.mkdir(exist_ok=True)
    data = build_data_json()
    out = DOCS_DIR / "data.json"
    out.write_text(json.dumps(data, indent=2))
    n = len(data["cells"])
    print(f"  data.json updated — {n} cell(s) across {len(data['models'])} model(s)")


# ── Markdown leaderboard ───────────────────────────────────────────────────────

def load_cells_for_mode(mode: str = "lite") -> dict[tuple, dict]:
    return {
        (c["model"], c["benchmark"]): c
        for c in load_all_cells()
        if c.get("mode") == mode
    }


TEXT_BENCHES  = ["music_theory_bench", "ziqi_eval"]
IMAGE_BENCHES = ["wildscore"]
AUDIO_BENCHES = ["muchomusic", "cmi_bench"]
ABC_BENCHES   = ["ssmr_bench"]


def _modality_score(model: str, bench_keys: list[str], cells: dict, require_all: bool = False) -> float | None:
    """Weighted average for a modality group. Returns None if require_all and any bench missing."""
    group = [b for b in BENCH_CATALOGUE if b["key"] in bench_keys]
    present = [b for b in group if (model, b["key"]) in cells]
    if require_all and len(present) < len(group):
        return None
    if not present:
        return None
    total_w = sum(b["weight"] for b in present)
    return sum(cells[(model, b["key"])]["accuracy"] * b["weight"] for b in present) / total_w


def build_table(mode: str = "lite") -> str:
    cells = load_cells_for_mode(mode)
    if not cells:
        return f"No results found for mode={mode}."

    models  = sorted({m for m, _ in cells})
    benches = [b["key"] for b in BENCH_CATALOGUE if any((m, b["key"]) in cells for m in models)]

    header = ["Model", "Text", "ABC", "Image", "Audio"] + [
        f"{b['short']} ({b['modality']})"
        for b in BENCH_CATALOGUE if b["key"] in benches
    ]
    sep = [":---"] + [":---:" for _ in range(4 + len(benches))]
    rows = [header, sep]

    for model in models:
        text_s  = _modality_score(model, TEXT_BENCHES,  cells, require_all=True)
        abc_s   = _modality_score(model, ABC_BENCHES,   cells, require_all=False)
        image_s = _modality_score(model, IMAGE_BENCHES, cells, require_all=True)
        audio_s = _modality_score(model, AUDIO_BENCHES, cells, require_all=True)
        dn = display_name(model)
        label = f"`{dn}` [T]" if infer_thinking(model) else f"`{dn}`"
        row = [
            label,
            f"**{text_s:.1%}**"  if text_s  is not None else "—",
            f"**{abc_s:.1%}**"   if abc_s   is not None else "—",
            f"**{image_s:.1%}**" if image_s is not None else "—",
            f"**{audio_s:.1%}**" if audio_s is not None else "—",
        ]
        for b_key in benches:
            cell = cells.get((model, b_key))
            row.append(f"{cell['accuracy']:.1%} ({cell['n']}q)" if cell else "—")
        rows.append(row)

    lines = [" | ".join(r) for r in rows]
    note = (
        f"\n> Mode: `{mode}` — lite uses standardized fixed-seed samples "
        f"(reproducible, comparable across contributors).  \n"
        f"> Submit your results via PR: run any cell, "
        f"commit `results/{{model}}/{{bench}}_{mode}.json` and `docs/data.json`.\n"
    )
    return "\n".join(lines) + "\n" + note


def update_readme(table: str):
    readme = REPO_ROOT / "README.md"
    if not readme.exists():
        print("README.md not found.")
        return
    content = readme.read_text()
    start, end = "<!-- LEADERBOARD_START -->", "<!-- LEADERBOARD_END -->"
    if start not in content:
        print("No leaderboard markers in README.md.")
        return
    updated = re.sub(
        rf"{re.escape(start)}.*?{re.escape(end)}",
        f"{start}\n{table}\n{end}",
        content, flags=re.DOTALL,
    )
    readme.write_text(updated)
    print("README.md leaderboard section updated.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    args = set(sys.argv[1:])
    mode = next((a.split("=")[1] for a in args if a.startswith("--mode=")), "lite")
    do_readme = "--update-readme" in args or "--all" in args
    do_data   = "--update-data"   in args or "--all" in args

    table = build_table(mode)
    print(f"=== Leaderboard (mode={mode}) ===\n")
    print(table)

    if do_readme:
        update_readme(table)
    if do_data:
        write_data_json()


if __name__ == "__main__":
    main()
