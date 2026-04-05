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
        "source":    "arXiv:2407.05830",
        "hf":        "nicolaus625/CMI-bench",
        "weight":    0.15,
        "n_full":    None,
        "lite_n":    50,
    },
]

# ── Model catalogue ────────────────────────────────────────────────────────────
# Inferred from model key prefix; used to annotate provider in data.json.

def infer_provider(model_key: str) -> str:
    if model_key.startswith("gpt-") or model_key == "o3":
        return "OpenAI"
    if model_key.startswith("claude-"):
        return "Anthropic"
    if model_key.startswith("gemini-"):
        return "Google"
    if model_key.startswith("deepseek-"):
        return "DeepSeek"
    if model_key.startswith(("llama-", "qwen", "mistral")):
        return "DeepInfra"
    return "Unknown"


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
        {"key": m, "provider": infer_provider(m)}
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


def build_table(mode: str = "lite") -> str:
    cells = load_cells_for_mode(mode)
    if not cells:
        return f"No results found for mode={mode}."

    models  = sorted({m for m, _ in cells})
    benches = [b["key"] for b in BENCH_CATALOGUE if any((m, b["key"]) in cells for m in models)]

    header = ["Model"] + [
        f"{b['short']} ({b['modality']})"
        for b in BENCH_CATALOGUE if b["key"] in benches
    ]
    sep = [":---"] + [":---:" for _ in benches]
    rows = [header, sep]

    for model in models:
        row = [f"`{model}`"]
        for b_key in benches:
            cell = cells.get((model, b_key))
            row.append(f"**{cell['accuracy']:.1%}** ({cell['n']}q)" if cell else "—")
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
