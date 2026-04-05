#!/usr/bin/env python3
"""
Generate leaderboard table from all result cells in results/.

Usage:
    python scripts/gen_leaderboard.py              # print to stdout
    python scripts/gen_leaderboard.py --update-readme  # update README.md leaderboard section
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# Benchmark display order and labels
BENCH_ORDER = [
    ("music_theory_bench", "MusicTheory"),
    ("ziqi_eval",          "ZIQI-Eval"),
    ("wildscore",          "WildScore"),
    ("muchomusic",         "MuChoMusic"),
    ("cmi_bench",          "CMI-Bench"),
]

MODALITY = {
    "music_theory_bench": "text",
    "ziqi_eval":          "text",
    "wildscore":          "image",
    "muchomusic":         "audio",
    "cmi_bench":          "audio",
}


def load_cells(mode: str = "lite") -> dict[tuple, dict]:
    """Return {(model, benchmark): cell_dict}"""
    cells = {}
    for model_dir in sorted(RESULTS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        for cell_file in sorted(model_dir.glob(f"*_{mode}.json")):
            bench = cell_file.stem.replace(f"_{mode}", "")
            try:
                cell = json.loads(cell_file.read_text())
                cells[(model, bench)] = cell
            except Exception:
                pass
    return cells


def build_table(mode: str = "lite") -> str:
    cells = load_cells(mode)
    if not cells:
        return f"No results found for mode={mode}."

    models = sorted(set(m for m, _ in cells))
    benches = [b for b, _ in BENCH_ORDER if any((m, b) in cells for m in models)]

    # Header
    header = ["Model"] + [f"{label} ({MODALITY[b]})" for b, label in BENCH_ORDER if b in benches]
    sep    = [":---"] + [":---:" for _ in benches]
    rows   = [header, sep]

    for model in models:
        row = [f"`{model}`"]
        for bench in benches:
            cell = cells.get((model, bench))
            if cell:
                acc = cell["accuracy"]
                n   = cell["n"]
                row.append(f"**{acc:.1%}** ({n}q)")
            else:
                row.append("—")
        rows.append(row)

    lines = [" | ".join(r) for r in rows]
    note = (f"\n> Mode: `{mode}` — lite uses standardized fixed-seed samples "
            f"(reproducible, comparable across contributors).\n"
            f"> Submit your results via PR: run any cell, commit `results/{{model}}/{{bench}}_{mode}.json`.\n")
    return "\n".join(lines) + "\n" + note


def update_readme(table: str):
    readme = REPO_ROOT / "README.md"
    if not readme.exists():
        print("README.md not found, skipping update.")
        return
    content = readme.read_text()
    marker_start = "<!-- LEADERBOARD_START -->"
    marker_end   = "<!-- LEADERBOARD_END -->"
    if marker_start not in content:
        print("No leaderboard markers found in README.md. Add:\n"
              "  <!-- LEADERBOARD_START -->\n  <!-- LEADERBOARD_END -->")
        return
    new_section = f"{marker_start}\n{table}\n{marker_end}"
    import re
    updated = re.sub(
        rf"{re.escape(marker_start)}.*?{re.escape(marker_end)}",
        new_section, content, flags=re.DOTALL
    )
    readme.write_text(updated)
    print(f"README.md leaderboard section updated.")


def main():
    update = "--update-readme" in sys.argv
    mode = "lite"
    for arg in sys.argv[1:]:
        if arg.startswith("--mode="):
            mode = arg.split("=")[1]

    table = build_table(mode)
    print(f"=== Leaderboard (mode={mode}) ===\n")
    print(table)

    if update:
        update_readme(table)


if __name__ == "__main__":
    main()
