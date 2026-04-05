#!/usr/bin/env python3
"""
MuTheoryEval — unified runner

Result files are stored per (model, benchmark, mode) cell:
    results/{model}/{benchmark}_lite.json    ← standardized sample, comparable across contributors
    results/{model}/{benchmark}_full.json    ← full dataset

Usage:
    python run.py --model gemini-3.1-flash-lite --benchmark all
    python run.py --model gemini-3.1-flash-lite --benchmark all --mode full
    python run.py --model gemini-3.1-flash-lite --benchmark wildscore,muchomusic
    python run.py --model all --benchmark music_theory_bench
    python run.py --estimate --model gemini-3.1-flash-lite --benchmark all
    python run.py --list-models
    python run.py --list-benchmarks

Contributing results via PR:
    Run any subset of (model, benchmark, mode) cells, commit the JSON files,
    open a PR. Existing cells are skipped unless --force is passed.

Background run (recommended for long jobs):
    See CLAUDE.md → "Running benchmarks" for the canonical workflow.
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Leaderboard updater — imported lazily to avoid circular issues at module level
def _update_leaderboard():
    try:
        sys.path.insert(0, str(Path(__file__).parent / "scripts"))
        from gen_leaderboard import write_data_json
        write_data_json()
    except Exception as e:
        print(f"  [warn] Could not update docs/data.json: {e}")

from benchmarks import REGISTRY as BENCH_REGISTRY, WEIGHTS
from benchmarks import MULTIMODAL_REGISTRY, MULTIMODAL_WEIGHTS
from models import REGISTRY as MODEL_REGISTRY

SYSTEM_PROMPT = (
    "You are a music theory expert. Answer the following multiple-choice question. "
    "Respond with only the letter of the correct answer (A, B, C, or D)."
)

# Cost & speed baseline per model (input $/MTok, output $/MTok, req/min observed)
MODEL_PRICING = {
    "gemini-3.1-flash-lite": {"input_per_mtok": 0.10, "output_per_mtok": 0.40, "rpm": 120},
    "gemini-3.1-flash":      {"input_per_mtok": 0.30, "output_per_mtok": 1.20, "rpm": 60},
    "gemini-3.1-pro":        {"input_per_mtok": 1.25, "output_per_mtok": 5.00, "rpm": 20},
    "gemini-2.5-flash":      {"input_per_mtok": 0.30, "output_per_mtok": 1.20, "rpm": 60},
    "gemini-2.5-pro":        {"input_per_mtok": 1.25, "output_per_mtok": 5.00, "rpm": 20},
    "gpt-5.4-nano":          {"input_per_mtok": 0.20, "output_per_mtok": 1.25, "rpm": 60},
    "gpt-5.4-mini":          {"input_per_mtok": 0.75, "output_per_mtok": 4.50, "rpm": 60},
    "gpt-5.4":               {"input_per_mtok": 2.50, "output_per_mtok": 15.0, "rpm": 30},
    "claude-haiku-4-5":      {"input_per_mtok": 1.00, "output_per_mtok": 5.00, "rpm": 60},
    "claude-sonnet-4-6":     {"input_per_mtok": 3.00, "output_per_mtok": 15.0, "rpm": 30},
    "deepseek-chat":         {"input_per_mtok": 0.27, "output_per_mtok": 1.10, "rpm": 60},
}
AVG_PROMPT_TOKENS = 150
AVG_OUTPUT_TOKENS = 3


def model_dir_name(model_name: str) -> str:
    """Convert model name to a safe directory name."""
    return re.sub(r"[^\w.\-]", "_", model_name)


def result_path(out_dir: Path, model_name: str, bench_name: str, mode: str) -> Path:
    """Return the path for a (model, benchmark, mode) result cell."""
    return out_dir / model_dir_name(model_name) / f"{bench_name}_{mode}.json"


def result_exists(out_dir: Path, model_name: str, bench_name: str, mode: str) -> bool:
    return result_path(out_dir, model_name, bench_name, mode).exists()


def estimate(model_name: str, n_questions: int) -> dict:
    p = MODEL_PRICING.get(model_name, {"input_per_mtok": 1.0, "output_per_mtok": 4.0, "rpm": 30})
    cost = (n_questions * AVG_PROMPT_TOKENS / 1_000_000 * p["input_per_mtok"]
          + n_questions * AVG_OUTPUT_TOKENS / 1_000_000 * p["output_per_mtok"])
    minutes = n_questions / p["rpm"]
    return {"cost_usd": cost, "minutes": minutes, "rpm": p["rpm"]}


def lite_n(meta: dict) -> int | None:
    """Return the standardized lite sample size for a benchmark (None = run all)."""
    return meta.get("lite_n")


def lite_seed(meta: dict) -> int:
    return meta.get("lite_seed", 42)


def n_for_mode(meta: dict, mode: str) -> int:
    if mode == "lite":
        return lite_n(meta) or meta.get("n_questions") or 0
    else:
        return meta.get("n_questions") or 0


def print_plan(model_name: str, bench_names: list[str], mode: str,
               out_dir: Path, force: bool = False):
    all_reg = {**BENCH_REGISTRY, **MULTIMODAL_REGISTRY}
    print(f"\n{'─'*62}")
    print(f"  Run plan: {model_name}  [mode={mode}]")
    print(f"{'─'*62}")
    total_cost, total_min = 0.0, 0.0
    for bn in bench_names:
        if bn not in all_reg:
            continue
        meta = all_reg[bn].METADATA
        if meta.get("status") == "UNRELEASED":
            print(f"  {'SKIP (unreleased)':<18} {bn}")
            continue
        modality = meta.get("modality", "text")
        n = n_for_mode(meta, mode)
        if not n:
            print(f"  {'?':<18} {bn}  [n/a — {modality}]")
            continue
        done = result_exists(out_dir, model_name, bn, mode)
        if done and not force:
            print(f"  {'✓ already done':<18} {bn}  ({n}q)  [skip]")
            continue
        est = estimate(model_name, n)
        total_cost += est["cost_usd"]
        total_min += est["minutes"]
        tag = f"[{modality}]" if modality != "text" else ""
        prefix = "  ↻ redo" if (done and force) else "  →"
        print(f"  {prefix:<18} {bn:<28} {n:>5}q  ~${est['cost_usd']:.3f}  ~{est['minutes']:.1f}min  {tag}")
    print(f"{'─'*62}")
    print(f"  {'TOTAL (to run)':<46}  ~${total_cost:.3f}  ~{total_min:.1f}min")
    print(f"{'─'*62}\n")
    return total_cost, total_min


def run_benchmark(model_name: str, bench_name: str, mode: str,
                  out_dir: Path, bench_registry: dict, force: bool = False) -> dict:
    bench = bench_registry[bench_name]
    meta = bench.METADATA

    if meta.get("status") == "UNRELEASED":
        print(f"\n[SKIP] {bench_name} — UNRELEASED")
        return {"skipped": True, "reason": "UNRELEASED"}

    # Check if already done
    cell_path = result_path(out_dir, model_name, bench_name, mode)
    if cell_path.exists() and not force:
        existing = json.loads(cell_path.read_text())
        acc = existing.get("accuracy", 0)
        print(f"\n[SKIP] {bench_name} — already done ({acc:.1%}). Use --force to rerun.")
        return existing

    n_expected = n_for_mode(meta, mode)
    est = estimate(model_name, n_expected) if n_expected else {"cost_usd": 0, "minutes": 0, "rpm": 30}
    modality = meta.get("modality", "text")
    modality_tag = f"  [{modality}]" if modality != "text" else ""
    n_str = f"{n_expected}q" if n_expected else "?q"

    print(f"\n{'─'*62}")
    print(f"  {model_name} × {bench_name}  [{mode}]{modality_tag}")
    print(f"  {n_str}  est. ${est['cost_usd']:.3f}  est. {est['minutes']:.1f}min  ({est.get('rpm', '?')} rpm)")
    print(f"{'─'*62}")

    model = MODEL_REGISTRY[model_name]()

    # Load dataset with mode-appropriate sampling
    try:
        if mode == "lite":
            ln = lite_n(meta)
            ls = lite_seed(meta)
            if ln:
                dataset = bench.load(sample=ln, seed=ls)
            else:
                dataset = bench.load()
        else:
            dataset = bench.load(sample=None)
    except (NotImplementedError, FileNotFoundError, TypeError) as e:
        # TypeError: some load() don't accept sample= kwarg — try without
        try:
            dataset = bench.load()
        except (NotImplementedError, FileNotFoundError) as e2:
            print(f"  SKIP: {e2}")
            return {"skipped": True, "reason": str(e2)}

    n = len(dataset)

    # Checkpoint path (in model subdir)
    cell_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_file = cell_path.parent / f".checkpoint_{bench_name}_{mode}.json"

    predictions, references = [], []
    t_start = time.time()
    start_idx = 0

    if checkpoint_file.exists():
        try:
            ck = json.loads(checkpoint_file.read_text())
            predictions = ck["predictions"]
            references = ck["references"]
            start_idx = len(predictions)
            print(f"  Resuming from checkpoint at {start_idx}/{n}")
        except Exception:
            pass

    for i, item in enumerate(dataset):
        if i < start_idx:
            continue

        prompt = bench.format_prompt(item)

        media = None
        if hasattr(bench, "get_media"):
            try:
                media = bench.get_media(item)
            except FileNotFoundError as e:
                print(f"\n  SKIP: {e}")
                return {"skipped": True, "reason": str(e)}

        raw = model.complete(prompt, system=SYSTEM_PROMPT, media=media)
        pred = model.extract_choice(raw)

        if hasattr(bench, "get_answer"):
            ref = bench.get_answer(item)
        else:
            ref = item.get("answer", item.get("label", ""))
            ref = ref.strip().upper() if isinstance(ref, str) else str(ref)

        predictions.append(pred)
        references.append(ref)

        done = i + 1
        elapsed = time.time() - t_start
        speed = (done - start_idx) / elapsed if elapsed > 0 else 0
        remaining = (n - done) / speed if speed > 0 else 0
        correct_so_far = sum(p == r for p, r in zip(predictions, references))
        acc_so_far = correct_so_far / done

        bar_w = 25
        filled = int(bar_w * done / n)
        bar = "█" * filled + "░" * (bar_w - filled)
        eta_str = f"{int(remaining//60)}m{int(remaining%60):02d}s" if speed > 0 else "--"
        print(f"\r  [{bar}] {done}/{n}  acc={acc_so_far:.1%}  ETA {eta_str}   ", end="", flush=True)

        if done % 50 == 0:
            checkpoint_file.write_text(json.dumps({"predictions": predictions, "references": references}))

    print()

    result = bench.score(predictions, references)
    elapsed_total = time.time() - t_start

    # Detect reasoning-native models (always-on thinking, can't be disabled)
    _reasoning_prefixes = ("o1", "o3", "o4", "deepseek-reasoner", "deepseek-r1")
    is_reasoning_native = any(
        model_name == p or model_name.startswith(p + "-")
        for p in _reasoning_prefixes
    )

    cell = {
        "model": model_name,
        "benchmark": bench_name,
        "mode": mode,
        "accuracy": result["accuracy"],
        "n": result["n"],
        "seed": lite_seed(meta) if mode == "lite" else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(elapsed_total),
        "cost_est_usd": round(estimate(model_name, result["n"])["cost_usd"], 4),
        # Thinking configuration — for reproducibility and fair-comparison transparency
        "thinking": "native" if is_reasoning_native else False,
    }

    print(f"  ✓ accuracy: {cell['accuracy']:.1%}  ({cell['n']}q)  "
          f"took {int(elapsed_total//60)}m{int(elapsed_total%60):02d}s  ~${cell['cost_est_usd']:.3f}")

    cell_path.write_text(json.dumps(cell, indent=2))
    print(f"  Saved → {cell_path}")

    if checkpoint_file.exists():
        checkpoint_file.unlink()

    _update_leaderboard()

    return cell


def weighted_score(cells: list[dict], weights: dict) -> float | None:
    valid = [c for c in cells if not c.get("skipped") and c.get("benchmark") in weights]
    if not valid:
        return None
    total_weight = sum(weights[c["benchmark"]] for c in valid)
    if total_weight == 0:
        return None
    return sum(c["accuracy"] * weights[c["benchmark"]] for c in valid) / total_weight


def main():
    parser = argparse.ArgumentParser(description="MuTheoryEval runner")
    parser.add_argument("--model", default="gemini-3.1-flash-lite",
                        help="Model key(s), comma-separated or 'all'")
    parser.add_argument("--benchmark", default="all",
                        help="Benchmark name(s), comma-separated, 'all', or 'multimodal'")
    parser.add_argument("--mode", default="lite", choices=["lite", "full"],
                        help="lite = standardized sample (default); full = entire dataset")
    parser.add_argument("--estimate", action="store_true",
                        help="Print cost/time estimate and exit without running")
    parser.add_argument("--force", action="store_true",
                        help="Rerun and overwrite existing result cells")
    parser.add_argument("--multimodal", action="store_true",
                        help="Include multimodal (image/audio) benchmarks")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--list-benchmarks", action="store_true")
    parser.add_argument("--out", default="results", help="Output directory root")
    args = parser.parse_args()

    ALL_BENCH_REGISTRY = {**BENCH_REGISTRY, **MULTIMODAL_REGISTRY}
    out_dir = Path(args.out)

    if args.list_models:
        print("Available models:")
        for m in MODEL_REGISTRY:
            pricing = MODEL_PRICING.get(m, {})
            rpm = pricing.get("rpm", "?")
            inp = pricing.get("input_per_mtok", "?")
            print(f"  {m:<30} ${inp}/MTok in  {rpm} rpm")
        return

    if args.list_benchmarks:
        print("\nAvailable benchmarks:")
        for b, mod in ALL_BENCH_REGISTRY.items():
            meta = mod.METADATA
            status = meta.get("status", "OK")
            modality = meta.get("modality", "text")
            n_lite = n_for_mode(meta, "lite")
            n_full = meta.get("n_questions") or 0
            est = estimate("gemini-3.1-flash-lite", n_lite) if n_lite else {"cost_usd": 0, "minutes": 0}
            print(f"  {b:<28} lite={n_lite or '?':>5}q  full={n_full or '?':>6}q  "
                  f"w={meta['weight']}  [{modality}]  ~${est['cost_usd']:.3f}/lite  [{status}]")
        return

    models = list(MODEL_REGISTRY.keys()) if args.model == "all" else [m.strip() for m in args.model.split(",")]

    if args.benchmark == "all":
        benchmarks = list(BENCH_REGISTRY.keys())
    elif args.benchmark == "multimodal":
        benchmarks = list(MULTIMODAL_REGISTRY.keys())
    else:
        benchmarks = [b.strip() for b in args.benchmark.split(",")]

    if args.multimodal:
        benchmarks = list(dict.fromkeys(benchmarks + list(MULTIMODAL_REGISTRY.keys())))

    for model_name in models:
        print_plan(model_name, benchmarks, args.mode, out_dir, force=args.force)

    if args.estimate:
        return

    for model_name in models:
        if model_name not in MODEL_REGISTRY:
            print(f"Unknown model: {model_name}. Use --list-models.")
            continue

        all_cells = []
        t_model_start = time.time()

        for bench_name in benchmarks:
            if bench_name not in ALL_BENCH_REGISTRY:
                print(f"Unknown benchmark: {bench_name}. Use --list-benchmarks.")
                continue
            cell = run_benchmark(model_name, bench_name, args.mode,
                                 out_dir, ALL_BENCH_REGISTRY, force=args.force)
            if not cell.get("skipped"):
                all_cells.append(cell)

        elapsed = time.time() - t_model_start

        # Print summary of this model's cells
        text_cells = [c for c in all_cells if c["benchmark"] in WEIGHTS]
        mm_cells   = [c for c in all_cells if c["benchmark"] in MULTIMODAL_WEIGHTS]
        ws_text = weighted_score(text_cells, WEIGHTS)
        ws_mm   = weighted_score(mm_cells, MULTIMODAL_WEIGHTS)

        print(f"\n{'═'*62}")
        print(f"  {model_name}  [{args.mode}]")
        if ws_text is not None:
            print(f"  Text weighted score:  {ws_text:.1%}")
        if ws_mm is not None:
            print(f"  Modal weighted score: {ws_mm:.1%}")
        print(f"  Total time: {int(elapsed//60)}m{int(elapsed%60):02d}s")
        print(f"{'═'*62}")


if __name__ == "__main__":
    main()
