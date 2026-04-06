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
import hashlib
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

from benchmarks import REGISTRY as BENCH_REGISTRY, WEIGHTS, ABC_WEIGHTS
from benchmarks import MULTIMODAL_REGISTRY, MULTIMODAL_WEIGHTS
from benchmarks.answer_formats import ANSWER_FORMATS, get_format, get_format_name
from models import REGISTRY as MODEL_REGISTRY


# ── Bad-case JSONL helpers ────────────────────────────────────────────────────

def _write_errors_jsonl(path: Path, sample_data: list[dict]) -> None:
    """Overwrite the errors JSONL with all wrong items from sample_data.

    Called on checkpoint-write (keeps file in sync) and on resume (rebuilds
    from checkpoint so the file never has stale entries from a prior crash).
    """
    lines = [json.dumps(s) for s in sample_data if not s.get("correct", True)]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _append_error_line(path: Path, s: dict) -> None:
    """Stream-append one wrong item to the errors JSONL immediately after eval."""
    with open(path, "a") as f:
        f.write(json.dumps(s) + "\n")

# Cost & speed baseline per model (input $/MTok, output $/MTok, req/min observed)
MODEL_PRICING = {
    # Verified against LiteLLM model_prices_and_context_window.json (2026-04)
    "gemini-3.1-flash-lite": {"input_per_mtok": 0.25, "output_per_mtok": 1.50,  "rpm": 120},
    "gemini-3.1-flash":         {"input_per_mtok": 0.50, "output_per_mtok": 3.00, "rpm": 60},
    "gemini-3.1-flash-minimal": {"input_per_mtok": 0.50, "output_per_mtok": 3.00, "rpm": 60},  # same model, minimal thinking
    "gemini-3.1-pro":        {"input_per_mtok": 2.00, "output_per_mtok": 12.00, "rpm": 20},
    "gemini-2.5-flash":      {"input_per_mtok": 0.30, "output_per_mtok": 1.20,  "rpm": 60},
    "gemini-2.5-pro":        {"input_per_mtok": 1.25, "output_per_mtok": 5.00,  "rpm": 20},
    "gpt-5.4-nano":          {"input_per_mtok": 0.20, "output_per_mtok": 1.25, "rpm": 60},
    "gpt-5.4-mini":          {"input_per_mtok": 0.75, "output_per_mtok": 4.50, "rpm": 60},
    "gpt-5.4":               {"input_per_mtok": 2.50, "output_per_mtok": 15.0, "rpm": 30},
    "claude-haiku-4-5":       {"input_per_mtok": 1.00, "output_per_mtok": 5.00,  "rpm": 60},
    "claude-sonnet-4-6":      {"input_per_mtok": 3.00, "output_per_mtok": 15.0,  "rpm": 30},
    "claude-sonnet-4-6-xt8k": {"input_per_mtok": 3.00, "output_per_mtok": 15.0,  "rpm": 30},  # thinking tokens at output rate
    "claude-opus-4-6":        {"input_per_mtok": 5.00, "output_per_mtok": 25.0,  "rpm": 20},
    "claude-opus-4-6-xt8k":   {"input_per_mtok": 5.00, "output_per_mtok": 25.0,  "rpm": 20},
    "deepseek-chat":          {"input_per_mtok": 0.27, "output_per_mtok": 1.10,  "rpm": 60},
    # Qwen3.5-Omni (DashScope international) — free during 90-day preview (Apr 2026)
    # Post-preview estimate based on Qwen3-Omni-Flash rates: audio $3.81/MTok (7 tok/s)
    # Text: $0.43 in / $1.66 out (text-only mode); $3.06 out (multimodal mode)
    "qwen3.5-omni-plus":  {"input_per_mtok": 0.0,  "output_per_mtok": 0.0,   "rpm": 20},  # free preview
    "qwen3.5-omni-flash": {"input_per_mtok": 0.0,  "output_per_mtok": 0.0,   "rpm": 30},  # free preview
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

    # Apply answer-format overrides (system prompt, max_output_tokens)
    fmt_name = get_format_name(bench)
    fmt = ANSWER_FORMATS.get(fmt_name, ANSWER_FORMATS["MCQ"])
    # Override model's max_output_tokens to match the format requirement.
    # For thinking models, preserve the larger of the two values.
    fmt_max = fmt["max_output_tokens"]
    model_max = model.config.get("max_output_tokens", fmt_max)
    model.config["max_output_tokens"] = max(fmt_max, model_max)

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

    # Checkpoint + errors paths (in model subdir)
    cell_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_file = cell_path.parent / f".checkpoint_{bench_name}_{mode}.json"
    errors_path     = cell_path.parent / f"{bench_name}_{mode}.errors.jsonl"

    # On --force, clear any stale errors file so it reflects the fresh run only
    if force and errors_path.exists():
        errors_path.unlink()

    # Format-error predicate — defined here (needs fmt in scope, used in loop)
    def _is_format_error(s: dict) -> bool:
        return fmt["is_format_error"](s["pred"]) or s["stop_reason"] == "max_tokens"

    predictions, references = [], []
    _sample_data: list[dict] = []  # per-question data for bad-case analysis
    t_start = time.time()
    start_idx = 0
    # Accumulated token usage across all questions
    _usage_totals: dict = {"prompt_tokens": 0, "completion_tokens": 0, "thinking_tokens": 0}
    _has_thinking_tokens = False

    if checkpoint_file.exists():
        try:
            ck = json.loads(checkpoint_file.read_text())
            predictions = ck["predictions"]
            references = ck["references"]
            _sample_data = ck.get("sample_data", [])
            start_idx = len(predictions)
            print(f"  Resuming from checkpoint at {start_idx}/{n}")
            # Rebuild JSONL from checkpoint so it's in sync (no stale entries)
            _write_errors_jsonl(errors_path, _sample_data)
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

        raw = model.complete(prompt, system=fmt["system_prompt"], media=media)
        pred = fmt["extract"](raw)

        # Accumulate token usage reported by the API
        q_output_tokens: int | None = None
        if model.last_usage:
            _usage_totals["prompt_tokens"]     += model.last_usage.get("prompt_tokens") or 0
            _usage_totals["completion_tokens"] += model.last_usage.get("completion_tokens") or 0
            tk = model.last_usage.get("thinking_tokens")
            if tk is not None:
                _usage_totals["thinking_tokens"] += tk
                _has_thinking_tokens = True
            q_output_tokens = model.last_usage.get("completion_tokens")

        if hasattr(bench, "get_answer"):
            ref = bench.get_answer(item)
        else:
            ref = item.get("answer", item.get("label", ""))
            ref = ref.strip().upper() if isinstance(ref, str) else str(ref)

        predictions.append(pred)
        references.append(ref)

        # Collect per-question data for bad-case analysis
        max_out = model.config.get("max_output_tokens", 16) if model.config else 16
        stop_reason_est = (
            "max_tokens" if (q_output_tokens is not None and q_output_tokens >= max_out)
            else "end_turn"
        )
        item_score = fmt["compare"](pred, ref)
        _s: dict = {
            "idx":          i,
            "subject":      item.get("subject", "") if hasattr(item, "get") else "",
            "stem":         (item.get("stem", item.get("question", "")) or "")[:80] if hasattr(item, "get") else "",
            "raw":          raw[:120],
            "pred":         pred,
            "ref":          ref,
            "correct":      item_score == 1.0,
            "score":        item_score,
            "stop_reason":  stop_reason_est,
            "output_tokens": q_output_tokens,
        }
        _s["format_error"] = _is_format_error(_s)
        _sample_data.append(_s)

        # Stream-append wrong items to JSONL immediately (real-time diagnostics)
        if not _s["correct"]:
            _append_error_line(errors_path, _s)

        done = i + 1
        elapsed = time.time() - t_start
        speed = (done - start_idx) / elapsed if elapsed > 0 else 0
        remaining = (n - done) / speed if speed > 0 else 0
        score_so_far = sum(fmt["compare"](p, r) for p, r in zip(predictions, references))
        acc_so_far = score_so_far / done

        bar_w = 25
        filled = int(bar_w * done / n)
        bar = "█" * filled + "░" * (bar_w - filled)
        eta_str = f"{int(remaining//60)}m{int(remaining%60):02d}s" if speed > 0 else "--"
        print(f"\r  [{bar}] {done}/{n}  acc={acc_so_far:.1%}  ETA {eta_str}   ", end="", flush=True)

        if done % 50 == 0:
            checkpoint_file.write_text(json.dumps({
                "predictions": predictions,
                "references":  references,
                "sample_data": _sample_data,
            }))
            # Overwrite JSONL from sample_data to remove any duplication
            # that could arise if the process is killed and resumed
            _write_errors_jsonl(errors_path, _sample_data)

    print()

    result = bench.score(predictions, references)
    elapsed_total = time.time() - t_start
    n_answered = result["n"]

    # Build config block from model's self-reported configuration
    system_prompt_hash = hashlib.sha256(fmt["system_prompt"].encode()).hexdigest()[:8]
    config_block = {
        **model.config,
        "answer_format":     fmt_name,
        "system_prompt_hash": system_prompt_hash,
    }

    # Build usage block from accumulated per-question API usage
    usage_block: dict | None = None
    if _usage_totals["prompt_tokens"] > 0 or _usage_totals["completion_tokens"] > 0:
        usage_block = {
            "total_prompt_tokens":     _usage_totals["prompt_tokens"],
            "total_completion_tokens": _usage_totals["completion_tokens"],
        }
        if _has_thinking_tokens:
            usage_block["total_thinking_tokens"]   = _usage_totals["thinking_tokens"]
            usage_block["avg_thinking_tokens_per_q"] = round(
                _usage_totals["thinking_tokens"] / n_answered, 1
            ) if n_answered else 0

    # Build per-task accuracy breakdown (only when ≥2 distinct non-empty subjects)
    from collections import defaultdict as _defaultdict
    _task_buckets: dict = _defaultdict(lambda: {"n": 0, "correct": 0})
    for _s in _sample_data:
        _subj = (_s.get("subject") or "").strip()
        if _subj:
            _task_buckets[_subj]["n"] += 1
            if _s["correct"]:
                _task_buckets[_subj]["correct"] += 1
    task_accuracy: dict | None = None
    if len(_task_buckets) >= 2:
        task_accuracy = {
            k: {"n": v["n"], "correct": v["correct"],
                "accuracy": round(v["correct"] / v["n"], 4) if v["n"] else 0.0}
            for k, v in sorted(_task_buckets.items())
        }

    # Error count summary
    _n_fmt  = sum(1 for s in _sample_data if s.get("format_error"))
    _n_wrong = sum(1 for s in _sample_data if not s["correct"] and not s.get("format_error"))
    _n_ok    = sum(1 for s in _sample_data if s["correct"])
    error_counts = {"format_error": _n_fmt, "wrong": _n_wrong, "correct": _n_ok}

    cell: dict = {
        "model": model_name,
        "benchmark": bench_name,
        "mode": mode,
        "accuracy": result["accuracy"],
        "n": n_answered,
        "seed": lite_seed(meta) if mode == "lite" else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(elapsed_total),
        "cost_est_usd": round(estimate(model_name, n_answered)["cost_usd"], 4),
        "config": config_block,
    }
    if usage_block:
        cell["usage"] = usage_block
    if task_accuracy:
        cell["task_accuracy"] = task_accuracy
    cell["error_counts"] = error_counts

    print(f"  ✓ accuracy: {cell['accuracy']:.1%}  ({cell['n']}q)  "
          f"took {int(elapsed_total//60)}m{int(elapsed_total%60):02d}s  ~${cell['cost_est_usd']:.3f}")

    cell_path.write_text(json.dumps(cell, indent=2))
    print(f"  Saved → {cell_path}")
    # Write final JSONL (authoritative, overwrites any streaming state)
    _write_errors_jsonl(errors_path, _sample_data)
    print(f"  Errors → {errors_path}  ({error_counts['wrong']} wrong, {error_counts['format_error']} fmt-err)")

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
        abc_cells  = [c for c in all_cells if c["benchmark"] in ABC_WEIGHTS]
        mm_cells   = [c for c in all_cells if c["benchmark"] in MULTIMODAL_WEIGHTS]
        ws_text = weighted_score(text_cells, WEIGHTS)
        ws_abc  = weighted_score(abc_cells,  ABC_WEIGHTS)
        ws_mm   = weighted_score(mm_cells, MULTIMODAL_WEIGHTS)

        print(f"\n{'═'*62}")
        print(f"  {model_name}  [{args.mode}]")
        if ws_text is not None:
            print(f"  Text weighted score:  {ws_text:.1%}")
        if ws_abc is not None:
            print(f"  ABC weighted score:   {ws_abc:.1%}")
        if ws_mm is not None:
            print(f"  Modal weighted score: {ws_mm:.1%}")
        print(f"  Total time: {int(elapsed//60)}m{int(elapsed%60):02d}s")
        print(f"{'═'*62}")


if __name__ == "__main__":
    main()
