#!/usr/bin/env python3
"""
MuTheoryEval — unified runner

Usage:
    python run.py --model gemini-3.1-flash-lite --benchmark music_theory_bench
    python run.py --model gemini-3.1-flash-lite --benchmark all
    python run.py --model gemini-3.1-flash-lite,gpt-5.4-mini --benchmark all
    python run.py --estimate --model gemini-3.1-flash-lite --benchmark all
    python run.py --list-models
    python run.py --list-benchmarks

Background run (recommended for long jobs):
    See CLAUDE.md → "Running benchmarks" for the canonical workflow.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from benchmarks import REGISTRY as BENCH_REGISTRY, WEIGHTS
from benchmarks import MULTIMODAL_REGISTRY, MULTIMODAL_WEIGHTS
from models import REGISTRY as MODEL_REGISTRY

SYSTEM_PROMPT = (
    "You are a music theory expert. Answer the following multiple-choice question. "
    "Respond with only the letter of the correct answer (A, B, C, or D)."
)

# Cost & speed baseline per model (input $/MTok, output $/MTok, req/min observed)
# Flash-lite prices: $0.10 input / $0.40 output per MTok (as of 2026-04)
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
# avg tokens per question (prompt ~150 tok, answer ~3 tok)
AVG_PROMPT_TOKENS = 150
AVG_OUTPUT_TOKENS = 3


def estimate(model_name: str, n_questions: int) -> dict:
    p = MODEL_PRICING.get(model_name, {"input_per_mtok": 1.0, "output_per_mtok": 4.0, "rpm": 30})
    cost = (n_questions * AVG_PROMPT_TOKENS / 1_000_000 * p["input_per_mtok"]
          + n_questions * AVG_OUTPUT_TOKENS / 1_000_000 * p["output_per_mtok"])
    minutes = n_questions / p["rpm"]
    return {"cost_usd": cost, "minutes": minutes, "rpm": p["rpm"]}


def print_plan(model_name: str, bench_names: list[str]):
    from benchmarks import MULTIMODAL_REGISTRY
    all_reg = {**BENCH_REGISTRY, **MULTIMODAL_REGISTRY}
    print(f"\n{'─'*55}")
    print(f"  Run plan: {model_name}")
    print(f"{'─'*55}")
    total_cost, total_min = 0.0, 0.0
    for bn in bench_names:
        if bn not in all_reg:
            continue
        meta = all_reg[bn].METADATA
        if meta.get("status") == "UNRELEASED":
            print(f"  {'SKIP':<22} {bn} (UNRELEASED)")
            continue
        modality = meta.get("modality", "text")
        n = meta.get("default_sample") or meta.get("n_questions") or 0
        if not n:
            print(f"  {bn:<28}    ?q  [n/a — {modality}]")
            continue
        est = estimate(model_name, n)
        total_cost += est["cost_usd"]
        total_min += est["minutes"]
        tag = f"[{modality}]" if modality != "text" else ""
        print(f"  {bn:<28} {n:>5}q  ~${est['cost_usd']:.3f}  ~{est['minutes']:.1f}min  {tag}")
    print(f"{'─'*55}")
    print(f"  {'TOTAL':<28}        ~${total_cost:.3f}  ~{total_min:.1f}min")
    print(f"{'─'*55}\n")
    return total_cost, total_min


def run_benchmark(model_name: str, bench_name: str, out_dir: Path,
                  bench_registry: dict | None = None) -> dict:
    if bench_registry is None:
        bench_registry = BENCH_REGISTRY
    bench = bench_registry[bench_name]
    meta = bench.METADATA

    if meta.get("status") == "UNRELEASED":
        print(f"\n[SKIP] {bench_name} — UNRELEASED")
        return {"skipped": True, "reason": "UNRELEASED"}

    n_expected = meta.get("default_sample") or meta.get("n_questions") or 0
    est = estimate(model_name, n_expected) if n_expected else {"cost_usd": 0, "minutes": 0, "rpm": 30}
    print(f"\n{'─'*55}")
    print(f"  {model_name} × {bench_name}")
    modality = meta.get("modality", "text")
    modality_tag = f"  [{modality}]" if modality != "text" else ""
    n_str = f"{n_expected}q" if n_expected else "?q"
    print(f"  {n_str}  est. ${est['cost_usd']:.3f}  est. {est['minutes']:.1f}min  ({est.get('rpm', '?')} rpm){modality_tag}")
    print(f"{'─'*55}")

    model = MODEL_REGISTRY[model_name]()

    try:
        dataset = bench.load()
    except (NotImplementedError, FileNotFoundError) as e:
        print(f"  SKIP: {e}")
        return {"skipped": True, "reason": str(e)}

    n = len(dataset)
    predictions, references = [], []
    t_start = time.time()
    checkpoint_file = out_dir / f".checkpoint_{model_name.replace('/', '_')}_{bench_name}.json"

    # Resume from checkpoint if exists
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

        # Collect media (images/audio) if benchmark provides them
        media = None
        if hasattr(bench, "get_media"):
            try:
                media = bench.get_media(item)
            except FileNotFoundError as e:
                print(f"\n  SKIP: {e}")
                return {"skipped": True, "reason": str(e)}

        raw = model.complete(prompt, system=SYSTEM_PROMPT, media=media)
        pred = model.extract_choice(raw)

        # Use get_answer() for benchmarks with integer-indexed answers
        if hasattr(bench, "get_answer"):
            ref = bench.get_answer(item)
        else:
            ref = item.get("answer", item.get("label", ""))
            if isinstance(ref, str):
                ref = ref.strip().upper()
            else:
                ref = str(ref)
        predictions.append(pred)
        references.append(ref)

        done = i + 1
        elapsed = time.time() - t_start
        speed = (done - start_idx) / elapsed if elapsed > 0 else 0
        remaining = (n - done) / speed if speed > 0 else 0
        correct_so_far = sum(p == r for p, r in zip(predictions, references))
        acc_so_far = correct_so_far / done

        # Progress bar
        bar_w = 25
        filled = int(bar_w * done / n)
        bar = "█" * filled + "░" * (bar_w - filled)
        eta_str = f"{int(remaining//60)}m{int(remaining%60):02d}s" if speed > 0 else "--"
        print(f"\r  [{bar}] {done}/{n}  acc={acc_so_far:.1%}  ETA {eta_str}   ", end="", flush=True)

        # Checkpoint every 50
        if done % 50 == 0:
            checkpoint_file.write_text(json.dumps({"predictions": predictions, "references": references}))

    print()  # newline after progress bar

    result = bench.score(predictions, references)
    result["model"] = model_name
    result["benchmark"] = bench_name
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    result["elapsed_sec"] = round(time.time() - t_start)
    result["cost_est_usd"] = round(estimate(model_name, n)["cost_usd"], 4)

    elapsed_total = time.time() - t_start
    print(f"  ✓ accuracy: {result['accuracy']:.1%}  ({result['n']}q)  "
          f"took {int(elapsed_total//60)}m{int(elapsed_total%60):02d}s  ~${result['cost_est_usd']:.3f}")

    # Clean up checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    return result


def weighted_score(results: list[dict]) -> float:
    valid = [r for r in results if not r.get("skipped") and r["benchmark"] in WEIGHTS]
    total_weight = sum(WEIGHTS[r["benchmark"]] for r in valid)
    if total_weight == 0:
        return 0.0
    return sum(r["accuracy"] * WEIGHTS[r["benchmark"]] for r in valid) / total_weight


def main():
    parser = argparse.ArgumentParser(description="MuTheoryEval runner")
    parser.add_argument("--model", default="gemini-3.1-flash-lite",
                        help="Model key(s), comma-separated or 'all'")
    parser.add_argument("--benchmark", default="music_theory_bench",
                        help="Benchmark name(s), comma-separated or 'all' or 'multimodal'")
    parser.add_argument("--estimate", action="store_true",
                        help="Print cost/time estimate and exit without running")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--list-benchmarks", action="store_true")
    parser.add_argument("--multimodal", action="store_true",
                        help="Include multimodal (image/audio) benchmarks")
    parser.add_argument("--out", default="results", help="Output directory")
    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for m in MODEL_REGISTRY:
            pricing = MODEL_PRICING.get(m, {})
            rpm = pricing.get("rpm", "?")
            inp = pricing.get("input_per_mtok", "?")
            print(f"  {m:<30} ${inp}/MTok in  {rpm} rpm")
        return

    if args.list_benchmarks:
        all_benches = {**BENCH_REGISTRY, **MULTIMODAL_REGISTRY}
        print("\nAvailable benchmarks:")
        for b, mod in all_benches.items():
            meta = mod.METADATA
            status = meta.get("status", "OK")
            modality = meta.get("modality", "text")
            n = meta.get("default_sample") or meta.get("n_questions") or 0
            n_str = f"{n}q" if n else "?q"
            est = estimate("gemini-3.1-flash-lite", n) if n else {"cost_usd": 0, "minutes": 0}
            print(f"  {b:<28} {n_str:>6}  w={meta['weight']}  [{modality}]  "
                  f"~${est['cost_usd']:.3f}  ~{est['minutes']:.1f}min  [{status}]")
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

    # Build a combined lookup for resolving benchmark names
    ALL_BENCH_REGISTRY = {**BENCH_REGISTRY, **MULTIMODAL_REGISTRY}

    # Always show plan first
    for model_name in models:
        total_cost, total_min = print_plan(model_name, benchmarks)

    if args.estimate:
        return

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    for model_name in models:
        if model_name not in MODEL_REGISTRY:
            print(f"Unknown model: {model_name}. Use --list-models.")
            continue

        all_results = []
        t_model_start = time.time()
        for bench_name in benchmarks:
            if bench_name not in ALL_BENCH_REGISTRY:
                print(f"Unknown benchmark: {bench_name}. Use --list-benchmarks.")
                continue
            result = run_benchmark(model_name, bench_name, out_dir, ALL_BENCH_REGISTRY)
            all_results.append(result)

            # Save partial results after each benchmark
            partial_file = out_dir / f"{model_name.replace('/', '_')}_partial.json"
            partial_file.write_text(json.dumps({
                "model": model_name,
                "status": "in_progress",
                "results": all_results,
            }, indent=2))

        ws = weighted_score(all_results)
        elapsed = time.time() - t_model_start
        print(f"\n{'═'*55}")
        print(f"  {model_name}")
        print(f"  Weighted score: {ws:.1%}  |  Total time: {int(elapsed//60)}m{int(elapsed%60):02d}s")
        print(f"{'═'*55}")

        out_file = out_dir / f"{model_name.replace('/', '_')}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        out_file.write_text(json.dumps({
            "model": model_name,
            "weighted_score": ws,
            "elapsed_sec": round(elapsed),
            "results": all_results,
        }, indent=2))
        print(f"  Saved → {out_file}")

        # Remove partial file now that we have the final
        partial_file = out_dir / f"{model_name.replace('/', '_')}_partial.json"
        if partial_file.exists():
            partial_file.unlink()


if __name__ == "__main__":
    main()
