#!/usr/bin/env python3
"""
MuTheoryEval — unified runner

Usage:
    python run.py --model gpt-4o --benchmark music_theory_bench
    python run.py --model gpt-4o --benchmark all
    python run.py --model gpt-4o,claude-3-5-sonnet --benchmark music_theory_bench
    python run.py --list-models
    python run.py --list-benchmarks
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from benchmarks import REGISTRY as BENCH_REGISTRY, WEIGHTS
from models import REGISTRY as MODEL_REGISTRY

SYSTEM_PROMPT = (
    "You are a music theory expert. Answer the following multiple-choice question. "
    "Respond with only the letter of the correct answer (A, B, C, or D)."
)


def run_benchmark(model_name: str, bench_name: str) -> dict:
    print(f"\n=== {model_name} × {bench_name} ===")

    bench = BENCH_REGISTRY[bench_name]
    model = MODEL_REGISTRY[model_name]()

    try:
        dataset = bench.load()
    except NotImplementedError as e:
        print(f"  SKIP: {e}")
        return {"skipped": True, "reason": str(e)}

    predictions, references = [], []
    for i, item in enumerate(dataset):
        prompt = bench.format_prompt(item)
        raw = model.complete(prompt, system=SYSTEM_PROMPT)
        pred = model.extract_choice(raw)
        ref = item.get("answer", item.get("label", "")).strip().upper()
        predictions.append(pred)
        references.append(ref)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(dataset)}...")

    result = bench.score(predictions, references)
    result["model"] = model_name
    result["benchmark"] = bench_name
    result["timestamp"] = datetime.utcnow().isoformat()
    print(f"  accuracy: {result['accuracy']:.1%} ({result['n']} questions)")
    return result


def weighted_score(results: list[dict]) -> float:
    total_weight = sum(WEIGHTS[r["benchmark"]] for r in results if not r.get("skipped"))
    if total_weight == 0:
        return 0.0
    return sum(r["accuracy"] * WEIGHTS[r["benchmark"]] for r in results if not r.get("skipped")) / total_weight


def main():
    parser = argparse.ArgumentParser(description="MuTheoryEval runner")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name(s), comma-separated or 'all'")
    parser.add_argument("--benchmark", default="music_theory_bench", help="Benchmark name(s), comma-separated or 'all'")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--list-benchmarks", action="store_true")
    parser.add_argument("--out", default="results", help="Output directory")
    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for m in MODEL_REGISTRY:
            print(f"  {m}")
        return

    if args.list_benchmarks:
        print("Available benchmarks:")
        for b, mod in BENCH_REGISTRY.items():
            meta = mod.METADATA
            print(f"  {b} — {meta['n_questions']} questions, weight={meta['weight']}")
        return

    models = list(MODEL_REGISTRY.keys()) if args.model == "all" else args.model.split(",")
    benchmarks = list(BENCH_REGISTRY.keys()) if args.benchmark == "all" else args.benchmark.split(",")

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    for model_name in models:
        model_name = model_name.strip()
        if model_name not in MODEL_REGISTRY:
            print(f"Unknown model: {model_name}. Use --list-models.")
            continue

        all_results = []
        for bench_name in benchmarks:
            bench_name = bench_name.strip()
            if bench_name not in BENCH_REGISTRY:
                print(f"Unknown benchmark: {bench_name}. Use --list-benchmarks.")
                continue
            result = run_benchmark(model_name, bench_name)
            all_results.append(result)

        ws = weighted_score(all_results)
        print(f"\n>>> {model_name} weighted score: {ws:.1%}")

        out_file = out_dir / f"{model_name.replace('/', '_')}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_file, "w") as f:
            json.dump({"model": model_name, "weighted_score": ws, "results": all_results}, f, indent=2)
        print(f"    saved → {out_file}")


if __name__ == "__main__":
    main()
