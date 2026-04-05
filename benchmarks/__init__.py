from . import music_theory_bench, ziqi_eval, abc_eval
from . import ssmr_bench, msu_bench, wildscore

# Text-only benchmarks — runnable with any LLM
TEXT_REGISTRY = {
    "music_theory_bench": music_theory_bench,  # ✅ HF: m-a-p/MusicTheoryBench
    "ziqi_eval":          ziqi_eval,           # ✅ HF: MYTH-Lab/ZIQI-Eval
    # abc_eval: UNRELEASED — anonymous link expired, no public repo as of 2026-04
    # ssmr_bench: UNRELEASED — anonymous link expired, no public repo as of 2026-04
}

# VLM-only benchmarks — require image input (score sheets)
VLM_REGISTRY = {
    "msu_bench":  msu_bench,   # ⚠️  dataset TBC + VLM required
    "wildscore":  wildscore,   # ⚠️  dataset TBC + VLM required
}

# Default registry for run.py (text-only)
REGISTRY = TEXT_REGISTRY

WEIGHTS = {k: m.METADATA["weight"] for k, m in REGISTRY.items()}
