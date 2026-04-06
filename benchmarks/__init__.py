from . import music_theory_bench, ziqi_eval, abc_eval, ssmr_bench
from . import msu_bench, wildscore, muchomusic, cmi_bench

# Text-only benchmarks — runnable with any LLM
TEXT_REGISTRY = {
    "music_theory_bench": music_theory_bench,  # ✅ HF: m-a-p/MusicTheoryBench
    "ziqi_eval":          ziqi_eval,           # ✅ HF: MYTH-Lab/ZIQI-Eval
    # abc_eval: UNRELEASED — anonymous link expired, no public repo as of 2026-04
}

# ABC notation benchmarks — pure text input, tests symbolic music understanding
ABC_REGISTRY = {
    "ssmr_bench": ssmr_bench,  # ✅ HF: Sylence/SSMR-Bench — 1600q ABC MCQ, 9 task types
    # ssmr_bench (UNRELEASED stub removed): now fully integrated
}

# VLM benchmarks — require image input (music score sheets)
VLM_REGISTRY = {
    "wildscore": wildscore,   # ✅ HF: GM77/WildScore — needs: python scripts/download_wildscore.py
    # msu_bench: dataset status unclear — kept as stub
}

# Audio-Language Model benchmarks — require audio input
ALM_REGISTRY = {
    "muchomusic": muchomusic,  # ✅ HF: lmms-lab/muchomusic (audio inline, ~862 MB)
    "cmi_bench":  cmi_bench,   # ✅ HF: nicolaus625/CMI-bench — needs: python scripts/download_cmibench.py
}

# Combined multimodal registry
MULTIMODAL_REGISTRY = {**VLM_REGISTRY, **ALM_REGISTRY}

# Default registry for run.py --benchmark all (text + abc)
REGISTRY = {**TEXT_REGISTRY, **ABC_REGISTRY}

# Weights: text-only (for backward-compat "Text weighted score" in run.py summary)
WEIGHTS = {k: m.METADATA["weight"] for k, m in TEXT_REGISTRY.items()}

# ABC weights (separate, for "ABC weighted score" in run.py summary)
ABC_WEIGHTS = {k: m.METADATA["weight"] for k, m in ABC_REGISTRY.items()}

MULTIMODAL_WEIGHTS = {k: m.METADATA["weight"] for k, m in MULTIMODAL_REGISTRY.items()}
