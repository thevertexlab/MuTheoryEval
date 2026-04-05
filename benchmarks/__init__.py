from . import music_theory_bench, ziqi_eval, abc_eval, ssmr_bench
from . import msu_bench, wildscore, muchomusic, cmi_bench

# Text-only benchmarks — runnable with any LLM
TEXT_REGISTRY = {
    "music_theory_bench": music_theory_bench,  # ✅ HF: m-a-p/MusicTheoryBench
    "ziqi_eval":          ziqi_eval,           # ✅ HF: MYTH-Lab/ZIQI-Eval
    # abc_eval: UNRELEASED — anonymous link expired, no public repo as of 2026-04
    # ssmr_bench: UNRELEASED — anonymous link expired, no public repo as of 2026-04
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

# Default registry for run.py (text-only)
REGISTRY = TEXT_REGISTRY

WEIGHTS = {k: m.METADATA["weight"] for k, m in REGISTRY.items()}
MULTIMODAL_WEIGHTS = {k: m.METADATA["weight"] for k, m in MULTIMODAL_REGISTRY.items()}
