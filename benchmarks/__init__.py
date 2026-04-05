from . import music_theory_bench, ziqi_eval, abc_eval
from . import ssmr_bench, msu_bench, wildscore

# Text-only benchmarks — runnable with any LLM
TEXT_REGISTRY = {
    "music_theory_bench": music_theory_bench,  # ✅ dataset available
    "ziqi_eval":          ziqi_eval,           # ✅ dataset available
    "abc_eval":           abc_eval,            # ⚠️  dataset TBC
    "ssmr_bench":         ssmr_bench,          # ⚠️  dataset TBC
}

# VLM-only benchmarks — require image input (score sheets)
VLM_REGISTRY = {
    "msu_bench":  msu_bench,   # ⚠️  dataset TBC + VLM required
    "wildscore":  wildscore,   # ⚠️  dataset TBC + VLM required
}

# Default registry for run.py (text-only)
REGISTRY = TEXT_REGISTRY

WEIGHTS = {k: m.METADATA["weight"] for k, m in REGISTRY.items()}
