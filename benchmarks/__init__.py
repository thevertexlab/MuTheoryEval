from . import music_theory_bench, ziqi_eval, abc_eval

REGISTRY = {
    "music_theory_bench": music_theory_bench,
    "ziqi_eval": ziqi_eval,
    "abc_eval": abc_eval,
}

WEIGHTS = {k: m.METADATA["weight"] for k, m in REGISTRY.items()}
