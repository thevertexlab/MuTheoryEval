"""
SSMR-Bench — Synthetic Sheet Music Reasoning Benchmark (textual modality)
Source: arXiv:2509.04059 (Wang et al., USTC / Shanghai AI Lab, 2025)
Dataset: huggingface.co/datasets/Sylence/SSMR-Bench
GitHub: github.com/Linzwcs/AutoMusicTheoryQA

1,600 textual MCQ questions across 9 task types, programmatically generated
from ABC notation using music theory rules:
  BarLinePlacementQuestion     183q
  ChordIdentificationQuestion   44q
  ChordKeyRootIdentification   200q
  ChordsCompletionQuestion     156q
  IntervalNumberQuestion       199q
  NoteCompletionByInterval     201q
  ScaleIdentificationFromAbc   352q
  ScaleSelectionQuestion        48q
  TimeSignatureQuestion        217q

Input: ABC notation snippet + music theory question → 4-option MCQ
Answer format: single letter A–D

Modality: abc — pure text, ABC notation as LLM input
Weight in ABC aggregate: 1.0 (sole benchmark in modality)
"""

import random

METADATA = {
    "name":              "SSMR-Bench",
    "source":            "arXiv:2509.04059",
    "hf_dataset":        "Sylence/SSMR-Bench",
    "n_questions":       1600,
    "lite_n":            200,
    "lite_seed":         42,
    "answer_format":     "MCQ",
    "modality":          "abc",
    "weight":            1.0,
}


def load(split="Test_textual", sample=200, seed=42) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("Sylence/SSMR-Bench", split=split)

    rng = random.Random(seed)
    raw = list(ds)
    rng.shuffle(raw)
    if sample and sample < len(raw):
        raw = raw[:sample]

    items = []
    for row in raw:
        correct = row["correct_answer"]
        options = [correct, row["incorrect_answer1"], row["incorrect_answer2"], row["incorrect_answer3"]]
        rng.shuffle(options)
        correct_letter = chr(65 + options.index(correct))
        items.append({
            "subject":         row["class_name"],
            "question":        row["question"],
            "abc_context":     row["abc_context"] or "",
            "category":        row.get("category", ""),
            "difficulty":      row.get("difficulty", ""),
            "_options":        options,
            "_correct_letter": correct_letter,
        })
    return items


def format_prompt(item: dict) -> str:
    # abc_context stores literal \n sequences from the HF dataset — decode to real newlines
    abc = item["abc_context"].strip().replace("\\n", "\n")
    opts = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(item["_options"]))
    abc_block = f"ABC Notation:\n{abc}\n\n" if abc else ""
    return f"{abc_block}{item['question']}\n\n{opts}"


def get_answer(item: dict) -> str:
    return item["_correct_letter"]


def score(predictions: list[str], references: list[str]) -> dict:
    correct = sum(p.strip().upper() == r.strip().upper() for p, r in zip(predictions, references))
    return {"accuracy": correct / len(references), "n": len(references)}
