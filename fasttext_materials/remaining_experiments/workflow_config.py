"""Paths and immutable experiment specifications for individual runs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSIGNMENT = ROOT.parent
RESULTS = ROOT / "results"
CACHE = ROOT / ".cache"
BASELINE = ASSIGNMENT / "phase3/results/lstm-s42-h64-l512-b32-lr0.001-e30"
BASELINE_256 = ASSIGNMENT / "phase3/results/lstm-s42-h64-l256-b32-lr0.001-e30"
MEAN_BASELINE = ASSIGNMENT / "phase2/results/seed-42_epochs-30/lr-mean"
PROJECT1 = ASSIGNMENT.parent.parent / "NLP1-Assignment1/project_starter/results/baseline"
MODEL = ASSIGNMENT / "models/cc.en.300.bin"
DATA = ASSIGNMENT / "data"
SEEDS = (42, 123, 2026)


def manifest():
    return json.loads((ROOT / "experiments.json").read_text(encoding="utf-8"))


def specification(experiment, seed=42, alternative=None):
    doc = manifest()
    if experiment == "F-alternative":
        if alternative not in {"A2", "A3", "B1", "C1", "C3", "D1", "D3", "E2", "E3"}:
            raise ValueError("F-alternative requires --alternative A2/A3/B1/C1/C3/D1/D3/E2/E3.")
        experiment = alternative
    elif alternative is not None:
        raise ValueError("--alternative is only valid for F-alternative.")
    if experiment not in doc["experiments"]:
        raise ValueError(f"Unknown experiment {experiment}.")
    if seed not in SEEDS:
        raise ValueError(f"Seed must be one of {SEEDS}.")
    row = dict(doc["experiments"][experiment])
    if row.get("baseline_alias") and seed != 42:
        raise ValueError("Baseline aliases refer to the existing seed-42 run; use F-base for other seeds.")
    return {**doc["common"], **row, "experiment": experiment, "seed": seed}


def result_directory(experiment, seed):
    return RESULTS / experiment / f"seed-{seed}"
