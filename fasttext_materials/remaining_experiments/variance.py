#!/usr/bin/env python3
"""Summarize individually completed seeds; never launches training."""
import argparse
import os
import json
from pathlib import Path

import workflow_config as cfg
os.environ.setdefault("XDG_CACHE_HOME", str(cfg.CACHE))
os.environ.setdefault("MPLCONFIGDIR", str(cfg.CACHE / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bridge
from data_access import pooled_cache, read_json, reserve_output
import training


def locate(group, seed):
    if group == "F-base" and seed == 42:
        return cfg.BASELINE
    if group == "F-mean" and seed == 42:
        return cfg.MEAN_BASELINE
    return cfg.result_directory(group, seed)


def load_record(group, seed):
    path = locate(group, seed)
    if not (path / "metrics.json").is_file():
        return None
    metrics = read_json(path / "metrics.json")
    if group == "F-base" and seed == 42:
        original = read_json(path / "run_config.json")
        spec = cfg.specification("F-base", seed)
        context = {"split": original["split"], "cache_identity": original["sequence_cache"]["manifest"]["identity"],
                   "cache_files_sha256": original["sequence_cache"]["manifest"]["files_sha256"],
                   "selected_training_review_ids_sha256": original["split"]["training_indices_sha256_le_int64"]}
    elif group == "F-mean" and seed == 42:
        original = read_json(path.parent / "run_config.json")
        spec = cfg.specification("F-mean", seed)
        _, _, pooled_path = pooled_cache()
        context = {"split": original["split"], "pooled_feature_identity": original["features"]["identity"],
                   "pooled_file_sha256": {name: bridge.legacy.phase1.file_sha256(pooled_path / f"{name}_mean.npy")
                                          for name in ("train", "validation")}}
    else:
        context = read_json(path / "run_config.json")
        spec = context["specification"]
    if spec["seed"] != seed or spec["experiment"] != group:
        raise ValueError(f"Unexpected run metadata at {path}")
    return {"group": group, "seed": seed, "source": str(path.resolve()),
            "validation_accuracy": float(metrics["validation"]["accuracy"]),
            "validation_macro_f1": float(metrics["validation"]["macro_f1"]),
            "best_epoch": metrics.get("best_epoch"), "spec": spec, "context": context}


def signature(record):
    spec = {k: v for k, v in record["spec"].items() if k not in ("seed", "description")}
    ctx = record["context"]
    relevant = {key: ctx.get(key) for key in ("split", "cache_identity", "cache_files_sha256",
                                               "selected_training_review_ids_sha256", "pooled_feature_identity", "pooled_file_sha256")}
    return json.dumps({"spec": spec, "context": relevant}, sort_keys=True)


def summarize(group, records):
    if records and len({signature(record) for record in records}) != 1:
        raise ValueError(f"{group} runs differ in split, vocabulary, preprocessing, initialization or settings; refusing to treat them as repeats.")
    values = [r["validation_macro_f1"] for r in records]
    return {"group": group, "seeds": [r["seed"] for r in records], "n": len(values),
            "validation_macro_f1_mean": float(np.mean(values)) if values else None,
            "validation_macro_f1_sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None}


def aggregate(alternative, include_mean=False):
    groups = ["F-base", alternative] + (["F-mean"] if include_mean else [])
    records = {group: [record for seed in cfg.SEEDS if (record := load_record(group, seed)) is not None] for group in groups}
    summaries = [summarize(group, records[group]) for group in groups]
    if records["F-base"] and records[alternative]:
        base_context = records["F-base"][0]["context"]
        alt_context = records[alternative][0]["context"]
        for key in ("split", "cache_identity", "cache_files_sha256"):
            if base_context[key] != alt_context[key]:
                raise ValueError(f"Alternative and baseline are not paired on the same verified data/cache: {key}")
    baseline = {row["seed"]: row for row in records["F-base"]}
    pairs = [{"seed": row["seed"], "baseline_macro_f1": baseline[row["seed"]]["validation_macro_f1"],
              "alternative_macro_f1": row["validation_macro_f1"],
              "alternative_minus_baseline": row["validation_macro_f1"] - baseline[row["seed"]]["validation_macro_f1"]}
             for row in records[alternative] if row["seed"] in baseline]
    compact = [{k: row[k] for k in ("group", "seed", "source", "validation_accuracy", "validation_macro_f1", "best_epoch")}
               for group in groups for row in records[group]]
    differences = [row["alternative_minus_baseline"] for row in pairs]
    return {"individual_runs": compact, "summary": summaries, "paired_differences": pairs,
            "paired_difference_summary": {"n": len(differences),
                "mean": float(np.mean(differences)) if differences else None,
                "sample_std": float(np.std(differences, ddof=1)) if len(differences) > 1 else None},
            "missing": [{"group": g, "seed": seed} for g in groups for seed in cfg.SEEDS
                        if not any(row["seed"] == seed for row in records[g])],
            "note": "Sample standard deviation (ddof=1); missing runs are not invented. "
                    "Pairs share validation and cache; the explicitly selected design change may alter training data or model settings. "
                    "F-mean uses full-review mean pooling and is not an LSTM repeat. Its seed-42 original "
                    "did not record array checksums, so its historical cache bytes cannot be independently verified; "
                    "the current arrays are hashed for comparisons with later runs."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alternative", required=True, choices=("A2", "A3", "B1", "C1", "C3", "D1", "D3", "E2", "E3"))
    parser.add_argument("--include-mean", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"Output exists: {args.output}")
    result = aggregate(args.alternative, args.include_mean)
    output = reserve_output(args.output)
    training.write_json(output / "variance.json", result)
    pd.DataFrame(result["individual_runs"]).to_csv(output / "individual_runs.csv", index=False)
    pd.DataFrame(result["summary"]).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(result["paired_differences"], columns=["seed", "baseline_macro_f1", "alternative_macro_f1", "alternative_minus_baseline"]).to_csv(output / "paired_differences.csv", index=False)
    shown = [row for row in result["summary"] if row["n"]]
    if shown:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar([r["group"] for r in shown], [r["validation_macro_f1_mean"] for r in shown],
                      yerr=[r["validation_macro_f1_sample_std"] or 0 for r in shown], capsize=4)
        ax.bar_label(bars, labels=[f"{r['validation_macro_f1_mean']:.4f}\nn={r['n']}" for r in shown], padding=3)
        ax.set(ylabel="Validation macro-F1", ylim=(0, 1.1), title="Completed runs: mean and sample standard deviation")
        fig.tight_layout(); fig.savefig(output / "variance.png", dpi=160); plt.close(fig)
    print(json.dumps(result["summary"], indent=2)); print(f"Saved {output}")


if __name__ == "__main__":
    main()
