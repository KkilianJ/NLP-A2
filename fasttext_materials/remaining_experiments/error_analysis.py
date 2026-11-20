#!/usr/bin/env python3
"""Compare two saved models on aligned validation reviews; no training."""
import argparse
import os
from pathlib import Path

import workflow_config as cfg
os.environ.setdefault("XDG_CACHE_HOME", str(cfg.CACHE))
os.environ.setdefault("MPLCONFIGDIR", str(cfg.CACHE / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch

from data_access import read_json, reserve_output
from inference import gap_metrics, predict_validation
import training
from validation_features import validation_features


def resolve_model(value):
    aliases = {"lstm512": cfg.BASELINE, "lstm256": cfg.BASELINE_256, "lr-mean": cfg.MEAN_BASELINE,
               "A1": cfg.BASELINE, "B2": cfg.BASELINE, "C2": cfg.BASELINE, "D2": cfg.BASELINE,
               "E1": cfg.BASELINE, "F-base": cfg.BASELINE, "F-mean": cfg.MEAN_BASELINE}
    if value in aliases:
        return aliases[value]
    if value in cfg.manifest()["experiments"]:
        return cfg.result_directory(value, 42)
    return Path(value)


def subgroup_metrics(frame):
    """A macro-F1 is omitted for single-true-class groups; counts remain visible."""
    frame = frame.copy()
    frame["length_bin"] = pd.cut(frame.token_length, [-1, 128, 256, 512, np.inf],
                                 labels=["0–128", "129–256", "257–512", ">512"])
    oov_bins = []
    for field in ("oov_fraction", "oov_fraction_256", "oov_fraction_512"):
        if field in frame:
            name = field.replace("fraction", "bin")
            frame[name] = pd.cut(pd.to_numeric(frame[field], errors="coerce"), [-0.000001, 0.01, 0.05, 1.0],
                                 labels=["≤1%", ">1–5%", ">5%"])
            oov_bins.append(name)
    for name in ("left", "right"):
        frame[f"{name}_confidence_bin"] = pd.cut(frame[f"{name}_confidence"], [0.499999, 0.6, 0.8, 1.000001],
                                                  labels=["0.50–0.60", "0.60–0.80", "0.80–1.00"])
    fields = ("length_bin", "negation_indicator", *oov_bins, "truncated_256", "truncated_512",
              "left_confidence_bin", "right_confidence_bin")
    rows = []
    for field in fields:
        for value, part in frame.groupby(field, observed=True, dropna=False):
            if not len(part):
                continue
            y = part.label.to_numpy(dtype=int)
            supports = np.bincount(y, minlength=2)
            row = {"grouping": field, "group": str(value), "n": len(part), "negative_support": int(supports[0]),
                   "positive_support": int(supports[1]), "small_group_under_30": len(part) < 30,
                   "macro_f1_note": "both classes present" if np.all(supports) else "omitted: only one true class"}
            for side in ("left", "right"):
                pred = part[f"{side}_prediction"].to_numpy(dtype=int)
                row[f"{side}_accuracy"] = float(accuracy_score(y, pred))
                row[f"{side}_macro_f1"] = float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)) if np.all(supports) else None
            row["accuracy_right_minus_left"] = row["right_accuracy"] - row["left_accuracy"]
            row["macro_f1_right_minus_left"] = (row["right_macro_f1"] - row["left_macro_f1"]
                                                   if row["left_macro_f1"] is not None and row["right_macro_f1"] is not None else None)
            row["left_only_correct"] = int((part.left_correct & ~part.right_correct).sum())
            row["right_only_correct"] = int((~part.left_correct & part.right_correct).sum())
            rows.append(row)
    return rows


def pair_predictions(left, right, features):
    if not np.array_equal(left["ids"], right["ids"]) or not np.array_equal(left["labels"], right["labels"]):
        raise ValueError("Models do not have identical ordered validation IDs and labels.")
    base = pd.DataFrame({"review_id": left["ids"], "label": left["labels"]})
    if features.review_id.duplicated().any():
        raise ValueError("Duplicate review IDs in validation features.")
    frame = base.merge(features, on="review_id", how="left", validate="one_to_one", suffixes=("", "_feature"))
    if frame.token_length.isna().any() or not np.array_equal(frame.label, frame.label_feature):
        raise ValueError("Features do not align with the models' validation IDs and labels.")
    frame = frame.drop(columns="label_feature")
    for side, result in (("left", left), ("right", right)):
        p = result["probabilities"]
        frame[f"{side}_probability_positive"] = p
        frame[f"{side}_prediction"] = (p >= 0.5).astype(int)
        frame[f"{side}_confidence"] = np.maximum(p, 1 - p)
        frame[f"{side}_correct"] = frame[f"{side}_prediction"] == frame.label
        frame[f"{side}_error_type"] = np.select(
            [(frame.label == 0) & (frame[f"{side}_prediction"] == 1),
             (frame.label == 1) & (frame[f"{side}_prediction"] == 0)],
            ["false_positive", "false_negative"], default="correct")
    return frame


def save_plots(groups, frame, output):
    by_length = pd.DataFrame([r for r in groups if r["grouping"] == "length_bin"])
    if not by_length.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5)); x = np.arange(len(by_length)); width = .36
        ax.bar(x-width/2, by_length.left_accuracy, width, label="Left")
        ax.bar(x+width/2, by_length.right_accuracy, width, label="Right")
        ax.set(xticks=x, xticklabels=[f"{r.group}\nn={r.n}" for r in by_length.itertuples()],
               xlabel="Original regex token length", ylabel="Validation accuracy", ylim=(0, 1.05),
               title="Paired accuracy by review length")
        ax.legend(); fig.tight_layout(); fig.savefig(output / "accuracy_by_length.png", dpi=160); plt.close(fig)
    counts = [int((frame.left_correct & frame.right_correct).sum()), int((frame.left_correct & ~frame.right_correct).sum()),
              int((~frame.left_correct & frame.right_correct).sum()), int((~frame.left_correct & ~frame.right_correct).sum())]
    fig, ax = plt.subplots(figsize=(7, 4)); bars = ax.bar(["Both correct", "Left only", "Right only", "Both wrong"], counts)
    ax.bar_label(bars); ax.set(ylabel="Reviews", title="Paired validation outcomes")
    fig.tight_layout(); fig.savefig(output / "paired_outcomes.png", dpi=160); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", default="lstm512", help="lstm512, lstm256, lr-mean, experiment ID, or run directory")
    parser.add_argument("--right", default="lr-mean")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=cfg.DATA)
    parser.add_argument("--model", type=Path, default=cfg.MODEL)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--variance-summary", type=Path, help="optional variance.json from variance.py")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Predicting validation reviews on {device}; no training.", flush=True)
    left = predict_validation(resolve_model(args.left), device)
    right = predict_validation(resolve_model(args.right), device)
    features, feature_meta = validation_features(args.data_dir, args.model, args.feature_cache)
    frame = pair_predictions(left, right, features)
    groups = subgroup_metrics(frame)
    summary = {"left": {"source": left["run_dir"], "kind": left["kind"], "saved_gaps": gap_metrics(left["metrics"])},
               "right": {"source": right["run_dir"], "kind": right["kind"], "saved_gaps": gap_metrics(right["metrics"])},
               "review_count": len(frame), "feature_provenance": feature_meta,
               "false_positive_and_negative_counts": {side: {"false_positive": int(((frame.label == 0) & (frame[f'{side}_prediction'] == 1)).sum()),
                                                               "false_negative": int(((frame.label == 1) & (frame[f'{side}_prediction'] == 0)).sum())}
                                                      for side in ("left", "right")},
               "variance": read_json(args.variance_summary) if args.variance_summary else None,
               "note": "Excerpts are evidence, not explanations. Negation is a surface heuristic; OOV is direct-vocabulary absence. "
                       "Single-class macro-F1 is omitted and small subgroups are marked. Results use validation only."}
    for side in ("left", "right"):
        summary[side]["recomputed_validation_accuracy"] = float(accuracy_score(frame.label, frame[f"{side}_prediction"]))
        summary[side]["recomputed_validation_macro_f1"] = float(f1_score(frame.label, frame[f"{side}_prediction"], labels=[0, 1], average="macro", zero_division=0))
        saved = summary[side]["saved_gaps"]
        delta = summary[side]["recomputed_validation_macro_f1"] - saved["validation_macro_f1"]
        summary[side]["recomputed_minus_saved_macro_f1"] = delta
        summary[side]["checkpoint_score_matches_saved_within_1e-6"] = abs(delta) <= 1e-6
    output = reserve_output(args.output)
    frame.to_csv(output / "per_example.csv", index=False)
    frame[frame.left_correct != frame.right_correct].to_csv(output / "disagreements.csv", index=False)
    frame[(~frame.left_correct) | (~frame.right_correct)].to_csv(output / "errors.csv", index=False)
    pd.DataFrame(groups).to_csv(output / "subgroups.csv", index=False)
    training.write_json(output / "summary.json", summary)
    save_plots(groups, frame, output)
    print(f"Saved paired validation analysis to {output}. No training or test evaluation.")


if __name__ == "__main__":
    main()
