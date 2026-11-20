"""Summarize completed runs and quantify paired prediction differences."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from finish_glove_study import save, scores


def macro_f1_from_counts(matrix):
    numerator = 2 * np.diagonal(matrix, axis1=-2, axis2=-1)
    denominator = matrix.sum(axis=-2) + matrix.sum(axis=-1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float),
                     where=denominator > 0).mean(axis=-1)


def paired_bootstrap(predictions, samples=10000):
    # The eight joint label/prediction cells preserve the pairing under resampling.
    cells = 4 * predictions["label"].to_numpy() + 2 * predictions["meanpool"].to_numpy() + predictions["gru"].to_numpy()
    observed = np.bincount(cells, minlength=8)
    sampled = np.random.default_rng(42).multinomial(len(cells), observed / len(cells), size=samples)
    matrices = sampled.reshape(-1, 2, 2, 2)
    differences = macro_f1_from_counts(matrices.sum(axis=-2)) - macro_f1_from_counts(matrices.sum(axis=-1))
    return np.quantile(differences, (.025, .975)).tolist()



def compare_stronger_baseline(predictions, directory):
    with np.load(directory / "trainable_meanpool_validation_predictions.npz") as baseline:
        if not np.array_equal(baseline["project1_row"], predictions["project1_row"]):
            raise ValueError("Row IDs differ across prediction files.")
        if not np.array_equal(baseline["label"], predictions["label"]):
            raise ValueError("Labels differ across prediction files.")
        predictions = predictions.assign(meanpool=baseline["meanpool"])
    old = json.loads((directory / "error_analysis.json").read_text())
    lower, upper = old["training_length_tertiles"]
    lengths = predictions["tokens"]
    oov_rate = (lengths - predictions["glove_hits"]) / lengths.clip(lower=1)
    masks = {
        "all": np.ones(len(predictions), dtype=bool),
        "short": lengths <= lower,
        "medium": (lengths > lower) & (lengths <= upper),
        "long": lengths > upper,
        "negation_present": predictions["negation"] == 1,
        "negation_absent": predictions["negation"] == 0,
        "contrast_present": predictions["contrast"] == 1,
        "contrast_absent": predictions["contrast"] == 0,
        "truncated": predictions["truncated"] == 1,
        "not_truncated": predictions["truncated"] == 0,
        "above_training_median_oov_rate": oov_rate > old["training_median_oov_rate"],
        "at_or_below_training_median_oov_rate": oov_rate <= old["training_median_oov_rate"],
    }
    groups = {}
    for name, mask in masks.items():
        group = predictions.loc[mask]
        if group.empty:
            continue
        y, m, g = (group[column].to_numpy() for column in ("label", "meanpool", "gru"))
        groups[name] = {"n": len(group), "meanpool": scores(y, m), "gru": scores(y, g),
                        "only_meanpool_correct": int(((m == y) & (g != y)).sum()),
                        "only_gru_correct": int(((m != y) & (g == y)).sum())}
    result = {"gru_minus_trainable_meanpool_ci95": paired_bootstrap(predictions), "groups": groups}
    save(directory / "trainable_baseline_error_analysis.json", result)
    predictions.to_csv(directory / "trainable_baseline_predictions.csv", index=False)
    return result


def draw_curves(runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    colors = {"frozen_full": "#186c70", "tuned_full": "#a75324", "frozen_half": "#72539e"}
    labels = {"frozen_full": "Frozen · full", "tuned_full": "Fine-tuned · full", "frozen_half": "Frozen · half"}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharey=True)
    fig.subplots_adjust(left=.085, right=.97, bottom=.23, top=.72, wspace=.17)
    fig.text(.085, .92, "BiGRU learning curves", fontsize=20, weight="bold")
    fig.text(.085, .85, "Three seeds per condition · dots mark the inner-validation checkpoints",
             fontsize=10, color="#475569")
    for ax, (metric, title) in zip(axes, (
        ("online_train_macro_f1", "Training (online per batch)"),
        ("inner_macro_f1", "Inner validation"),
    )):
        for run in runs:
            key = run["setup"]
            rows = run["history"]
            ax.plot([row["epoch"] for row in rows], [row[metric] for row in rows],
                    color=colors[key], alpha=.62, lw=1.7,
                    label=labels[key] if run["seed"] == 42 else None)
            if metric == "inner_macro_f1":
                best = run["best"]
                ax.scatter(best["epoch"], best[metric], color=colors[key], s=24, zorder=3)
        ax.set_title(title, loc="left", fontsize=12, weight="semibold")
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="y", color="#e2e8f0")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Macro-F1")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(.52, .03), ncol=3, frameon=False)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("final_glove_study"))
    parser.add_argument("--project1", type=Path, required=True, help="Project 1 experiment.py")
    args = parser.parse_args()
    data = json.loads((args.study / "recurrent_results.json").read_text())
    if "final_test" not in data:
        raise ValueError("Recurrent study has not completed.")
    predictions = pd.read_csv(args.study / "validation_predictions.csv")
    assert predictions["project1_row"].is_unique and len(predictions) == data["project1_validation_reviews"]
    setup = data["final_choice"]["setup"]
    runs = {(r["setup"], r["seed"]): r["project1_validation"]["macro_f1"] for r in data["runs"]}
    seeds = data["seed_list"]
    deltas = {}
    for name, left, right in (
        ("tuned_minus_frozen_full", "tuned_full", "frozen_full"),
        ("full_minus_half_frozen", "frozen_full", "frozen_half"),
    ):
        values = [runs[left, s] - runs[right, s] for s in seeds]
        deltas[name] = {"by_seed": dict(zip(seeds, values)), "mean": float(np.mean(values)),
                        "sample_sd": float(np.std(values, ddof=1))}
    matched_path = args.study / "matched_trainable_meanpool.json"
    if matched_path.exists():
        matched = json.loads(matched_path.read_text())
        if matched["fingerprint"] != data["fingerprint"] or matched["limit"] != data["final_choice"]["token_limit"]:
            raise ValueError("Matched trainable MeanPool has different inputs.")
        mean_by_seed = {row["seed"]: row["project1_validation_macro_f1"] for row in matched["runs"]}
        values = [runs["tuned_full", seed] - mean_by_seed[seed] for seed in seeds]
        deltas["tuned_gru_minus_tuned_meanpool"] = {
            "by_seed": dict(zip(seeds, values)), "mean": float(np.mean(values)),
            "sample_sd": float(np.std(values, ddof=1)),
        }
    pairs = (predictions["gru"] == predictions["label"], predictions["meanpool"] == predictions["label"])
    baseline_path = args.project1.parent / "results/baseline/validation_scores.csv"
    prior = pd.read_csv(baseline_path)
    prior = prior[prior["model"] != "kmeans"].sort_values("score", ascending=False)
    summary = {
        "selected_gru_validation_ci_scope": "bootstrap over reviews for the fixed seed 42 model pair",
        "selected_gru_minus_meanpool_macro_f1_ci95": paired_bootstrap(predictions),
        "only_gru_correct": int((pairs[0] & ~pairs[1]).sum()),
        "only_meanpool_correct": int((pairs[1] & ~pairs[0]).sum()),
        "paired_runs": deltas,
        "selected_setup": setup,
        "project1_historical_validation": {
            "best_supervised": prior.iloc[0].to_dict(),
            "best_supervised_without_bigrams": prior[prior["representation"] != "bigrams"].iloc[0].to_dict(),
        },
    }
    if (args.study / "trainable_meanpool_validation_predictions.npz").exists():
        comparison = compare_stronger_baseline(predictions, args.study)
        summary["stronger_baseline_fixed_seed"] = {
            "ci95": comparison["gru_minus_trainable_meanpool_ci95"], **comparison["groups"]["all"],
        }
    save(args.study / "paired_summary.json", summary)
    draw_curves(data["runs"], args.study / "gru_learning_curves.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
