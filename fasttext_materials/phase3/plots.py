"""Plot saved history and the best checkpoint without loading or training a model."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")


def line_plot(history, columns, names, ylabel, title, best, path):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    selected = history[history.epoch == best]
    if len(selected) != 1:
        raise ValueError("The best epoch is missing from the saved history.")
    for col, name in zip(columns, names):
        value = float(selected[col].iloc[0])
        ax.plot(history.epoch, history[col], marker="o", markersize=3,
                label=f"{name} (best epoch: {value:.4f})")
        ax.scatter([best], [value], s=42, zorder=3)
    ax.axvline(best, color="gray", linestyle="--", linewidth=1, label=f"Best checkpoint: epoch {best}")
    ax.set(xlabel="Epoch", ylabel=ylabel, title=title)
    if ylabel in ("Accuracy", "Macro-F1"):
        ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def confusion_plot(matrix, path, title):
    counts = np.asarray(matrix, dtype=int)
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    image = ax.imshow(counts, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Reviews")
    for row in range(2):
        for col in range(2):
            color = "white" if counts[row, col] > counts.max() / 2 else "black"
            ax.text(col, row, f"{counts[row, col]:,}", ha="center", va="center", color=color, fontsize=12)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Negative", "Positive"],
           yticklabels=["Negative", "Positive"], xlabel="Predicted label", ylabel="True label", title=title)
    ax.grid(False)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def head_comparison(run_dir, metrics, configuration, graph_dir):
    """Compare the saved 512-token linear baseline with the best MLP checkpoint."""
    reference = json.loads((run_dir / "linear_baseline_snapshot.json").read_text(encoding="utf-8"))
    train, val = metrics["training"], metrics["validation"]
    rows = [
        {"model": reference["model"], "validation_accuracy": reference["validation_accuracy"],
         "validation_macro_f1": reference["validation_macro_f1"], "best_epoch": reference["best_epoch"],
         "train_accuracy": reference["train_accuracy"], "train_macro_f1": reference["train_macro_f1"],
         "training_validation_accuracy_gap": reference["training_validation_accuracy_gap"],
         "training_validation_macro_f1_gap": reference["training_validation_macro_f1_gap"]},
        {"model": f"LSTM + MLP (64), {metrics['max_sequence_length']} tokens",
         "validation_accuracy": val["accuracy"], "validation_macro_f1": val["macro_f1"],
         "best_epoch": metrics["best_epoch"], "train_accuracy": train["accuracy"],
         "train_macro_f1": train["macro_f1"],
         "training_validation_accuracy_gap": train["accuracy"] - val["accuracy"],
         "training_validation_macro_f1_gap": train["macro_f1"] - val["macro_f1"]},
    ]
    pd.DataFrame(rows).to_csv(run_dir / "head_comparison.csv", index=False)
    (run_dir / "head_comparison.json").write_text(json.dumps({"comparable": reference["same_full_data_and_cache"],
                                                               "note": reference["note"], "rows": rows}, indent=2) + "\n", encoding="utf-8")
    names = [f"Linear, 512 tokens\nepoch {rows[0]['best_epoch']}",
             f"MLP (64), {metrics['max_sequence_length']} tokens\nepoch {rows[1]['best_epoch']}"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.7))
    for ax, field, heading in zip(axes, ("validation_accuracy", "validation_macro_f1"),
                                  ("Validation accuracy", "Validation macro-F1")):
        values = [row[field] for row in rows]
        bars = ax.bar(names, values, color=["#347bb7", "#e1812c"])
        ax.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=3)
        ax.set(ylim=(0, 1.08), ylabel="Score", title=heading)
    subtitle = ("Same project validation split and cached sequences" if reference["same_full_data_and_cache"]
                else "SMOKE SUBSET: NOT COMPARABLE TO FULL BASELINE")
    fig.suptitle(f"LSTM classification heads\n{subtitle}", fontsize=12)
    fig.tight_layout(); fig.savefig(graph_dir / "head_comparison.png", dpi=160); plt.close(fig)
    print("\nLinear-head versus MLP-head comparison:")
    print(pd.DataFrame(rows)[["model", "validation_accuracy", "validation_macro_f1", "best_epoch",
                              "training_validation_accuracy_gap", "training_validation_macro_f1_gap"]]
          .to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)


def regenerate(run_dir: Path):
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    baseline = json.loads((run_dir / "baseline_snapshot.json").read_text(encoding="utf-8"))
    configuration = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    history = pd.read_csv(run_dir / "history.csv")
    best = metrics["best_epoch"]
    graph_dir = run_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    head = metrics.get("head", "linear")  # old saved runs predate the head field
    head_label = "LSTM" if head == "linear" else f"LSTM + MLP ({metrics['head_hidden_size']})"
    title = f"{head_label} h={metrics['hidden_size']}, max length={metrics['max_sequence_length']}"
    line_plot(history, ["training_loss", "validation_loss"], ["Training", "Validation"], "BCE loss",
              f"{title}: loss", best, graph_dir / "loss_by_epoch.png")
    line_plot(history, ["training_accuracy", "validation_accuracy"], ["Training", "Validation"], "Accuracy",
              f"{title}: accuracy", best, graph_dir / "accuracy_by_epoch.png")
    line_plot(history, ["training_macro_f1", "validation_macro_f1"], ["Training", "Validation"], "Macro-F1",
              f"{title}: macro-F1", best, graph_dir / "macro_f1_by_epoch.png")
    confusion_plot(metrics["validation"]["confusion_matrix"], graph_dir / "confusion_matrix.png",
                   f"{title}: validation, best epoch {best}")

    rows = baseline["rows"] + [{"experiment": "lstm", "validation_accuracy": metrics["validation"]["accuracy"],
                                "validation_macro_f1": metrics["validation"]["macro_f1"]}]
    pd.DataFrame(rows).to_csv(run_dir / "comparison.csv", index=False)
    (run_dir / "comparison.json").write_text(json.dumps({"smoke": configuration["smoke"], "rows": rows}, indent=2) + "\n", encoding="utf-8")
    labels = ["LR-Mean", "LR-Max", "MLP-Mean", "MLP-Max", f"{head_label}\nh={metrics['hidden_size']}, L={metrics['max_sequence_length']}"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, field, heading in zip(axes, ("validation_accuracy", "validation_macro_f1"),
                                  ("Validation accuracy", "Validation macro-F1")):
        values = [r[field] for r in rows]
        bars = ax.bar(labels, values, color=["#8ebbe0"] * 4 + ["#e1812c"])
        ax.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=3, fontsize=9)
        ax.set(ylim=(0, 1.08), ylabel="Score", title=heading)
        ax.tick_params(axis="x", labelrotation=15)
    subtitle = "SMOKE SUBSET: NOT COMPARABLE TO FULL BASELINES" if configuration["smoke"] else "Same project validation split; LSTM best checkpoint"
    fig.suptitle(f"Phase 2 baselines and Phase 3 LSTM\n{subtitle}", fontsize=12)
    fig.tight_layout(); fig.savefig(graph_dir / "phase2_comparison.png", dpi=160); plt.close(fig)
    if head == "mlp":
        head_comparison(run_dir, metrics, configuration, graph_dir)
    print("\nValidation comparison" + (" (smoke subset is not directly comparable)" if configuration["smoke"] else "") + ":")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)
    print(f"Saved graphs and comparison table in {run_dir.resolve()}", flush=True)
