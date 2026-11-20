"""Regenerate all charts and comparison tables from saved JSON/CSV results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import settings
from training import metric_row, write_json

plt.style.use("seaborn-v0_8-whitegrid")


def confusion_plot(matrix, output, title):
    matrix = np.asarray(matrix, dtype=int)
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Reviews")
    for row in range(2):
        for col in range(2):
            color = "white" if matrix[row, col] > matrix.max() / 2 else "black"
            ax.text(col, row, f"{matrix[row, col]:,}", ha="center", va="center", color=color, fontsize=12)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Negative", "Positive"],
           yticklabels=["Negative", "Positive"], xlabel="Predicted label", ylabel="True label", title=title)
    ax.grid(False)
    fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def metrics_bars(result, output):
    values = [result["validation"]["accuracy"], result["validation"]["macro_f1"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Accuracy", "Macro-F1"], values, color=["#347bb7", "#ef9040"])
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=3)
    ax.set(ylim=(0, 1.08), ylabel="Validation score", title=f"{result['experiment']}: validation scores")
    fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def epoch_plot(history, columns, labels, ylabel, title, best_epoch, output):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for column, label in zip(columns, labels):
        selected = history.loc[history.epoch == best_epoch, column].iloc[0]
        ax.plot(history.epoch, history[column], marker="o", markersize=3,
                label=f"{label} (best epoch: {selected:.4f})")
        ax.scatter([best_epoch], [selected], s=45, zorder=4)
    ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label=f"Best checkpoint: epoch {best_epoch}")
    ax.set(xlabel="Epoch", ylabel=ylabel, title=title)
    if "accuracy" in ylabel.lower() or "f1" in ylabel.lower():
        ax.set_ylim(0, 1.02)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def per_model_plots(directory, result):
    graph_dir = directory / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    confusion_plot(result["validation"]["confusion_matrix"], graph_dir / "confusion_matrix.png",
                   f"{result['experiment']}: validation confusion matrix")
    if result["model"] == "Logistic Regression":
        metrics_bars(result, graph_dir / "validation_scores.png")
        return
    history = pd.read_csv(directory / "epochs.csv")
    best = result["best_epoch"]
    epoch_plot(history, ["training_loss", "validation_loss"], ["Train", "Validation"],
               "BCE loss", f"{result['experiment']}: loss", best, graph_dir / "loss_by_epoch.png")
    epoch_plot(history, ["training_accuracy", "validation_accuracy"], ["Train", "Validation"],
               "Accuracy", f"{result['experiment']}: accuracy", best, graph_dir / "accuracy_by_epoch.png")
    epoch_plot(history, ["validation_macro_f1"], ["Validation"], "Macro-F1",
               f"{result['experiment']}: validation macro-F1", best, graph_dir / "macro_f1_by_epoch.png")


def comparison_bar(results, key, output, title):
    names = [r["experiment"] for r in results]
    values = [r["validation"][key] for r in results]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, values, color=["#347bb7", "#83b5dd", "#e2843b", "#f4c181"][:len(names)])
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=3)
    ax.set(ylim=(0, 1.08), ylabel="Validation score", title=title)
    fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def pooling_comparison(results, output):
    by_key = {(r["model"], r["pooling"]): r for r in results}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    models = ["Logistic Regression", "MLP (64)"]
    for ax, key, title in zip(axes, ["accuracy", "macro_f1"], ["Validation accuracy", "Validation macro-F1"]):
        for offset, pool, color in ((-.18, "mean", "#347bb7"), (.18, "max", "#ef9040")):
            xs, values = [], []
            for x, model in enumerate(models):
                if (model, pool) in by_key:
                    xs.append(x + offset); values.append(by_key[(model, pool)]["validation"][key])
            if values:
                bars = ax.bar(xs, values, width=.34, label=pool.capitalize(), color=color)
                ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=3, fontsize=9)
        ax.set(xticks=[0, 1], xticklabels=models, ylim=(0, 1.08), title=title, ylabel="Score")
        ax.legend(frameon=False)
    fig.suptitle("Mean versus max pooling (completed experiments)")
    fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def regenerate(run_dir: Path):
    results = []
    for experiment in settings.EXPERIMENTS:
        directory = run_dir / experiment
        file = directory / "metrics.json"
        if file.is_file():
            result = json.loads(file.read_text(encoding="utf-8"))
            per_model_plots(directory, result)
            results.append(result)
    if not results:
        raise ValueError(f"No completed experiment metrics found in {run_dir}")
    rows = [metric_row(result) for result in results]
    pd.DataFrame(rows).to_csv(run_dir / "comparison.csv", index=False)
    write_json(run_dir / "comparison.json", {"complete": len(results) == 4, "experiments": rows})
    graph_dir = run_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    comparison_bar(results, "accuracy", graph_dir / "validation_accuracy_comparison.png",
                   "Validation accuracy: completed experiments")
    comparison_bar(results, "macro_f1", graph_dir / "validation_macro_f1_comparison.png",
                   "Validation macro-F1: completed experiments")
    pooling_comparison(results, graph_dir / "mean_vs_max.png")
    print("\nComparison of completed experiments:")
    print(pd.DataFrame(rows)[["model", "pooling", "train_accuracy", "validation_accuracy", "validation_macro_f1"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)
    print(f"Results and graphs: {run_dir.resolve()}", flush=True)
