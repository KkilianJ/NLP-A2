"""Save measured data and a compact coverage plot; never write the report template."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from examples import expansion_rows
import settings


def write_results(output, rows, examples, occurrences, reviews, metadata):
    pd.DataFrame(rows).to_csv(output / "coverage.csv", index=False)
    (output / "coverage.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    columns = ["category", "review_id", "matched_text", "excerpt", *settings.VARIANTS]
    pd.DataFrame(examples, columns=columns).to_csv(output / "tokenization_examples.csv", index=False)
    pd.DataFrame(expansion_rows(occurrences, reviews),
                 columns=["original_token", "expanded_tokens", "occurrences", "reviews_with_token"]
                 ).to_csv(output / "negation_expansions.csv", index=False)
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False,
                                                          allow_nan=False) + "\n", encoding="utf-8")
    plot_coverage(rows, output / "coverage.png", metadata["scope"])


def plot_coverage(rows, destination, scope):
    labels = ["Whitespace", "Regex", "Regex +\nnegation"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    x, width = np.arange(len(rows)), 0.36
    for offset, key, label, color in ((-width / 2, "direct_token_fraction", "Token-weighted", "#347bb7"),
                                      (width / 2, "direct_type_fraction", "Type-weighted", "#e1812c")):
        values = [100 * row[key] if row[key] is not None else 0 for row in rows]
        bars = axes[0].bar(x + offset, values, width, label=label, color=color)
        axes[0].bar_label(bars, labels=[f"{value:.2f}%" if row[key] is not None else "N/A"
                                       for value, row in zip(values, rows)], padding=3, fontsize=8)
    axes[0].set(title="Direct stored-vocabulary coverage", ylabel="Percentage", xticks=x, xticklabels=labels,
                ylim=(0, 110))
    axes[0].legend(fontsize=8)
    key = "oov_nonzero_occurrence_fraction"
    values = [100 * row[key] if row[key] is not None else 0 for row in rows]
    bars = axes[1].bar(x, values, color="#53a276")
    axes[1].bar_label(bars, labels=[f"{value:.2f}%" if row[key] is not None else "N/A"
                                    for value, row in zip(values, rows)], padding=3, fontsize=8)
    axes[1].set(title="OOV occurrences with a nonzero vector", ylabel="Percentage of OOV occurrences",
                xticks=x, xticklabels=labels, ylim=(0, 110))
    title = ("published source train — DESCRIPTIVE ONLY; overlaps project holdout"
             if scope == "published-train-descriptive" else "project training partition")
    fig.suptitle(f"fastText tokenization comparison — {title}")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
