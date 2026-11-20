"""Measure the validation contribution of OOV bigrams with L2 logistic regression."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from compare_oov_bigrams import load_features
from prepare_meanpool import load_project1, load_reviews
from vocabulary import read_glove

C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
INNER_SEED = 42


def tune(fit, inner, labels, inner_labels):
    best_model = None
    best_score = -1.0
    best_c = None
    grid = []
    for c in C_GRID:
        model = LogisticRegression(C=c, solver="lbfgs", max_iter=1000)
        model.fit(fit, labels)
        score = float(f1_score(inner_labels, model.predict(inner), average="macro"))
        grid.append({"C": c, "inner_macro_f1": score, "iterations": int(model.n_iter_.max())})
        if score > best_score:
            best_model, best_score, best_c = model, score, c
    return best_model, best_c, grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project1", type=Path, required=True)
    parser.add_argument("--glove", type=Path, required=True)
    parser.add_argument("--arrow-dir", type=Path)
    parser.add_argument("--meanpool-dir", type=Path, default=Path("meanpool_output"))
    parser.add_argument("--output", type=Path, default=Path("oov_logistic_results.json"))
    args = parser.parse_args()

    project = load_project1(args.project1)
    frame, (train_ids, val_ids, _), fingerprint = load_reviews(project, args.arrow_dir)
    glove = read_glove(args.glove)
    train, labels, train_dicts = load_features(args.meanpool_dir / "train.npz", frame, train_ids, glove)
    val, val_labels, val_dicts = load_features(args.meanpool_dir / "validation.npz", frame, val_ids, glove)
    fit, inner = train_test_split(
        np.arange(len(labels)), test_size=0.10, stratify=labels, random_state=INNER_SEED,
    )
    scaler = StandardScaler().fit(train[fit])
    dense = tuple(csr_matrix(scaler.transform(x)) for x in (train[fit], train[inner], val))
    vectorizer = DictVectorizer(dtype=np.float32)
    chars = (
        vectorizer.fit_transform([train_dicts[i] for i in fit]),
        vectorizer.transform([train_dicts[i] for i in inner]),
        vectorizer.transform(val_dicts),
    )
    augmented = tuple(hstack((a, b), format="csr") for a, b in zip(dense, chars))
    result = {
        "ordered_sha256": fingerprint,
        "fitting_reviews": len(fit),
        "inner_validation_reviews": len(inner),
        "project1_validation_reviews": len(val_labels),
        "inner_split_seed": INNER_SEED,
        "oov_bigram_features": chars[0].shape[1],
        "models": {},
    }
    predictions = {}
    for name, (x_fit, x_inner, x_val) in (("meanpool", dense), ("meanpool_plus_oov_bigrams", augmented)):
        model, c, grid = tune(x_fit, x_inner, labels[fit], labels[inner])
        predictions[name] = model.predict(x_val)
        result["models"][name] = {
            "selected_C": c,
            "grid": grid,
            "fitting_macro_f1": float(f1_score(labels[fit], model.predict(x_fit), average="macro")),
            "project1_validation_macro_f1": float(f1_score(val_labels, predictions[name], average="macro")),
        }
    baseline = result["models"]["meanpool"]["project1_validation_macro_f1"]
    enhanced = result["models"]["meanpool_plus_oov_bigrams"]["project1_validation_macro_f1"]
    result["project1_validation_macro_f1_delta"] = enhanced - baseline
    base_correct = predictions["meanpool"] == val_labels
    augmented_correct = predictions["meanpool_plus_oov_bigrams"] == val_labels
    result["only_meanpool_correct"] = int(np.count_nonzero(base_correct & ~augmented_correct))
    result["only_augmented_correct"] = int(np.count_nonzero(augmented_correct & ~base_correct))
    report = json.dumps(result, indent=2)
    args.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
