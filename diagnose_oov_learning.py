"""Trace the 16x16 MLP and choose epochs on a shared inner validation split."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from compare_oov_bigrams import SEEDS, load_features
from prepare_meanpool import load_project1, load_reviews
from vocabulary import read_glove

EPOCHS = 100


def trace(train, inner, outer, labels, seed):
    train_labels, inner_labels, outer_labels = labels
    model = MLPClassifier(
        hidden_layer_sizes=(16, 16), activation="relu", solver="adam",
        alpha=1e-4, random_state=np.random.RandomState(seed),
    )
    history = []
    best = None
    for epoch in range(1, EPOCHS + 1):
        model.partial_fit(train, train_labels, classes=np.array([0, 1]))
        row = {
            "epoch": epoch,
            "train_macro_f1": float(f1_score(train_labels, model.predict(train), average="macro")),
            "inner_macro_f1": float(f1_score(inner_labels, model.predict(inner), average="macro")),
        }
        history.append(row)
        if best is None or row["inner_macro_f1"] > best["inner_macro_f1"]:
            best = row
            checkpoint = deepcopy(model)
    return {
        "history": history,
        "best": best,
        "outer_macro_f1_at_best": float(f1_score(outer_labels, checkpoint.predict(outer), average="macro")),
        "outer_macro_f1_at_100": float(f1_score(outer_labels, model.predict(outer), average="macro")),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project1", type=Path, required=True)
    parser.add_argument("--glove", type=Path, required=True)
    parser.add_argument("--arrow-dir", type=Path)
    parser.add_argument("--meanpool-dir", type=Path, default=Path("meanpool_output"))
    parser.add_argument("--output", type=Path, default=Path("oov_learning_curves.json"))
    args = parser.parse_args()

    project = load_project1(args.project1)
    frame, (train_ids, outer_ids, _), fingerprint = load_reviews(project, args.arrow_dir)
    glove = read_glove(args.glove)
    train, labels, dictionaries = load_features(args.meanpool_dir / "train.npz", frame, train_ids, glove)
    outer, outer_labels, outer_dicts = load_features(args.meanpool_dir / "validation.npz", frame, outer_ids, glove)
    fit, inner = train_test_split(np.arange(len(labels)), test_size=0.10, stratify=labels, random_state=42)
    scaler = StandardScaler().fit(train[fit])
    dense = tuple(csr_matrix(scaler.transform(rows)) for rows in (train[fit], train[inner], outer))
    vectorizer = DictVectorizer(dtype=np.float32)
    chars = (
        vectorizer.fit_transform([dictionaries[i] for i in fit]),
        vectorizer.transform([dictionaries[i] for i in inner]),
        vectorizer.transform(outer_dicts),
    )
    augmented = tuple(hstack((x, y), format="csr") for x, y in zip(dense, chars))
    label_sets = (labels[fit], labels[inner], outer_labels)
    result = {
        "ordered_sha256": fingerprint,
        "inner_split_seed": 42,
        "fitting_reviews": len(fit),
        "inner_validation_reviews": len(inner),
        "outer_validation_reviews": len(outer_ids),
        "oov_bigram_features": len(vectorizer.feature_names_),
        "runs": [],
    }
    print(f"Prepared {len(fit):,} fitting and {len(inner):,} inner validation reviews", flush=True)
    for seed in SEEDS:
        run = {"seed": seed}
        for name, features in (("meanpool", dense), ("oov_bigrams", augmented)):
            run[name] = trace(*features, label_sets, seed)
            scores = {key: value for key, value in run[name].items() if key != "history"}
            print(seed, name, json.dumps(scores), flush=True)
        result["runs"].append(run)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
