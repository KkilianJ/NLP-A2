"""Compare masked GloVe MeanPool with and without binary OOV bigram features."""

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from prepare_meanpool import load_project1, load_reviews, tokenize
from vocabulary import read_glove

SEEDS = (42, 43, 44)


def oov_bigram_features(words, glove):
    return {
        word[start : start + 2]: 1
        for word in words
        if word not in glove
        for start in range(len(word) - 1)
    }


def load_features(path, frame, indices, glove):
    with np.load(path) as archive:
        labels = frame.label.iloc[indices].map({"negative": 0, "positive": 1}).to_numpy()
        if not np.array_equal(archive["row_indices"], indices) or not np.array_equal(archive["labels"], labels):
            raise ValueError(f"Project 1 row indices or labels do not match {path}")
        pooled = archive["zero_masked"]
        expected_oov_counts = archive["token_counts"] - archive["glove_covered_counts"]

    reviews = tokenize(frame.text.iloc[indices])
    oov_counts = [sum(word not in glove for word in words) for words in reviews]
    if not np.array_equal(oov_counts, expected_oov_counts):
        raise ValueError(f"GloVe coverage does not match the saved MeanPool data in {path}")
    return pooled, labels, [oov_bigram_features(words, glove) for words in reviews]


def score_model(train, labels, validation, validation_labels, seed, hidden_layers):
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers, activation="relu", solver="adam",
        alpha=1e-4, max_iter=100, random_state=seed,
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train, labels)
    predictions = model.predict(validation)
    return {
        "macro_f1": float(f1_score(validation_labels, predictions, average="macro")),
        "train_macro_f1": float(f1_score(labels, model.predict(train), average="macro")),
        "epochs": int(model.n_iter_),
        "convergence_warning": any(issubclass(item.category, ConvergenceWarning) for item in captured),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project1", type=Path, required=True)
    parser.add_argument("--glove", type=Path, required=True)
    parser.add_argument("--arrow-dir", type=Path)
    parser.add_argument("--meanpool-dir", type=Path, default=Path("meanpool_output"))
    parser.add_argument("--output", type=Path, default=Path("oov_bigram_results.json"))
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[16])
    args = parser.parse_args()
    if any(width < 1 for width in args.hidden_layers):
        parser.error("--hidden-layers requires positive widths")
    hidden_layers = tuple(args.hidden_layers)

    project = load_project1(args.project1)
    frame, (train_indices, validation_indices, _), fingerprint = load_reviews(project, args.arrow_dir)
    manifest = json.loads((args.meanpool_dir / "manifest.json").read_text())
    if manifest["ordered_sha256"] != fingerprint or manifest["tokenization"] != "regex_keep_contractions":
        raise ValueError("Saved MeanPool manifest does not match the expected dataset and tokenization.")

    glove = read_glove(args.glove)
    train, train_labels, train_dicts = load_features(args.meanpool_dir / "train.npz", frame, train_indices, glove)
    val, val_labels, val_dicts = load_features(args.meanpool_dir / "validation.npz", frame, validation_indices, glove)
    scaler = StandardScaler()
    train = csr_matrix(scaler.fit_transform(train))
    val = csr_matrix(scaler.transform(val))
    vectorizer = DictVectorizer(dtype=np.float32)
    train_bigrams = vectorizer.fit_transform(train_dicts)
    val_bigrams = vectorizer.transform(val_dicts)
    train_augmented = hstack((train, train_bigrams), format="csr")
    val_augmented = hstack((val, val_bigrams), format="csr")
    print(f"Prepared {train_bigrams.shape[1]:,} OOV bigram features from training reviews", flush=True)

    runs = []
    for seed in SEEDS:
        baseline = score_model(train, train_labels, val, val_labels, seed, hidden_layers)
        augmented = score_model(train_augmented, train_labels, val_augmented, val_labels, seed, hidden_layers)
        run = {
            "seed": seed,
            "meanpool": baseline,
            "meanpool_plus_oov_bigrams": augmented,
            "macro_f1_delta": augmented["macro_f1"] - baseline["macro_f1"],
        }
        runs.append(run)
        print(json.dumps(run), flush=True)

    scores = {
        "meanpool": [run["meanpool"]["macro_f1"] for run in runs],
        "meanpool_plus_oov_bigrams": [run["meanpool_plus_oov_bigrams"]["macro_f1"] for run in runs],
        "paired_delta": [run["macro_f1_delta"] for run in runs],
    }
    result = {
        "dataset_revision": project.IMDB_REVISION,
        "ordered_sha256": fingerprint,
        "project1_split_seed": project.SEED,
        "train_reviews": len(train_labels),
        "validation_reviews": len(val_labels),
        "oov_bigram_features": train_bigrams.shape[1],
        "validation_reviews_with_known_oov_bigrams": int(np.count_nonzero(val_bigrams.getnnz(axis=1))),
        "mlp_hidden_layers": list(hidden_layers),
        "runs": runs,
        "summary": {
            name: {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1))}
            for name, values in scores.items()
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
