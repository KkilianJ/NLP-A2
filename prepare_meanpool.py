"""Build paired review embeddings on the original Project 1 IMDb split."""

import argparse
import importlib.util
import json
from pathlib import Path

from datasets import Dataset, load_dataset
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge

from complete_embeddings import ALPHA, DIMENSION, character_ngrams, fit_ridge, load_vectors
from vocabulary import TOKEN, clean, lexical_tokens

SPLIT_SIZES = (34707, 7437, 7438)


def load_project1(path):
    spec = importlib.util.spec_from_file_location("project1_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_reviews(project1, arrow_dir):
    names = ("train", "test")
    if arrow_dir is None:
        data = load_dataset("mteb/imdb", revision=project1.IMDB_REVISION)
    else:
        data = {name: Dataset.from_file(str(arrow_dir / f"imdb-{name}.arrow")) for name in names}
    frames = [
        data[name].select_columns(["text", "label_text"]).to_pandas()
        for name in names
    ]
    frame = pd.concat(frames, ignore_index=True).rename(columns={"label_text": "label"})
    fingerprint = project1.ordered_data_sha256(frame)
    if fingerprint != project1.IMDB_ORDERED_SHA256:
        raise ValueError("IMDb row-order fingerprint differs from Project 1.")
    indices = project1.split_indices(frame)
    if tuple(map(len, indices)) != SPLIT_SIZES:
        raise ValueError("Recreated split sizes differ from Project 1.")
    return frame, indices, fingerprint


def tokenize(texts):
    return [lexical_tokens(TOKEN.findall(clean(text))) for text in texts]


def pool_reviews(reviews, glove, predicted):
    masked = np.zeros((len(reviews), DIMENSION), dtype=np.float32)
    completed = np.zeros_like(masked)
    token_counts = np.zeros(len(reviews), dtype=np.int32)
    covered_counts = np.zeros_like(token_counts)
    for row, words in enumerate(reviews):
        known_sum = np.zeros(DIMENSION, dtype=np.float64)
        predicted_sum = np.zeros_like(known_sum)
        covered = 0
        for word in words:
            vector = glove.get(word)
            if vector is None:
                predicted_sum += predicted[word]
            else:
                known_sum += vector
                covered += 1
        token_counts[row] = len(words)
        covered_counts[row] = covered
        if covered:
            masked[row] = known_sum / covered
        if words:
            completed[row] = (known_sum + predicted_sum) / len(words)
    return masked, completed, token_counts, covered_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project1", type=Path, required=True, help="Project 1 experiment.py")
    parser.add_argument("--glove", type=Path, required=True)
    parser.add_argument("--arrow-dir", type=Path, help="Optional directory with cached IMDb Arrow files")
    parser.add_argument("--output", type=Path, default=Path("meanpool_output"))
    args = parser.parse_args()

    project1 = load_project1(args.project1)
    frame, (train_indices, val_indices, test_indices), fingerprint = load_reviews(project1, args.arrow_dir)
    print(f"Verified Project 1 row order and split sizes {SPLIT_SIZES}", flush=True)
    train_reviews = tokenize(frame.text.iloc[train_indices])
    val_reviews = tokenize(frame.text.iloc[val_indices])
    train_words = {word for review in train_reviews for word in review}
    val_words = {word for review in val_reviews for word in review}
    all_words = train_words | val_words
    glove = load_vectors(args.glove, all_words)

    fitting_words = sorted(train_words & glove.keys())
    vectorizer = CountVectorizer(analyzer=character_ngrams, binary=True)
    model = Ridge(alpha=ALPHA, solver="lsqr", tol=1e-4)
    print(f"Fitting ridge on {len(fitting_words):,} GloVe-covered training word types", flush=True)
    fit_ridge(vectorizer, model, fitting_words, glove)
    oov = sorted(all_words - glove.keys())
    predicted = {}
    for start in range(0, len(oov), 4096):
        batch = oov[start : start + 4096]
        features = vectorizer.transform(batch).astype(np.float64)
        predicted.update(zip(batch, model.predict(features).astype(np.float32)))

    args.output.mkdir(parents=True, exist_ok=True)
    split_report = {}
    for name, indices, reviews in (
        ("train", train_indices, train_reviews),
        ("validation", val_indices, val_reviews),
    ):
        masked, completed, tokens, covered = pool_reviews(reviews, glove, predicted)
        labels = frame.label.iloc[indices].map({"negative": 0, "positive": 1}).to_numpy(dtype=np.int8)
        np.savez_compressed(
            args.output / f"{name}.npz",
            row_indices=indices,
            labels=labels,
            zero_masked=masked,
            ridge_completed=completed,
            token_counts=tokens,
            glove_covered_counts=covered,
        )
        split_report[name] = {
            "reviews": len(indices),
            "reviews_with_oov": int(np.count_nonzero(tokens > covered)),
            "reviews_without_glove_hits": int(np.count_nonzero(covered == 0)),
        }
        print(f"Saved {name}: {masked.shape}", flush=True)

    manifest = {
        "dataset": "mteb/imdb",
        "revision": project1.IMDB_REVISION,
        "ordered_sha256": fingerprint,
        "seed": project1.SEED,
        "dimensions": DIMENSION,
        "tokenization": "regex_keep_contractions",
        "glove_file": args.glove.name,
        "ridge_alpha": ALPHA,
        "ridge_fitting_types": len(fitting_words),
        "labels": {"negative": 0, "positive": 1},
        "splits": split_report,
        "held_out_test_reviews": len(test_indices),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
