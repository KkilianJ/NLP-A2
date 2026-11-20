"""Evaluate character n-gram ridge regression for missing IMDb word embeddings."""

import argparse
from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from vocabulary import TOKEN, clean, lexical_tokens

DIMENSION = 100
ALPHA = 1.0
SEED = 42


def character_ngrams(word):
    marked = f"^{word}$"
    return [
        marked[start : start + size]
        for size in (3, 4, 5)
        for start in range(len(marked) - size + 1)
    ]


@contextmanager
def glove_lines(path):
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive, archive.open("glove.6B.100d.txt") as lines:
            yield lines
    else:
        with path.open("rb") as lines:
            yield lines


def load_vectors(path, words):
    vectors = {}
    with glove_lines(path) as lines:
        for line in lines:
            fields = line.split(maxsplit=1)
            if not fields:
                continue
            word = fields[0].decode("utf-8")
            if word in words:
                vector = np.fromstring(fields[1].decode("ascii"), sep=" ", dtype=np.float32)
                if vector.shape != (DIMENSION,) or not np.isfinite(vector).all():
                    raise ValueError(f"Invalid {DIMENSION}d GloVe vector for {word!r}")
                vectors[word] = vector
    return vectors


def reconstruction_metrics(targets, predictions):
    targets = targets.astype(np.float64)
    predictions = predictions.astype(np.float64)
    denominators = np.linalg.norm(targets, axis=1) * np.linalg.norm(predictions, axis=1)
    valid = denominators > 0
    similarities = np.sum(targets[valid] * predictions[valid], axis=1) / denominators[valid]
    return {
        "rmse": float(np.sqrt(np.mean((predictions - targets) ** 2))),
        "mean_cosine": float(similarities.mean()) if valid.any() else None,
        "cosine_defined_words": int(valid.sum()),
    }


def fit_ridge(vectorizer, model, words, vectors):
    features = vectorizer.fit_transform(words).astype(np.float64)
    model.fit(features, np.stack([vectors[word] for word in words]).astype(np.float64))


def run(texts, glove_path, output):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    counts = Counter()
    for text in texts:
        counts.update(lexical_tokens(TOKEN.findall(clean(text))))
    vectors = load_vectors(glove_path, counts)
    known = sorted(vectors)
    if len(known) < 2:
        raise ValueError("At least two IMDb word types must have GloVe vectors.")
    oov = sorted(counts.keys() - vectors.keys())
    train_words, held_out_words = train_test_split(known, test_size=0.10, random_state=SEED)
    targets = np.stack([vectors[word] for word in held_out_words])

    vectorizer = CountVectorizer(analyzer=character_ngrams, binary=True)
    model = Ridge(alpha=ALPHA, solver="lsqr", tol=1e-4)
    fit_ridge(vectorizer, model, train_words, vectors)
    held_out_features = vectorizer.transform(held_out_words).astype(np.float64)
    held_out_predictions = model.predict(held_out_features)
    report = {
        "dataset": "mteb/imdb",
        "split": "train",
        "tokenization": "regex_keep_contractions",
        "glove_file": glove_path.name,
        "dimensions": DIMENSION,
        "character_ngram_sizes": [3, 4, 5],
        "alpha": ALPHA,
        "split_seed": SEED,
        "glove_covered_types": len(known),
        "imdb_oov_types": len(oov),
        "ridge_training_types": len(train_words),
        "evaluation_types": len(held_out_words),
        "evaluation_types_without_known_ngrams": int(np.sum(held_out_features.getnnz(axis=1) == 0)),
        "zero_vector": reconstruction_metrics(targets, np.zeros_like(targets)),
        "ridge": reconstruction_metrics(targets, held_out_predictions),
    }

    # Fit on all available reference words only after computing held-out metrics.
    fit_ridge(vectorizer, model, known, vectors)
    oov_vectors = np.empty((len(oov), DIMENSION), dtype=np.float32)
    without_ngrams = 0
    for start in range(0, len(oov), 4096):
        batch = oov[start : start + 4096]
        features = vectorizer.transform(batch).astype(np.float64)
        without_ngrams += int(np.sum(features.getnnz(axis=1) == 0))
        oov_vectors[start : start + len(batch)] = model.predict(features)
    report["imdb_oov_types_without_known_ngrams"] = without_ngrams

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "oov_vectors.npz",
        words=np.asarray(oov, dtype=str),
        vectors=oov_vectors,
        zero_vector=np.zeros(DIMENSION, dtype=np.float32),
    )
    (output / "held_out_words.txt").write_text("\n".join(held_out_words) + "\n", encoding="utf-8")
    formatted = json.dumps(report, ensure_ascii=False, indent=2)
    (output / "embedding_results.json").write_text(formatted + "\n", encoding="utf-8")
    print(formatted)
    print(f"Saved to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glove", type=Path, required=True, help="GloVe 6B ZIP or extracted 100d text file")
    parser.add_argument("--output", type=Path, default=Path("embedding_output"))
    args = parser.parse_args()
    if not args.glove.is_file():
        parser.error(f"GloVe file does not exist: {args.glove}")
    from datasets import load_dataset

    train = load_dataset("mteb/imdb", split="train")
    run(train["text"], args.glove, args.output)


if __name__ == "__main__":
    main()
