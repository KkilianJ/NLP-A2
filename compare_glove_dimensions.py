"""Screen GloVe 6B dimensions for a frozen MeanPool classifier."""
import argparse
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from finish_glove_study import save, scores
from prepare_meanpool import load_project1, load_reviews, tokenize


def load_dimension(path, dimension, words):
    vectors = {}
    with ZipFile(path) as archive, archive.open(f"glove.6B.{dimension}d.txt") as lines:
        for line in lines:
            token, _, coefficients = line.partition(b" ")
            word = token.decode("utf-8")
            if word in words:
                vector = np.fromstring(coefficients.decode("ascii"), sep=" ", dtype=np.float32)
                if vector.size != dimension:
                    raise ValueError(f"Invalid {dimension}d vector for {word!r}.")
                vectors[word] = vector
    return vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project1", type=Path, required=True)
    parser.add_argument("--glove", type=Path, required=True)
    parser.add_argument("--arrow-dir", type=Path)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("final_glove_study/dimension_screen.json"))
    args = parser.parse_args()
    project = load_project1(args.project1)
    frame, (train, _, _), fingerprint = load_reviews(project, args.arrow_dir)
    labels = frame.label.iloc[train].map({"negative": 0, "positive": 1}).to_numpy(dtype=np.int64)
    fit, inner = train_test_split(np.arange(len(train)), test_size=.10, stratify=labels, random_state=42)
    docs = tokenize(frame.text.iloc[train])
    words = {word for doc in docs for word in doc}
    report = {"ordered_sha256": fingerprint, "inner_split_seed": 42, "token_limit": args.limit, "dimensions": []}
    reference_vocabulary = None
    for dimension in (50, 100, 200, 300):
        vectors = load_dimension(args.glove, dimension, words)
        if reference_vocabulary is None:
            reference_vocabulary = set(vectors)
        if vectors.keys() != reference_vocabulary:
            raise ValueError("GloVe vocabularies differ across dimensions; fixed-input contrast is invalid.")
        features = np.zeros((len(docs), dimension), dtype=np.float32)
        for i, doc in enumerate(docs):
            covered = [vectors[word] for word in doc if word in vectors][:args.limit]
            if covered:
                features[i] = np.mean(covered, axis=0, dtype=np.float64)
        grid = []
        for c in (0.001, 0.01, 0.1, 1, 10):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000))
            model.fit(features[fit], labels[fit])
            if model[-1].n_iter_.max() >= model[-1].max_iter:
                raise RuntimeError(f"Logistic regression did not converge for {dimension}d, C={c}.")
            grid.append({"C": c, **scores(labels[inner], model.predict(features[inner]))})
        row = {"dimension": dimension, "classifier_parameters": dimension + 1,
               "best": max(grid, key=lambda item: item["macro_f1"]), "grid": grid}
        report["dimensions"].append(row)
        save(args.output, report)
        print(dimension, row["best"], flush=True)


if __name__ == "__main__":
    main()
