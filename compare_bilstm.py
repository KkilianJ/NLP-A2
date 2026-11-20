"""Run one BiLSTM against the same-split, same-seed GloVe BiGRU and MeanPool."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn

from compare_gru import Classifier, batches, evaluate
from complete_embeddings import load_vectors
from finish_glove_study import Configuration, predict_gru, save, scores, train_gru
from prepare_meanpool import load_project1, load_reviews, tokenize
from summarize_glove_study import paired_bootstrap


class BiLSTM(Classifier):
    def __init__(self, weights, hidden_size):
        nn.Module.__init__(self)
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=True, padding_idx=0)
        self.recurrent = True
        self.forward_gru = nn.LSTM(weights.shape[1], hidden_size, batch_first=True)
        self.backward_gru = nn.LSTM(weights.shape[1], hidden_size, batch_first=True)
        self.head = nn.Linear(2 * hidden_size, 1)


def paired_scores(rows, mask):
    group = rows.loc[mask]
    y = group["label"].to_numpy()
    lstm_correct = group["lstm"].to_numpy() == y
    result = {"reviews": len(group), "lstm": scores(y, group["lstm"])}
    for name in ("meanpool", "gru"):
        correct = group[name].to_numpy() == y
        result[name] = {**scores(y, group[name]), "only_this_correct": int((correct & ~lstm_correct).sum()),
                        "only_lstm_correct": int((lstm_correct & ~correct).sum())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project1", type=Path, required=True)
    parser.add_argument("--glove", type=Path, required=True)
    parser.add_argument("--arrow-dir", type=Path)
    parser.add_argument("--study", type=Path, default=Path("final_glove_study"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = perf_counter()
    prior = json.loads((args.study / "recurrent_results.json").read_text())
    reference = next(run for run in prior["runs"] if run["setup"] == "tuned_full" and run["seed"] == 42)
    config = Configuration(**reference["configuration"])
    project = load_project1(args.project1)
    frame, (train, validation, _), fingerprint = load_reviews(project, args.arrow_dir)
    if fingerprint != prior["fingerprint"]:
        raise ValueError("IMDb order differs from the BiGRU experiment.")

    row_ids = np.concatenate((train, validation))
    labels = frame.label.iloc[row_ids].map({"negative": 0, "positive": 1}).to_numpy(dtype=np.int64)
    fit, inner = train_test_split(np.arange(len(train)), test_size=.10, stratify=labels[:len(train)],
                                  random_state=prior["inner_seed"])
    outer = np.arange(len(train), len(row_ids))
    docs = tokenize(frame.text.iloc[row_ids])
    vectors = load_vectors(args.glove, {word for doc in docs for word in doc})
    word_ids = {word: i for i, word in enumerate(vectors, 1)}
    weights = torch.zeros(len(vectors) + 1, config_dimension := len(next(iter(vectors.values()))))
    weights[1:] = torch.from_numpy(np.stack(list(vectors.values())))
    sequences = [[word_ids[word] for word in doc if word in word_ids][:config.limit] for doc in docs]
    print(f"Prepared {len(row_ids):,} reviews; training seed 42", flush=True)
    model, log = train_gru(config, weights, sequences, labels, fit, inner, 42,
                           prior["confirmation_epoch_cap"], model_factory=BiLSTM)
    predictions = predict_gru(model, sequences, labels, outer)
    assert np.isclose(scores(labels[outer], predictions)["macro_f1"],
                      evaluate(model, batches(sequences, labels, outer, prior["batch_size"], "cpu")))
    rows = pd.read_csv(args.study / "trainable_baseline_predictions.csv")
    assert np.array_equal(rows["project1_row"], row_ids[outer])
    assert np.array_equal(rows["label"], labels[outer])
    rows["lstm"] = predictions
    earlier_errors = json.loads((args.study / "error_analysis.json").read_text())
    low, high = earlier_errors["training_length_tertiles"]
    length = rows["tokens"]
    oov_rate = (length - rows["glove_hits"]) / length.clip(lower=1)
    high_oov = oov_rate > earlier_errors["training_median_oov_rate"]
    masks = {
        "all": np.ones(len(rows), dtype=bool),
        "short": length <= low,
        "medium": (length > low) & (length <= high),
        "long": length > high,
        "negation_present": rows["negation"] == 1,
        "negation_absent": rows["negation"] == 0,
        "contrast_present": rows["contrast"] == 1,
        "contrast_absent": rows["contrast"] == 0,
        "truncated": rows["truncated"] == 1,
        "not_truncated": rows["truncated"] == 0,
        "high_oov_rate": high_oov,
        "low_oov_rate": ~high_oov,
    }
    report = {
        "fingerprint": fingerprint, "seed": 42, "configuration": reference["configuration"],
        "embedding_dimension": config_dimension, "fit_reviews": len(fit), "inner_reviews": len(inner),
        "epoch_cap": prior["confirmation_epoch_cap"], "batch_size": prior["batch_size"],
        "optimizer": prior["optimizer"], "patience": prior["patience"], "test_evaluated": False,
        "gru_reference_trainable_parameters": reference["trainable_parameters"],
        **log, "groups": {key: paired_scores(rows, mask) for key, mask in masks.items() if mask.any()},
        "paired_bootstrap_lstm_minus_gru_macro_f1": paired_bootstrap(
            rows.assign(meanpool=rows["gru"], gru=rows["lstm"])),
        "runtime_seconds": perf_counter() - started,
    }
    rows.to_csv(args.study / "bilstm_validation_predictions.csv", index=False)
    save(args.study / "bilstm_results.json", report)
    print(json.dumps({"best": log["best"], "overall": report["groups"]["all"],
                      "runtime_seconds": report["runtime_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
