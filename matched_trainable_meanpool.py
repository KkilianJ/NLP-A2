"""Run the earlier trainable MeanPool baseline on exactly the GRU's token inputs."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
import torch

from compare_initialization import BATCH_SIZE, EPOCHS, LEARNING_RATE, PATIENCE, SEEDS, fit_model
from complete_embeddings import DIMENSION, load_vectors
from finish_glove_study import predict_gru, save, scores
from prepare_meanpool import load_project1, load_reviews, tokenize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project1", type=Path, required=True)
    parser.add_argument("--glove", type=Path, required=True)
    parser.add_argument("--arrow-dir", type=Path)
    parser.add_argument("--study", type=Path, default=Path("final_glove_study"))
    parser.add_argument("--save-predictions", action="store_true", help="Reproduce seed 42 and save predictions")
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate the locked seed 42 checkpoint on Project 1 test")
    args = parser.parse_args()
    torch.set_num_threads(2)

    recurrent = json.loads((args.study / "recurrent_results.json").read_text())
    limit = recurrent["selected_configuration"]["limit"]
    project = load_project1(args.project1)
    frame, (train, validation, test), fingerprint = load_reviews(project, args.arrow_dir)
    if fingerprint != recurrent["fingerprint"]:
        raise ValueError("The recurrent study and local dataset fingerprints differ.")
    row_ids = np.concatenate((train, validation))
    labels = frame.label.iloc[row_ids].map({"negative": 0, "positive": 1}).to_numpy(dtype=np.int64)
    fit, inner = train_test_split(
        np.arange(len(train)), test_size=.10, stratify=labels[:len(train)], random_state=42,
    )
    outer = np.arange(len(train), len(row_ids))
    docs = tokenize(frame.text.iloc[row_ids])
    glove = load_vectors(args.glove, {word for doc in docs for word in doc})
    ids = {word: i for i, word in enumerate(glove, 1)}
    weights = torch.zeros(len(glove) + 1, DIMENSION)
    weights[1:] = torch.from_numpy(np.stack(list(glove.values())))
    sequences = [[ids[word] for word in doc if word in ids][:limit] for doc in docs]
    if args.save_predictions or args.evaluate_test:
        predictions = []
        checkpoint = []

        def collect(model):
            predictions.extend(predict_gru(model, sequences, labels, outer))
            checkpoint.append(model)

        run = fit_model(weights, sequences, labels, (fit, inner, outer), 42, "meanpool", False,
                        on_checkpoint=collect)
        previous = json.loads((args.study / "matched_trainable_meanpool.json").read_text())["runs"][0]
        if not np.isclose(run["project1_validation_macro_f1"], previous["project1_validation_macro_f1"], atol=1e-12):
            raise ValueError("Matched baseline's seed 42 result did not reproduce.")
        if args.save_predictions:
            np.savez_compressed(args.study / "trainable_meanpool_validation_predictions.npz",
                                project1_row=row_ids[outer], label=labels[outer], meanpool=predictions)
        if args.evaluate_test:
            model = checkpoint[0]
            test_docs = tokenize(frame.text.iloc[test])
            extra = load_vectors(args.glove, {word for doc in test_docs for word in doc} - ids.keys())
            if extra:
                ids.update({word: i for i, word in enumerate(extra, len(ids) + 1)})
                all_weights = torch.cat((model.embedding.weight.detach(), torch.from_numpy(np.stack(list(extra.values())))))
                model.embedding = torch.nn.Embedding.from_pretrained(all_weights, freeze=True, padding_idx=0)
            test_sequences = [[ids[word] for word in doc if word in ids][:limit] for doc in test_docs]
            test_labels = frame.label.iloc[test].map({"negative": 0, "positive": 1}).to_numpy(dtype=np.int64)
            test_predictions = predict_gru(model, test_sequences, test_labels, np.arange(len(test)))
            save(args.study / "matched_trainable_meanpool_test.json", {
                "fingerprint": fingerprint, "seed": 42, "limit": limit, "test_reviews": len(test),
                "selection": "best inner-validation epoch using fixed seed 42",
                **scores(test_labels, test_predictions),
            })
        return

    runs = [
        {"seed": seed, **fit_model(weights, sequences, labels, (fit, inner, outer), seed, "meanpool", False)}
        for seed in SEEDS
    ]
    values = [run["project1_validation_macro_f1"] for run in runs]
    save(args.study / "matched_trainable_meanpool.json", {
        "fingerprint": fingerprint, "limit": limit, "optimizer": "Adagrad", "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE, "epoch_cap": EPOCHS, "patience": PATIENCE, "runs": runs,
        "mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1)),
    })


if __name__ == "__main__":
    main()
