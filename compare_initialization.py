"""Compare random and GloVe initialization for trainable MeanPool embeddings."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn

from compare_gru import batches, evaluate
from complete_embeddings import DIMENSION, load_vectors
from prepare_meanpool import load_project1, load_reviews, tokenize

SEEDS = (42, 43, 44)
MAX_TOKENS = 128
BATCH_SIZE = 256
EPOCHS = 12
PATIENCE = 3
LEARNING_RATE = 0.05


class MeanPool(nn.Module):
    def __init__(self, weights, freeze=False):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            weights.clone(), freeze=freeze, padding_idx=0, sparse=True,
        )
        self.head = nn.Linear(weights.shape[1], 1)

    def forward(self, ids, lengths):
        average = self.embedding(ids).sum(dim=1) / lengths.clamp_min(1).unsqueeze(1)
        return self.head(average).squeeze(1)


def fit_model(weights, sequences, labels, splits, seed, name, freeze, on_checkpoint=None):
    fit, inner, validation = splits
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MeanPool(weights, freeze=freeze)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adagrad(parameters, lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    best_score = -1.0
    best_epoch = 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for ids, lengths, target in batches(sequences, labels, fit, BATCH_SIZE, "cpu", rng):
            optimizer.zero_grad()
            loss = criterion(model(ids, lengths), target.float())
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(target)
        score = evaluate(model, batches(sequences, labels, inner, BATCH_SIZE, "cpu"))
        history.append({"epoch": epoch, "train_loss": loss_sum / len(fit), "inner_macro_f1": score})
        if score > best_score:
            best_score, best_epoch = score, epoch
            checkpoint = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if epoch - best_epoch >= PATIENCE:
            break
    model.load_state_dict(checkpoint)
    if on_checkpoint is not None:
        on_checkpoint(model)
    score = evaluate(model, batches(sequences, labels, validation, BATCH_SIZE, "cpu"))
    print(f"Seed {seed}, {name}: epoch {best_epoch}, validation macro-F1 {score:.4f}", flush=True)
    return {"best_epoch": best_epoch, "project1_validation_macro_f1": score, "history": history}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project1", type=Path, required=True)
    parser.add_argument("--glove", type=Path, required=True)
    parser.add_argument("--arrow-dir", type=Path)
    parser.add_argument("--freeze", action="store_true", help="Train only the classifier head")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        name = "frozen_initialization_results.json" if args.freeze else "initialization_results.json"
        args.output = Path(name)
    torch.set_num_threads(4)

    project = load_project1(args.project1)
    frame, (train_ids, val_ids, _), fingerprint = load_reviews(project, args.arrow_dir)
    row_ids = np.concatenate((train_ids, val_ids))
    labels = frame.label.iloc[row_ids].map({"negative": 0, "positive": 1}).to_numpy(dtype=np.int64)
    fit, inner = train_test_split(
        np.arange(len(train_ids)), test_size=0.10, stratify=labels[: len(train_ids)], random_state=42,
    )
    val = np.arange(len(train_ids), len(row_ids))
    tokenized = tokenize(frame.text.iloc[row_ids])
    vocabulary = sorted({word for i in fit for word in tokenized[i][:MAX_TOKENS]})
    word_ids = {word: i for i, word in enumerate(vocabulary, start=1)}
    sequences = [
        [word_ids[word] for word in words[:MAX_TOKENS] if word in word_ids]
        for words in tokenized
    ]
    glove = load_vectors(args.glove, word_ids)
    covered_ids = [word_ids[word] for word in glove]
    covered_vectors = torch.from_numpy(np.stack(list(glove.values())))
    scale = float(covered_vectors.std())
    result = {
        "ordered_sha256": fingerprint,
        "inner_split_seed": 42,
        "fitting_reviews": len(fit),
        "inner_validation_reviews": len(inner),
        "project1_validation_reviews": len(val),
        "tokenization": "regex_keep_contractions",
        "max_tokens": MAX_TOKENS,
        "dimensions": DIMENSION,
        "embeddings_frozen": args.freeze,
        "training_vocabulary_types": len(vocabulary),
        "types_initialized_from_glove": len(glove),
        "random_initialization": {"distribution": "normal", "mean": 0.0, "sd": scale},
        "optimizer": "Adagrad",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epoch_cap": EPOCHS,
        "patience": PATIENCE,
        "runs": [],
    }
    print(f"Vocabulary: {len(vocabulary):,} types; {len(glove):,} GloVe hits", flush=True)
    for seed in SEEDS:
        generator = torch.Generator().manual_seed(seed)
        random = torch.randn(len(vocabulary) + 1, DIMENSION, generator=generator) * scale
        random[0].zero_()
        pretrained = random.clone()
        pretrained[covered_ids] = covered_vectors
        run = {"seed": seed}
        for name, weights in (("random", random), ("glove", pretrained)):
            run[name] = fit_model(weights, sequences, labels, (fit, inner, val), seed, name, args.freeze)
        run["glove_minus_random"] = (
            run["glove"]["project1_validation_macro_f1"] - run["random"]["project1_validation_macro_f1"]
        )
        result["runs"].append(run)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    scores = {
        name: [row[name]["project1_validation_macro_f1"] for row in result["runs"]]
        for name in ("random", "glove")
    }
    scores["glove_minus_random"] = [row["glove_minus_random"] for row in result["runs"]]
    result["summary"] = {
        key: {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1))}
        for key, values in scores.items()
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
