"""Compare frozen GloVe MeanPool and a bidirectional GRU on identical sequences."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn

from complete_embeddings import load_vectors
from prepare_meanpool import load_project1, load_reviews, tokenize


class Classifier(nn.Module):
    def __init__(self, weights, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=True, padding_idx=0)
        self.recurrent = hidden_size is not None
        if self.recurrent:
            self.forward_gru = nn.GRU(weights.shape[1], hidden_size, batch_first=True)
            self.backward_gru = nn.GRU(weights.shape[1], hidden_size, batch_first=True)
        self.head = nn.Linear(2 * hidden_size if self.recurrent else weights.shape[1], 1)

    def forward(self, ids, lengths):
        x = self.embedding(ids)
        if not self.recurrent:
            pooled = x.sum(dim=1) / lengths.clamp_min(1).unsqueeze(1)
        else:
            rows = torch.arange(len(ids), device=ids.device)
            last = lengths.clamp_min(1) - 1
            offsets = torch.arange(ids.shape[1], device=ids.device)
            reverse_index = (last.unsqueeze(1) - offsets).clamp_min(0)
            reversed_ids = ids.gather(1, reverse_index).masked_fill(offsets >= lengths.unsqueeze(1), 0)
            forward, _ = self.forward_gru(x)
            backward, _ = self.backward_gru(self.embedding(reversed_ids))
            pooled = torch.cat((forward[rows, last], backward[rows, last]), dim=1)
            pooled = pooled.masked_fill(lengths.unsqueeze(1) == 0, 0)
        return self.head(pooled).squeeze(1)


def batches(sequences, labels, indices, batch_size, device, rng=None):
    order = rng.permutation(indices) if rng is not None else indices
    for start in range(0, len(order), batch_size * 16):
        block = sorted(order[start : start + batch_size * 16], key=lambda i: len(sequences[i]))
        groups = [block[i : i + batch_size] for i in range(0, len(block), batch_size)]
        if rng is not None:
            rng.shuffle(groups)
        for group in groups:
            lengths = np.array([len(sequences[i]) for i in group], dtype=np.int64)
            ids = np.zeros((len(group), max(1, lengths.max())), dtype=np.int64)
            for j, i in enumerate(group):
                ids[j, : lengths[j]] = sequences[i]
            yield (
                torch.from_numpy(ids).to(device),
                torch.from_numpy(lengths).to(device),
                torch.from_numpy(labels[group]).to(device),
            )


def evaluate(model, data):
    model.eval()
    expected, predicted = [], []
    with torch.inference_mode():
        for ids, lengths, labels in data:
            predicted.extend((model(ids, lengths) >= 0).cpu().tolist())
            expected.extend(labels.cpu().tolist())
    return float(f1_score(expected, predicted, average="macro"))


def train_model(name, hidden_size, weights, sequences, labels, splits, args):
    fit, inner, validation = splits
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = Classifier(weights, hidden_size).to(args.device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    best_score, best_epoch = -1.0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for ids, lengths, target in batches(sequences, labels, fit, args.batch_size, args.device, rng):
            optimizer.zero_grad()
            loss = criterion(model(ids, lengths), target.float())
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            loss_sum += loss.detach().item() * len(target)
        score = evaluate(model, batches(sequences, labels, inner, args.batch_size, args.device))
        row = {"epoch": epoch, "train_loss": loss_sum / len(fit), "inner_macro_f1": score}
        history.append(row)
        print(name, json.dumps(row), flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
            checkpoint = deepcopy(model.head.state_dict()) if hidden_size is None else {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
                if key != "embedding.weight"
            }
        if epoch - best_epoch >= args.patience:
            break
    if hidden_size is None:
        model.head.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint, strict=False)
    return {
        "trainable_parameters": sum(p.numel() for p in parameters),
        "best_epoch": best_epoch,
        "best_inner_macro_f1": best_score,
        "project1_validation_macro_f1": evaluate(
            model, batches(sequences, labels, validation, args.batch_size, args.device),
        ),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project1", type=Path, required=True)
    parser.add_argument("--glove", type=Path, required=True)
    parser.add_argument("--arrow-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("gru_comparison.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()
    if min(args.max_tokens, args.batch_size, args.epochs, args.patience, args.threads) < 1:
        parser.error("Token limit, batch size, epochs, patience and threads must be positive.")
    torch.set_num_threads(args.threads)

    project = load_project1(args.project1)
    frame, (train_ids, validation_ids, _), fingerprint = load_reviews(project, args.arrow_dir)
    row_ids = np.concatenate((train_ids, validation_ids))
    raw_tokens = tokenize(frame.text.iloc[row_ids])
    glove = load_vectors(args.glove, {word for review in raw_tokens for word in review})
    word_ids = {word: i for i, word in enumerate(glove, start=1)}
    weights = torch.zeros(len(glove) + 1, len(next(iter(glove.values()))))
    weights[1:] = torch.from_numpy(np.stack(list(glove.values())))
    sequences = []
    truncated = 0
    for review in raw_tokens:
        covered = [word_ids[word] for word in review if word in word_ids]
        truncated += len(covered) > args.max_tokens
        sequences.append(covered[: args.max_tokens])
    labels = frame.label.iloc[row_ids].map({"negative": 0, "positive": 1}).to_numpy(dtype=np.int64)
    fit, inner = train_test_split(
        np.arange(len(train_ids)), test_size=0.10, stratify=labels[:len(train_ids)], random_state=42,
    )
    validation = np.arange(len(train_ids), len(row_ids))
    result = {
        "ordered_sha256": fingerprint,
        "training_seed": args.seed,
        "inner_split_seed": 42,
        "fitting_reviews": len(fit),
        "inner_validation_reviews": len(inner),
        "project1_validation_reviews": len(validation),
        "tokenization": "regex_keep_contractions",
        "oov_policy": "drop",
        "max_glove_covered_tokens": args.max_tokens,
        "truncated_reviews_across_splits": truncated,
        "device": args.device,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epoch_cap": args.epochs,
        "patience": args.patience,
        "models": {},
    }
    print(f"Prepared {len(row_ids):,} reviews; truncated {truncated:,}", flush=True)
    for name, hidden_size in (("meanpool", None), ("bidirectional_gru", 64)):
        result["models"][name] = train_model(
            name, hidden_size, weights, sequences, labels, (fit, inner, validation), args,
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output.read_text(), flush=True)


if __name__ == "__main__":
    main()
