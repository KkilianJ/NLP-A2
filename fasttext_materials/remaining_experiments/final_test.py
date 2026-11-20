#!/usr/bin/env python3
"""Evaluate one explicitly locked saved model on the held-out project test set."""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader

import bridge
from data_access import read_json, reserve_output
from error_analysis import resolve_model
from inference import load_lstm, validate_lr_run
import training
from validation_features import model_digest
import workflow_config as cfg


def reconstruct_indices(labels):
    indices = np.arange(len(labels))
    train, other = train_test_split(indices, test_size=.30, stratify=labels, random_state=42)
    validation, test = train_test_split(other, test_size=.5, stratify=np.asarray(labels)[other], random_state=42)
    rng = np.random.default_rng(42)
    return tuple(rng.permutation(x) for x in (train, validation, test))


def load_test_partition(data_dir):
    # The existing loader verifies both pinned files, ordered content, and the
    # original split before any held-out text is returned here.
    _, source = bridge.legacy.phase1.load_training_partition(data_dir, smoke=False)
    original = read_json(cfg.BASELINE / "run_config.json")["split"]
    if source["full_training_indices_sha256_le_int64"] != original["training_indices_sha256_le_int64"]:
        raise ValueError("Original training split differs; refusing final evaluation.")
    frames = [pd.read_parquet(data_dir / name, columns=["text", "label_text"]) for name in ("train.parquet", "test.parquet")]
    df = pd.concat(frames, ignore_index=True)
    train, validation, test = reconstruct_indices(df.label_text)
    if (bridge.legacy.indices_sha(train) != original["training_indices_sha256_le_int64"] or
            bridge.legacy.indices_sha(validation) != original["validation_indices_sha256_le_int64"] or len(test) != 7438):
        raise ValueError("The reconstructed held-out split does not match the original project split.")
    rows = df.iloc[test]
    labels = rows.label_text.map({"negative": 0, "positive": 1})
    if labels.isna().any():
        raise ValueError("Unexpected held-out label.")
    return np.asarray(test), rows.text.tolist(), labels.to_numpy(dtype=np.int64), original


def random_unseen_vector(token, seed, dimension=300, std=.05):
    digest = hashlib.sha256(f"unseen-random-v1\0{seed}\0{token}".encode("utf-8")).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return (generator.standard_normal(dimension) * std).astype(np.float32)


def lstm_probabilities(run_dir, texts, labels, binary_path, device, loaded=None):
    model, cache, manifest, policy, checkpoint = loaded or load_lstm(run_dir, device="cpu")
    tokens = json.loads((cache / "tokens.json").read_text(encoding="utf-8"))
    lookup = {token: i for i, token in enumerate(tokens) if i}
    limit = manifest["identity"]["max_length"]
    extra = []
    binary = None
    sequences = []
    for index, text in enumerate(texts, 1):
        ids = []
        for token in bridge.legacy.phase1.tokenize(text, "regex")[:limit]:
            token_id = lookup.get(token)
            if token_id is None:
                token_id = len(lookup) + 1
                lookup[token] = token_id
                if policy == "random_trainable":
                    spec = checkpoint["specification"]
                    vector = random_unseen_vector(token, spec["seed"], std=spec["random_embedding_std"])
                else:
                    if binary is None:
                        if model_digest(binary_path) != manifest["identity"]["model_sha256"]:
                            raise ValueError("Final evaluation requires the original fastText binary.")
                        import fasttext
                        binary = fasttext.load_model(str(binary_path))
                    vector = np.asarray(binary.get_word_vector(token), dtype=np.float32)
                if vector.shape != (300,) or not np.isfinite(vector).all():
                    raise ValueError(f"Invalid held-out token vector for {token!r}")
                extra.append(vector)
            ids.append(token_id)
        sequences.append((torch.tensor(ids, dtype=torch.long), int(labels[index - 1])))
        if index % 500 == 0 or index == len(texts):
            print(f"  Encoded {index:,}/{len(texts):,} held-out reviews", flush=True)
    if extra:
        matrix = torch.cat([model.embedding.weight.detach().cpu(), torch.from_numpy(np.stack(extra))], dim=0)
        model.embedding = nn.Embedding.from_pretrained(matrix, freeze=True, padding_idx=0)
    model.to(device).eval()
    loader = DataLoader(sequences, batch_size=32, shuffle=False, collate_fn=bridge.sequence_dataset.collate_reviews)
    probabilities = []
    with torch.no_grad():
        for ids, lengths, _ in loader:
            probabilities.append(torch.sigmoid(model(ids.to(device), lengths)).cpu().numpy())
    return np.concatenate(probabilities), {"family": "lstm", "embedding_policy": policy, "sequence_limit": limit,
                                           "new_token_types": len(extra),
                                           "new_token_policy": ("seeded token-hash random normal, not a fastText vector"
                                                                if policy == "random_trainable" else
                                                                "original binary subword vector; no fine-tuning for unseen tokens")}


def lr_probabilities(run_dir, texts, binary_path):
    validate_lr_run(run_dir)
    model_digest(binary_path)
    import fasttext
    binary = fasttext.load_model(str(binary_path))
    pool = bridge.load_phase2_module("embeddings").pool_vectors
    @lru_cache(maxsize=100_000)
    def vector(token):
        return binary.get_word_vector(token)
    features = np.empty((len(texts), 300), dtype=np.float32)
    for index, text in enumerate(texts):
        words = bridge.legacy.phase1.tokenize(text, "regex")
        features[index] = pool(vector(word) for word in words)[0]
        if (index + 1) % 500 == 0 or index + 1 == len(texts):
            print(f"  Pooled {index + 1:,}/{len(texts):,} held-out reviews", flush=True)
    pipeline = joblib.load(Path(run_dir) / "model.joblib")  # trusted local artifact only
    return pipeline.predict_proba(features)[:, list(pipeline.classes_).index(1)], {"family": "lr_mean_full_review"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="one locked model run directory or lstm512/lstm256/lr-mean alias")
    parser.add_argument("--confirm-model-locked", action="store_true", help="confirm selection is complete before test access")
    parser.add_argument("--data-dir", type=Path, default=cfg.DATA)
    parser.add_argument("--model", type=Path, default=cfg.MODEL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.confirm_model_locked:
        parser.error("Final-test access requires --confirm-model-locked after model selection is complete.")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    run_dir = resolve_model(args.run)
    if not (run_dir / "metrics.json").is_file():
        parser.error(f"Completed model metrics not found: {run_dir}")
    lr_model = (run_dir / "model.joblib").is_file()
    loaded = None
    if lr_model:
        validate_lr_run(run_dir)
    else:
        if not (run_dir / "best_model.pt").is_file():
            parser.error(f"Best checkpoint not found: {run_dir}")
        loaded = load_lstm(run_dir, device="cpu")
    if (lr_model or loaded[3] != "random_trainable") and not args.model.is_file():
        parser.error(f"Original fastText binary is required for unseen tokens: {args.model}")
    ids, texts, labels, split = load_test_partition(args.data_dir)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if lr_model:
        probabilities, details = lr_probabilities(run_dir, texts, args.model)
        model_file = run_dir / "model.joblib"
    else:
        probabilities, details = lstm_probabilities(run_dir, texts, labels, args.model, device, loaded=loaded)
        model_file = run_dir / "best_model.pt"
    scores = bridge.legacy.evaluate(labels, probabilities)
    output = reserve_output(args.output)
    pd.DataFrame({"review_id": ids, "true_label": labels, "probability_positive": probabilities,
                  "prediction": (probabilities >= .5).astype(int)}).to_csv(output / "test_predictions.csv", index=False)
    training.write_json(output / "metrics.json", {"test": scores, "n": len(ids), "model": str(run_dir.resolve()),
                        "test_indices_sha256_le_int64": bridge.legacy.indices_sha(ids), "split": split, "details": details,
                        "model_checkpoint_sha256": bridge.legacy.phase1.file_sha256(model_file),
                        "environment": training.versions(), "threshold": .5,
                        "note": "One explicitly locked model; do not use test scores for further model selection."})
    bridge.load_phase3_plotting().confusion_plot(scores["confusion_matrix"], output / "confusion_matrix.png",
                                                "Held-out project test: locked model")
    print(f"Saved locked-model held-out evaluation to {output}.")


if __name__ == "__main__":
    main()
