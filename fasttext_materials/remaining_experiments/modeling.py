"""Controlled extensions of the original final-valid-state LSTM."""
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split

import bridge


def training_positions(labels, fraction, subset_seed=42):
    labels = np.asarray(labels)
    indices = np.arange(len(labels))
    if fraction == 1.0:
        return indices
    if fraction != 0.5:
        raise ValueError("Only the requested 50% stratified subset is supported.")
    selected, _ = train_test_split(indices, train_size=0.5, stratify=labels, random_state=subset_seed)
    return np.asarray(selected, dtype=np.int64)


def make_model(weights, spec, initialize_random=True):
    model = bridge.sequence_model.SentimentLSTM(weights, hidden_size=spec["hidden_size"], head="linear")
    if spec["dropout"]:
        model.classifier = nn.Sequential(nn.Dropout(spec["dropout"]), model.classifier)
    if spec["embedding"] == "random_trainable" and initialize_random:
        # Independent generator preserves the LSTM's seed-42 initialization.
        generator = torch.Generator(device="cpu").manual_seed(spec["seed"])
        with torch.no_grad():
            model.embedding.weight.normal_(spec["random_embedding_mean"], spec["random_embedding_std"], generator=generator)
            model.embedding.weight[0].zero_()
    if spec["embedding"] not in ("pretrained_frozen", "pretrained_trainable", "random_trainable"):
        raise ValueError("Unknown embedding initialization policy.")
    model.embedding.weight.requires_grad_(spec["embedding"] != "pretrained_frozen")
    return model


def restore_model(checkpoint, base_weights):
    spec = checkpoint["specification"]
    if spec["embedding"] != "pretrained_frozen" and "embedding_table" not in checkpoint:
        raise ValueError("A trainable-embedding checkpoint must contain its selected embedding weights.")
    embedding = checkpoint.get("embedding_table", base_weights)
    model = make_model(embedding.clone(), spec, initialize_random=False)
    status = model.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
    if set(status.missing_keys) != {"embedding.weight"} or status.unexpected_keys:
        raise ValueError(f"Unexpected checkpoint keys: {status}")
    return model
