"""Load trusted completed checkpoints and predict, without fitting anything."""
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

import bridge
from data_access import pooled_cache, read_json, sequence_cache
from modeling import restore_model
import workflow_config as cfg


def validate_lr_run(run_dir):
    run_dir = Path(run_dir)
    metrics = read_json(run_dir / "metrics.json")
    if metrics.get("model") != "Logistic Regression" or metrics.get("pooling") != "mean":
        raise ValueError("Expected a completed mean-pooling logistic-regression model.")
    path = run_dir / "run_config.json"
    if not path.is_file():
        path = run_dir.parent / "run_config.json"
    config = read_json(path)
    _, expected, root = pooled_cache()
    actual_identity = config.get("pooled_feature_identity") or config.get("features", {}).get("identity")
    if actual_identity != expected["features"]["identity"] or config["split"] != expected["split"]:
        raise ValueError("Mean-pooling model metadata does not match the original features and split.")
    if "pooled_file_sha256" in config:
        for name, digest in config["pooled_file_sha256"].items():
            if bridge.legacy.phase1.file_sha256(root / f"{name}_mean.npy") != digest:
                raise ValueError("Mean-pooling feature array differs from the saved run checksum.")
    return metrics


def load_lstm(run_dir, device="cpu"):
    run_dir = Path(run_dir)
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("format") == "remaining-lstm-v1":
        cache, manifest, _ = sequence_cache()
        config = read_json(run_dir / "run_config.json")
        if (checkpoint["cache_key"] != cache.name or checkpoint["tokens_sha256"] != manifest["tokens_sha256"]
                or checkpoint["embedding_table_sha256"] != manifest["embedding_table_sha256"]
                or config["cache_files_sha256"] != manifest["files_sha256"]):
            raise ValueError("Checkpoint and original sequence/vector cache do not match.")
        if checkpoint["specification"] != config["specification"]:
            raise ValueError("Checkpoint configuration differs from its saved run metadata.")
        weights = torch.from_numpy(np.array(np.load(cache / "embedding_table.npy", mmap_mode="r"), copy=True))
        model = restore_model(checkpoint, weights)
        limit = checkpoint["specification"]["max_sequence_length"]
        embedding_policy = checkpoint["specification"]["embedding"]
    else:
        cache, manifest, config = sequence_cache(run_dir)
        if (checkpoint["tokens_sha256"] != manifest["tokens_sha256"] or
                checkpoint["embedding_table_sha256"] != manifest["embedding_table_sha256"]):
            raise ValueError("Old checkpoint references a different vector cache.")
        model = bridge.sequence_model.model_from_checkpoint(checkpoint)
        limit = checkpoint["max_sequence_length"]
        embedding_policy = "pretrained_frozen"
    if limit != manifest["identity"]["max_length"]:
        raise ValueError("Checkpoint sequence limit differs from its cache.")
    if checkpoint["epoch"] != read_json(run_dir / "metrics.json")["best_epoch"]:
        raise ValueError("Checkpoint epoch differs from the selected metrics.")
    return model.to(device).eval(), cache, manifest, embedding_policy, checkpoint


def predict_validation(run_dir, device="cpu"):
    run_dir = Path(run_dir)
    metrics = read_json(run_dir / "metrics.json")
    if (run_dir / "model.joblib").is_file():
        validate_lr_run(run_dir)
        arrays, _, _ = pooled_cache()
        cache, _, _ = sequence_cache()
        pipeline = joblib.load(run_dir / "model.joblib")  # trusted local artifact only
        classes = list(pipeline.classes_)
        if 1 not in classes:
            raise ValueError("Positive class is missing from the logistic model.")
        probabilities = pipeline.predict_proba(arrays["validation"])[:, classes.index(1)]
        kind = "lr_mean_full_review"
    else:
        model, cache, _, _, _ = load_lstm(run_dir, device=device)
        dataset = bridge.sequence_dataset.SequenceDataset(cache, "validation")
        loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=bridge.sequence_dataset.collate_reviews)
        probabilities = []
        with torch.no_grad():
            for token_ids, lengths, _ in loader:
                probabilities.append(torch.sigmoid(model(token_ids.to(device), lengths)).cpu().numpy())
        probabilities = np.concatenate(probabilities)
        kind = "lstm"
    ids = np.load(cache / "validation_review_ids.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(cache / "validation_labels.npy", mmap_mode="r", allow_pickle=False)
    if len(ids) != len(probabilities) or not np.isfinite(probabilities).all():
        raise ValueError("Prediction count or values are invalid.")
    return {"run_dir": str(run_dir.resolve()), "kind": kind, "ids": np.asarray(ids),
            "labels": np.asarray(labels), "probabilities": np.asarray(probabilities), "metrics": metrics}


def gap_metrics(metrics):
    return {"training_accuracy": metrics["training"]["accuracy"], "validation_accuracy": metrics["validation"]["accuracy"],
            "accuracy_gap": metrics["training"]["accuracy"] - metrics["validation"]["accuracy"],
            "training_macro_f1": metrics["training"]["macro_f1"], "validation_macro_f1": metrics["validation"]["macro_f1"],
            "macro_f1_gap": metrics["training"]["macro_f1"] - metrics["validation"]["macro_f1"],
            "note": "For B1 the training metric uses the selected half, not the full training set."}
