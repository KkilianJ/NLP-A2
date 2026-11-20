"""Training and validation for the single frozen-embedding LSTM experiment."""
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

import experiment_settings as cfg
from legacy import evaluate
from sequence_dataset import SequenceDataset, collate_reviews
from sequence_model import SentimentLSTM


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class EarlyStopping:
    """Strict improvement resets patience; equal scores keep the earlier epoch."""
    def __init__(self, patience):
        self.patience = patience
        self.best_score = float("-inf")
        self.best_epoch = None
        self.stale_epochs = 0

    def update(self, score, epoch):
        if not np.isfinite(score):
            raise ValueError("Validation macro-F1 is non-finite.")
        if score > self.best_score:
            self.best_score, self.best_epoch, self.stale_epochs = float(score), epoch, 0
            return True
        self.stale_epochs += 1
        return False

    @property
    def should_stop(self):
        return self.stale_epochs >= self.patience


def score_model(model, loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = total = 0
    labels_all, probabilities_all = [], []
    with torch.no_grad():
        for token_ids, lengths, labels in loader:
            token_ids = token_ids.to(device)
            logits = model(token_ids, lengths)
            labels_device = labels.to(device)
            total_loss += float(criterion(logits, labels_device).item())
            total += len(labels)
            labels_all.append(labels.numpy().astype(np.int64))
            probabilities_all.append(torch.sigmoid(logits).cpu().numpy())
    if not total:
        raise ValueError("Cannot evaluate an empty partition.")
    return total_loss / total, evaluate(np.concatenate(labels_all), np.concatenate(probabilities_all))


def fit(cache_dir, manifest, output, epochs, seed, batch_size, learning_rate, hidden_size,
        head="linear", head_hidden_size=64):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on {device}; maximum {epochs} epochs; early-stopping patience {cfg.EARLY_STOP_PATIENCE}.", flush=True)
    train_data = SequenceDataset(cache_dir, "train")
    validation_data = SequenceDataset(cache_dir, "validation")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator,
                              collate_fn=collate_reviews, num_workers=0)
    train_eval_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False,
                                   collate_fn=collate_reviews, num_workers=0)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False,
                                   collate_fn=collate_reviews, num_workers=0)
    embedding = np.load(cache_dir / "embedding_table.npy", mmap_mode="r", allow_pickle=False)
    # Copy once; torch cannot safely own a read-only memory map.
    weights = torch.from_numpy(np.array(embedding, dtype=np.float32, copy=True))
    model = SentimentLSTM(weights, hidden_size=hidden_size, head=head,
                          head_hidden_size=head_hidden_size).to(device)
    if model.embedding.weight.requires_grad:
        raise RuntimeError("The pretrained embedding lookup must remain frozen.")
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=learning_rate, weight_decay=0.0)
    criterion = nn.BCEWithLogitsLoss()
    history = []
    stopping = EarlyStopping(cfg.EARLY_STOP_PATIENCE)
    early_stopped = False
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        progress_every = max(1, (len(train_loader) + 3) // 4)
        for step, (token_ids, lengths, labels) in enumerate(train_loader, 1):
            token_ids, labels = token_ids.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(token_ids, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            if step % progress_every == 0 or step == len(train_loader):
                print(f"  Epoch {epoch:02d}/{epochs:02d}: batch {step}/{len(train_loader)}", flush=True)
        train_loss, train_metrics = score_model(model, train_eval_loader, device)
        val_loss, val_metrics = score_model(model, validation_loader, device)
        row = {"epoch": epoch, "training_loss": train_loss, "validation_loss": val_loss,
               "training_accuracy": train_metrics["accuracy"], "validation_accuracy": val_metrics["accuracy"],
               "training_macro_f1": train_metrics["macro_f1"], "validation_macro_f1": val_metrics["macro_f1"],
               "training_validation_accuracy_gap": train_metrics["accuracy"] - val_metrics["accuracy"],
               "training_validation_macro_f1_gap": train_metrics["macro_f1"] - val_metrics["macro_f1"]}
        history.append(row)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        write_json(output / "history.json", history)
        if stopping.update(val_metrics["macro_f1"], epoch):
            # The frozen table is already stored once in the cache. Save only
            # trainable recurrent/classifier weights, with cache fingerprints.
            state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
                     if name != "embedding.weight"}
            torch.save({"trainable_state_dict": state, "epoch": epoch, "hidden_size": hidden_size,
                        "head": head, "head_hidden_size": head_hidden_size if head == "mlp" else None,
                        "embedding_dimension": cfg.DIMENSION, "seed": seed, "threshold": cfg.THRESHOLD,
                        "max_sequence_length": manifest["identity"]["max_length"],
                        "sequence_cache_key": cache_dir.name,
                        "embedding_table_sha256": manifest["embedding_table_sha256"],
                        "tokens_sha256": manifest["tokens_sha256"],
                        "label_mapping": {"negative": 0, "positive": 1}}, output / "best_model.pt")
        print(f"  Epoch {epoch:02d}/{epochs:02d} | loss {train_loss:.4f}/{val_loss:.4f} "
              f"| accuracy {train_metrics['accuracy']:.4f}/{val_metrics['accuracy']:.4f} "
              f"| macro-F1 {train_metrics['macro_f1']:.4f}/{val_metrics['macro_f1']:.4f} (train/val) "
              f"| best epoch {stopping.best_epoch}, patience {stopping.stale_epochs}/{cfg.EARLY_STOP_PATIENCE}", flush=True)
        if stopping.should_stop:
            early_stopped = True
            print("Early stopping: five consecutive epochs without a strict validation macro-F1 improvement.", flush=True)
            break
    training_seconds = time.perf_counter() - start
    # Explicitly restore the best saved state before computing final metrics.
    checkpoint = torch.load(output / "best_model.pt", map_location="cpu", weights_only=True)
    if checkpoint["embedding_table_sha256"] != manifest["embedding_table_sha256"]:
        raise ValueError("Best checkpoint references a different embedding table.")
    status = model.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
    if set(status.missing_keys) != {"embedding.weight"} or status.unexpected_keys:
        raise ValueError(f"Unexpected checkpoint keys: {status}")
    train_loss, train_metrics = score_model(model, train_eval_loader, device)
    val_loss, val_metrics = score_model(model, validation_loader, device)
    # Package the selected trainable weights, frozen vectors, and exact token-ID
    # mapping together once at the end. This makes a completed checkpoint usable
    # even if the feature cache is later moved; no original 6.7-GiB model copy.
    checkpoint["embedding_table"] = weights
    checkpoint["tokens"] = json.loads((cache_dir / "tokens.json").read_text(encoding="utf-8"))
    checkpoint["tokenizer"] = "Phase 1 regex; preserve case and HTML markup"
    temporary_checkpoint = output / ".best_model.pt.part"
    torch.save(checkpoint, temporary_checkpoint)
    temporary_checkpoint.replace(output / "best_model.pt")
    result = {"model": "LSTM", "head": head, "head_hidden_size": head_hidden_size if head == "mlp" else None,
              "device": str(device), "seed": seed, "threshold": cfg.THRESHOLD,
              "hidden_size": hidden_size, "max_sequence_length": manifest["identity"]["max_length"],
              "best_epoch": stopping.best_epoch, "epochs_completed": len(history), "early_stopped": early_stopped,
              "checkpoint_selection": "strictly highest validation macro-F1; earliest epoch wins ties",
              "patience": cfg.EARLY_STOP_PATIENCE, "training_seconds": training_seconds,
              "time_definition": "Training loop including per-epoch train/validation evaluation and checkpoint writes; excludes preprocessing and final best-checkpoint reevaluation.",
              "training_loss": train_loss, "validation_loss": val_loss,
              "training_validation_accuracy_gap": train_metrics["accuracy"] - val_metrics["accuracy"],
              "training_validation_macro_f1_gap": train_metrics["macro_f1"] - val_metrics["macro_f1"],
              "training": train_metrics, "validation": val_metrics}
    write_json(output / "metrics.json", result)
    flat = {"model": "LSTM", "head": head,
            "head_hidden_size": head_hidden_size if head == "mlp" else None,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"], "validation_accuracy": val_metrics["accuracy"],
            "validation_macro_f1": val_metrics["macro_f1"], "validation_loss": val_loss,
            "training_validation_accuracy_gap": train_metrics["accuracy"] - val_metrics["accuracy"],
            "training_validation_macro_f1_gap": train_metrics["macro_f1"] - val_metrics["macro_f1"],
            "negative_precision": val_metrics["per_class"]["negative"]["precision"],
            "negative_recall": val_metrics["per_class"]["negative"]["recall"],
            "positive_precision": val_metrics["per_class"]["positive"]["precision"],
            "positive_recall": val_metrics["per_class"]["positive"]["recall"],
            "tn": val_metrics["confusion_matrix"][0][0], "fp": val_metrics["confusion_matrix"][0][1],
            "fn": val_metrics["confusion_matrix"][1][0], "tp": val_metrics["confusion_matrix"][1][1],
            "best_epoch": stopping.best_epoch, "epochs_completed": len(history), "training_seconds": training_seconds}
    pd.DataFrame([flat]).to_csv(output / "metrics.csv", index=False)
    return result
