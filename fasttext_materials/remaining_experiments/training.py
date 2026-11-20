"""Single-run LSTM training on the unchanged cached project split."""
import hashlib
import importlib.metadata
import json
import platform
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

import bridge
from modeling import make_model, training_positions
import workflow_config as cfg


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def versions():
    packages = ("torch", "numpy", "pandas", "scikit-learn", "fasttext", "matplotlib")
    return {"python": sys.version, "platform": platform.platform(),
            **{name: importlib.metadata.version(name) for name in packages}}


def plot_history(output, result):
    plots = bridge.load_phase3_plotting()
    history = pd.read_csv(output / "history.csv")
    graphs = output / "graphs"
    graphs.mkdir(exist_ok=True)
    title = f"{result['experiment']} (seed {result['seed']}), best epoch {result['best_epoch']}"
    for key, ylabel in (("loss", "BCE loss"), ("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")):
        plots.line_plot(history, [f"training_{key}", f"validation_{key}"], ["Training", "Validation"],
                        ylabel, f"{title}: {ylabel}", result["best_epoch"], graphs / f"{key}_by_epoch.png")
    plots.confusion_plot(result["validation"]["confusion_matrix"], graphs / "confusion_matrix.png", title)


def train_lstm(spec, cache, manifest, original_config, output):
    seed = spec["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train = bridge.sequence_dataset.SequenceDataset(cache, "train")
    val = bridge.sequence_dataset.SequenceDataset(cache, "validation")
    positions = training_positions(train.labels, spec["train_fraction"], spec["subset_seed"])
    selected = Subset(train, positions.tolist())
    ids = np.load(cache / "train_review_ids.npy", mmap_mode="r", allow_pickle=False)[positions]
    np.save(output / "training_review_ids.npy", ids, allow_pickle=False)
    training_ids_hash = bridge.legacy.indices_sha(ids)
    generator = torch.Generator().manual_seed(seed)
    loader_args = {"batch_size": spec["batch_size"], "collate_fn": bridge.sequence_dataset.collate_reviews,
                   "num_workers": spec["data_loader_workers"]}
    train_loader = DataLoader(selected, shuffle=True, generator=generator, **loader_args)
    train_eval = DataLoader(selected, shuffle=False, **loader_args)
    val_loader = DataLoader(val, shuffle=False, **loader_args)
    weights = torch.from_numpy(np.array(np.load(cache / "embedding_table.npy", mmap_mode="r"), dtype=np.float32, copy=True))
    model = make_model(weights, spec).to(device)
    trainable_embedding = spec["embedding"] != "pretrained_frozen"
    if model.embedding.weight.requires_grad != trainable_embedding:
        raise ValueError("Embedding trainability does not match the configuration.")
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=spec["learning_rate"], weight_decay=spec["weight_decay"],
                                 betas=tuple(spec["adam_betas"]), eps=spec["adam_epsilon"])
    criterion = nn.BCEWithLogitsLoss(reduction=spec["loss_reduction"])
    identity = {"split": original_config["split"], "cache_key": cache.name,
                "cache_identity": manifest["identity"], "cache_files_sha256": manifest["files_sha256"],
                "selected_training_review_ids_sha256": training_ids_hash, "selected_training_count": len(ids),
                "vocabulary_includes_validation_tokens": True,
                "embedding_policy": "Exact pretrained token vectors include fastText OOV subwords; "
                    "trainable lookup rows are fine-tuned, not the original fastText n-gram model. "
                    "Random mode uses the same IDs but no pretrained/subword information.",
                "embedding_checksum_note": "The cache embedding checksum identifies the original source table; "
                    "for trainable runs the selected changed table is stored in and covered by the checkpoint SHA-256.",
                "random_initialization": "independent seeded normal(0, 0.05), padding row zero",
                "specification": spec, "environment": versions(), "device": str(device),
                "workflow_source_sha256": {path.name: bridge.legacy.phase1.file_sha256(path) for path in
                    (cfg.ROOT / "experiments.json", cfg.ROOT / "workflow_config.py", cfg.ROOT / "modeling.py",
                     cfg.ROOT / "training.py", cfg.ROOT / "data_access.py", cfg.ROOT / "run.py")},
                "checkpoint_selection": "strictly greatest validation macro-F1; earliest epoch wins ties"}
    write_json(output / "run_config.json", identity)
    stopper = bridge.lstm_training.EarlyStopping(spec["patience"])
    history = []
    start = time.perf_counter()
    print(f"Training {spec['experiment']} seed {seed} on {device}, {len(selected):,} training reviews; "
          f"embedding={spec['embedding']}", flush=True)
    for epoch in range(1, spec["epochs"] + 1):
        model.train()
        progress = max(1, (len(train_loader) + 3) // 4)
        for step, (token_ids, lengths, labels) in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            logits = model(token_ids.to(device), lengths)
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            if step % progress == 0 or step == len(train_loader):
                print(f"  epoch {epoch:02d}/{spec['epochs']}: batch {step}/{len(train_loader)}", flush=True)
        train_loss, train_scores = bridge.lstm_training.score_model(model, train_eval, device)
        val_loss, val_scores = bridge.lstm_training.score_model(model, val_loader, device)
        history.append({"epoch": epoch, "training_loss": train_loss, "validation_loss": val_loss,
                        "training_accuracy": train_scores["accuracy"], "validation_accuracy": val_scores["accuracy"],
                        "training_macro_f1": train_scores["macro_f1"], "validation_macro_f1": val_scores["macro_f1"]})
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        write_json(output / "history.json", history)
        if stopper.update(val_scores["macro_f1"], epoch):
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k != "embedding.weight"}
            checkpoint = {"format": "remaining-lstm-v1", "specification": spec, "epoch": epoch,
                          "trainable_state_dict": state, "cache_key": cache.name,
                          "embedding_table_sha256": manifest["embedding_table_sha256"],
                          "tokens_sha256": manifest["tokens_sha256"]}
            if trainable_embedding:
                checkpoint["embedding_table"] = model.embedding.weight.detach().cpu().clone()
            temp = output / ".best_model.pt.part"
            torch.save(checkpoint, temp)
            temp.replace(output / "best_model.pt")
        print(f"  epoch {epoch:02d}: loss {train_loss:.4f}/{val_loss:.4f}, "
              f"accuracy {train_scores['accuracy']:.4f}/{val_scores['accuracy']:.4f}, "
              f"macro-F1 {train_scores['macro_f1']:.4f}/{val_scores['macro_f1']:.4f}; "
              f"best={stopper.best_epoch}, patience={stopper.stale_epochs}/{spec['patience']}", flush=True)
        if stopper.should_stop:
            print("Early stopping after five epochs without strict validation macro-F1 improvement.", flush=True)
            break
    seconds = time.perf_counter() - start
    checkpoint = torch.load(output / "best_model.pt", map_location="cpu", weights_only=True)
    if trainable_embedding:
        with torch.no_grad():
            model.embedding.weight.copy_(checkpoint["embedding_table"].to(device))
    status = model.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
    if set(status.missing_keys) != {"embedding.weight"} or status.unexpected_keys:
        raise ValueError(f"Invalid best checkpoint: {status}")
    train_loss, train_scores = bridge.lstm_training.score_model(model, train_eval, device)
    val_loss, val_scores = bridge.lstm_training.score_model(model, val_loader, device)
    result = {"experiment": spec["experiment"], "seed": seed, "device": str(device),
              "best_epoch": stopper.best_epoch, "epochs_completed": len(history), "training_seconds": seconds,
              "time_definition": "Training loop including per-epoch train/validation evaluation and checkpoint writes; "
                                 "excludes cache loading and final best-checkpoint reevaluation.",
              "training_loss": train_loss, "validation_loss": val_loss, "training": train_scores,
              "validation": val_scores, "training_validation_accuracy_gap": train_scores["accuracy"] - val_scores["accuracy"],
              "training_validation_macro_f1_gap": train_scores["macro_f1"] - val_scores["macro_f1"],
              "checkpoint_sha256": bridge.legacy.phase1.file_sha256(output / "best_model.pt")}
    write_json(output / "metrics.json", result)
    flat = {k: result[k] for k in ("experiment", "seed", "best_epoch", "epochs_completed", "training_seconds",
                                  "training_validation_accuracy_gap", "training_validation_macro_f1_gap")}
    flat.update({"training_accuracy": train_scores["accuracy"], "training_macro_f1": train_scores["macro_f1"],
                 "validation_accuracy": val_scores["accuracy"], "validation_macro_f1": val_scores["macro_f1"],
                 "negative_precision": val_scores["per_class"]["negative"]["precision"],
                 "negative_recall": val_scores["per_class"]["negative"]["recall"],
                 "positive_precision": val_scores["per_class"]["positive"]["precision"],
                 "positive_recall": val_scores["per_class"]["positive"]["recall"],
                 "tn": val_scores["confusion_matrix"][0][0], "fp": val_scores["confusion_matrix"][0][1],
                 "fn": val_scores["confusion_matrix"][1][0], "tp": val_scores["confusion_matrix"][1][1]})
    pd.DataFrame([flat]).to_csv(output / "metrics.csv", index=False)
    plot_history(output, result)
    print(f"Best epoch {stopper.best_epoch}: validation accuracy {val_scores['accuracy']:.4f}, "
          f"macro-F1 {val_scores['macro_f1']:.4f}; saved to {output}", flush=True)
