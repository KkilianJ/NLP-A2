"""Train one classifier at a time from cached frozen pooled features."""
import importlib.metadata
import json
import random
import threading
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metrics import evaluate
import settings


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def metric_row(result):
    train, val = result["training"], result["validation"]
    return {"experiment": result["experiment"], "model": result["model"], "pooling": result["pooling"],
            "train_accuracy": train["accuracy"], "validation_accuracy": val["accuracy"],
            "validation_macro_f1": val["macro_f1"],
            "validation_negative_precision": val["per_class"]["negative"]["precision"],
            "validation_negative_recall": val["per_class"]["negative"]["recall"],
            "validation_positive_precision": val["per_class"]["positive"]["precision"],
            "validation_positive_recall": val["per_class"]["positive"]["recall"],
            "tn": val["confusion_matrix"][0][0], "fp": val["confusion_matrix"][0][1],
            "fn": val["confusion_matrix"][1][0], "tp": val["confusion_matrix"][1][1],
            "training_seconds": result["training_seconds"], "best_epoch": result.get("best_epoch")}


def save_result(output, result):
    write_json(output / "metrics.json", result)
    pd.DataFrame([metric_row(result)]).to_csv(output / "metrics.csv", index=False)


def train_logistic(experiment, pooling, x_train, y_train, x_val, y_val, seed, output):
    print(f"\n[{experiment}] Fitting train-only StandardScaler and logistic regression...", flush=True)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=settings.LR_C, max_iter=settings.LR_MAX_ITER,
                                          class_weight=None, random_state=seed, verbose=1)),
    ])
    start = time.perf_counter()
    finished = threading.Event()
    def show_progress():
        while not finished.wait(10):
            print(f"  {experiment}: optimization running ({time.perf_counter() - start:.0f}s)...", flush=True)
    ticker = threading.Thread(target=show_progress, daemon=True)
    ticker.start()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline.fit(x_train, y_train)
    finally:
        finished.set()
        ticker.join()
    seconds = time.perf_counter() - start
    messages = [str(w.message) for w in caught]
    for message in messages:
        print(f"  Warning: {message}", flush=True)
    train_metrics = evaluate(y_train, pipeline.predict_proba(x_train)[:, 1])
    validation_metrics = evaluate(y_val, pipeline.predict_proba(x_val)[:, 1])
    joblib.dump(pipeline, output / "model.joblib")
    result = {"experiment": experiment, "model": "Logistic Regression", "pooling": pooling,
              "seed": seed, "threshold": settings.THRESHOLD, "training_seconds": seconds,
              "time_definition": "Pipeline fit including scaler fit; excludes feature extraction and final evaluation.",
              "hyperparameters": {"C": settings.LR_C, "max_iter": settings.LR_MAX_ITER,
                                  "class_weight": None, "random_state": seed},
              "training": train_metrics, "validation": validation_metrics, "warnings": messages,
              "versions": {"numpy": np.__version__, "scikit_learn": importlib.metadata.version("scikit-learn")},
              "iterations": pipeline.named_steps["classifier"].n_iter_.tolist()}
    save_result(output, result)
    print(f"  Finished in {seconds:.1f}s | train accuracy {train_metrics['accuracy']:.4f} | "
          f"validation accuracy {validation_metrics['accuracy']:.4f} | "
          f"validation macro-F1 {validation_metrics['macro_f1']:.4f}", flush=True)
    return result


def train_mlp(experiment, pooling, x_train, y_train, x_val, y_val, seed, epochs, output):
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("PyTorch is missing. Install phase2/requirements.txt before running MLP experiments.") from exc

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n[{experiment}] Training on {device} for at most {epochs} epochs...", flush=True)
    # Copies avoid writable-buffer issues with cached read-only memory maps.
    train_ds = TensorDataset(torch.tensor(np.asarray(x_train), dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(np.asarray(x_val), dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.float32))
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=settings.MLP_BATCH_SIZE, shuffle=True, generator=gen)
    train_eval_loader = DataLoader(train_ds, batch_size=settings.EVAL_BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=settings.EVAL_BATCH_SIZE, shuffle=False)
    model = nn.Sequential(nn.Linear(settings.DIMENSION, settings.MLP_HIDDEN), nn.ReLU(),
                          nn.Linear(settings.MLP_HIDDEN, 1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.MLP_LR,
                                 weight_decay=settings.MLP_WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    def score_loader(loader, truth):
        model.eval()
        total_loss, total = 0.0, 0
        probabilities = []
        with torch.no_grad():
            for features, labels in loader:
                features, labels = features.to(device), labels.to(device)
                logits = model(features).squeeze(-1)
                loss = criterion(logits, labels)
                total_loss += float(loss.item()) * labels.numel()
                total += labels.numel()
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
        return total_loss / total, evaluate(truth, np.concatenate(probabilities))

    history, best = [], None
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for step, (features, labels) in enumerate(train_loader, 1):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features).squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            progress_every = max(1, (len(train_loader) + 3) // 4)
            if step % progress_every == 0 or step == len(train_loader):
                print(f"  Epoch {epoch:02d}/{epochs:02d}: batch {step}/{len(train_loader)}", flush=True)
        train_loss, train_scores = score_loader(train_eval_loader, y_train)
        val_loss, val_scores = score_loader(val_loader, y_val)
        history.append({"epoch": epoch, "training_loss": train_loss, "validation_loss": val_loss,
                        "training_accuracy": train_scores["accuracy"],
                        "validation_accuracy": val_scores["accuracy"],
                        "validation_macro_f1": val_scores["macro_f1"]})
        pd.DataFrame(history).to_csv(output / "epochs.csv", index=False)
        write_json(output / "epochs.json", history)
        if best is None or val_scores["macro_f1"] > best["validation"]["macro_f1"]:
            best = {"epoch": epoch, "training": train_scores, "validation": val_scores,
                    "training_loss": train_loss, "validation_loss": val_loss}
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            checkpoint = {"state_dict": state, "epoch": epoch, "seed": seed,
                          "input_size": settings.DIMENSION, "hidden_size": settings.MLP_HIDDEN,
                          "output_size": 1, "threshold": settings.THRESHOLD, "pooling": pooling}
            torch.save(checkpoint, output / "best_model.pt")
        print(f"  Epoch {epoch:02d}/{epochs:02d} | loss {train_loss:.4f}/{val_loss:.4f} "
              f"(train/val) | accuracy {train_scores['accuracy']:.4f}/{val_scores['accuracy']:.4f} "
              f"| val macro-F1 {val_scores['macro_f1']:.4f}", flush=True)
    seconds = time.perf_counter() - start
    result = {"experiment": experiment, "model": "MLP (64)", "pooling": pooling,
              "seed": seed, "device": str(device), "threshold": settings.THRESHOLD,
              "versions": {"numpy": np.__version__, "torch": torch.__version__,
                           "scikit_learn": importlib.metadata.version("scikit-learn")},
              "best_epoch": best["epoch"], "training_seconds": seconds,
              "time_definition": "Entire training loop including per-epoch train/validation evaluation; excludes feature extraction.",
              "hyperparameters": {"architecture": "Linear(300,64) -> ReLU -> Linear(64,1)",
                                  "optimizer": "Adam", "learning_rate": settings.MLP_LR,
                                  "batch_size": settings.MLP_BATCH_SIZE, "evaluation_batch_size": settings.EVAL_BATCH_SIZE, "epochs": epochs,
                                  "loss": "BCEWithLogitsLoss", "dropout": 0.0,
                                  "weight_decay": settings.MLP_WEIGHT_DECAY},
              "checkpoint_selection": "highest validation macro-F1; earliest epoch wins ties",
              "best_training_loss": best["training_loss"], "best_validation_loss": best["validation_loss"],
              "training": best["training"], "validation": best["validation"]}
    save_result(output, result)
    print(f"  Best checkpoint: epoch {best['epoch']} | val accuracy {best['validation']['accuracy']:.4f} | "
          f"val macro-F1 {best['validation']['macro_f1']:.4f} | {seconds:.1f}s", flush=True)
    return result
