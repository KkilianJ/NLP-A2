"""Snapshot the four saved Phase 2 validation baselines with split checks."""
import hashlib
import inspect
import importlib.metadata
import json
from pathlib import Path
import pandas as pd

from legacy import phase1, phase1_config


def load_baselines(csv_path: Path, split_info):
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing Phase 2 comparison CSV: {csv_path}")
    config_path = csv_path.parent / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing Phase 2 run metadata needed to verify the split: {config_path}")
    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    previous = metadata["split"]
    for key in ("training_indices_sha256_le_int64", "validation_indices_sha256_le_int64", "ordered_source_sha256"):
        if previous[key] != split_info[key]:
            raise ValueError(f"Phase 2 baseline has a different {key}; refusing comparison.")
    if metadata["features"]["identity"]["model_sha256"] != phase1_config.MODEL_SHA256:
        raise ValueError("Phase 2 used a different embedding binary.")
    feature_identity = metadata["features"]["identity"]
    current_tokenizer_sha = hashlib.sha256(inspect.getsource(phase1.tokenize).encode()).hexdigest()
    if (feature_identity["tokenizer"] != "Phase 1 regex (case and HTML preserved)"
            or feature_identity["tokenizer_function_sha256"] != current_tokenizer_sha
            or feature_identity["regex_pattern"] != phase1.WORD_RE.pattern):
        raise ValueError("Phase 2 baseline tokenizer does not match the current Phase 1 regex implementation.")
    if metadata["experiment_settings"]["threshold"] != 0.5:
        raise ValueError("Phase 2 baselines used a different classification threshold.")
    frame = pd.read_csv(csv_path)
    names = ("lr-mean", "lr-max", "mlp-mean", "mlp-max")
    if len(frame) != 4 or set(frame.experiment) != set(names):
        raise ValueError("Phase 2 comparison must contain exactly the four expected baselines.")
    rows = []
    for name in names:
        record = frame.loc[frame.experiment == name].iloc[0]
        acc, f1 = float(record.validation_accuracy), float(record.validation_macro_f1)
        if not (0 <= acc <= 1 and 0 <= f1 <= 1):
            raise ValueError(f"Invalid validation scores for {name}.")
        rows.append({"experiment": name, "validation_accuracy": acc, "validation_macro_f1": f1})
    return {"source_csv": str(csv_path.resolve()), "source_config": str(config_path.resolve()),
            "split_sha256": {key: previous[key] for key in
                             ("training_indices_sha256_le_int64", "validation_indices_sha256_le_int64")},
            "rows": rows}


def load_linear_reference(directory: Path):
    """Read the existing 512-token linear LSTM, never modifying its outputs."""
    import math
    config_path, metrics_path = directory / "run_config.json", directory / "metrics.json"
    history_path, checkpoint_path = directory / "history.csv", directory / "best_model.pt"
    for path in (config_path, metrics_path, history_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing previous linear-head baseline artifact: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    h = config["hyperparameters"]
    expected = {"embedding_dimension": 300, "hidden_size": 64, "lstm_layers": 1,
                "bidirectional": False, "embeddings_frozen": True, "classifier": "Linear(hidden_size, 1)",
                "output_dimension": 1, "max_sequence_length": 512, "batch_size": 32,
                "learning_rate": 0.001, "optimizer": "Adam", "loss": "BCEWithLogitsLoss",
                "epochs_max": 30, "dropout": 0.0, "weight_decay": 0.0,
                "threshold": 0.5, "early_stopping_patience": 5}
    differences = [key for key, value in expected.items() if h.get(key) != value]
    if differences or config.get("seed") != 42 or config.get("smoke") or h.get("head", "linear") != "linear":
        raise ValueError(f"The comparison baseline does not match the existing 512-token linear experiment: {differences}")
    identity = config["sequence_cache"]["manifest"]["identity"]
    expected_identity = {"model_sha256": phase1_config.MODEL_SHA256, "max_length": 512, "smoke": False,
                         "tokenizer_function_sha256": hashlib.sha256(inspect.getsource(phase1.tokenize).encode()).hexdigest(),
                         "regex_pattern": phase1.WORD_RE.pattern, "regex_flags": phase1.WORD_RE.flags,
                         "fasttext_version": importlib.metadata.version("fasttext"),
                         "numpy_version": importlib.metadata.version("numpy")}
    identity_differences = [key for key, value in expected_identity.items() if identity.get(key) != value]
    if identity_differences:
        raise ValueError("The previous linear run's preprocessing environment differs from the current one: "
                         + ", ".join(identity_differences))
    if metrics.get("model") != "LSTM" or metrics.get("seed") != 42 or metrics.get("max_sequence_length") != 512:
        raise ValueError("The saved baseline metrics do not describe the expected 512-token LSTM.")
    accuracy = float(metrics["validation"]["accuracy"])
    f1 = float(metrics["validation"]["macro_f1"])
    train_accuracy = float(metrics["training"]["accuracy"])
    train_f1 = float(metrics["training"]["macro_f1"])
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (accuracy, f1, train_accuracy, train_f1)):
        raise ValueError("The saved linear baseline contains invalid scores.")
    history = pd.read_csv(history_path)
    best = history.loc[history.validation_macro_f1.idxmax()]
    if int(best.epoch) != metrics["best_epoch"] or not math.isclose(float(best.validation_macro_f1), f1, abs_tol=1e-7):
        raise ValueError("The saved linear baseline's best epoch disagrees with its history.")
    return {"directory": str(directory.resolve()), "config": config, "metrics": metrics}


def snapshot_linear_reference(reference, split_info, manifest, smoke=False):
    """Confirm comparable source/split and, for a full run, identical sequence cache."""
    config, metrics = reference["config"], reference["metrics"]
    for key in ("training_indices_sha256_le_int64", "validation_indices_sha256_le_int64", "ordered_source_sha256"):
        if config["split"][key] != split_info[key]:
            raise ValueError(f"The linear baseline has a different {key}; refusing comparison.")
    previous = config["sequence_cache"]["manifest"]
    if not smoke:
        if previous["identity"] != manifest["identity"] or previous["files_sha256"] != manifest["files_sha256"]:
            raise ValueError("The linear baseline and MLP do not use identical cached sequences and vectors.")
    elif previous["identity"]["model_sha256"] != manifest["identity"]["model_sha256"]:
        raise ValueError("The smoke run and linear baseline use different fastText models.")
    train, val = metrics["training"], metrics["validation"]
    return {"source_directory": reference["directory"], "model": "LSTM + Linear, 512 tokens",
            "best_epoch": metrics["best_epoch"], "train_accuracy": train["accuracy"],
            "train_macro_f1": train["macro_f1"], "validation_accuracy": val["accuracy"],
            "validation_macro_f1": val["macro_f1"],
            "training_validation_accuracy_gap": train["accuracy"] - val["accuracy"],
            "training_validation_macro_f1_gap": train["macro_f1"] - val["macro_f1"],
            "same_full_data_and_cache": not smoke,
            "note": ("Smoke subset is not comparable with the full-validation baseline." if smoke else
                     "Same fixed split, sequence cache, optimizer settings and checkpoint-selection rule.")}
