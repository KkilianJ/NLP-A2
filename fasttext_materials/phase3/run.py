#!/usr/bin/env python3
"""Train one frozen-fastText LSTM or regenerate plots from a saved run."""
import argparse
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
from pathlib import Path
import sys
import time

import experiment_settings as cfg

os.environ.setdefault("XDG_CACHE_HOME", str(cfg.CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(cfg.CACHE_DIR / "matplotlib"))


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def default_run_dir(args):
    prefix = "smoke" if args.smoke else "lstm"
    if args.head == "mlp":
        prefix += f"-mlp{args.head_hidden_size}"
    return cfg.RESULTS_DIR / (f"{prefix}-s{args.seed}-h{args.hidden_size}-l{args.max_sequence_length}"
                              f"-b{args.batch_size}-lr{args.learning_rate:g}-e{args.epochs}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=cfg.MAX_EPOCHS, help="maximum epochs, 1–30 (default: 30)")
    parser.add_argument("--learning-rate", type=float, default=cfg.LEARNING_RATE, help="Adam learning rate (default: 0.001)")
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE, help="training and evaluation batch size (default: 32)")
    parser.add_argument("--hidden-size", type=int, default=cfg.HIDDEN_SIZE, help="LSTM hidden size (default: 64)")
    parser.add_argument("--head", choices=("linear", "mlp"), default="linear",
                        help="classification head (default: linear; mlp is the controlled 512-token experiment)")
    parser.add_argument("--head-hidden-size", type=int, default=cfg.HEAD_HIDDEN_SIZE,
                        help="MLP head hidden units; fixed to 64 for the controlled comparison")
    parser.add_argument("--max-sequence-length", type=int, default=None,
                        help="retain first N tokens (default: 256 for linear, 512 for mlp)")
    parser.add_argument("--seed", type=int, default=cfg.SEED,
                        help="training and smoke-sampling seed; the project split remains seed 42")
    parser.add_argument("--smoke", action="store_true", help="use 64 training and 32 validation reviews")
    parser.add_argument("--model", type=Path, default=cfg.MODEL_PATH, help="existing original cc.en.300.bin")
    parser.add_argument("--data-dir", type=Path, default=cfg.DATA_DIR, help="directory with pinned source parquet files")
    parser.add_argument("--baseline-csv", type=Path, default=cfg.BASELINE_CSV,
                        help="Phase 2 comparison CSV; its run_config.json must be alongside it")
    parser.add_argument("--linear-baseline-dir", type=Path, default=cfg.LINEAR_BASELINE_DIR,
                        help="previous 512-token linear LSTM result directory, used for the MLP comparison")
    parser.add_argument("--run-dir", type=Path, help="new or empty output directory")
    parser.add_argument("--plots-only", action="store_true", help="regenerate plots from saved metrics, without training")
    args = parser.parse_args(argv)
    if args.max_sequence_length is None:
        args.max_sequence_length = cfg.MLP_MAX_LENGTH if args.head == "mlp" else cfg.MAX_LENGTH
    if not 1 <= args.epochs <= cfg.MAX_EPOCHS:
        parser.error("--epochs must be between 1 and 30.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be a positive finite number.")
    if min(args.batch_size, args.hidden_size, args.max_sequence_length) < 1:
        parser.error("--batch-size, --hidden-size, and --max-sequence-length must be positive.")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must be between 0 and 4294967295.")
    if args.head_hidden_size != cfg.HEAD_HIDDEN_SIZE:
        parser.error("The controlled MLP head has exactly 64 hidden units; --head-hidden-size must be 64.")
    if args.head == "mlp" and not args.plots_only:
        expected = {"learning_rate": cfg.LEARNING_RATE, "batch_size": cfg.BATCH_SIZE,
                    "hidden_size": cfg.HIDDEN_SIZE, "max_sequence_length": cfg.MLP_MAX_LENGTH, "seed": cfg.SEED}
        if not args.smoke:
            expected["epochs"] = cfg.MAX_EPOCHS
        mismatches = [name for name, value in expected.items() if getattr(args, name) != value]
        if mismatches:
            parser.error("The MLP-head comparison keeps the previous 512-token settings fixed; "
                         "remove conflicting options: " + ", ".join(mismatches))
    output = args.run_dir or default_run_dir(args)
    if args.plots_only:
        if not output.is_dir():
            parser.error(f"Saved run directory not found: {output}. Use --run-dir if it has a different name.")
        from plots import regenerate
        regenerate(output)
        return
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Output already exists and is not empty: {output}. Choose a new --run-dir.")
    if not args.model.is_file():
        parser.error(f"Original fastText binary not found: {args.model}")
    if not args.baseline_csv.is_file():
        parser.error(f"Phase 2 baseline comparison not found: {args.baseline_csv}")
    for name in ("torch", "fasttext"):
        if importlib.util.find_spec(name) is None:
            parser.error(f"Missing {name}. Use the Phase 2 environment or install phase3/requirements.txt.")

    import numpy as np
    import pandas as pd
    from baselines import load_baselines, load_linear_reference, snapshot_linear_reference
    from legacy import load_partitions
    from sequence_preparation import prepare_sequences, sample_partition
    from lstm_training import fit, write_json
    from plots import regenerate

    linear_reference = load_linear_reference(args.linear_baseline_dir) if args.head == "mlp" else None
    print("Reconstructing the original Phase 1/2 split (fixed seed 42)...", flush=True)
    train, validation, split_info = load_partitions(args.data_dir)
    baseline = load_baselines(args.baseline_csv, split_info)
    if args.smoke:
        rng = np.random.default_rng(args.seed)
        train = sample_partition(train, cfg.SMOKE_TRAIN_REVIEWS, rng)
        validation = sample_partition(validation, cfg.SMOKE_VALIDATION_REVIEWS, rng)
        print("SMOKE RUN: tiny train/validation subsets; scores are not comparable to full Phase 2 results.", flush=True)
    print(f"Preparing {len(train.ids):,} training and {len(validation.ids):,} validation sequences.", flush=True)
    prepare_start = time.perf_counter()
    cache_dir, manifest, reused = prepare_sequences(train, validation, split_info, args.model,
                                                     args.max_sequence_length, args.smoke)
    preparation_seconds = time.perf_counter() - prepare_start
    linear_snapshot = (snapshot_linear_reference(linear_reference, split_info, manifest, args.smoke)
                       if linear_reference is not None else None)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Output became nonempty during preprocessing: {output}; choose a new --run-dir.")
    output.mkdir(parents=True, exist_ok=True)
    run_config = {"smoke": args.smoke, "seed": args.seed, "split_seed": cfg.SPLIT_SEED,
                  "training_reviews_selected": len(train.ids), "validation_reviews_selected": len(validation.ids),
                  "hyperparameters": {"embedding_dimension": cfg.DIMENSION, "hidden_size": args.hidden_size,
                                      "lstm_layers": 1, "bidirectional": False, "embeddings_frozen": True,
                                      "head": args.head,
                                      "head_hidden_size": args.head_hidden_size if args.head == "mlp" else None,
                                      "classifier": ("Linear(hidden_size, 64) -> ReLU -> Linear(64, 1)"
                                                     if args.head == "mlp" else "Linear(hidden_size, 1)"),
                                      "output_dimension": 1,
                                      "max_sequence_length": args.max_sequence_length, "batch_size": args.batch_size,
                                      "learning_rate": args.learning_rate, "optimizer": "Adam", "loss": "BCEWithLogitsLoss",
                                      "epochs_max": args.epochs, "dropout": 0.0, "weight_decay": 0.0,
                                      "threshold": cfg.THRESHOLD, "early_stopping_patience": cfg.EARLY_STOP_PATIENCE},
                  "sequence_policy": "Phase 1 regex; retain case and markup; first N tokens; dynamic right padding; last valid output",
                  "checkpoint_policy": "highest validation macro-F1, strictly greater; earliest epoch wins ties",
                  "split": split_info, "sequence_cache": {"path": str(cache_dir.resolve()), "reused": reused,
                                                              "manifest": manifest},
                  "preparation_seconds": preparation_seconds,
                  "environment": {"python": sys.version, "platform": platform.platform(),
                                  "torch": package_version("torch"), "numpy": package_version("numpy"),
                                  "fasttext": package_version("fasttext"), "pandas": package_version("pandas"),
                                  "scikit_learn": package_version("scikit-learn")}}
    write_json(output / "run_config.json", run_config)
    write_json(output / "baseline_snapshot.json", baseline)
    if linear_snapshot is not None:
        write_json(output / "linear_baseline_snapshot.json", linear_snapshot)
    write_json(output / "sequence_statistics.json", manifest["statistics"])
    pd.DataFrame([{"partition": name, **values} for name, values in manifest["statistics"].items()])\
      .to_csv(output / "sequence_statistics.csv", index=False)
    result = fit(cache_dir, manifest, output, args.epochs, args.seed, args.batch_size,
                 args.learning_rate, args.hidden_size, head=args.head, head_hidden_size=args.head_hidden_size)
    val = result["validation"]
    print(f"\nBest checkpoint epoch: {result['best_epoch']} | validation accuracy: {val['accuracy']:.4f} "
          f"| validation macro-F1: {val['macro_f1']:.4f}")
    for label in ("negative", "positive"):
        values = val["per_class"][label]
        print(f"  {label.capitalize()}: precision {values['precision']:.4f}, recall {values['recall']:.4f}, support {values['support']}")
    print(f"  Confusion matrix [[TN, FP], [FN, TP]]: {val['confusion_matrix']}")
    print(f"  Training accuracy: {result['training']['accuracy']:.4f} | "
          f"train − validation accuracy gap: {result['training_validation_accuracy_gap']:.4f}")
    print(f"  Training macro-F1: {result['training']['macro_f1']:.4f} | "
          f"train − validation macro-F1 gap: {result['training_validation_macro_f1_gap']:.4f}")
    print(f"  Training time: {result['training_seconds']:.1f}s", flush=True)
    regenerate(output)
    print("No held-out test evaluation was performed.", flush=True)


if __name__ == "__main__":
    main()
