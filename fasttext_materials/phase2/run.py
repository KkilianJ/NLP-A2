#!/usr/bin/env python3
"""Train the requested four Phase 2 classifiers or regenerate saved plots."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

import settings

# Never write matplotlib/font caches into Phase 1.
os.environ.setdefault("XDG_CACHE_HOME", str(settings.CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(settings.CACHE_DIR / "matplotlib"))

CHOICES = ("all", "lr", "mlp", *settings.EXPERIMENTS)


def selected_experiments(choice):
    if choice == "all":
        return list(settings.EXPERIMENTS)
    if choice == "lr":
        return [name for name in settings.EXPERIMENTS if name.startswith("lr-")]
    if choice == "mlp":
        return [name for name in settings.EXPERIMENTS if name.startswith("mlp-")]
    return [choice]


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def print_details(result):
    val = result["validation"]
    print(f"  Validation confusion matrix [[TN, FP], [FN, TP]]: {val['confusion_matrix']}")
    for label in ("negative", "positive"):
        metrics = val["per_class"][label]
        print(f"  {label.capitalize()}: precision {metrics['precision']:.4f}, "
              f"recall {metrics['recall']:.4f}, support {metrics['support']}")
    print(f"  Training time: {result['training_seconds']:.1f}s", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=CHOICES, default="all",
                        help="all, lr, mlp, or a single named experiment (default: all)")
    parser.add_argument("--seed", type=int, default=settings.SEED,
                        help="classifier/training seed; the original data split always uses 42")
    parser.add_argument("--epochs", type=int, default=settings.MLP_MAX_EPOCHS,
                        help="maximum MLP epochs, 1 through 30 (default: 30)")
    parser.add_argument("--reuse-cache", action=argparse.BooleanOptionalAction, default=True,
                        help="reuse/create pooled feature cache (default); --no-reuse-cache recomputes without saving")
    parser.add_argument("--run-dir", type=Path,
                        help="result group for separate runs or graph regeneration")
    parser.add_argument("--model", type=Path, default=settings.DEFAULT_MODEL,
                        help="existing original cc.en.300.bin")
    parser.add_argument("--data-dir", type=Path, default=settings.DEFAULT_DATA_DIR,
                        help="directory containing the pinned train.parquet and test.parquet source files")
    parser.add_argument("--plots-only", action="store_true", help="regenerate plots and comparison from saved metrics; no training")
    args = parser.parse_args(argv)
    if not 1 <= args.epochs <= settings.MLP_MAX_EPOCHS:
        parser.error("--epochs must be between 1 and 30.")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must be between 0 and 4294967295.")
    run_dir = args.run_dir or settings.RESULTS_DIR / f"seed-{args.seed}_epochs-{args.epochs}"

    if args.plots_only:
        from plots import regenerate
        if not run_dir.is_dir():
            parser.error(f"Result directory does not exist: {run_dir}")
        regenerate(run_dir)
        return

    selected = selected_experiments(args.experiment)
    if any(name.startswith("mlp-") for name in selected) and importlib.util.find_spec("torch") is None:
        parser.error("PyTorch is not installed. See phase2/README.md before starting MLP experiments.")
    if not args.model.is_file():
        parser.error(f"The original fastText binary is missing: {args.model}")
    existing = [name for name in selected if (run_dir / name).exists()]
    if existing:
        parser.error(f"Experiment directories already exist: {', '.join(existing)}. "
                     "Choose another --run-dir or remove only your incomplete run.")
    base_config = {"seed": args.seed, "mlp_epochs": args.epochs, "split_seed": settings.SPLIT_SEED,
                   "tokenizer": "Phase 1 regex; original case and HTML tags preserved",
                   "embedding": "original cc.en.300.bin, frozen, 300 dimensions with OOV subwords",
                   "pooling_zero_policy": "exclude zero vectors for both methods; empty/all-zero -> zero vector",
                   "threshold": settings.THRESHOLD}
    config_file = run_dir / "run_config.json"
    if config_file.exists():
        old = json.loads(config_file.read_text(encoding="utf-8"))
        if old.get("experiment_settings") != base_config:
            parser.error("This --run-dir was created with different settings. Choose another directory.")

    from data import load_partitions
    from embeddings import extract_pooled
    from training import train_logistic, train_mlp, write_json
    from plots import regenerate

    print("Reconstructing the existing Phase 1 train/validation split (split seed 42)...", flush=True)
    train, validation, split_info = load_partitions(args.data_dir)
    start = time.perf_counter()
    arrays, feature_info = extract_pooled(train, validation, split_info, args.model, args.reuse_cache)
    extraction_seconds = time.perf_counter() - start
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {"experiment_settings": base_config, "split": split_info,
                  "features": feature_info, "feature_preparation_seconds": extraction_seconds,
                  "environment": {"python": sys.version, "platform": platform.platform(),
                                  "numpy": package_version("numpy"), "scikit_learn": package_version("scikit-learn"),
                                  "fasttext": package_version("fasttext"), "torch": package_version("torch")}}
    if config_file.exists():
        old = json.loads(config_file.read_text(encoding="utf-8"))
        if old["split"] != split_info or old["features"]["identity"] != feature_info["identity"]:
            parser.error("This --run-dir has different split or embedding provenance; choose another directory.")
    else:
        write_json(config_file, run_config)

    for name in selected:
        model, pooling = settings.EXPERIMENTS[name]
        directory = run_dir / name
        directory.mkdir(parents=False, exist_ok=False)
        x_train = arrays[f"train_{pooling}"]
        x_val = arrays[f"validation_{pooling}"]
        if model == "Logistic Regression":
            result = train_logistic(name, pooling, x_train, train.labels, x_val, validation.labels, args.seed, directory)
        else:
            result = train_mlp(name, pooling, x_train, train.labels, x_val, validation.labels,
                               args.seed, args.epochs, directory)
        print_details(result)
        # Keep previously completed results reviewable if a later run is interrupted.
        regenerate(run_dir)
    print("\nTraining complete. No held-out test evaluation was performed.", flush=True)


if __name__ == "__main__":
    main()
