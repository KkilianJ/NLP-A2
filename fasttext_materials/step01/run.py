#!/usr/bin/env python3
"""Step 01: compare fastText coverage; default to the project training split."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

from examples import select_examples
from measurement import measure_coverage
from phase1_adapter import phase1, phase1_config
from reporting import write_results
import settings
from tokenization import count_records


def model_sha256(path, force_rehash=False):
    """Reuse Phase 1's fingerprint read-only when the file version matches."""
    stat = path.stat()
    info = {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    cache = phase1_config.CACHE_DIR / "model_fingerprint.json"
    if not force_rehash and cache.is_file():
        try:
            old = json.loads(cache.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            old = {}
        if all(old.get(key) == value for key, value in info.items()) and old.get("sha256") == phase1_config.MODEL_SHA256:
            return old["sha256"], "Phase 1 fingerprint cache (read-only, matched file path/size/mtime)"
    print("Hashing the existing model binary for verification...", flush=True)
    digest = phase1.file_sha256(path)
    if digest != phase1_config.MODEL_SHA256:
        raise ValueError("The binary does not match the original Phase 1 fastText SHA-256.")
    return digest, "computed from binary in this run"


def load_records(data_dir, scope):
    if scope == "project-train":
        if phase1_config.SEED != settings.PROJECT_SPLIT_SEED:
            raise ValueError("The original project split seed changed; refusing to select a different partition.")
        records, info = phase1.load_training_partition(data_dir, smoke=False)
        if len(records) != settings.EXPECTED_TRAIN_COUNT or info["full_training_indices_sha256_le_int64"] != settings.EXPECTED_TRAIN_IDS_SHA256:
            raise ValueError("Reconstructed training partition does not match the original project split.")
        return records, {"protocol": "project training partition; development-safe scope", **info,
                         "source_files_sha256": dict(phase1_config.SOURCE_SHA256), "held_out_analysis": False}
    if scope != "published-train-descriptive":
        raise ValueError(f"Unknown dataset scope: {scope}")
    path = data_dir / "train.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing pinned source file: {path}")
    if phase1.file_sha256(path) != phase1_config.SOURCE_SHA256[path.name]:
        raise ValueError("Published-source file does not match the pinned Phase 1 source SHA-256.")
    frame = pd.read_parquet(path, columns=["text"])
    if len(frame) != settings.EXPECTED_SOURCE_TRAIN_COUNT:
        raise ValueError("Unexpected number of published source-train reviews.")
    records = list(enumerate(frame["text"].tolist()))
    return records, {"protocol": "published source-train partition; descriptive only, not a project split",
                     "source_file": str(path.resolve()), "source_file_sha256": phase1_config.SOURCE_SHA256[path.name],
                     "source_review_count": len(records),
                     "selected_review_ids_and_text_sha256": phase1.ordered_sha256(records),
                     "held_out_analysis": True,
                     "warning": "This source partition overlaps the project's validation/test sets. "
                                "Do not use these descriptive results for model-development decisions."}


def version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("project-train", "published-train-descriptive"), default="project-train",
                        help="data protocol; default: original project training partition")
    parser.add_argument("--acknowledge-project-holdout-overlap", action="store_true",
                        help="required for descriptive published-source analysis; never use it for model decisions")
    parser.add_argument("--data-dir", type=Path, default=phase1_config.DATA_DIR)
    parser.add_argument("--model", type=Path, default=phase1_config.MODEL_PATH)
    parser.add_argument("--output", type=Path, help="new output directory; existing directories are protected")
    parser.add_argument("--example-seed", type=int, default=settings.SEED,
                        help="seed for choosing example excerpts; the project split always uses seed 42")
    parser.add_argument("--verify-model-hash", action="store_true",
                        help="rehash the original binary even when Phase 1's cached fingerprint matches")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.scope == "published-train-descriptive" and not args.acknowledge_project_holdout_overlap:
        parser.error("The published source-train file overlaps project validation/test reviews. "
                     "For descriptive-only analysis, add --acknowledge-project-holdout-overlap.")
    if args.scope == "project-train" and args.acknowledge_project_holdout_overlap:
        parser.error("The holdout-overlap acknowledgement is only for published-train-descriptive scope.")
    output = args.output or settings.RESULTS_DIR / f"{args.scope}-seed{args.example_seed}"
    if output.exists():
        parser.error(f"Output already exists: {output}. Choose a new --output; existing results are never overwritten.")
    if not args.model.is_file():
        parser.error(f"Original fastText binary not found: {args.model}")
    if args.scope == "published-train-descriptive":
        print("DESCRIPTIVE ONLY: published-source rows overlap project validation/test. "
              "Do not use this scope for preprocessing or model selection.", flush=True)
    else:
        print("Reconstructing the original seed-42 project training partition...", flush=True)
    records, data_info = load_records(args.data_dir, args.scope)
    print(f"Selected {len(records):,} reviews for scope {args.scope}.", flush=True)
    digest, fingerprint_method = model_sha256(args.model, args.verify_model_hash)
    print("Loading the original fastText binary once; this requires several GB of RAM...", flush=True)
    import fasttext
    model = fasttext.load_model(str(args.model))
    if model.get_dimension() != 300:
        raise ValueError("Expected the original 300-dimensional fastText binary.")
    print("Counting tokenizations on the selected reviews...", flush=True)
    start = time.perf_counter()
    counters, occurrences, reviews = count_records(records)
    examples = select_examples(records, args.example_seed)
    print("Checking direct vocabulary membership and OOV vectors once per unique token...", flush=True)
    rows, _ = measure_coverage(model, counters)
    elapsed = time.perf_counter() - start
    args_info = model.f.getArgs()
    metadata = {"step": "01", "created_utc": datetime.now(timezone.utc).isoformat(), "scope": args.scope,
                "data_directory": str(args.data_dir.resolve()), "output_directory": str(output.resolve()),
                "example_seed": args.example_seed, "project_split_seed": phase1_config.SEED,
                "data": data_info, "dataset_revision": phase1_config.DATASET_REVISION,
                "model": {"path": str(args.model.resolve()), "sha256": digest, "fingerprint_method": fingerprint_method,
                          "dimensions": model.get_dimension(), "minn": args_info.minn, "maxn": args_info.maxn},
                "tokenizers": list(settings.VARIANTS), "regex_pattern": phase1.WORD_RE.pattern,
                "regex_flags": int(phase1.WORD_RE.flags),
                "phase1_tokenize_sha256": hashlib.sha256(inspect.getsource(phase1.tokenize).encode()).hexdigest(),
                "negation_mapping_to_auxiliary": settings.NEGATIONS,
                "negation_rule": "replace listed tokens by mapped auxiliary plus not",
                "negation_casing": "lower/title/upper preserved; mixed case unchanged",
                "oov_nonzero_definition": "all returned components are finite and at least one is exactly nonzero",
                "prior_phase1_oov_definition": "Euclidean vector norm > 1e-12; direct membership definition unchanged",
                "elapsed_seconds_excluding_input_loading": elapsed,
                "environment": {"python": sys.version, "platform": platform.platform(),
                                "numpy": np.__version__, "pandas": pd.__version__, "fasttext": version("fasttext"),
                                "matplotlib": version("matplotlib"), "scikit_learn": version("scikit-learn")},
                "source_file_sha256": {path.name: phase1.file_sha256(path) for path in
                                       (settings.ROOT / "settings.py", settings.ROOT / "tokenization.py",
                                        settings.ROOT / "measurement.py", settings.ROOT / "examples.py",
                                        settings.ROOT / "reporting.py", settings.ROOT / "phase1_adapter.py",
                                        settings.ROOT / "run.py")},
                "note": "Nonzero OOV vectors indicate availability, not semantic quality or direct vocabulary membership."}
    # Atomic creation refuses every pre-existing directory, including old or partial runs.
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir(exist_ok=False)
    except FileExistsError:
        parser.error(f"Output appeared during the run: {output}. Choose a new --output.")
    write_results(output, rows, examples, occurrences, reviews, metadata)
    print("\nMeasured coverage (fractions):", flush=True)
    for row in rows:
        print(f"  {row['variant']}: {row['token_occurrences']:,} tokens, {row['token_types']:,} types; "
              f"direct token={row['direct_token_fraction']}, direct type={row['direct_type_fraction']}, "
              f"OOV nonzero fraction={row['oov_nonzero_occurrence_fraction']}", flush=True)
    print(f"Saved Step 01 outputs to {output.resolve()}. Existing research reports were not modified.", flush=True)


if __name__ == "__main__":
    main()
