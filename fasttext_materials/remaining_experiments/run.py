#!/usr/bin/env python3
"""Run exactly one configured experiment; seed-42 baselines are reused."""
import argparse
from pathlib import Path

import numpy as np

import bridge
from data_access import pooled_cache, read_json, reserve_output, sequence_cache
import training
import workflow_config as cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    names = list(cfg.manifest()["experiments"])
    parser.add_argument("--experiment", required=True, choices=names)
    parser.add_argument("--seed", type=int, choices=cfg.SEEDS, default=42)
    parser.add_argument("--alternative", help="for F-alternative: A2/A3/B1/C1/C3/D1/D3/E2/E3")
    parser.add_argument("--output", type=Path, help="new output directory; defaults to results/ID/seed-N")
    args = parser.parse_args(argv)
    try:
        spec = cfg.specification(args.experiment, args.seed, args.alternative)
    except ValueError as exc:
        parser.error(str(exc))
    if spec.get("baseline_alias") or (spec["experiment"] == "F-base" and args.seed == 42):
        if args.output:
            parser.error("The existing baseline is read-only; --output is not used for baseline aliases.")
        reference = bridge.baselines.load_linear_reference(cfg.BASELINE)
        print(f"Reusing existing seed-42 baseline: {reference['directory']}")
        print(f"Validation macro-F1: {reference['metrics']['validation']['macro_f1']:.6f}")
        return
    if spec["experiment"] == "F-mean" and args.seed == 42:
        if args.output:
            parser.error("The existing mean-pooling baseline is read-only; --output is not used.")
        metrics = read_json(cfg.MEAN_BASELINE / "metrics.json")
        print(f"Reusing existing Phase 2 LR-Mean baseline: {cfg.MEAN_BASELINE}")
        print(f"Validation macro-F1: {metrics['validation']['macro_f1']:.6f}")
        return
    output = args.output or cfg.result_directory(spec["experiment"], args.seed)
    if output.exists():
        parser.error(f"Output exists: {output}. Choose a new --output; existing results are protected.")
    bridge.baselines.load_linear_reference(cfg.BASELINE)
    cache, manifest, original = sequence_cache()
    if manifest["identity"]["max_length"] != 512:
        raise ValueError("Expected the existing 512-token cache.")
    if spec.get("model_family") == "lr_mean":
        arrays, phase2_config, pooled_path = pooled_cache()
        train_labels = np.load(cache / "train_labels.npy", mmap_mode="r")
        val_labels = np.load(cache / "validation_labels.npy", mmap_mode="r")
        output = reserve_output(output)
        training.write_json(output / "run_config.json", {"specification": spec, "split": original["split"],
                            "pooled_feature_identity": phase2_config["features"]["identity"],
                            "pooled_cache": str(pooled_path),
                            "pooled_file_sha256": {name: bridge.legacy.phase1.file_sha256(pooled_path / f"{name}_mean.npy")
                                                    for name in ("train", "validation")},
                            "environment": training.versions(),
                            "note": "Same Phase 2 full-review mean cache identity and training-only scaler; "
                                    "the original Phase 2 run did not save array checksums."})
        module = bridge.load_phase2_module("training")
        module.train_logistic("F-mean", "mean", arrays["train"], train_labels,
                              arrays["validation"], val_labels, args.seed, output)
    else:
        output = reserve_output(output)
        training.train_lstm(spec, cache, manifest, original, output)
    print("No held-out test evaluation was performed.", flush=True)


if __name__ == "__main__":
    main()
