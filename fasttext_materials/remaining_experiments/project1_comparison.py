#!/usr/bin/env python3
"""Compare saved Project 1 and Project 2 validation artifacts without training."""
import argparse
import os
from pathlib import Path

import workflow_config as cfg
os.environ.setdefault("XDG_CACHE_HOME", str(cfg.CACHE))
os.environ.setdefault("MPLCONFIGDIR", str(cfg.CACHE / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_access import read_json, reserve_output
import training


def compare(project1_dir, project2_dirs):
    project1_dir = Path(project1_dir)
    old = read_json(project1_dir / "config.json")
    scores = pd.read_csv(project1_dir / "validation_scores.csv")
    if old.get("metric") != "macro_f1" or not {"representation", "model", "score"}.issubset(scores.columns):
        raise ValueError("Project 1 results do not declare validation macro-F1 in the expected format.")
    reference = read_json(cfg.BASELINE / "run_config.json")["split"]
    old_source = old.get("imdb_ordered_sha256") or old.get("ordered_source_sha256")
    old_val = old.get("validation_indices_sha256_le_int64")
    verified = bool(old_source and old_val and old_source == reference["ordered_source_sha256"] and
                    old_val == reference["validation_indices_sha256_le_int64"])
    mismatch = bool((old_source and old_source != reference["ordered_source_sha256"]) or
                    (old_val and old_val != reference["validation_indices_sha256_le_int64"]))
    status = "verified_same_split" if verified else ("different_split" if mismatch else "unverified_split_fingerprints")
    rows = [{"project": "Project 1", "model": f"{r.representation} / {r.model}",
             "validation_macro_f1": float(r.score), "validation_accuracy": None,
             "source": str((project1_dir / "validation_scores.csv").resolve()), "split_status": status}
            for r in scores.itertuples()]
    for path in project2_dirs:
        path = Path(path)
        if not (path / "metrics.json").is_file():
            continue
        metrics = read_json(path / "metrics.json")
        config_path = path / "run_config.json"
        if not config_path.exists():
            config_path = path.parent / "run_config.json"
        config = read_json(config_path)
        split = config["split"]
        same = all(split.get(key) == reference[key] for key in
                   ("ordered_source_sha256", "training_indices_sha256_le_int64", "validation_indices_sha256_le_int64"))
        spec = config.get("specification", {})
        label = f"{spec['experiment']} / seed {spec['seed']}" if "experiment" in spec else path.name
        rows.append({"project": "Project 2", "model": label,
                     "validation_macro_f1": float(metrics["validation"]["macro_f1"]),
                     "validation_accuracy": float(metrics["validation"]["accuracy"]),
                     "source": str(path.resolve()), "split_status": "verified_project2_split" if same else "different_or_unverified"})
    return {"project1_config": old, "project1_split_status": status, "rows": rows,
            "note": "Project 1 saved config may omit fingerprints and validation accuracy; missing evidence is not fabricated. "
                    "Comparisons use validation, not final test. Matching seed and sizes alone do not prove matching rows."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project1-dir", type=Path, default=cfg.PROJECT1)
    parser.add_argument("--project2-dir", type=Path, action="append", help="additional completed run directory; repeatable")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"Output exists: {args.output}")
    for path in args.project2_dir or []:
        if not (path / "metrics.json").is_file():
            parser.error(f"Requested Project 2 metrics do not exist: {path}")
    phase2 = cfg.MEAN_BASELINE.parent
    dirs = [phase2 / name for name in ("lr-mean", "lr-max", "mlp-mean", "mlp-max")]
    dirs += [cfg.BASELINE_256, cfg.BASELINE, cfg.ASSIGNMENT / "phase3/results/lstm-mlp64-s42-h64-l512-b32-lr0.001-e30"]
    dirs += args.project2_dir or []
    result = compare(args.project1_dir, dirs)
    output = reserve_output(args.output)
    training.write_json(output / "comparison.json", result)
    pd.DataFrame(result["rows"]).to_csv(output / "comparison.csv", index=False)
    selected = sorted(result["rows"], key=lambda row: row["validation_macro_f1"], reverse=True)
    fig, ax = plt.subplots(figsize=(9, max(4, len(selected) * .35)))
    ax.barh([f"{r['project']}: {r['model']}" for r in selected], [r["validation_macro_f1"] for r in selected])
    ax.invert_yaxis(); ax.set(xlabel="Saved validation macro-F1", xlim=(0, 1.1),
                              title=f"Project 1 split status: {result['project1_split_status']}")
    fig.tight_layout(); fig.savefig(output / "comparison.png", dpi=160); plt.close(fig)
    print(f"Project 1 split verification: {result['project1_split_status']}; saved {output}")


if __name__ == "__main__":
    main()
