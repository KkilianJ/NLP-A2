"""Validate and reuse the existing Phase 3 sequence and Phase 2 pooling caches."""
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

import bridge
import workflow_config as cfg


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sequence_cache(run_dir=cfg.BASELINE):
    config = read_json(Path(run_dir) / "run_config.json")
    saved = config["sequence_cache"]["manifest"]
    identity = saved["identity"]
    reference = read_json(cfg.BASELINE / "run_config.json")
    for key in ("ordered_source_sha256", "training_indices_sha256_le_int64", "validation_indices_sha256_le_int64"):
        if config["split"][key] != reference["split"][key]:
            raise ValueError(f"Incompatible project split: {key}")
    source = hashlib.sha256(inspect.getsource(bridge.legacy.phase1.tokenize).encode()).hexdigest()
    if (identity["tokenizer_function_sha256"] != source or identity["regex_pattern"] != bridge.legacy.phase1.WORD_RE.pattern
            or identity["regex_flags"] != bridge.legacy.phase1.WORD_RE.flags
            or identity["source_sha256"] != bridge.legacy.phase1_config.ORDERED_DATA_SHA256
            or identity["model_sha256"] != bridge.legacy.phase1_config.MODEL_SHA256 or identity["smoke"]):
        raise ValueError("Sequence cache model, tokenizer, or scope does not match Phase 1.")
    path = cfg.ASSIGNMENT / "phase3/.cache/sequences" / Path(config["sequence_cache"]["path"]).name
    if not path.is_dir():
        raise FileNotFoundError(f"Required existing sequence cache is missing: {path}; restore the historical cache.")
    disk = bridge.sequence_preparation._load_cache(path, identity, config["split"]["training_count"],
                                                   config["split"]["validation_count"])
    if disk["files_sha256"] != saved["files_sha256"]:
        raise ValueError("Sequence cache checksums differ from the saved original run.")
    return path, disk, config


def pooled_cache():
    config = read_json(cfg.MEAN_BASELINE.parent / "run_config.json")
    reference = read_json(cfg.BASELINE / "run_config.json")
    for key in ("ordered_source_sha256", "training_indices_sha256_le_int64", "validation_indices_sha256_le_int64"):
        if config["split"][key] != reference["split"][key]:
            raise ValueError(f"Phase 2 pooled features have a different split: {key}")
    identity = config["features"]["identity"]
    if (identity["model_sha256"] != bridge.legacy.phase1_config.MODEL_SHA256 or
        identity["dataset_order_sha256"] != bridge.legacy.phase1_config.ORDERED_DATA_SHA256 or
        identity["regex_pattern"] != bridge.legacy.phase1.WORD_RE.pattern or
        identity["regex_flags"] != bridge.legacy.phase1.WORD_RE.flags or
        identity["tokenizer_function_sha256"] != hashlib.sha256(inspect.getsource(bridge.legacy.phase1.tokenize).encode()).hexdigest()):
        raise ValueError("Phase 2 pooled features have a different model or tokenizer.")
    root = cfg.ASSIGNMENT / "phase2/.cache/embeddings" / Path(config["features"]["cache_path"]).name
    saved = read_json(root / "manifest.json")
    if saved["identity"] != identity:
        raise ValueError("Pooled cache identity differs from Phase 2 metadata.")
    arrays = {}
    for name, count in (("train", config["split"]["training_count"]), ("validation", config["split"]["validation_count"])):
        array = np.load(root / f"{name}_mean.npy", mmap_mode="r", allow_pickle=False)
        if array.shape != (count, 300) or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"Invalid pooled feature array: {name}")
        arrays[name] = array
    return arrays, config, root


def reserve_output(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"Output already exists: {path}. Choose a new output directory.") from exc
    return path
