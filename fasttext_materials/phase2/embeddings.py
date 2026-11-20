"""Frozen fastText lookup and reusable train/validation pooled arrays."""
from functools import lru_cache
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from data import indices_sha
from phase1_imports import phase1, phase1_config
import settings


def pool_vectors(vectors, dimension=settings.DIMENSION):
    """Mean/max over finite nonzero token vectors; empty/all-zero -> zeros.

    There is no padding. A zero vector is excluded from both aggregations so it
    cannot dilute the mean or mask negative coordinates in the maximum.
    """
    usable = []
    zero_count = 0
    for vector in vectors:
        arr = np.asarray(vector, dtype=np.float32)
        if arr.shape != (dimension,) or not np.isfinite(arr).all():
            raise ValueError("A token vector has the wrong shape or a non-finite value.")
        if not np.any(arr):
            zero_count += 1
        else:
            usable.append(arr)
    if not usable:
        zero = np.zeros(dimension, dtype=np.float32)
        return zero, zero.copy(), zero_count
    matrix = np.stack(usable)
    return (matrix.mean(axis=0, dtype=np.float64).astype(np.float32),
            matrix.max(axis=0).astype(np.float32), zero_count)


def verified_model_digest(model_path: Path):
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing fastText binary: {model_path}")
    settings.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fingerprint_file = settings.CACHE_DIR / "model_fingerprint.json"
    stat = model_path.stat()
    identity = {"path": str(model_path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    previous = {}
    if fingerprint_file.is_file():
        try:
            previous = json.loads(fingerprint_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    if all(previous.get(k) == v for k, v in identity.items()) and previous.get("sha256"):
        digest = previous["sha256"]
    else:
        print("Checking the original fastText binary SHA-256 (once per file version)...", flush=True)
        digest = phase1.file_sha256(model_path)
        fingerprint_file.write_text(json.dumps({**identity, "sha256": digest}, indent=2) + "\n", encoding="utf-8")
    if digest != phase1_config.MODEL_SHA256:
        raise ValueError("Model SHA-256 does not match the original Phase 1 cc.en.300.bin.")
    return digest


def cache_identity(train, validation, split_info, model_sha):
    tokenizer_sha = hashlib.sha256(inspect.getsource(phase1.tokenize).encode("utf-8")).hexdigest()
    return {"cache_version": settings.CACHE_VERSION, "model_sha256": model_sha,
            "dataset_order_sha256": split_info["ordered_source_sha256"],
            "training_indices_sha256": indices_sha(train.ids),
            "validation_indices_sha256": indices_sha(validation.ids),
            "tokenizer": "Phase 1 regex (case and HTML preserved)",
            "tokenizer_function_sha256": tokenizer_sha, "regex_pattern": phase1.WORD_RE.pattern,
            "regex_flags": phase1.WORD_RE.flags, "dimension": settings.DIMENSION,
            "fasttext_version": importlib.metadata.version("fasttext"), "numpy_version": np.__version__,
            "zero_vector_policy": "exclude from both poolings; all-zero/empty review maps to zero"}


def _load_cache(path, identity, train_count, validation_count):
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Incomplete embedding cache at {path}. Remove it or use --no-reuse-cache.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("identity") != identity:
        raise ValueError(f"Embedding cache metadata does not match {path}.")
    arrays = {}
    for part, count in (("train", train_count), ("validation", validation_count)):
        for pooling in ("mean", "max"):
            name = f"{part}_{pooling}"
            array = np.load(path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            if array.shape != (count, settings.DIMENSION) or array.dtype != np.float32:
                raise ValueError(f"Invalid cached feature shape or dtype: {name}")
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite values in cached features: {name}")
            arrays[name] = array
    return arrays, manifest


def extract_pooled(train, validation, split_info, model_path: Path, reuse_cache=True):
    model_sha = verified_model_digest(model_path)
    identity = cache_identity(train, validation, split_info, model_sha)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    cache_root = settings.CACHE_DIR / "embeddings"
    cache_path = cache_root / key
    if reuse_cache and cache_path.exists():
        arrays, manifest = _load_cache(cache_path, identity, len(train.ids), len(validation.ids))
        print(f"Reusing pooled embeddings from {cache_path}", flush=True)
        return arrays, {**manifest, "cache_path": str(cache_path), "reused": True}

    try:
        import fasttext
    except ImportError as exc:
        raise RuntimeError("fasttext is missing. Install phase2/requirements.txt first.") from exc
    print("Loading the original fastText binary once; this can require several GB of RAM...", flush=True)
    model = fasttext.load_model(str(model_path))
    if model.get_dimension() != settings.DIMENSION:
        raise ValueError("Expected 300-dimensional fastText vectors.")

    @lru_cache(maxsize=settings.TOKEN_VECTOR_CACHE_SIZE)
    def token_vector(token):
        # The binary API synthesizes subword vectors for OOV strings automatically.
        vector = np.asarray(model.get_word_vector(token), dtype=np.float32)
        if vector.shape != (settings.DIMENSION,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid fastText vector for token {token!r}")
        return vector

    arrays, statistics = {}, {}
    for name, partition in (("train", train), ("validation", validation)):
        n = len(partition.texts)
        mean = np.zeros((n, settings.DIMENSION), dtype=np.float32)
        maximum = np.zeros_like(mean)
        counts = {"reviews": n, "tokens": 0, "zero_vector_tokens": 0,
                  "empty_tokenized_reviews": 0, "reviews_without_usable_vectors": 0}
        print(f"Extracting {name} embeddings: 0/{n}", flush=True)
        for i, text in enumerate(partition.texts):
            tokens = phase1.tokenize(text, "regex")
            means, maxima, zero = pool_vectors((token_vector(token) for token in tokens))
            mean[i], maximum[i] = means, maxima
            counts["tokens"] += len(tokens)
            counts["zero_vector_tokens"] += zero
            counts["empty_tokenized_reviews"] += not tokens
            counts["reviews_without_usable_vectors"] += zero == len(tokens)
            if (i + 1) % settings.EXTRACT_PROGRESS_EVERY == 0 or i + 1 == n:
                print(f"  {name}: {i + 1:,}/{n:,} reviews", flush=True)
        arrays[f"{name}_mean"], arrays[f"{name}_max"] = mean, maximum
        statistics[name] = counts
    token_vector.cache_clear()
    del model
    manifest = {"identity": identity, "statistics": statistics}
    if reuse_cache:
        cache_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=cache_root))
        try:
            for name, array in arrays.items():
                np.save(temporary / f"{name}.npy", array, allow_pickle=False)
            (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            try:
                os.rename(temporary, cache_path)
            except OSError:
                # Another process may have published the same cache concurrently.
                if not cache_path.is_dir():
                    raise
                shutil.rmtree(temporary)
            arrays, manifest = _load_cache(cache_path, identity, len(train.ids), len(validation.ids))
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        print(f"Saved pooled embeddings to {cache_path}", flush=True)
    return arrays, {**manifest, "cache_path": str(cache_path) if reuse_cache else None, "reused": False}
