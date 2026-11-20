"""Cache exact token-to-vector mappings and compact, truncated token sequences."""
from array import array
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np

import experiment_settings as cfg
from legacy import phase1, phase1_config, indices_sha


def sample_partition(partition, count, rng):
    selected = rng.choice(len(partition.ids), size=min(count, len(partition.ids)), replace=False)
    return SimpleNamespace(ids=partition.ids[selected], texts=[partition.texts[i] for i in selected],
                           labels=partition.labels[selected])


def encode_texts(texts, token_to_id, tokens, max_length, name="reviews", progress_every=cfg.PROGRESS_EVERY):
    """Use exact Phase 1 regex, retain the first max_length tokens and collect lengths."""
    ids = array("i")
    if ids.itemsize != 4:
        raise RuntimeError("A 32-bit signed integer array is required for compact sequence storage.")
    offsets = [0]
    total = kept = truncated = empty = 0
    for index, text in enumerate(texts, 1):
        words = phase1.tokenize(text, "regex")
        n = len(words)
        total += n
        truncated += n > max_length
        empty += n == 0
        words = words[:max_length]
        kept += len(words)
        for word in words:
            word_id = token_to_id.get(word)
            if word_id is None:
                word_id = len(tokens)
                if word_id >= 2**31:
                    raise ValueError("Vocabulary exceeds the supported int32 ID range.")
                token_to_id[word] = word_id
                tokens.append(word)
            ids.append(word_id)
        offsets.append(len(ids))
        if progress_every and (index % progress_every == 0 or index == len(texts)):
            print(f"  {name}: {index:,}/{len(texts):,} reviews", flush=True)
    n_reviews = len(texts)
    statistics = {"reviews": n_reviews, "empty_reviews": empty,
                  "reviews_longer_than_limit": truncated,
                  "reviews_truncated_percent": 100.0 * truncated / n_reviews if n_reviews else 0.0,
                  "tokens_before_truncation": total, "tokens_retained": kept,
                  "tokens_discarded": total - kept,
                  "tokens_discarded_percent": 100.0 * (total - kept) / total if total else 0.0,
                  "max_sequence_length": max_length}
    return np.asarray(ids, dtype=np.int32), np.asarray(offsets, dtype=np.int64), statistics


def verified_model_digest(model_path: Path):
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing original fastText binary: {model_path}")
    cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fingerprint_file = cfg.CACHE_DIR / "model_fingerprint.json"
    stat = model_path.stat()
    identity = {"path": str(model_path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    old = {}
    if fingerprint_file.is_file():
        try:
            old = json.loads(fingerprint_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    if all(old.get(k) == v for k, v in identity.items()) and old.get("sha256"):
        digest = old["sha256"]
    else:
        print("Checking the original fastText binary SHA-256 (once per file version)...", flush=True)
        digest = phase1.file_sha256(model_path)
        fingerprint_file.write_text(json.dumps({**identity, "sha256": digest}, indent=2) + "\n", encoding="utf-8")
    if digest != phase1_config.MODEL_SHA256:
        raise ValueError("The binary does not match the original Phase 1 fastText model SHA-256.")
    return digest


def cache_identity(train, validation, split_info, model_sha, max_length, smoke):
    return {"version": cfg.CACHE_VERSION, "source_sha256": split_info["ordered_source_sha256"],
            "training_review_ids_sha256": indices_sha(train.ids),
            "validation_review_ids_sha256": indices_sha(validation.ids),
            "model_sha256": model_sha, "smoke": smoke, "max_length": max_length,
            "truncation": "first max_length tokens; right-truncate the rest",
            "tokenizer": "Phase 1 regex; preserve case and HTML markup",
            "tokenizer_function_sha256": hashlib.sha256(inspect.getsource(phase1.tokenize).encode()).hexdigest(),
            "regex_pattern": phase1.WORD_RE.pattern, "regex_flags": phase1.WORD_RE.flags,
            "fasttext_version": importlib.metadata.version("fasttext"), "numpy_version": np.__version__,
            "embedding_dimension": cfg.DIMENSION, "zero_token_policy": "retain tokens with zero vectors"}


def _load_cache(cache_dir, identity, train_count, validation_count):
    manifest_file = cache_dir / "manifest.json"
    if not manifest_file.is_file():
        raise ValueError(f"Incomplete sequence cache: {cache_dir}")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("identity") != identity:
        raise ValueError("Sequence cache identity does not match the requested run.")
    for filename, expected_sha in manifest["files_sha256"].items():
        if phase1.file_sha256(cache_dir / filename) != expected_sha:
            raise ValueError(f"Sequence cache file failed its SHA-256 check: {filename}")
    table = np.load(cache_dir / "embedding_table.npy", mmap_mode="r", allow_pickle=False)
    if table.ndim != 2 or table.shape[1] != cfg.DIMENSION or table.dtype != np.float32:
        raise ValueError("Invalid embedding table shape or dtype.")
    if len(table) != manifest["vocabulary_size_including_padding"] or np.any(table[0]) or not np.isfinite(table).all():
        raise ValueError("Invalid padding row, vocabulary size, or non-finite embeddings.")
    for name, count in (("train", train_count), ("validation", validation_count)):
        ids = np.load(cache_dir / f"{name}_ids.npy", mmap_mode="r", allow_pickle=False)
        offsets = np.load(cache_dir / f"{name}_offsets.npy", mmap_mode="r", allow_pickle=False)
        labels = np.load(cache_dir / f"{name}_labels.npy", mmap_mode="r", allow_pickle=False)
        review_ids = np.load(cache_dir / f"{name}_review_ids.npy", mmap_mode="r", allow_pickle=False)
        identity_key = "training_review_ids_sha256" if name == "train" else "validation_review_ids_sha256"
        if len(review_ids) != count or indices_sha(review_ids) != identity[identity_key]:
            raise ValueError(f"Invalid {name} review IDs in sequence cache.")
        if ids.dtype != np.int32 or len(labels) != count or len(offsets) != count + 1:
            raise ValueError(f"Invalid {name} cached sequence arrays.")
        lengths = np.diff(offsets)
        if offsets[0] != 0 or offsets[-1] != len(ids) or np.any(lengths < 0) or np.any(lengths > identity["max_length"]):
            raise ValueError(f"Invalid {name} sequence offsets or truncation lengths.")
        if len(ids) and (ids.min() < 1 or ids.max() >= len(table)):
            raise ValueError(f"Invalid {name} token ID.")
    return manifest


def print_statistics(name, stats):
    print(f"  {name}: {stats['reviews_longer_than_limit']:,}/{stats['reviews']:,} reviews longer than "
          f"{stats['max_sequence_length']} tokens ({stats['reviews_truncated_percent']:.2f}%); "
          f"discarded {stats['tokens_discarded']:,}/{stats['tokens_before_truncation']:,} tokens "
          f"({stats['tokens_discarded_percent']:.2f}%).", flush=True)


def prepare_sequences(train, validation, split_info, model_path, max_length, smoke):
    digest = verified_model_digest(model_path)
    identity = cache_identity(train, validation, split_info, digest, max_length, smoke)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    cache_root = cfg.CACHE_DIR / "sequences"
    cache_dir = cache_root / key
    if cache_dir.is_dir():
        manifest = _load_cache(cache_dir, identity, len(train.ids), len(validation.ids))
        print(f"Reusing sequence/vector cache: {cache_dir}", flush=True)
        for name, stats in manifest["statistics"].items():
            print_statistics(name, stats)
        return cache_dir, manifest, True

    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=cache_root))
    try:
        token_to_id = {}
        tokens = [None]  # Index 0 is exclusively padding, never an OOV fallback.
        statistics = {}
        for name, partition in (("train", train), ("validation", validation)):
            print(f"Tokenizing {name} reviews, retaining order...", flush=True)
            flat, offsets, stats = encode_texts(partition.texts, token_to_id, tokens, max_length, name)
            np.save(temporary / f"{name}_ids.npy", flat, allow_pickle=False)
            np.save(temporary / f"{name}_offsets.npy", offsets, allow_pickle=False)
            np.save(temporary / f"{name}_labels.npy", np.asarray(partition.labels, dtype=np.int8), allow_pickle=False)
            np.save(temporary / f"{name}_review_ids.npy", np.asarray(partition.ids, dtype=np.int64), allow_pickle=False)
            statistics[name] = stats
            print_statistics(name, stats)
            del flat, offsets
        del token_to_id
        (temporary / "tokens.json").write_text(json.dumps(tokens, ensure_ascii=False), encoding="utf-8")

        try:
            import fasttext
        except ImportError as exc:
            raise RuntimeError("fasttext is missing; use the Phase 2 Python environment.") from exc
        print("Loading the original fastText binary once to construct the frozen lookup...", flush=True)
        model = fasttext.load_model(str(model_path))
        if model.get_dimension() != cfg.DIMENSION:
            raise ValueError("Expected 300-dimensional fastText embeddings.")
        table = np.lib.format.open_memmap(temporary / "embedding_table.npy", mode="w+", dtype=np.float32,
                                          shape=(len(tokens), cfg.DIMENSION))
        table[0] = 0
        oov = np.zeros(len(tokens), dtype=bool)
        zero = np.zeros(len(tokens), dtype=bool)
        for index, token in enumerate(tokens[1:], 1):
            vector = np.asarray(model.get_word_vector(token), dtype=np.float32)
            if vector.shape != (cfg.DIMENSION,) or not np.isfinite(vector).all():
                raise ValueError(f"Invalid embedding for token {token!r}.")
            table[index] = vector
            oov[index] = model.get_word_id(token) < 0
            zero[index] = not np.any(vector)
            if index % 10000 == 0 or index == len(tokens) - 1:
                print(f"  fastText vectors: {index:,}/{len(tokens) - 1:,}", flush=True)
        table.flush()
        del table, model
        for name in ("train", "validation"):
            flat = np.load(temporary / f"{name}_ids.npy", mmap_mode="r", allow_pickle=False)
            statistics[name]["retained_oov_token_occurrences"] = int(np.count_nonzero(oov[flat]))
            statistics[name]["retained_zero_vector_token_occurrences"] = int(np.count_nonzero(zero[flat]))
        filenames = ["embedding_table.npy", "tokens.json"] + [
            f"{name}_{suffix}.npy" for name in ("train", "validation")
            for suffix in ("ids", "offsets", "labels", "review_ids")]
        file_hashes = {name: phase1.file_sha256(temporary / name) for name in filenames}
        manifest = {"identity": identity, "statistics": statistics,
                    "vocabulary_size_including_padding": len(tokens),
                    "unique_oov_tokens": int(oov.sum()), "unique_zero_vector_tokens": int(zero.sum()),
                    "embedding_table_sha256": file_hashes["embedding_table.npy"],
                    "tokens_sha256": file_hashes["tokens.json"], "files_sha256": file_hashes}
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        try:
            os.rename(temporary, cache_dir)
        except OSError:
            if not cache_dir.is_dir():
                raise
            shutil.rmtree(temporary)
        manifest = _load_cache(cache_dir, identity, len(train.ids), len(validation.ids))
        print(f"Saved sequence/vector cache: {cache_dir}", flush=True)
        return cache_dir, manifest, False
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
