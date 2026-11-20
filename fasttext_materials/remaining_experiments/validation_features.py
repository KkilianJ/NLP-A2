"""Optional, explicitly invoked validation-only features for paired error analysis."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pandas as pd

import bridge
from data_access import read_json, reserve_output, sequence_cache
import workflow_config as cfg


def model_digest(path):
    stat = path.stat()
    identity = {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    for old_path in (cfg.ASSIGNMENT / "phase3/.cache/model_fingerprint.json", cfg.ASSIGNMENT / ".cache/model_fingerprint.json"):
        if old_path.is_file():
            try:
                old = read_json(old_path)
                if all(old.get(k) == v for k, v in identity.items()) and old.get("sha256") == bridge.legacy.phase1_config.MODEL_SHA256:
                    return old["sha256"]
            except (OSError, ValueError):
                pass
    digest = bridge.legacy.phase1.file_sha256(path)
    if digest != bridge.legacy.phase1_config.MODEL_SHA256:
        raise ValueError("fastText binary differs from the original model.")
    return digest


def has_negation(tokens):
    for token in tokens:
        word = token.lower().replace("’", "'")
        if word in {"not", "no", "never", "neither", "nor", "cannot"} or word.endswith("n't") or "n't'" in word:
            return True
    return False


def oov_statistics(tokens, is_oov):
    flags = [is_oov(token) for token in tokens]
    result = {"oov_occurrences": sum(flags), "oov_fraction": sum(flags) / len(flags) if flags else np.nan}
    for limit in (256, 512):
        retained = flags[:limit]
        result[f"oov_occurrences_{limit}"] = sum(retained)
        result[f"oov_fraction_{limit}"] = sum(retained) / len(retained) if retained else np.nan
    return result


def validation_features(data_dir=cfg.DATA, model_path=cfg.MODEL, cache_dir=None):
    cache, manifest, original = sequence_cache()
    feature_dir = Path(cache_dir) if cache_dir else cfg.CACHE / "validation_features_v1"
    identity = {"validation_ids_sha256": original["split"]["validation_indices_sha256_le_int64"],
                "source_sha256": original["split"]["ordered_source_sha256"],
                "model_sha256": manifest["identity"]["model_sha256"],
                "tokenizer_sha256": manifest["identity"]["tokenizer_function_sha256"], "version": 1}
    if feature_dir.exists():
        metadata = read_json(feature_dir / "metadata.json")
        if metadata["identity"] != identity or bridge.legacy.phase1.file_sha256(feature_dir / "features.csv") != metadata["features_sha256"]:
            raise ValueError("Existing validation feature cache does not match its provenance; choose a new --feature-cache.")
        print(f"Reusing validation features: {feature_dir}", flush=True)
        return pd.read_csv(feature_dir / "features.csv", keep_default_na=False), metadata
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Original fastText binary required for OOV statistics: {model_path}")
    print("Reconstructing original project validation partition; no held-out predictions...", flush=True)
    _, validation, split = bridge.legacy.load_partitions(Path(data_dir))
    if split["validation_indices_sha256_le_int64"] != identity["validation_ids_sha256"]:
        raise ValueError("Validation IDs do not match the saved runs.")
    if model_digest(Path(model_path)) != identity["model_sha256"]:
        raise ValueError("Unexpected fastText binary.")
    import fasttext
    print("Loading fastText once for direct-vocabulary membership...", flush=True)
    model = fasttext.load_model(str(model_path))
    @lru_cache(maxsize=200_000)
    def is_oov(token):
        return model.get_word_id(token) < 0
    rows = []
    for i, (review_id, text, label) in enumerate(zip(validation.ids, validation.texts, validation.labels), 1):
        tokens = bridge.legacy.phase1.tokenize(text, "regex")
        rows.append({"review_id": int(review_id), "label": int(label), "token_length": len(tokens),
                     "negation_indicator": has_negation(tokens), **oov_statistics(tokens, is_oov),
                     "truncated_256": len(tokens) > 256, "truncated_512": len(tokens) > 512,
                     "excerpt_first_280_characters": text[:280]})
        if i % 500 == 0 or i == len(validation.ids):
            print(f"  Validation review features: {i:,}/{len(validation.ids):,}", flush=True)
    feature_dir = reserve_output(feature_dir)
    pd.DataFrame(rows).to_csv(feature_dir / "features.csv", index=False)
    metadata = {"identity": identity, "features_sha256": bridge.legacy.phase1.file_sha256(feature_dir / "features.csv"),
                "note": "OOV is exact stored-vocabulary absence on full original regex tokens and their retained prefixes, not a semantic quality score. "
                        "Negation is a surface-form indicator, not a verified explanation of an error."}
    (feature_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return pd.read_csv(feature_dir / "features.csv", keep_default_na=False), metadata
