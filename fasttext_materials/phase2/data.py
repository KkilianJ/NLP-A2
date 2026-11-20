"""Reconstruct and verify the existing Phase 1 train/validation partitions."""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from phase1_imports import phase1, phase1_config
import settings


@dataclass
class Partition:
    ids: np.ndarray
    texts: list[str]
    labels: np.ndarray


def indices_sha(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def load_partitions(data_dir: Path):
    # Phase 1 verifies the source file and ordered text/label fingerprints and
    # reconstructs its fixed split. It returns only the training review records.
    records, original = phase1.load_training_partition(data_dir, smoke=False)
    if original["full_training_indices_sha256_le_int64"] != "3fe569c54add383b4b7249ea3465297678599f30dc8c0d48d43ee7186f09ec82":
        raise ValueError("The reconstructed training split differs from the saved Phase 1 split.")
    if phase1_config.SEED != settings.SPLIT_SEED:
        raise ValueError("The Phase 1 split seed has changed; refusing to create a different split.")

    paths = [data_dir / name for name in ("train.parquet", "test.parquet")]
    df = pd.concat([pd.read_parquet(p, columns=["text", "label_text"]) for p in paths], ignore_index=True)
    train_idx, other = train_test_split(np.arange(len(df)), test_size=0.30,
                                        stratify=df.label_text, random_state=settings.SPLIT_SEED)
    val_idx, heldout_idx = train_test_split(other, test_size=0.5, stratify=df.label_text.iloc[other],
                                            random_state=settings.SPLIT_SEED)
    rng = np.random.default_rng(settings.SPLIT_SEED)
    train_idx = rng.permutation(train_idx)
    val_idx = rng.permutation(val_idx)
    if not np.array_equal(train_idx, np.asarray([row[0] for row in records])):
        raise ValueError("Training IDs disagree with Phase 1; split was not reused.")
    if len(train_idx) != 34707 or len(val_idx) != 7437 or len(heldout_idx) != 7438:
        raise ValueError("Unexpected split sizes; refusing to proceed.")
    del heldout_idx, other

    def part(indices):
        rows = df.iloc[indices]
        labels = rows.label_text.map({"negative": 0, "positive": 1})
        if labels.isna().any():
            raise ValueError("Unexpected IMDb class labels; expected negative and positive.")
        return Partition(np.asarray(indices, dtype=np.int64), rows.text.tolist(), labels.to_numpy(dtype=np.int64))

    train, validation = part(train_idx), part(val_idx)
    if train.texts != [text for _, text in records]:
        raise ValueError("Training text order disagrees with Phase 1.")
    info = {"source_revision": phase1_config.DATASET_REVISION,
            "ordered_source_sha256": phase1_config.ORDERED_DATA_SHA256,
            "split_seed": settings.SPLIT_SEED, "training_count": len(train.ids),
            "validation_count": len(validation.ids), "heldout_test_count": 7438,
            "training_indices_sha256_le_int64": indices_sha(train.ids),
            "validation_indices_sha256_le_int64": indices_sha(validation.ids),
            "label_mapping": {"negative": 0, "positive": 1}}
    return train, validation, info
