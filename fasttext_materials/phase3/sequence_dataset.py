"""Compact ragged token IDs and dynamic right-padding for each batch."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, cache_dir: Path, split: str):
        self.ids = np.load(cache_dir / f"{split}_ids.npy", mmap_mode="r", allow_pickle=False)
        self.offsets = np.load(cache_dir / f"{split}_offsets.npy", mmap_mode="r", allow_pickle=False)
        self.labels = np.load(cache_dir / f"{split}_labels.npy", mmap_mode="r", allow_pickle=False)
        if len(self.offsets) != len(self.labels) + 1 or self.offsets[0] != 0 or self.offsets[-1] != len(self.ids):
            raise ValueError(f"Invalid {split} ragged-array cache.")
        if np.any(np.diff(self.offsets) < 0):
            raise ValueError(f"Invalid {split} offsets.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        # Copy only this review; the on-disk IDs remain read-only.
        sequence = torch.tensor(np.array(self.ids[start:end], dtype=np.int64, copy=True), dtype=torch.long)
        return sequence, int(self.labels[index])


def collate_reviews(rows):
    sequences, labels = zip(*rows)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    width = max(1, int(lengths.max().item()))
    padded = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        padded[row, :len(sequence)] = sequence
    return padded, lengths, torch.tensor(labels, dtype=torch.float32)
