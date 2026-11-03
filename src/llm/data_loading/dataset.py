import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedDataset(Dataset):
    """Memory-mapped dataset of tokenized, fixed-length windows into a packed corpus."""

    def __init__(self, path: str | Path, config: Any, dtype: np.dtype = np.uint16):
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(
                f"No packed corpus at {self.path}. Build it first."
            )

        self.dtype = dtype
        self.tokens = np.memmap(self.path, dtype=self.dtype, mode="r")
        self.meta: dict[str, Any] | None = None

        self.window = config.max_seq_len + 1
        self.num_windows = len(self.tokens) // self.window

        if self.num_windows == 0:
            raise ValueError(
                f"Corpus is too short. Need at least {self.window} tokens, got {len(self.tokens)}"
            )

    @classmethod
    def from_meta(cls, path: str | Path, config: Any) -> "PackedDataset":
        """Load a packed dataset using metadata from meta.json if available."""
        path = Path(path)
        meta_path = path.parent / "meta.json"

        dtype = np.uint16
        meta = None
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if "dtype" in meta:
                    dtype = getattr(np, meta["dtype"], np.uint16)
            except Exception:
                pass

        dataset = cls(path, config, dtype=dtype)
        dataset.meta = meta
        return dataset

    @property
    def num_tokens(self) -> int:
        """Total number of tokens in the memory-mapped corpus."""
        return len(self.tokens)

    def __len__(self) -> int:
        """Total number of complete windows in the corpus."""
        return self.num_windows

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return a single window of tokens as an int64 tensor."""
        start = index * self.window
        window = self.tokens[start : start + self.window]
        return {"input_ids": torch.from_numpy(window.astype(np.int64))}
