"""Chia tập theo FCBFormer và đọc/ghi file chia tập."""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from medseg.paths import DATA

SPLITS_DIR = DATA / "splits"
SPLIT_NAMES = ("train", "val", "test", "all")


def fcbformer_split(n: int, seed: int = 42) -> dict[str, np.ndarray]:
    """Tái tạo đúng split_ids() của FCBFormer: 80/10/10 trên chỉ số 0..n-1."""
    test_size = int(round(0.1 * n))
    ids = np.linspace(0, n - 1, n).astype("int")
    train_ids, test_ids = train_test_split(ids, test_size=test_size, random_state=seed)
    train_ids, val_ids = train_test_split(train_ids, test_size=test_size, random_state=seed)
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def split_path(dataset: str, split: str) -> Path:
    return SPLITS_DIR / dataset / f"{split}.txt"


def write_split(dataset: str, split: str, names: list[str]) -> Path:
    path = split_path(dataset, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + "\n", encoding="utf-8")
    return path


def load_split(dataset: str, split: str) -> list[str]:
    """Đọc danh sách tên mẫu (không đuôi file) của một tập."""
    if split not in SPLIT_NAMES:
        raise ValueError(f"split phải thuộc {SPLIT_NAMES}, nhận: {split}")
    path = split_path(dataset, split)
    if not path.exists():
        raise FileNotFoundError(f"Chưa có {path}. Chạy scripts/make_splits.py trước.")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
