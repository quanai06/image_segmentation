"""Kiểm thử DataLoader: tái lập được và các worker không trùng augmentation."""

import pytest
import torch

from medseg.data import build_dataloader
from medseg.data.dataset import processed_root

pytestmark = pytest.mark.skipif(
    not processed_root("kvasir_seg", 256).exists(), reason="chưa chạy prepare_data.py"
)


def first_batches(k: int = 3, seed: int = 42, workers: int = 2) -> list[dict]:
    it = iter(build_dataloader("kvasir_seg", "train", batch_size=4, num_workers=workers, seed=seed))
    return [next(it) for _ in range(k)]


def test_reproducible_with_workers():
    a, b = first_batches(), first_batches()
    for x, y in zip(a, b):
        assert x["name"] == y["name"]
        assert torch.equal(x["image"], y["image"]) and torch.equal(x["mask"], y["mask"])


def test_seed_changes_batches():
    a, b = first_batches(seed=1), first_batches(seed=2)
    assert a[0]["name"] != b[0]["name"]


def test_batch_format():
    x = first_batches(k=1, workers=0)[0]
    assert x["image"].shape == (4, 3, 256, 256) and x["mask"].shape == (4, 1, 256, 256)
    assert set(x["mask"].unique().tolist()) <= {0.0, 1.0}


def test_eval_loader_keeps_all_samples():
    dl = build_dataloader("cvc_clinicdb", "all", batch_size=16, num_workers=0)
    assert not dl.drop_last
    assert sum(len(b["name"]) for b in dl) == 612
