"""Kiểm thử pipeline dữ liệu. Chạy: pytest -q"""

import cv2
import numpy as np
import pytest
import torch

from medseg.data.dataset import SegmentationDataset, processed_root
from medseg.data.io import read_mask, resize_pair
from medseg.data.splits import fcbformer_split, load_split, split_path
from medseg.paths import ROOT

DATASETS = {"kvasir_seg": (800, 100, 100), "cvc_clinicdb": (490, 61, 61)}


def has_split(name: str) -> bool:
    return split_path(name, "train").exists()


def has_processed(name: str, size: int = 256) -> bool:
    return processed_root(name, size).exists() and has_split(name)


def test_root_is_repo():
    assert (ROOT / "pyproject.toml").exists()


@pytest.mark.parametrize("n,expected", [(1000, (800, 100, 100)), (612, (490, 61, 61))])
def test_fcbformer_split_sizes(n, expected):
    parts = fcbformer_split(n, seed=42)
    assert tuple(len(parts[s]) for s in ("train", "val", "test")) == expected
    joined = np.concatenate([parts[s] for s in ("train", "val", "test")])
    assert sorted(joined.tolist()) == list(range(n))


def test_fcbformer_split_deterministic():
    a, b = fcbformer_split(1000), fcbformer_split(1000)
    assert all((a[s] == b[s]).all() for s in a)


def test_read_mask_binarizes_jpeg_noise(tmp_path):
    m = np.zeros((40, 40), np.uint8)
    m[10:30, 10:30] = 255
    path = tmp_path / "m.jpg"
    cv2.imwrite(str(path), m, [cv2.IMWRITE_JPEG_QUALITY, 30])
    out = read_mask(path)
    assert set(np.unique(out)) <= {0, 1}
    assert abs(int(out.sum()) - 400) < 40


def test_resize_pair_keeps_mask_binary():
    img = np.random.randint(0, 255, (300, 200, 3), np.uint8)
    msk = np.zeros((300, 200), np.uint8)
    msk[50:150, 40:120] = 1
    img_r, msk_r = resize_pair(img, msk, 64)
    assert img_r.shape == (64, 64, 3) and msk_r.shape == (64, 64)
    assert set(np.unique(msk_r)) <= {0, 1}


def test_cvc_tiff_read_as_color():
    # TIFF của CVC khai báo sai photometric: OpenCV đọc thành ảnh xám
    path = ROOT / "data/raw/cvc_clinicdb/CVC-ClinicDB/Original/1.tif"
    if not path.exists():
        pytest.skip("chưa tải CVC-ClinicDB")
    from medseg.data.io import read_image
    img = read_image(path)
    assert img.shape == (288, 384, 3)
    assert not np.array_equal(img[..., 0], img[..., 1])
    assert img[..., 0].mean() > img[..., 2].mean()  # nội soi: kênh đỏ trội


@pytest.mark.parametrize("name", DATASETS)
def test_split_files(name):
    if not has_split(name):
        pytest.skip("chưa chạy make_splits.py")
    parts = {s: set(load_split(name, s)) for s in ("train", "val", "test", "all")}
    assert tuple(len(parts[s]) for s in ("train", "val", "test")) == DATASETS[name]
    assert not parts["train"] & parts["val"] and not parts["train"] & parts["test"] and not parts["val"] & parts["test"]
    assert parts["train"] | parts["val"] | parts["test"] == parts["all"]


@pytest.mark.parametrize("name", DATASETS)
@pytest.mark.parametrize("split", ["train", "test"])
def test_dataset_item(name, split):
    if not has_processed(name):
        pytest.skip("chưa chạy prepare_data.py")
    ds = SegmentationDataset(name, split)
    item = ds[0]
    assert item["image"].shape == (3, 256, 256) and item["image"].dtype == torch.float32
    assert item["mask"].shape == (1, 256, 256) and item["mask"].dtype == torch.float32
    assert set(torch.unique(item["mask"]).tolist()) <= {0.0, 1.0}


@pytest.mark.parametrize("name", DATASETS)
def test_eval_transform_is_deterministic(name):
    if not has_processed(name):
        pytest.skip("chưa chạy prepare_data.py")
    ds = SegmentationDataset(name, "test")
    assert torch.equal(ds[0]["image"], ds[0]["image"])
    assert torch.equal(ds[0]["mask"], ds[0]["mask"])
