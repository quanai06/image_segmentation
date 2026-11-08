"""Kiểm tra dữ liệu đã chuẩn hóa và xuất thống kê, hình minh họa cho báo cáo.

Kiểm tra:
- file chia tập: không chồng lấn, hợp lại đúng bằng all.txt
- ảnh/mặt nạ đã xử lý: đủ file, đúng kích thước, mặt nạ nhị phân
- thống kê kích thước tổn thương theo từng tập
- ảnh gần trùng giữa test và train (rò rỉ do khung hình video liên tiếp)

Kết quả: report/tables/*.csv và report/figures/*.png

Chạy từ gốc repo:
    python scripts/check_data.py
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medseg.config import load_data_config, load_dataset_specs  # noqa: E402
from medseg.data.dataset import SegmentationDataset, processed_root  # noqa: E402
from medseg.data.splits import load_split  # noqa: E402
from medseg.paths import DATA, REPORT, to_root  # noqa: E402

FIG = REPORT / "figures"
TAB = REPORT / "tables"
SPLITS = ("train", "val", "test")
SMALL = 0.05  # tổn thương nhỏ: < 5% diện tích ảnh


def check_splits(name: str) -> dict[str, list[str]]:
    parts = {s: load_split(name, s) for s in (*SPLITS, "all")}
    sets = {s: set(v) for s, v in parts.items()}
    for s, v in parts.items():
        assert len(v) == len(sets[s]), f"{name}/{s}: có tên trùng lặp"
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            assert not sets[a] & sets[b], f"{name}: {a} và {b} chồng lấn"
    assert sets["train"] | sets["val"] | sets["test"] == sets["all"], f"{name}: hợp các tập khác all"
    return parts


def check_files(name: str, size: int, names: list[str]) -> None:
    root = processed_root(name, size)
    for n in names:
        img = cv2.imread(str(root / "images" / f"{n}.png"))
        msk = cv2.imread(str(root / "masks" / f"{n}.png"), cv2.IMREAD_GRAYSCALE)
        assert img is not None and msk is not None, f"{name}/{n}: thiếu file"
        assert img.shape == (size, size, 3) and msk.shape == (size, size), f"{name}/{n}: sai kích thước"
        assert set(np.unique(msk)) <= {0, 255}, f"{name}/{n}: mặt nạ không nhị phân"


def split_stats(name: str, parts: dict, manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in (*SPLITS, "all"):
        m = manifest.set_index("name").loc[parts[s]]
        fg = m["fg_ratio_orig"]
        rows.append({
            "dataset": name, "split": s, "n": len(m),
            "fg_mean": fg.mean(), "fg_median": fg.median(), "fg_min": fg.min(), "fg_max": fg.max(),
            "small_lesion_pct": 100 * (fg < SMALL).mean(),
            "multi_lesion_pct": 100 * (m["n_components"] > 1).mean(),
        })
    return pd.DataFrame(rows)


def thumbnails(name: str, size: int, names: list[str]) -> np.ndarray:
    root = processed_root(name, size)
    out = []
    for n in names:
        g = cv2.imread(str(root / "images" / f"{n}.png"), cv2.IMREAD_GRAYSCALE)
        t = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        t -= t.mean()
        out.append(t / (np.linalg.norm(t) + 1e-8))
    return np.stack(out)


def near_duplicates(name: str, size: int, parts: dict) -> dict:
    """Tương quan lớn nhất giữa mỗi ảnh test và mọi ảnh train (ảnh thu nhỏ 32x32)."""
    tr, te = thumbnails(name, size, parts["train"]), thumbnails(name, size, parts["test"])
    best = (te @ tr.T).max(axis=1)
    return {
        "dataset": name,
        "test_n": len(best),
        "median_max_corr": float(np.median(best)),
        "pct_corr_ge_0.95": float(100 * (best >= 0.95).mean()),
        "pct_corr_ge_0.90": float(100 * (best >= 0.90).mean()),
        "_best": best,
    }


def overlay(img: np.ndarray, msk: np.ndarray) -> np.ndarray:
    out = img.copy()
    edge = cv2.morphologyEx(msk, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    out[msk > 0] = (0.6 * out[msk > 0] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
    out[edge] = (255, 255, 0)
    return out


def plot_samples(name: str, size: int, parts: dict, n: int = 6, seed: int = 0) -> Path:
    root = processed_root(name, size)
    rng = np.random.default_rng(seed)
    picks = rng.choice(parts["train"], n, replace=False)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.6))
    for j, p in enumerate(picks):
        img = cv2.cvtColor(cv2.imread(str(root / "images" / f"{p}.png")), cv2.COLOR_BGR2RGB)
        msk = (cv2.imread(str(root / "masks" / f"{p}.png"), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        axes[0, j].imshow(img)
        axes[1, j].imshow(overlay(img, msk))
        axes[0, j].set_title(p[:12], fontsize=8)
        for a in axes[:, j]:
            a.axis("off")
    axes[0, 0].set_ylabel("ảnh")
    fig.suptitle(f"{name}: ảnh {size}x{size} (trên) và mặt nạ (dưới)", fontsize=10)
    fig.tight_layout()
    path = FIG / f"{name}_samples.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_augment(name: str, size: int, n: int = 7, seed: int = 0) -> Path:
    ds = SegmentationDataset(name, "train", train=True, normalize="none", seed=seed)
    clean = SegmentationDataset(name, "train", train=False, normalize="none")
    idx = int(np.random.default_rng(seed).integers(len(ds)))
    fig, axes = plt.subplots(1, n + 1, figsize=(2.2 * (n + 1), 2.6))
    items = [clean[idx]] + [ds[idx] for _ in range(n)]
    for j, it in enumerate(items):
        img = (it["image"].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        msk = it["mask"][0].numpy().astype(np.uint8)
        axes[j].imshow(overlay(img, msk))
        axes[j].set_title("gốc" if j == 0 else f"aug {j}", fontsize=9)
        axes[j].axis("off")
    fig.suptitle(f"{name}: augmentation khi train (viền vàng = mặt nạ)", fontsize=10)
    fig.tight_layout()
    path = FIG / f"{name}_augment.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    specs = load_dataset_specs()
    cfg = load_data_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(specs), choices=list(specs))
    parser.add_argument("--size", type=int, default=cfg["image_size"])
    args = parser.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    all_stats, dup_rows, manifests = [], [], {}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for name in args.datasets:
        print(f"[{name}]")
        parts = check_splits(name)
        print(f"  chia tập hợp lệ: " + ", ".join(f"{s}={len(parts[s])}" for s in SPLITS))
        check_files(name, args.size, parts["all"])
        print(f"  {len(parts['all'])} cặp ảnh/mặt nạ {args.size}x{args.size}, mặt nạ nhị phân")
        manifest = pd.read_csv(DATA / "metadata" / f"{name}_manifest.csv", dtype={"name": str})
        manifests[name] = manifest
        all_stats.append(split_stats(name, parts, manifest))
        dup = near_duplicates(name, args.size, parts)
        best = dup.pop("_best")
        dup_rows.append(dup)
        print(f"  ảnh test có ảnh train tương quan >= 0.95: {dup['pct_corr_ge_0.95']:.1f}%")
        ax[1].hist(best, bins=40, alpha=0.6, label=name)
        ax[0].hist(manifest["fg_ratio_orig"], bins=40, alpha=0.6, label=name)
        print("  hình:", to_root(plot_samples(name, args.size, parts)), to_root(plot_augment(name, args.size)))

    ax[0].set_xlabel("tỷ lệ diện tích tổn thương")
    ax[0].set_ylabel("số ảnh")
    ax[0].set_title("Kích thước tổn thương")
    ax[1].set_xlabel("tương quan lớn nhất với một ảnh train")
    ax[1].set_title("Ảnh test gần trùng ảnh train")
    for a in ax:
        a.legend()
    fig.tight_layout()
    fig.savefig(FIG / "data_overview.png", dpi=120)
    plt.close(fig)

    stats = pd.concat(all_stats).round(4)
    stats.to_csv(TAB / "split_stats.csv", index=False)
    dups = pd.DataFrame(dup_rows).round(4)
    dups.to_csv(TAB / "near_duplicates.csv", index=False)
    sizes = pd.concat([
        m.assign(dataset=n)[["dataset", "orig_width", "orig_height"]] for n, m in manifests.items()
    ]).groupby("dataset").agg(["min", "median", "max"])
    sizes.to_csv(TAB / "image_sizes.csv")
    print("\n" + stats.to_string(index=False))
    print("\n" + dups.to_string(index=False))
    print("\nĐã ghi report/tables/ và report/figures/")


if __name__ == "__main__":
    main()
