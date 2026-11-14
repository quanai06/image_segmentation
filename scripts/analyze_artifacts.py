"""Phân tích artifact trong ảnh nội soi và tác dụng của các preset tiền xử lý.

Đo trên ảnh gốc (data/raw), dùng đúng các bộ dò trong src/medseg/data/preprocess.py:
- khung tối ngoài vùng nhìn thấy (FOV): diện tích, mức xám
- ô ScopeGuide ở góc dưới trái (đen / xanh)
- phản xạ sáng: diện tích, mật độ trong tổn thương so với nền
- tổn thương nằm ngoài vùng hợp lệ (kiểm tra tiền xử lý không che mất tổn thương)
- màu trong FOV (LAB) theo bộ dữ liệu
- khoảng cách miền: một bộ phân loại tuyến tính đoán nguồn ảnh (Kvasir hay CVC) từ thống kê
  màu toàn ảnh 256 x 256; so sánh khi không tiền xử lý và với từng preset

Kết quả:
    report/tables/artifacts_per_image.csv   từng ảnh
    report/tables/artifact_stats.csv        tổng hợp theo bộ
    report/tables/domain_gap.csv            độ chính xác đoán nguồn ảnh theo preset
    report/figures/preprocess_examples.png
    report/figures/color_shift.png

Chạy từ gốc repo (cần data/raw, xem download_data.py):
    python scripts/analyze_artifacts.py
    python scripts/analyze_artifacts.py --workers 8
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medseg.config import load_data_config, load_dataset_specs  # noqa: E402
from medseg.data.io import list_images, read_image, read_mask  # noqa: E402
from medseg.data.preprocess import (  # noqa: E402
    PreprocessParams,
    _green,
    highlight_mask,
    inpaint_highlights,
    lab_stats,
    normalize_color,
    overlay_region,
    valid_mask,
)
from medseg.data.splits import load_split  # noqa: E402
from medseg.paths import REPORT, from_root, to_root  # noqa: E402

FIG = REPORT / "figures"
TAB = REPORT / "tables"
VARIANTS = ("none", "clean", "clean_color")  # khớp preprocess_presets trong configs/data.yaml


def domain_features(img: np.ndarray, valid: np.ndarray, size: int) -> dict[str, np.ndarray]:
    """Đặc trưng để đoán nguồn ảnh, tính trên ảnh đã resize về size x size.

    global:    trung bình, độ lệch chuẩn RGB và LAB của toàn ảnh (gồm cả khung)
    fov_color: phân vị 5/25/50/75/95 của L, a, b chỉ trong vùng hợp lệ (bỏ qua hình học khung)
    """
    small = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    v = cv2.resize(valid.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST) > 0
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    flat_lab, rgb = lab.reshape(-1, 3), small.reshape(-1, 3).astype(np.float32)
    return {
        "global": np.concatenate([rgb.mean(0), rgb.std(0), flat_lab.mean(0), flat_lab.std(0)]),
        "fov_color": np.percentile(lab[v], [5, 25, 50, 75, 95], axis=0).ravel(),
    }


def analyze_one(job: tuple) -> dict:
    dataset, name, img_path, msk_path, params, size = job
    p = PreprocessParams.from_config(params)
    img = read_image(img_path)
    lesion = read_mask(msk_path) > 0
    valid, ov = valid_mask(img, p)
    overlay = overlay_region(img, p) & ~valid
    hl = highlight_mask(img, valid, p)
    bg = valid & ~lesion
    clean = img.copy()
    clean[~valid] = 0
    clean, _ = inpaint_highlights(clean, valid, p)
    clean_color = normalize_color(clean, valid, p)
    row = {
        "dataset": dataset, "name": name,
        "valid_frac": valid.mean(),
        "frame_level": float(np.median(img.max(axis=2)[~valid])) if (~valid).any() else np.nan,
        "overlay_frac": ov,
        "overlay_box": ov >= p.overlay_min,
        "overlay_green": bool(ov >= p.overlay_min and _green(img, p)[overlay].mean() > 0.1),
        "highlight_frac": hl.mean(),
        "highlight_in_lesion": hl[lesion].mean() if lesion.any() else np.nan,
        "highlight_in_bg": hl[bg].mean() if bg.any() else np.nan,
        "lesion_outside_valid": (lesion & ~valid).sum() / max(lesion.sum(), 1),
        **{f"raw_{k}": v for k, v in lab_stats(img, valid).items()},
        **{f"clean_{k}": v for k, v in lab_stats(clean, valid).items()},
    }
    feats = {}
    for v, x in zip(VARIANTS, (img, clean, clean_color)):
        for group, f in domain_features(x, valid, size).items():
            feats[(group, v)] = f
    feats[("geometry", "any")] = np.array([valid.mean(), ov])
    return row, feats


def domain_gap(feats: dict, labels: np.ndarray, seed: int = 0) -> pd.DataFrame:
    """Độ chính xác cân bằng (5-fold) của hồi quy logistic đoán nguồn ảnh. 0.5 = không phân biệt được."""
    rows = []
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    for (group, v), x in feats.items():
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        acc = cross_val_score(clf, x, labels, cv=cv, scoring="balanced_accuracy")
        rows.append({"features": group, "preset": v, "balanced_acc_mean": acc.mean(), "balanced_acc_std": acc.std()})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, d in df.groupby("dataset"):
        rows.append({
            "dataset": name, "n": len(d),
            "frame_level_median": d.frame_level.median(),
            "valid_frac_median": d.valid_frac.median(),
            "overlay_box_pct": 100 * d.overlay_box.mean(),
            "overlay_green_n": int(d.overlay_green.sum()),
            "highlight_frac_median": d.highlight_frac.median(),
            "highlight_frac_p95": d.highlight_frac.quantile(0.95),
            "highlight_lesion_vs_bg": d.highlight_in_lesion.mean() / d.highlight_in_bg.mean(),
            "lesion_outside_valid_mean": d.lesion_outside_valid.mean(),
            "lesion_outside_valid_max": d.lesion_outside_valid.max(),
            "n_lesion_outside_gt_1pct": int((d.lesion_outside_valid > 0.01).sum()),
            "L_mean_median": d.raw_L_mean.median(),
            "L_std_median": d.raw_L_std.median(),
            "a_mean_median": d.raw_a_mean.median(),
            "b_mean_median": d.raw_b_mean.median(),
        })
    return pd.DataFrame(rows)


def color_target(df: pd.DataFrame, dataset: str, names: list[str]) -> dict:
    """Đích chuẩn hóa màu: trung vị thống kê LAB (sau bước clean) trên tập train của một bộ."""
    d = df[(df.dataset == dataset) & df.name.isin(names)]
    return {k: round(float(d[f"clean_{k}"].median()), 1) for k in ("L_mean", "L_std", "a_mean", "b_mean")}


def plot_examples(picks: list[tuple[str, Path]], p: PreprocessParams) -> Path:
    cols = ["ảnh gốc", "phát hiện", "clean", "clean_color"]
    fig, axes = plt.subplots(len(picks), 4, figsize=(11, 2.4 * len(picks)))
    for i, (label, path) in enumerate(picks):
        img = read_image(path)
        valid, _ = valid_mask(img, p)
        hl = highlight_mask(img, valid, p)
        det = img.copy()
        det[~valid] = (0.35 * det[~valid] + 0.65 * np.array([40, 90, 255])).astype(np.uint8)
        det[hl] = (255, 0, 255)
        clean = img.copy()
        clean[~valid] = 0
        clean, _ = inpaint_highlights(clean, valid, p)
        for j, x in enumerate((img, det, clean, normalize_color(clean, valid, p))):
            axes[i, j].imshow(x)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(cols[j], fontsize=10)
        axes[i, 0].set_ylabel(label, fontsize=8)
    fig.suptitle("Tiền xử lý: xanh dương = ngoài vùng hợp lệ (khung, ô ScopeGuide), tím = phản xạ sáng/chữ", fontsize=10)
    fig.tight_layout()
    out = FIG / "preprocess_examples.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_color(df: pd.DataFrame, p: PreprocessParams) -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    for name, d in df.groupby("dataset"):
        ax[0].hist(d.raw_L_mean, bins=40, alpha=0.6, label=name)
        ax[1].scatter(d.raw_a_mean, d.raw_b_mean, s=4, alpha=0.5, label=name)
    ax[0].axvline(p.color_target["L_mean"], color="k", ls="--", lw=1, label="đích clean_color")
    ax[0].set_xlabel("độ sáng L trung bình trong FOV (0-255)")
    ax[0].set_ylabel("số ảnh")
    ax[0].set_title("Độ sáng theo bộ dữ liệu")
    ax[1].scatter([p.color_target["a_mean"]], [p.color_target["b_mean"]], c="k", marker="x", s=60, label="đích")
    ax[1].set_xlabel("a trung bình (đỏ)")
    ax[1].set_ylabel("b trung bình (vàng)")
    ax[1].set_title("Sắc độ trong FOV")
    for a in ax:
        a.legend(fontsize=8)
    fig.tight_layout()
    out = FIG / "color_shift.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    specs = load_dataset_specs()
    cfg = load_data_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(specs), choices=list(specs))
    parser.add_argument("--size", type=int, default=cfg["image_size"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--target-from", default="kvasir_seg", help="bộ dùng để đề xuất đích chuẩn hóa màu (tập train)")
    args = parser.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    params = cfg.get("preprocess_params", {})
    p = PreprocessParams.from_config(params)

    jobs = []
    for name in args.datasets:
        spec = specs[name]
        masks = {q.stem: q for q in list_images(from_root(spec["raw_masks"]))}
        for q in list_images(from_root(spec["raw_images"])):
            jobs.append((name, q.stem, q, masks[q.stem], params, args.size))
    print(f"phân tích {len(jobs)} ảnh với {args.workers} worker...")
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(analyze_one, jobs, chunksize=8))

    df = pd.DataFrame([r for r, _ in results])
    df.round(6).to_csv(TAB / "artifacts_per_image.csv", index=False)
    stats = summarize(df).round(4)
    stats.to_csv(TAB / "artifact_stats.csv", index=False)
    print("\n" + stats.T.to_string(header=False))

    if len(args.datasets) > 1:
        feats = {k: np.stack([f[k] for _, f in results]) for k in results[0][1]}
        gap = domain_gap(feats, df.dataset.to_numpy()).round(4)
        gap.to_csv(TAB / "domain_gap.csv", index=False)
        print("\nĐoán nguồn ảnh từ thống kê màu toàn ảnh (0.5 = không phân biệt được):")
        print(gap.to_string(index=False))

    target = color_target(df, args.target_from, load_split(args.target_from, "train"))
    print(f"\nĐích chuẩn hóa màu đề xuất (trung vị tập train {args.target_from}): {target}")
    if any(abs(target[k] - p.color_target[k]) > 1 for k in target):
        print(f"  khác configs/data.yaml ({p.color_target}): cập nhật preprocess_params.color_target")

    picks = []
    kv = df[df.dataset == "kvasir_seg"].sort_values("name")
    if len(kv):
        root = from_root(specs["kvasir_seg"]["raw_images"])
        chosen = [
            ("Kvasir: ô đen", kv[kv.overlay_box & ~kv.overlay_green]),
            ("Kvasir: ô xanh", kv[kv.overlay_green]),
            ("Kvasir: nhiều phản xạ", kv.sort_values("highlight_frac", ascending=False)),
        ]
        for label, d in chosen:
            if len(d):
                picks.append((label, next(root.glob(f"{d.iloc[0]['name']}.*"))))
    cvc = df[df.dataset == "cvc_clinicdb"]
    if len(cvc):
        root = from_root(specs["cvc_clinicdb"]["raw_images"])
        n = cvc.sort_values("highlight_frac").iloc[len(cvc) // 2]["name"]
        picks.append(("CVC-ClinicDB", next(root.glob(f"{n}.*"))))
    print("\nhình:", to_root(plot_examples(picks, p)), to_root(plot_color(df, p)))
    print("Đã ghi report/tables/{artifacts_per_image,artifact_stats,domain_gap}.csv")


if __name__ == "__main__":
    main()
