"""Phân tích lỗi theo từng ảnh: hiệu năng theo diện tích polyp (nhỏ / vừa / lớn) và các đặc điểm dữ liệu.

Mục tiêu: đối chiếu kết quả huấn luyện với dữ liệu để biết nhóm ảnh nào hay sai, sai kiểu gì
(bỏ sót, thiếu vùng, lấn biên, báo nhầm vùng khác) và lệch diện tích bao nhiêu. Metric tổng của
results.json cộng dồn pixel trên toàn tập nên bị polyp lớn chi phối; ở đây tính theo từng ảnh.

Hai bước:
1. Dự đoán (cần checkpoint): ghi mặt nạ dự đoán ra <work-dir>/analysis/pred/<dataset>/<tên>.png
2. Phân tích (chỉ cần mặt nạ dự đoán + data/processed): metric từng ảnh, đặc điểm ảnh, bảng, hình

Bước 2 không phụ thuộc mô hình: mô hình khác (U-Net, ...) chỉ cần ghi mặt nạ dự đoán 0/255 cùng
định dạng vào <work-dir>/analysis/pred/ là dùng chung được.

Chạy từ gốc repo:
    python scripts/analyze_errors.py --ckpt outputs/segformer_b2/kvasir_seg/best.pth   # dự đoán + phân tích
    python scripts/analyze_errors.py --work-dir outputs/segformer_b2/kvasir_seg         # chỉ phân tích lại
    python scripts/analyze_errors.py --work-dir ... --targets kvasir_seg/val cvc_clinicdb/all
    python scripts/analyze_errors.py --work-dir ... --bins 0.05 0.15                     # ngưỡng nhỏ/vừa/lớn

Kết quả trong <work-dir>/analysis/:
    per_image.csv        metric + đặc điểm từng ảnh
    size_bins.csv        hiệu năng theo nhóm diện tích
    factors.csv          Dice theo đặc điểm dữ liệu (nhiều polyp, sát mép FOV, tương phản, ...)
    worst_cases.csv      các ảnh tệ nhất
    summary.md           bảng tóm tắt để dán vào báo cáo
    perf_by_size.png     Dice, recall, precision, lệch diện tích, HD95, tỷ lệ ảnh hỏng theo nhóm diện tích
    dice_vs_area.png     Dice và tỷ lệ diện tích dự đoán / thật theo diện tích polyp (thang log)
    error_types.png      cơ cấu kiểu lỗi trong từng nhóm diện tích
    error_factors.png    Dice theo đặc điểm dữ liệu
    worst_<dataset>.png  ảnh tệ nhất, tô TP / FP / FN
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medseg.config import load_data_config, load_model_config  # noqa: E402
from medseg.data.dataset import processed_root  # noqa: E402
from medseg.data.preprocess import PreprocessParams, fov_mask, highlight_mask, overlay_region  # noqa: E402
from medseg.data.splits import load_split  # noqa: E402
from medseg.paths import from_root, to_root  # noqa: E402

SIZE_LABELS = ("nhỏ", "vừa", "lớn")
ERROR_TYPES = ("tốt", "thiếu vùng", "lấn biên", "báo nhầm", "bỏ sót")
ERROR_HELP = {
    "tốt": "Dice >= ngưỡng tốt",
    "thiếu vùng": "lỗi chính là FN: dự đoán nhỏ hơn hoặc lệch khỏi polyp",
    "lấn biên": "lỗi chính là FP dính liền polyp: tràn ra mô xung quanh",
    "báo nhầm": "lỗi chính là FP ở vùng tách rời polyp: nhận nhầm cấu trúc khác",
    "bỏ sót": "không pixel nào của polyp được dự đoán (TP = 0)",
}

# bảng màu đã kiểm tra mù màu (dataviz validate_palette.js): 2 tập đánh giá, 4 kiểu lỗi + xám cho "tốt"
TARGET_COLORS = ("#2a78d6", "#eb6834")
ERROR_COLORS = {"tốt": "#d9d8d3", "thiếu vùng": "#1baf7a", "lấn biên": "#eda100",
                "báo nhầm": "#e87ba4", "bỏ sót": "#6250d6"}
TP_C, FP_C, FN_C = (0x1b, 0xaf, 0x7a), (0xe3, 0x49, 0x48), (0x2a, 0x78, 0xd6)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


# ---------------------------------------------------------------- cấu hình

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None, help="checkpoint để dự đoán; bỏ trống thì dùng mặt nạ dự đoán đã có")
    p.add_argument("--work-dir", default=None, help="mặc định: thư mục chứa --ckpt")
    p.add_argument("--model", default="segformer_b2")
    p.add_argument("--config", default="config.json")
    p.add_argument("--targets", nargs="+", default=None,
                   help="dataset/split, mặc định: test của bộ train + bộ chuyển miền (cross_domain)")
    p.add_argument("--bins", nargs=2, type=float, default=(0.05, 0.15), metavar=("NHO", "LON"),
                   help="ngưỡng tỷ lệ diện tích polyp / ảnh: nhỏ < NHO <= vừa < LON <= lớn (mặc định 0.05 0.15)")
    p.add_argument("--good-dice", type=float, default=0.8, help="Dice từ ngưỡng này trở lên coi là tốt")
    p.add_argument("--fail-dice", type=float, default=0.5, help="Dice dưới ngưỡng này coi là hỏng")
    p.add_argument("--worst", type=int, default=12, help="số ảnh tệ nhất vẽ cho mỗi tập")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default=None, help="mặc định: common.device")
    p.add_argument("--to-report", action="store_true", help="chép hình và bảng sang report/figures, report/tables")
    return p.parse_args()


def load_run_config(work_dir: Path, model: str, config: str) -> dict:
    """Cấu hình đã dùng khi train (config_used.json, có preset tiền xử lý); không có thì đọc configs/."""
    used = work_dir / "config_used.json"
    if used.exists():
        with open(used, encoding="utf-8") as f:
            return json.load(f)
    cfg = load_model_config(model, config)
    return {"dataset": cfg["common"]["train_dataset"], **cfg}


def default_targets(cfg: dict) -> list[str]:
    common, ds = cfg["common"], cfg["dataset"]
    out = [f"{ds}/{common['test_split']}"]
    cross = common.get("cross_domain")
    if cross and cross["dataset"] != ds:
        out.append(f"{cross['dataset']}/{cross['split']}")
    return out


def target_label(target: str, train_ds: str) -> str:
    ds, split = target.split("/")
    pretty = {"kvasir_seg": "Kvasir-SEG", "cvc_clinicdb": "CVC-ClinicDB"}.get(ds, ds)
    kind = "cùng miền" if ds == train_ds else "chuyển miền"
    return f"{pretty} {split} ({kind})"


# ---------------------------------------------------------------- bước 1: dự đoán

def predict(cfg: dict, ckpt: Path, targets: list[str], pred_dir: Path, batch_size: int, device_name: str | None) -> None:
    import torch

    from medseg.data import build_dataloader
    from medseg.models import build_segformer
    from medseg.training import load_checkpoint, resolve_device

    common = cfg["common"]
    device = resolve_device(device_name or common["device"])
    model = build_segformer(cfg)
    it = load_checkpoint(ckpt, model, map_location="cpu")
    model.to(device).eval()
    print(f"checkpoint {to_root(ckpt)} ({it} iter) | device {device} | preprocess {common.get('preprocess', 'none')}")
    for target in targets:
        ds, split = target.split("/")
        out = pred_dir / ds
        out.mkdir(parents=True, exist_ok=True)
        loader = build_dataloader(
            ds, split, batch_size=batch_size, shuffle=False, num_workers=common["num_workers"],
            seed=common["seed"], image_size=common["image_size"], augment=common["augment"],
            normalize=common["normalize"], preprocess=common.get("preprocess", "none"),
        )
        n = 0
        with torch.no_grad():
            for batch in loader:
                pred = model.predict(batch["image"].to(device)).cpu().numpy().astype(np.uint8)
                for name, m in zip(batch["name"], pred):
                    cv2.imwrite(str(out / f"{name}.png"), m * 255)
                    n += 1
        print(f"  [{target}] {n} mặt nạ dự đoán -> {to_root(out)}")


# ---------------------------------------------------------------- bước 2: metric từng ảnh

def _surface(m: np.ndarray) -> np.ndarray:
    return m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """HD95 đối xứng (pixel, ảnh 256): max của phân vị 95 khoảng cách biên theo hai chiều."""
    if not pred.any() or not gt.any():
        return float("nan")
    sp, sg = _surface(pred), _surface(gt)
    to_g = cv2.distanceTransform((~sg).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    to_p = cv2.distanceTransform((~sp).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return float(max(np.percentile(to_g[sp], 95), np.percentile(to_p[sg], 95)))


def components(m: np.ndarray) -> tuple[int, np.ndarray]:
    n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    return n - 1, lab


def to_lab(img: np.ndarray) -> np.ndarray:
    """CIELAB thật (L 0-100, a/b quanh 0) từ ảnh RGB uint8, để tính ΔE."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255
    lab[..., 1:] -= 128
    return lab


def image_record(img: np.ndarray, gt: np.ndarray, pred: np.ndarray, p: PreprocessParams, good: float) -> dict:
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    gt_px, pred_px = int(gt.sum()), int(pred.sum())
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0

    # thành phần liên thông: polyp bị bỏ sót, vùng dự đoán không chạm polyp nào (báo nhầm)
    n_gt, lab_gt = components(gt)
    n_pr, lab_pr = components(pred)
    hit_gt = np.unique(lab_gt[pred & gt])
    hit_pr = np.unique(lab_pr[pred & gt])
    far = np.isin(lab_pr, hit_pr, invert=True) & pred
    fp_far = int(far.sum())
    fp_near = fp - fp_far

    # đặc điểm ảnh, tính trên đúng ảnh mô hình thấy
    fov = fov_mask(img, p)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    near_edge = cv2.dilate((~fov).astype(np.uint8), k).astype(bool)
    ring = cv2.dilate(gt.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))).astype(bool) & ~gt & fov
    lab = to_lab(img)
    contrast = float(np.linalg.norm(lab[gt].mean(0) - lab[ring].mean(0))) if ring.any() and gt.any() else float("nan")
    overlay = overlay_region(img, p) & fov
    has_overlay = overlay.mean() >= p.overlay_min
    ov_zone = cv2.dilate(overlay.astype(np.uint8), k).astype(bool)
    hl = highlight_mask(img, fov, p)

    if dice >= good:
        err = "tốt"
    elif tp == 0:
        err = "bỏ sót"
    elif fn >= fp:
        err = "thiếu vùng"
    elif fp_far > fp_near:
        err = "báo nhầm"
    else:
        err = "lấn biên"

    ys, xs = np.nonzero(gt)
    return {
        "gt_frac": gt_px / gt.size,
        "gt_px": gt_px, "pred_px": pred_px, "tp": tp, "fp": fp, "fn": fn,
        "dice": dice,
        "iou": tp / (tp + fp + fn) if (tp + fp + fn) else 1.0,
        "precision": tp / pred_px if pred_px else float("nan"),
        "recall": tp / gt_px if gt_px else float("nan"),
        "area_ratio": pred_px / gt_px if gt_px else float("nan"),
        "hd95": hd95(pred, gt),
        "error_type": err,
        "n_gt_comp": n_gt, "n_pred_comp": n_pr,
        "missed_comp": n_gt - int((hit_gt > 0).sum()),
        "fp_far_comp": n_pr - int((hit_pr > 0).sum()),
        "fp_far_px": fp_far, "fp_near_px": fp_near,
        "fp_in_overlay_px": int((pred & ~gt & ov_zone).sum()),
        "has_overlay": bool(has_overlay),
        "at_fov_edge": bool((gt & near_edge).sum() >= 5),
        "contrast_dE": contrast,
        "highlight_in_lesion": float(hl[gt].mean()) if gt.any() else float("nan"),
        "brightness_L": float(lab[..., 0][fov].mean()),
        "centroid_dist": float(np.hypot(ys.mean() / gt.shape[0] - 0.5, xs.mean() / gt.shape[1] - 0.5)) if gt.any() else float("nan"),
    }


def per_image(cfg: dict, targets: list[str], pred_dir: Path, bins: tuple[float, float], good: float) -> pd.DataFrame:
    common = cfg["common"]
    data_cfg = load_data_config()
    p = PreprocessParams.from_config(data_cfg.get("preprocess_params"))
    rows = []
    for target in targets:
        ds, split = target.split("/")
        root = processed_root(ds, common["image_size"], data_cfg["processed_dir"], common.get("preprocess", "none"))
        names = load_split(ds, split)
        missing = [n for n in names if not (pred_dir / ds / f"{n}.png").exists()]
        if missing:
            raise FileNotFoundError(f"[{target}] thiếu {len(missing)}/{len(names)} mặt nạ dự đoán trong "
                                    f"{to_root(pred_dir / ds)} (vd. {missing[0]}). Chạy lại với --ckpt.")
        for name in names:
            img = cv2.cvtColor(cv2.imread(str(root / "images" / f"{name}.png")), cv2.COLOR_BGR2RGB)
            gt = cv2.imread(str(root / "masks" / f"{name}.png"), cv2.IMREAD_GRAYSCALE) > 127
            pred = cv2.imread(str(pred_dir / ds / f"{name}.png"), cv2.IMREAD_GRAYSCALE) > 127
            rows.append({"target": target, "dataset": ds, "name": name, **image_record(img, gt, pred, p, good)})
    df = pd.DataFrame(rows)
    df["target"] = pd.Categorical(df.target, categories=targets, ordered=True)  # giữ thứ tự tập trong bảng
    df["size_bin"] = pd.Categorical(
        np.select([df.gt_frac < bins[0], df.gt_frac < bins[1]], SIZE_LABELS[:2], SIZE_LABELS[2]),
        categories=SIZE_LABELS, ordered=True)
    return df


# ---------------------------------------------------------------- tổng hợp

def boot_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Khoảng tin cậy 95% bootstrap của trung bình."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def size_table(df: pd.DataFrame, fail: float) -> pd.DataFrame:
    rows = []
    for (target, b), g in df.groupby(["target", "size_bin"], observed=True, sort=True):
        lo, hi = boot_ci(g.dice)
        rows.append({
            "target": target, "size_bin": b, "n": len(g),
            "gt_frac_median": g.gt_frac.median(),
            "dice_mean": g.dice.mean(), "dice_ci_lo": lo, "dice_ci_hi": hi, "dice_median": g.dice.median(),
            "dice_pooled": 2 * g.tp.sum() / (2 * g.tp.sum() + g.fp.sum() + g.fn.sum()),
            "iou_mean": g.iou.mean(),
            "precision_mean": g.precision.mean(), "recall_mean": g.recall.mean(),
            "area_ratio_median": g.area_ratio.median(),
            "over_seg_pct": 100 * (g.area_ratio > 1.25).mean(), "under_seg_pct": 100 * (g.area_ratio < 0.8).mean(),
            "hd95_median": g.hd95.median(),
            "fail_pct": 100 * (g.dice < fail).mean(),
            "miss_pct": 100 * (g.tp == 0).mean(),
            "fp_far_img_pct": 100 * (g.fp_far_comp > 0).mean(),
            **{f"type_{t}_pct": 100 * (g.error_type == t).mean() for t in ERROR_TYPES},
        })
    return pd.DataFrame(rows)


def factor_levels(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Các đặc điểm dữ liệu để chia nhóm. Ngưỡng tertile lấy trên mọi ảnh phân tích để so được giữa các tập."""
    def tertile(col: str, names: tuple[str, str, str]) -> pd.Series:
        x = df[col]
        q = x.quantile([1 / 3, 2 / 3]).to_numpy()
        # nhiều giá trị bằng nhau (vd. phản xạ = 0 ở phần lớn ảnh): mức thấp lấy cả giá trị bằng ngưỡng
        low = x <= q[0] if q[0] <= x.min() else x < q[0]
        mid = ~low & ((x <= q[1]) if q[1] <= q[0] else (x < q[1]))
        lab = np.select([low, mid], names[:2], names[2])
        return pd.Series(pd.Categorical(np.where(x.isna(), None, lab), categories=names, ordered=True), index=df.index)

    def binary(cond: pd.Series, names: tuple[str, str]) -> pd.Series:
        return pd.Series(pd.Categorical(np.where(cond, names[1], names[0]), categories=names, ordered=True), index=df.index)

    return {
        "số polyp": binary(df.n_gt_comp > 1, ("một", "nhiều")),
        "sát mép FOV": binary(df.at_fov_edge, ("không", "có")),
        "tương phản polyp/nền (ΔE)": tertile("contrast_dE", ("thấp", "vừa", "cao")),
        "phản xạ sáng trong polyp": tertile("highlight_in_lesion", ("ít", "vừa", "nhiều")),
        "độ sáng ảnh (L)": tertile("brightness_L", ("tối", "vừa", "sáng")),
        "ô ScopeGuide": binary(df.has_overlay, ("không", "có")),
    }


def factor_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor, levels in factor_levels(df).items():
        for target, g in df.groupby("target", sort=False, observed=True):
            lv = levels.loc[g.index]
            for level in levels.cat.categories:
                if not (lv == level).any():
                    continue
                s = g[lv == level]
                lo, hi = boot_ci(s.dice)
                rows.append({"factor": factor, "level": level, "target": target, "n": len(s),
                             "dice_mean": s.dice.mean(), "dice_ci_lo": lo, "dice_ci_hi": hi,
                             "gt_frac_median": s.gt_frac.median(),
                             "small_pct": 100 * (s.size_bin == SIZE_LABELS[0]).mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- hình

def style() -> None:
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "legend.frameon": False,
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    })


def _grouped_box(ax, df: pd.DataFrame, targets: list[str], col: str, colors, jitter=True, log=False) -> None:
    """Box mảnh + điểm từng ảnh, nhóm theo diện tích, mỗi tập một màu."""
    w = 0.8 / len(targets)
    rng = np.random.default_rng(0)
    for i, t in enumerate(targets):
        for j, b in enumerate(SIZE_LABELS):
            v = df[(df.target == t) & (df.size_bin == b)][col].dropna().to_numpy()
            if log:
                v = v[v > 0]
            if not len(v):
                continue
            x = j + (i - (len(targets) - 1) / 2) * w
            ax.boxplot(v, positions=[x], widths=w * 0.7, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor="none", edgecolor=colors[i], lw=1.2),
                       medianprops=dict(color=colors[i], lw=2), whiskerprops=dict(color=colors[i], lw=1),
                       capprops=dict(color=colors[i], lw=1))
            if jitter:
                ax.scatter(x + rng.uniform(-w * 0.28, w * 0.28, len(v)), v, s=9, color=colors[i],
                           alpha=0.45, lw=0, zorder=3)
    if log:
        ax.set_yscale("log")


def _mean_ci_lines(ax, tab: pd.DataFrame, targets: list[str], col: str, colors) -> None:
    vals = np.array([tab[tab.target == t].set_index("size_bin").reindex(SIZE_LABELS)[col].to_numpy(float) for t in targets])
    for i in range(len(targets)):
        ax.plot(range(3), vals[i], "o-", color=colors[i], lw=2, ms=7, mec="#fcfcfb", mew=1.5)
        for j, v in enumerate(vals[i]):
            if np.isnan(v):
                continue
            # giá trị cao nhất trong cột ghi phía trên, các giá trị khác ghi phía dưới, để nhãn không chồng
            top = v >= np.nanmax(vals[:, j])
            ax.annotate(f"{v:.2f}", (j, v), textcoords="offset points", xytext=(0, 8 if top else -15),
                        ha="center", fontsize=8, color=MUTED)


def _bin_ticks(ax, tab: pd.DataFrame, targets: list[str], bins) -> None:
    edges = [f"<{bins[0]:.0%}", f"{bins[0]:.0%}–{bins[1]:.0%}", f"≥{bins[1]:.0%}"]
    ns = ["/".join(str(int(tab[(tab.target == t) & (tab.size_bin == b)].n.sum())) for t in targets) for b in SIZE_LABELS]
    ax.set_xticks(range(3), [f"{b}\n{e}\nn={n}" for b, e, n in zip(SIZE_LABELS, edges, ns)])
    ax.set_xlim(-0.5, 2.5)
    ax.grid(axis="x", visible=False)


def plot_perf_by_size(df, tab, targets, labels, bins, fail, out: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    ax = axes.ravel()
    _grouped_box(ax[0], df, targets, "dice", TARGET_COLORS)
    for i, t in enumerate(targets):  # trung bình + CI 95% đè lên box
        g = tab[tab.target == t].set_index("size_bin").reindex(SIZE_LABELS)
        x = np.arange(3) + (i - (len(targets) - 1) / 2) * 0.8 / len(targets)
        ax[0].errorbar(x, g.dice_mean, yerr=[g.dice_mean - g.dice_ci_lo, g.dice_ci_hi - g.dice_mean],
                       fmt="D", color=INK, ms=5, capsize=3, lw=1, zorder=4)
    ax[0].set(title="Dice từng ảnh  (◆ trung bình ± CI 95%)", ylabel="Dice", ylim=(-0.03, 1.03))

    _mean_ci_lines(ax[1], tab, targets, "recall_mean", TARGET_COLORS)
    ax[1].set(title="Recall trung bình (bắt được bao nhiêu % polyp)", ylabel="recall", ylim=(0, 1.05))
    _mean_ci_lines(ax[2], tab, targets, "precision_mean", TARGET_COLORS)
    ax[2].set(title="Precision trung bình (dự đoán đúng bao nhiêu %)", ylabel="precision", ylim=(0, 1.05))

    _grouped_box(ax[3], df.assign(area_ratio=df.area_ratio.clip(1 / 64, 64)), targets, "area_ratio", TARGET_COLORS, log=True)
    ax[3].axhline(1, color=INK, lw=1, ls="--")
    ax[3].set(title="Lệch diện tích: dự đoán / thật", ylabel="tỷ lệ (log; >1 lấn ra, <1 thiếu)")
    ax[3].set_yticks([1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8], ["1/8", "1/4", "1/2", "1", "2", "4", "8"])
    ax[3].set_ylim(1 / 12, 12)

    _grouped_box(ax[4], df, targets, "hd95", TARGET_COLORS)
    ax[4].set(title="HD95 (pixel, ảnh 256; thấp hơn là tốt)", ylabel="HD95")
    ax[4].set_ylim(0, max(10, float(df.hd95.quantile(0.97)) * 1.1))

    w = 0.8 / len(targets)
    for i, t in enumerate(targets):
        g = tab[tab.target == t].set_index("size_bin").reindex(SIZE_LABELS)
        x = np.arange(3) + (i - (len(targets) - 1) / 2) * w
        ax[5].bar(x, g.fail_pct, w - 0.04, color=TARGET_COLORS[i], edgecolor="#fcfcfb", lw=2)
        ax[5].bar(x, g.miss_pct, w - 0.04, color="none", edgecolor=INK, hatch="///", lw=0)
        for xx, v in zip(x, g.fail_pct):
            if pd.notna(v):
                ax[5].annotate(f"{v:.0f}%", (xx, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    ax[5].set(title=f"% ảnh hỏng (Dice < {fail})", ylabel="% ảnh trong nhóm")
    ax[5].set_ylim(0, max(10, float(tab.fail_pct.max()) * 1.25))

    for a in ax:
        _bin_ticks(a, tab, targets, bins)
    handles = [Patch(facecolor=c, label=lab) for c, lab in zip(TARGET_COLORS, labels)]
    handles.append(Patch(facecolor="none", edgecolor=INK, hatch="///", label="trong đó bỏ sót hoàn toàn (TP = 0)"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Hiệu năng theo diện tích polyp (tỷ lệ diện tích polyp / ảnh 256 × 256)", y=1.035, fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = out / "perf_by_size.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _binned_median(x: np.ndarray, y: np.ndarray, n_bins: int = 12, min_n: int = 6):
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x <= hi)
        if m.sum() >= min_n:
            xs.append(np.exp(np.log(x[m]).mean()))
            ys.append(np.median(y[m]))
    return xs, ys


def plot_dice_vs_area(df, targets, labels, bins, out: Path) -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for i, t in enumerate(targets):
        g = df[df.target == t]
        c = TARGET_COLORS[i]
        ax[0].scatter(g.gt_frac, g.dice, s=14, color=c, alpha=0.4, lw=0)
        ax[0].plot(*_binned_median(g.gt_frac.to_numpy(), g.dice.to_numpy()), color=c, lw=2.2, label=labels[i])
        r = g.area_ratio.clip(1 / 64, 64)
        ax[1].scatter(g.gt_frac, r, s=14, color=c, alpha=0.4, lw=0)
        ax[1].plot(*_binned_median(g.gt_frac.to_numpy(), r.to_numpy()), color=c, lw=2.2, label=labels[i])
    for a in ax:
        a.set_xscale("log")
        for b in bins:
            a.axvline(b, color=MUTED, lw=1, ls=":")
        a.set_xlabel("diện tích polyp / ảnh (log)")
        a.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ylim = ax[0].get_ylim()
    for b, x in zip(SIZE_LABELS, [bins[0] * 0.55, np.sqrt(bins[0] * bins[1]), bins[1] * 1.8]):
        ax[0].text(x, 1.04, b, ha="center", fontsize=9, color=MUTED)
    ax[0].set(title="Dice theo diện tích polyp", ylabel="Dice", ylim=(min(-0.03, ylim[0]), 1.09))
    ax[1].axhline(1, color=INK, lw=1, ls="--")
    ax[1].set_yscale("log")
    ax[1].set_yticks([1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8], ["1/8", "1/4", "1/2", "1", "2", "4", "8"])
    ax[1].set_ylim(1 / 12, 12)
    ax[1].set(title="Lệch diện tích theo diện tích polyp", ylabel="dự đoán / thật (log)")
    ax[0].legend(loc="lower right", title="đường: trung vị theo nhóm diện tích")
    fig.tight_layout()
    path = out / "dice_vs_area.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_error_types(tab, targets, labels, bins, good, out: Path) -> Path:
    fig, axes = plt.subplots(1, len(targets), figsize=(5.2 * len(targets) + 2.5, 4.8), sharey=True, squeeze=False)
    for ax, t, lab in zip(axes[0], targets, labels):
        g = tab[tab.target == t].set_index("size_bin").reindex(SIZE_LABELS)
        bottom = np.zeros(3)
        for e in ERROR_TYPES:
            v = g[f"type_{e}_pct"].fillna(0).to_numpy()
            ax.bar(range(3), v, 0.62, bottom=bottom, color=ERROR_COLORS[e], edgecolor="#fcfcfb", lw=2, label=e)
            for j, (vv, bb) in enumerate(zip(v, bottom)):
                if vv >= 6:
                    ax.text(j, bb + vv / 2, f"{vv:.0f}%", ha="center", va="center", fontsize=8.5, color=INK)
            bottom += v
        _bin_ticks(ax, tab, [t], bins)
        ax.set(title=lab, ylim=(0, 100))
    axes[0][0].set_ylabel("% ảnh trong nhóm")
    handles = [Patch(facecolor=ERROR_COLORS[e], label=f"{e}: {ERROR_HELP[e]}".replace("ngưỡng tốt", str(good)))
               for e in ERROR_TYPES]
    fig.legend(handles=handles[::-1], loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8.5)
    fig.suptitle("Kiểu lỗi theo diện tích polyp", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = out / "error_types.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_factors(ftab, targets, labels, out: Path) -> Path:
    factors = list(dict.fromkeys(ftab.factor))
    fig, axes = plt.subplots(len(factors), 1, figsize=(9.5, 0.62 * len(targets) * len(ftab.drop_duplicates(["factor", "level"])) + 1.5),
                             sharex=True, gridspec_kw={"height_ratios": [ftab[ftab.factor == f].level.nunique() for f in factors]})
    axes = np.atleast_1d(axes)
    off = np.linspace(-0.18, 0.18, len(targets)) if len(targets) > 1 else [0]
    for ax, f in zip(axes, factors):
        g = ftab[ftab.factor == f]
        levels = list(dict.fromkeys(g.level))
        for i, t in enumerate(targets):
            s = g[g.target == t].set_index("level").reindex(levels)
            y = np.arange(len(levels)) + off[i]
            ok = (s.n.fillna(0) > 0).to_numpy()
            m, lo, hi = (s[c].to_numpy(float)[ok] for c in ("dice_mean", "dice_ci_lo", "dice_ci_hi"))
            ax.errorbar(m, y[ok], xerr=[np.nan_to_num(m - lo), np.nan_to_num(hi - m)],
                        fmt="o", color=TARGET_COLORS[i], ms=7, mec="#fcfcfb", mew=1.5, capsize=3, lw=1.2)
            for yy, (_, r) in zip(y[ok], s[ok].iterrows()):
                ax.annotate(f"n={int(r.n)}, nhỏ {r.small_pct:.0f}%", (r.dice_ci_hi, yy), textcoords="offset points",
                            xytext=(5, -3), fontsize=7.5, color=MUTED)
        ax.set_yticks(range(len(levels)), levels)
        ax.set_ylim(-0.6, len(levels) - 0.4)
        ax.invert_yaxis()
        ax.set_ylabel(f, rotation=0, ha="right", va="center", fontsize=9.5)
        ax.grid(axis="y", visible=False)
    axes[-1].set_xlabel("Dice trung bình ± CI 95%   (chú thích: số ảnh, % polyp nhỏ trong nhóm, để soi gây nhiễu do kích thước)")
    handles = [Patch(facecolor=c, label=lab) for c, lab in zip(TARGET_COLORS, labels)]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.55, 1.0))
    fig.suptitle("Dice theo đặc điểm dữ liệu", y=1.035, fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = out / "error_factors.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def overlay_errors(img: np.ndarray, gt: np.ndarray, pred: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    out = img.astype(np.float32).copy()
    for m, c in ((pred & gt, TP_C), (pred & ~gt, FP_C), (~pred & gt, FN_C)):
        out[m] = (1 - alpha) * out[m] + alpha * np.array(c, np.float32)
    return out.clip(0, 255).astype(np.uint8)


def plot_worst(df, cfg, target, label, pred_dir, k, out: Path) -> Path | None:
    common = cfg["common"]
    ds = target.split("/")[0]
    worst = df[df.target == target].nsmallest(k, "dice")
    if worst.empty:
        return None
    root = processed_root(ds, common["image_size"], load_data_config()["processed_dir"], common.get("preprocess", "none"))
    cols = 4
    rows = int(np.ceil(len(worst) / cols))
    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 2.9, rows * 6.0), squeeze=False)
    for idx, (_, r) in enumerate(worst.iterrows()):
        img = cv2.cvtColor(cv2.imread(str(root / "images" / f"{r['name']}.png")), cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(root / "masks" / f"{r['name']}.png"), cv2.IMREAD_GRAYSCALE) > 127
        pred = cv2.imread(str(pred_dir / ds / f"{r['name']}.png"), cv2.IMREAD_GRAYSCALE) > 127
        a_img, a_ov = axes[(idx // cols) * 2, idx % cols], axes[(idx // cols) * 2 + 1, idx % cols]
        a_img.imshow(img)
        a_img.set_title(f"{r['name'][:14]}\nDice {r.dice:.2f} · {r.size_bin} {r.gt_frac:.1%}", fontsize=8.5, fontweight="normal")
        a_ov.imshow(overlay_errors(img, gt, pred))
        a_ov.set_title(f"{r.error_type} · dự đoán/thật {r.area_ratio:.2f}", fontsize=8.5, fontweight="normal")
    for a in axes.ravel():
        a.axis("off")
    handles = [Patch(facecolor=np.array(c) / 255, label=l) for c, l in
               ((TP_C, "TP: đúng"), (FP_C, "FP: dự đoán thừa"), (FN_C, "FN: bỏ sót"))]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"{len(worst)} ảnh Dice thấp nhất — {label}", y=1.02, fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = out / f"worst_{ds}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- tóm tắt

def summary_md(df, tab, ftab, labels_of, bins, good, fail) -> str:
    lines = ["# Phân tích lỗi theo diện tích polyp", "",
             f"Nhóm diện tích (tỷ lệ polyp / ảnh 256 × 256): nhỏ < {bins[0]:.0%}, vừa {bins[0]:.0%}–{bins[1]:.0%}, "
             f"lớn ≥ {bins[1]:.0%}. Ảnh tốt: Dice ≥ {good}; ảnh hỏng: Dice < {fail}.", ""]
    for t in tab.target.unique():
        g = tab[tab.target == t]
        d = df[df.target == t]
        pooled = 2 * d.tp.sum() / (2 * d.tp.sum() + d.fp.sum() + d.fn.sum())
        lines += [f"## {labels_of[t]}", "",
                  f"Dice trung bình theo ảnh {d.dice.mean():.3f}, Dice cộng dồn pixel (kiểu results.json) {pooled:.3f}.", "",
                  "| Nhóm | n | Dice TB [CI 95%] | Dice cộng dồn | Recall | Precision | Dự đoán/thật (trung vị) | HD95 (trung vị) | % hỏng | % bỏ sót | Kiểu lỗi chính |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            types = {e: r[f"type_{e}_pct"] for e in ERROR_TYPES if e != "tốt"}
            main = max(types, key=types.get)
            main_txt = f"{main} ({types[main]:.0f}%)" if types[main] > 0 else "—"
            lines.append(f"| {r.size_bin} | {r.n} | {r.dice_mean:.3f} [{r.dice_ci_lo:.3f}, {r.dice_ci_hi:.3f}] | {r.dice_pooled:.3f} "
                         f"| {r.recall_mean:.3f} | {r.precision_mean:.3f} | {r.area_ratio_median:.2f} | {r.hd95_median:.1f} "
                         f"| {r.fail_pct:.0f}% | {r.miss_pct:.0f}% | {main_txt} |")
        fp = d.fp.sum()
        lines += ["", f"- FP nằm ở vùng tách rời polyp (báo nhầm): {d.fp_far_px.sum() / max(fp, 1):.0%} tổng FP; "
                      f"{(d.fp_far_comp > 0).mean():.0%} ảnh có ít nhất một vùng báo nhầm.",
                  f"- FP rơi vào ô ScopeGuide: {d.fp_in_overlay_px.sum() / max(fp, 1):.1%} tổng FP "
                  f"({int(d.has_overlay.sum())} ảnh có ô).",
                  f"- Polyp bỏ sót trong ảnh nhiều polyp: {int(d[d.n_gt_comp > 1].missed_comp.sum())} / "
                  f"{int(d[d.n_gt_comp > 1].n_gt_comp.sum())} vùng.", ""]
    lines += ["## Dice theo đặc điểm dữ liệu", "",
              "| Đặc điểm | Mức | Tập | n | Dice TB [CI 95%] | % polyp nhỏ |", "|---|---|---|---|---|---|"]
    for _, r in ftab.iterrows():
        lines.append(f"| {r.factor} | {r.level} | {labels_of[r.target]} | {r.n} | {r.dice_mean:.3f} "
                     f"[{r.dice_ci_lo:.3f}, {r.dice_ci_hi:.3f}] | {r.small_pct:.0f}% |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not args.ckpt and not args.work_dir:
        raise SystemExit("Cần --ckpt (dự đoán + phân tích) hoặc --work-dir (phân tích mặt nạ dự đoán đã có)")
    ckpt = from_root(args.ckpt) if args.ckpt else None
    work_dir = from_root(args.work_dir) if args.work_dir else ckpt.parent
    out = work_dir / "analysis"
    pred_dir = out / "pred"
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_run_config(work_dir, args.model, args.config)
    targets = args.targets or default_targets(cfg)
    labels_of = {t: target_label(t, cfg["dataset"]) for t in targets}
    labels = [labels_of[t] for t in targets]
    bins = tuple(sorted(args.bins))

    if ckpt:
        predict(cfg, ckpt, targets, pred_dir, args.batch_size, args.device)

    df = per_image(cfg, targets, pred_dir, bins, args.good_dice)
    tab = size_table(df, args.fail_dice)
    ftab = factor_table(df)
    df.to_csv(out / "per_image.csv", index=False)
    tab.to_csv(out / "size_bins.csv", index=False)
    ftab.to_csv(out / "factors.csv", index=False)
    df.sort_values("dice").groupby("target", sort=False, observed=True).head(args.worst).to_csv(out / "worst_cases.csv", index=False)
    (out / "summary.md").write_text(summary_md(df, tab, ftab, labels_of, bins, args.good_dice, args.fail_dice), encoding="utf-8")

    style()
    figs = [plot_perf_by_size(df, tab, targets, labels, bins, args.fail_dice, out),
            plot_dice_vs_area(df, targets, labels, bins, out),
            plot_error_types(tab, targets, labels, bins, args.good_dice, out),
            plot_factors(ftab, targets, labels, out)]
    figs += [p for t in targets if (p := plot_worst(df, cfg, t, labels_of[t], pred_dir, args.worst, out))]

    cols = ["target", "size_bin", "n", "dice_mean", "dice_pooled", "recall_mean", "precision_mean",
            "area_ratio_median", "hd95_median", "fail_pct", "miss_pct"]
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(tab[cols].to_string(index=False))
    print("\nbảng:", ", ".join(to_root(out / f) for f in ("per_image.csv", "size_bins.csv", "factors.csv", "worst_cases.csv", "summary.md")))
    print("hình:", ", ".join(to_root(p) for p in figs))

    if args.to_report:
        import shutil

        from medseg.paths import REPORT
        run = to_root(work_dir).replace("outputs/", "").replace("/", "_")
        for p in figs:
            shutil.copy2(p, REPORT / "figures" / f"errors_{run}_{p.name}")
        for f in ("size_bins.csv", "factors.csv"):
            shutil.copy2(out / f, REPORT / "tables" / f"errors_{run}_{f}")
        print(f"đã chép sang report/figures/errors_{run}_*.png, report/tables/errors_{run}_*.csv")


if __name__ == "__main__":
    main()
