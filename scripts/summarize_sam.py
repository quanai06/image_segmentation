"""Gộp kết quả SAM zero-shot -> report/tables/sam_*.csv và report/figures/sam_*.png.

    python scripts/summarize_sam.py
    python scripts/summarize_sam.py --split test --examples-model vit_b
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

from medseg.data.dataset import processed_root  # noqa: E402
from medseg.paths import OUTPUTS, REPORT  # noqa: E402
from medseg.sam import sam_config  # noqa: E402
from medseg.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger()

# Thứ tự tăng dần lượng thông tin của gợi ý
LADDER = ["auto_top", "auto_oracle", "point_top", "point_oracle", "points", "box_jitter_px", "box_jitter_rel", "box"]
SIZES = ["small", "medium", "large"]
EXAMPLE_PROTOCOLS = ["point_top", "points", "box_jitter_px", "box"]


def load_results(split: str | None) -> pd.DataFrame:
    files = sorted((OUTPUTS / "sam").glob("*/*/per_image.csv"))
    if not files:
        raise FileNotFoundError("Chưa có kết quả. Chạy scripts/run_sam_zeroshot.py trước.")
    df = pd.concat([pd.read_csv(f, dtype={"name": str}) for f in files], ignore_index=True)
    return df[df["split"] == split] if split else df


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "dataset", "split", "protocol"])
    out = g.agg(
        n=("dice", "size"),
        dice_mean=("dice", "mean"), dice_std=("dice", "std"),
        iou_mean=("iou", "mean"), iou_std=("iou", "std"),
        precision=("precision", "mean"), recall=("recall", "mean"),
        hd95_mean=("hd95", "mean"), hd95_nan=("hd95", lambda s: int(s.isna().sum())),
        empty_pred_pct=("empty_pred", lambda s: 100 * s.mean()),
        sec_image=("sec_image", "mean"),
    ).reset_index()
    out["order"] = out["protocol"].map(LADDER.index)
    return out.sort_values(["model", "dataset", "split", "order"]).drop(columns="order").round(4)


def size_table(df: pd.DataFrame) -> pd.DataFrame:
    t = df.groupby(["model", "dataset", "split", "protocol", "size_bin"])["dice"].agg(["mean", "size"]).reset_index()
    return t.rename(columns={"mean": "dice_mean", "size": "n"}).round(4)


def plot_ladder(summary: pd.DataFrame, path: Path) -> None:
    ladder = [p for p in LADDER if p in set(summary["protocol"])]
    fig, ax = plt.subplots(figsize=(9, 4))
    for (model, dataset, split), g in summary.groupby(["model", "dataset", "split"]):
        g = g.set_index("protocol").reindex([p for p in ladder if p in set(g["protocol"])])
        x = [ladder.index(p) for p in g.index]
        ci = 1.96 * g["dice_std"] / np.sqrt(g["n"])
        ax.errorbar(x, g["dice_mean"], yerr=ci, marker="o", capsize=3, label=f"{dataset} ({model}, {split})")
    ax.set_xticks(range(len(ladder)), ladder, rotation=30, ha="right")
    ax.set_ylabel("Dice trung bình (95% CI)")
    ax.set_ylim(0, 1)
    ax.set_title("SAM zero-shot: Dice theo lượng thông tin của gợi ý (trái -> phải: tăng dần)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_by_size(sizes: pd.DataFrame, path: Path) -> None:
    keys = sizes[["model", "dataset", "split"]].drop_duplicates().values.tolist()
    fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 3.8), squeeze=False)
    for ax, (model, dataset, split) in zip(axes[0], keys):
        g = sizes[(sizes.model == model) & (sizes.dataset == dataset) & (sizes.split == split)]
        prots = [p for p in LADDER if p in set(g["protocol"])]
        w = 0.8 / len(SIZES)
        for i, s in enumerate(SIZES):
            vals = g[g.size_bin == s].set_index("protocol").reindex(prots)
            n = int(vals["n"].max()) if vals["n"].notna().any() else 0
            ax.bar(np.arange(len(prots)) + (i - 1) * w, vals["dice_mean"], w, label=f"{s} (n={n})")
        ax.set_xticks(range(len(prots)), prots, rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(f"{dataset} ({model}, {split})")
        ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("Dice trung bình")
    fig.suptitle("SAM zero-shot theo kích thước tổn thương (nhỏ < 5% <= vừa < 20% <= lớn)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_examples(df: pd.DataFrame, model: str, dataset: str, split: str, size: int, path: Path) -> bool:
    mask_dir = OUTPUTS / "sam" / model / f"{dataset}_{split}" / "masks"
    prots = [p for p in EXAMPLE_PROTOCOLS if (mask_dir / p).exists()]
    if "box" not in prots:
        return False
    box = df[(df.model == model) & (df.dataset == dataset) & (df.split == split) & (df.protocol == "box")]
    ranked = box.sort_values("dice")
    picks = list(ranked.tail(2).name[::-1]) + list(ranked.head(2).name)
    root = processed_root(dataset, size)
    fig, axes = plt.subplots(len(picks), len(prots) + 2, figsize=(2.3 * (len(prots) + 2), 2.4 * len(picks)))
    for i, name in enumerate(picks):
        img = cv2.cvtColor(cv2.imread(str(root / "images" / f"{name}.png")), cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(root / "masks" / f"{name}.png"), cv2.IMREAD_GRAYSCALE) > 127
        axes[i, 0].imshow(img)
        axes[i, 1].imshow(gt, cmap="gray")
        axes[i, 0].set_ylabel("tốt" if i < 2 else "xấu")
        for j, p in enumerate(prots, start=2):
            pred = cv2.imread(str(mask_dir / p / f"{name}.png"), cv2.IMREAD_GRAYSCALE) > 127
            d = df[(df.model == model) & (df.dataset == dataset) & (df.split == split)
                   & (df.protocol == p) & (df.name == name)]["dice"].iloc[0]
            axes[i, j].imshow(img)
            axes[i, j].imshow(np.ma.masked_where(~pred, pred), alpha=0.5, cmap="autumn")
            axes[i, j].contour(gt, colors="lime", linewidths=0.8)
            axes[i, j].set_title(f"{p}\nDice {d:.2f}", fontsize=8)
        axes[i, 0].set_title("ảnh", fontsize=8)
        axes[i, 1].set_title("nhãn", fontsize=8)
        for a in axes[i]:
            a.set_xticks([])
            a.set_yticks([])
    fig.suptitle(f"{dataset}: 2 ca tốt nhất và 2 ca xấu nhất theo gợi ý hộp (viền xanh = nhãn)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default=None, help="chỉ tổng hợp một tập (mặc định: tất cả)")
    parser.add_argument("--examples-model", default="vit_b")
    parser.add_argument("--examples-split", default="test")
    args = parser.parse_args()
    setup_logging("summarize_sam", args)

    df = load_results(args.split)
    tables, figures = REPORT / "tables", REPORT / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    summary, sizes = summary_table(df), size_table(df)
    summary.to_csv(tables / "sam_zeroshot.csv", index=False)
    sizes.to_csv(tables / "sam_by_size.csv", index=False)
    plot_ladder(summary, figures / "sam_prompt_ladder.png")
    plot_by_size(sizes, figures / "sam_by_size.png")

    size = sam_config()["eval_size"]
    for dataset in sorted(df["dataset"].unique()):
        plot_examples(df, args.examples_model, dataset, args.examples_split, size, figures / f"sam_examples_{dataset}.png")

    view = summary.assign(dice=summary.dice_mean.map("{:.3f}".format) + " ± " + summary.dice_std.map("{:.3f}".format))
    log.info(view.pivot_table(index="protocol", columns=["dataset", "model", "split"], values="dice", aggfunc="first")
          .reindex([p for p in LADDER if p in set(view.protocol)]).to_string())
    log.info("\nĐã ghi report/tables/sam_*.csv và report/figures/sam_*.png")


if __name__ == "__main__":
    main()
