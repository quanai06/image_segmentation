"""Kiểm định thống kê và phân tích lỗi cho kết quả SAM zero-shot.

- So sánh từng cặp kiểu gợi ý trên cùng ảnh: Wilcoxon signed-rank, hiệu chỉnh Holm trong mỗi (mô hình, bộ dữ liệu).
- So sánh các mô hình (vd vit_b với vit_h) trên cùng ảnh và cùng kiểu gợi ý.
- Phân tích lỗi: tỷ lệ diện tích dự đoán / nhãn, số ca thất bại (Dice < 0.5) theo kích thước tổn thương.

    python scripts/analyze_sam.py --split test

Kết quả: report/tables/sam_stats.csv, sam_model_compare.csv, sam_failures.csv
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medseg.paths import OUTPUTS, REPORT  # noqa: E402
from medseg.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger()

PAIRS = [
    ("auto_top", "auto_oracle"),
    ("point_top", "point_oracle"),
    ("point_top", "points"),
    ("points", "box"),
    ("box_jitter_px", "box"),
    ("box_jitter_rel", "box"),
]
FAIL = 0.5


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted, running = np.empty(len(pvalues)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted.tolist()


def paired_test(a: pd.Series, b: pd.Series) -> dict:
    diff = b - a
    p = wilcoxon(a, b).pvalue if np.any(diff != 0) else 1.0
    return {"n": len(diff), "mean_a": a.mean(), "mean_b": b.mean(), "mean_diff": diff.mean(),
            "median_diff": diff.median(), "b_better_pct": 100 * (diff > 0).mean(), "p": p}


def protocol_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, dataset), g in df.groupby(["model", "dataset"]):
        wide = g.pivot(index="name", columns="protocol", values="dice")
        group = []
        for a, b in PAIRS:
            if a in wide and b in wide:
                both = wide[[a, b]].dropna()
                group.append({"model": model, "dataset": dataset, "a": a, "b": b, **paired_test(both[a], both[b])})
        for r, p_holm in zip(group, holm([r["p"] for r in group])):
            r["p_holm"] = p_holm
        rows += group
    return pd.DataFrame(rows)


def model_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    models = sorted(df["model"].unique())
    for (dataset, protocol), g in df.groupby(["dataset", "protocol"]):
        wide = g.pivot(index="name", columns="model", values="dice")
        for a, b in itertools.combinations(models, 2):
            if a in wide and b in wide:
                both = wide[[a, b]].dropna()
                rows.append({"dataset": dataset, "protocol": protocol, "a": a, "b": b, **paired_test(both[a], both[b])})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = holm(out["p"].tolist())
    return out


def failures(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["area_ratio"] = np.where(d["precision"] > 0, d["recall"] / d["precision"].where(d["precision"] > 0), np.nan)
    g = d.groupby(["model", "dataset", "protocol"])
    out = g.agg(
        n=("dice", "size"),
        fail_pct=("dice", lambda s: 100 * (s < FAIL).mean()),
        overseg_pct=("area_ratio", lambda s: 100 * (s > 2).mean()),
        underseg_pct=("area_ratio", lambda s: 100 * (s < 0.5).mean()),
        median_area_ratio=("area_ratio", "median"),
    ).reset_index()
    by_size = d.assign(fail=d["dice"] < FAIL).pivot_table(
        index=["model", "dataset", "protocol"], columns="size_bin", values="fail", aggfunc="mean") * 100
    by_size.columns = [f"fail_pct_{c}" for c in by_size.columns]
    return out.merge(by_size.reset_index(), on=["model", "dataset", "protocol"]).round(3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    setup_logging("analyze_sam", args)

    files = sorted((OUTPUTS / "sam").glob(f"*/*_{args.split}/per_image.csv"))
    if not files:
        raise FileNotFoundError("Chưa có kết quả SAM. Chạy scripts/run_sam_zeroshot.py trước.")
    df = pd.concat([pd.read_csv(f, dtype={"name": str}) for f in files], ignore_index=True)
    log.info(f"{len(files)} file kết quả, mô hình: {sorted(df['model'].unique())}")

    tables = REPORT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    stats, models, fails = protocol_stats(df), model_stats(df), failures(df)
    stats.round(5).to_csv(tables / "sam_stats.csv", index=False)
    fails.to_csv(tables / "sam_failures.csv", index=False)
    if not models.empty:
        models.round(5).to_csv(tables / "sam_model_compare.csv", index=False)

    cols = ["model", "dataset", "a", "b", "n", "mean_diff", "b_better_pct", "p_holm"]
    log.info("So sánh kiểu gợi ý (b - a, Wilcoxon, Holm):\n" + stats[cols].round(4).to_string(index=False))
    log.info("Phân tích lỗi:\n" + fails.to_string(index=False))
    if not models.empty:
        log.info("So sánh mô hình:\n" + models[["dataset", "protocol", "a", "b", "n", "mean_diff", "p_holm"]].round(4).to_string(index=False))
    log.info("Đã ghi report/tables/sam_stats.csv, sam_failures.csv" + (", sam_model_compare.csv" if not models.empty else ""))


if __name__ == "__main__":
    main()
