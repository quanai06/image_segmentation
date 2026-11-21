"""So sánh hai mô hình từ experiments.json; tùy chọn kiểm tra CSV gốc từ Kaggle.

python scripts/compare_experiments.py --plot
python scripts/compare_experiments.py --unet-dir <run_U-Net> --resnet34-dir <run_ResNet34>
"""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "report/results"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_splits(path: Path) -> dict[str, str]:
    result = {}
    for row in read_csv(path):
        name = row["name"].strip()
        label = row["split"].strip()
        label = "validation" if label == "val" else label
        if not name or name in result or label not in ("train", "validation", "test"):
            raise ValueError(f"{path}: ID rỗng/trùng hoặc nhãn split không hợp lệ")
        result[name] = label
    if set(result.values()) != {"train", "validation", "test"}:
        raise ValueError(f"{path}: cần đủ ba tập không rỗng")
    return result


def compare_splits(reference: Path, candidate: Path) -> dict:
    left, right = read_splits(reference), read_splits(candidate)
    return {
        "match": left == right,
        "missing": sorted(left.keys() - right.keys()),
        "extra": sorted(right.keys() - left.keys()),
        "changed": sorted(k for k in left.keys() & right.keys() if left[k] != right[k]),
    }


def paired_results(left_path: Path, right_path: Path, expected_count: int) -> list[dict]:
    def index(path):
        rows = read_csv(path)
        if len(rows) != expected_count:
            raise ValueError(f"{path}: cần đủ {expected_count} ảnh, có {len(rows)}")
        result = {}
        for row in rows:
            name = row["name"]
            if not name or name in result:
                raise ValueError(f"{path}: ID rỗng hoặc trùng")
            for key in ("dice", "iou"):
                value = float(row[key])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"{path}: {key} không hợp lệ")
            result[name] = row
        return result

    left, right = index(left_path), index(right_path)
    if left.keys() != right.keys():
        raise ValueError("Hai CSV không cùng ID test; không được bỏ qua các ảnh lệch")
    rows = []
    for name in sorted(left):
        a, b = left[name], right[name]
        if "gt_pixels" in a and "gt_pixels" in b and int(a["gt_pixels"]) != int(b["gt_pixels"]):
            raise ValueError(f"Diện tích mask thật khác nhau ở {name}")
        rows.append(dict(
            name=name, dice_unet=float(a["dice"]), dice_resnet34=float(b["dice"]),
            delta_dice=float(b["dice"]) - float(a["dice"]),
            iou_unet=float(a["iou"]), iou_resnet34=float(b["iou"]),
            delta_iou=float(b["iou"]) - float(a["iou"]),
        ))
    return rows


def summary_rows(data: dict) -> list[dict]:
    configs = [data["runs"][k]["run_config"] for k in ("unet_kvasir", "unet_resnet34_kvasir")]
    if [c["model"] for c in configs] != ["unet", "unet_resnet34"]:
        raise ValueError("Hai cấu hình phải là unet và unet_resnet34")
    for key in ("dataset", "image_size", "threshold", "loss", "mask_resize"):
        if configs[0][key] != configs[1][key]:
            raise ValueError(f"Hai run khác giao thức ở {key}")
    result = []
    for key in ("unet_kvasir", "unet_resnet34_kvasir"):
        run = data["runs"][key]
        config, test, scores = (run[k] for k in ("run_config", "test_metrics", "test_analysis_summary"))
        if scores["test_images"] != test["test_samples"]:
            raise ValueError(f"{key}: số ảnh test không khớp")
        for metric in ("dice", "iou"):
            if abs(scores[f"global_{metric}"] - test[f"test_{metric}"]) > 1e-6:
                raise ValueError(f"{key}: metric gộp không khớp")
        result.append(dict(
            model=config["model"], parameters=config["parameters"],
            normalization=config["normalization"], pretrained=config["pretrained"],
            best_epoch=test["best_epoch"], epochs_completed=config["epochs_completed"],
            best_validation_dice=test["best_validation_dice"], test_loss=test["test_loss"],
            global_dice=scores["global_dice"], global_iou=scores["global_iou"],
            mean_image_dice=scores["mean_image_dice"], mean_image_iou=scores["mean_image_iou"],
            median_image_dice=scores["median_image_dice"], test_images=scores["test_images"],
            train_seconds_logged=config["train_seconds_logged"],
        ))
    if result[0]["test_images"] != result[1]["test_images"]:
        raise ValueError("Hai run khác số ảnh test")
    return result


def verify_originals(data: dict, directories: tuple[Path, Path]) -> list[dict]:
    """Chỉ dùng CSV đầy đủ do người dùng tải từ Kaggle, không dùng preview."""
    a, b = directories
    split_paths = [p / "split_manifest.csv" for p in directories]
    if not compare_splits(*split_paths)["match"]:
        raise ValueError("Hai run khác ID train/validation/test")
    mapping = read_splits(split_paths[0])
    counts = {label: sum(v == label for v in mapping.values()) for label in ("train", "validation", "test")}
    if counts != {"train": 800, "validation": 100, "test": 100}:
        raise ValueError("Split Kvasir-SEG cần 800/100/100 ảnh")
    rows = paired_results(a / "test_per_image.csv", b / "test_per_image.csv", counts["test"])
    if {r["name"] for r in rows} != {name for name, label in mapping.items() if label == "test"}:
        raise ValueError("ID kết quả từng ảnh không trùng ID test trong split manifest")
    for key, suffix in (("unet_kvasir", "unet"), ("unet_resnet34_kvasir", "resnet34")):
        scores = data["runs"][key]["test_analysis_summary"]
        for metric in ("dice", "iou"):
            mean = sum(r[f"{metric}_{suffix}"] for r in rows) / len(rows)
            if abs(mean - scores[f"mean_image_{metric}"]) > 1e-6:
                raise ValueError(f"CSV {suffix} không khớp snapshot {metric}; kiểm tra checkpoint/threshold")
    return rows


def plot_comparison(rows: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    keys = ["global_dice", "global_iou", "mean_image_dice", "mean_image_iou"]
    labels = ["Global Dice", "Global IoU", "Mean image Dice", "Mean image IoU"]
    figure, axis = plt.subplots(figsize=(10, 4.5))
    for offset, row, label in ((-0.18, rows[0], "U-Net scratch"), (0.18, rows[1], "U-Net ResNet34 pretrained")):
        bars = axis.bar([i + offset for i in range(4)], [row[k] for k in keys], width=0.36, label=label)
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    axis.set_xticks(range(4), labels)
    axis.set_ylim(0, 1.1)
    axis.set_ylabel("Score")
    axis.legend(loc="upper center", ncol=2)
    figure.tight_layout()
    figure.savefig(out / "comparison.png", dpi=160)
    plt.close(figure)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=RESULTS / "experiments.json")
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--unet-dir", type=Path, help="thư mục kết quả gốc tải từ Kaggle")
    parser.add_argument("--resnet34-dir", type=Path, help="thư mục kết quả gốc tải từ Kaggle")
    args = parser.parse_args()
    if bool(args.unet_dir) != bool(args.resnet34_dir):
        parser.error("Cần truyền cả --unet-dir và --resnet34-dir")
    data = json.loads(args.data.read_text(encoding="utf-8"))
    rows = summary_rows(data)
    paired = verify_originals(data, (args.unet_dir, args.resnet34_dir)) if args.unet_dir else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "comparison.csv", rows)
    if args.plot:
        plot_comparison(rows, args.output_dir)
    if paired is not None:
        write_csv(args.output_dir / "paired_image_comparison.csv", paired)
        print("PASS: split gốc và đủ 100 ảnh test khớp; đã xuất paired_image_comparison.csv")
        print("Số ảnh Dice tăng:", sum(r["delta_dice"] > 1e-12 for r in paired))
        print("Số ảnh Dice giảm:", sum(r["delta_dice"] < -1e-12 for r in paired))
    else:
        print("Đã tạo bảng tổng hợp; chưa đối chiếu đủ CSV/split gốc từ Kaggle.")
    print("Kết quả:", args.output_dir)


if __name__ == "__main__":
    main()
