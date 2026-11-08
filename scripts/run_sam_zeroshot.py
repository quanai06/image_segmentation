"""SAM zero-shot với gợi ý sinh từ nhãn, chấm trên lưới eval_size (256) như các mô hình khác.

SAM chạy trên ảnh gốc; mặt nạ dự đoán được thu về 256 x 256 rồi so với nhãn trong data/processed.
Ghi từng ảnh vào CSV nên chạy lại sẽ tiếp tục từ chỗ dừng (dùng --overwrite để chạy lại từ đầu).

    python scripts/run_sam_zeroshot.py --model vit_b --limit 5
    python scripts/run_sam_zeroshot.py --model vit_h --split all --save-masks

Kết quả: outputs/sam/<model>/<dataset>_<split>/per_image.csv và masks/<protocol>/<tên>.png
"""

import argparse
import sys
import time
import zlib
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medseg.config import load_data_config, load_dataset_specs  # noqa: E402
from medseg.data.dataset import processed_root  # noqa: E402
from medseg.data.io import read_image, read_mask, resize_mask  # noqa: E402
from medseg.data.splits import load_split  # noqa: E402
from medseg.evaluation import segmentation_metrics  # noqa: E402
from medseg.paths import DATA, OUTPUTS, from_root  # noqa: E402
from medseg.sam import PROTOCOLS, SamZeroShot, load_sam, sam_config  # noqa: E402
from medseg.sam.prompts import regions  # noqa: E402
from medseg.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger()


def size_bin(fg: float, bins: list[float]) -> str:
    return "small" if fg < bins[0] else "medium" if fg < bins[1] else "large"


def run_dataset(runner: SamZeroShot, args, dataset: str, cfg: dict, threshold: int) -> None:
    size = cfg["eval_size"]
    manifest = pd.read_csv(DATA / "metadata" / f"{dataset}_manifest.csv", dtype={"name": str}).set_index("name")
    names = load_split(dataset, args.split)[: args.limit]
    out_dir = OUTPUTS / "sam" / args.model / f"{dataset}_{args.split}"
    csv_path = out_dir / "per_image.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite and csv_path.exists():
        csv_path.unlink()
    done = set(pd.read_csv(csv_path, dtype={"name": str})["name"]) if csv_path.exists() else set()
    todo = [n for n in names if n not in done]
    log.info(f"[{dataset}/{args.split}] {len(todo)} ảnh cần chạy ({len(done)} đã có) -> {csv_path.relative_to(OUTPUTS.parent)}")

    gt_dir = processed_root(dataset, size) / "masks"
    for name in tqdm(todo, desc=dataset):
        meta = manifest.loc[name]
        image = read_image(from_root(meta.raw_image))
        gt_raw = read_mask(from_root(meta.raw_mask), threshold)
        gt = cv2.imread(str(gt_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE) > 127

        start = time.perf_counter()
        preds = runner.predict(image, gt_raw, args.protocols, seed=cfg["seed"] + zlib.crc32(name.encode()))
        seconds = time.perf_counter() - start

        rows = []
        for protocol, mask in preds.items():
            pred = resize_mask(mask, size).astype(bool)
            rows.append({
                "dataset": dataset, "split": args.split, "model": args.model, "name": name, "protocol": protocol,
                **segmentation_metrics(pred, gt),
                "fg_ratio": meta.fg_ratio_orig,
                "size_bin": size_bin(meta.fg_ratio_orig, cfg["size_bins"]),
                "n_regions": len(regions(gt_raw, cfg["prompts"]["min_region_area"])),
                "sec_image": round(seconds, 3),
            })
            if args.save_masks:
                path = out_dir / "masks" / protocol / f"{name}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), pred.astype("uint8") * 255)
        pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        log.debug(f"{name} {seconds:.2f}s " + " ".join(f"{r['protocol']}={r['dice']:.3f}" for r in rows))

    results = pd.read_csv(csv_path, dtype={"name": str})
    results = results[results["name"].isin(names) & results["protocol"].isin(args.protocols)]
    summary = results.groupby("protocol")["dice"].agg(["mean", "std", "size"]).round(3)
    log.info(f"[{dataset}/{args.split}] Dice theo kiểu gợi ý ({len(set(results['name']))} ảnh):\n{summary.to_string()}")


def main() -> None:
    cfg = sam_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vit_b", choices=list(cfg["models"]))
    parser.add_argument("--datasets", nargs="+", default=list(load_dataset_specs()))
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=list(PROTOCOLS))
    parser.add_argument("--limit", type=int, default=None, help="chỉ chạy N ảnh đầu (gỡ lỗi)")
    parser.add_argument("--device", default=None, help="cuda | cpu | mps")
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    setup_logging("run_sam_zeroshot", args)

    runner = SamZeroShot(load_sam(args.model, args.device), cfg)
    log.info(f"SAM {args.model} trên {runner.sam.device}, gợi ý: {', '.join(args.protocols)}")
    log.debug(f"Cấu hình SAM: {cfg['prompts']} | automatic: {cfg['automatic']}")
    threshold = load_data_config()["mask_threshold"]
    for dataset in args.datasets:
        run_dataset(runner, args, dataset, cfg, threshold)


if __name__ == "__main__":
    main()
