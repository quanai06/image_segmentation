"""Chuẩn hóa dữ liệu thô -> data/processed/<dataset>/<size>/{images,masks}.

- Ảnh: RGB, resize INTER_AREA, lưu PNG.
- Mặt nạ: nhị phân hóa (> ngưỡng), resize INTER_AREA rồi ngưỡng 0.5, lưu PNG {0, 255}.
- Ghi manifest kiểm tra nguồn gốc vào data/metadata/<dataset>_manifest.csv.

Chạy từ gốc repo:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --datasets kvasir_seg --size 352
"""

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medseg.config import load_data_config, load_dataset_specs  # noqa: E402
from medseg.data.dataset import processed_root  # noqa: E402
from medseg.data.io import list_images, read_image, read_mask, resize_pair  # noqa: E402
from medseg.paths import DATA, from_root, to_root  # noqa: E402
from medseg.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(name: str, spec: dict, size: int, threshold: int) -> None:
    images = list_images(from_root(spec["raw_images"]))
    masks = {p.stem: p for p in list_images(from_root(spec["raw_masks"]))}
    missing = [p.stem for p in images if p.stem not in masks]
    if missing:
        raise ValueError(f"{name}: {len(missing)} ảnh không có mặt nạ, vd {missing[:3]}")

    out = processed_root(name, size)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    rows = []
    for img_path in tqdm(images, desc=name):
        msk_path = masks[img_path.stem]
        img, msk = read_image(img_path), read_mask(msk_path, threshold)
        if img.shape[:2] != msk.shape:
            raise ValueError(f"{name}/{img_path.name}: kích thước ảnh {img.shape[:2]} khác mặt nạ {msk.shape}")
        img_r, msk_r = resize_pair(img, msk, size)
        cv2.imwrite(str(out / "images" / f"{img_path.stem}.png"), cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / "masks" / f"{img_path.stem}.png"), msk_r * 255)
        rows.append({
            "name": img_path.stem,
            "raw_image": to_root(img_path),
            "raw_mask": to_root(msk_path),
            "orig_height": img.shape[0],
            "orig_width": img.shape[1],
            "fg_ratio_orig": round(float(msk.mean()), 6),
            f"fg_ratio_{size}": round(float(msk_r.mean()), 6),
            "n_components": cv2.connectedComponents(msk)[0] - 1,
            "sha256_image": file_sha256(img_path),
            "sha256_mask": file_sha256(msk_path),
        })

    meta = DATA / "metadata" / f"{name}_manifest.csv"
    meta.parent.mkdir(parents=True, exist_ok=True)
    with open(meta, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fg = np.array([r["fg_ratio_orig"] for r in rows])
    log.info(f"  {len(rows)} mẫu -> {to_root(out)}")
    log.info(f"  tỷ lệ tổn thương: min {fg.min():.4f}, trung vị {np.median(fg):.4f}, max {fg.max():.4f}")
    log.info(f"  manifest: {to_root(meta)}")


def main() -> None:
    specs = load_dataset_specs()
    cfg = load_data_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(specs), choices=list(specs))
    parser.add_argument("--size", type=int, default=cfg["image_size"])
    args = parser.parse_args()
    setup_logging("prepare_data", args)
    for name in args.datasets:
        log.info(f"[{name}]")
        prepare(name, specs[name], args.size, cfg["mask_threshold"])


if __name__ == "__main__":
    main()
