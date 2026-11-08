"""Tạo file chia tập cố định trong data/splits/<dataset>/{train,val,test,all}.txt.

Giao thức FCBFormer (Sanderson & Matuszewski, MIUA 2022):
sắp xếp file ảnh theo tên, rồi train_test_split(random_state=42) -> 80/10/10.
all.txt chứa toàn bộ mẫu, dùng cho test chuyển miền (vd train Kvasir, test CVC).

Chạy từ gốc repo:
    python scripts/make_splits.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medseg.config import load_dataset_specs  # noqa: E402
from medseg.data.io import list_images  # noqa: E402
from medseg.data.splits import fcbformer_split, write_split  # noqa: E402
from medseg.paths import from_root, to_root  # noqa: E402
from medseg.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger()


def main() -> None:
    specs = load_dataset_specs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(specs), choices=list(specs))
    args = parser.parse_args()
    setup_logging("make_splits", args)

    for name in args.datasets:
        spec = specs[name]
        if spec["split"]["protocol"] != "fcbformer":
            raise NotImplementedError(spec["split"]["protocol"])
        names = [p.stem for p in list_images(from_root(spec["raw_images"]))]
        parts = fcbformer_split(len(names), seed=spec["split"]["seed"])
        log.info(f"[{name}] {len(names)} mẫu")
        for split, ids in parts.items():
            path = write_split(name, split, sorted(names[i] for i in ids))
            log.info(f"  {split:5s} {len(ids):4d} -> {to_root(path)}")
        path = write_split(name, "all", names)
        log.info(f"  all   {len(names):4d} -> {to_root(path)}")


if __name__ == "__main__":
    main()
