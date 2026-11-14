"""Đóng gói code + trọng số pretrained thành Kaggle Dataset trong Kaggle/medseg-code/.

Dữ liệu ảnh KHÔNG đóng gói: notebook Kaggle lấy Kvasir-SEG và CVC-ClinicDB từ các dataset
có sẵn trên Kaggle, rồi chạy prepare_data.py như trên máy.

Chạy từ gốc repo (cần pretrained/mit_b2.pth, tạo bằng scripts/convert_hf_mit.py):
    python scripts/build_kaggle.py --kaggle-user <tên_kaggle>
Kết quả:
    Kaggle/medseg-code/       -> upload bằng `kaggle datasets create -p Kaggle/medseg-code --dir-mode zip`
    Kaggle/medseg-code.zip    -> hoặc kéo thả file zip này lên kaggle.com/datasets (giữ cấu trúc thư mục)
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medseg.paths import ROOT, to_root  # noqa: E402

# (nguồn, đích) tính từ gốc repo; chỉ những gì notebook cần để chuẩn bị dữ liệu, train và đánh giá
INCLUDE = [
    "src/medseg",
    "scripts",
    "configs",
    "data/splits",
    "data/metadata",
    "data/README.md",
    "pyproject.toml",
    "requirements.txt",
    "README.md",
    "pretrained/mit_b2.pth",
]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kaggle-user", default="YOUR_KAGGLE_USERNAME")
    p.add_argument("--slug", default="medseg-code")
    p.add_argument("--out", default="Kaggle/medseg-code")
    p.add_argument("--no-zip", action="store_true")
    args = p.parse_args()

    out = ROOT / args.out
    if not (ROOT / "pretrained/mit_b2.pth").exists():
        raise FileNotFoundError("Thiếu pretrained/mit_b2.pth. Chạy: python scripts/convert_hf_mit.py")
    if out.exists():
        shutil.rmtree(out)
    for rel in INCLUDE:
        src, dst = ROOT / rel, out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=IGNORE)
        else:
            shutil.copy2(src, dst)
    for d in ("data/raw", "data/processed", "outputs"):
        (out / d).mkdir(parents=True, exist_ok=True)
        (out / d / ".gitkeep").touch()  # Kaggle bỏ thư mục rỗng

    meta = {"title": args.slug, "id": f"{args.kaggle_user}/{args.slug}", "licenses": [{"name": "other"}]}
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"đã tạo {to_root(out)} ({size:.1f} MB)")
    if not args.no_zip:
        # gốc zip là nội dung thư mục, nên trên Kaggle code nằm ngay dưới /kaggle/input/<slug>/
        archive = shutil.make_archive(str(out), "zip", root_dir=out)
        print(f"đã tạo {to_root(archive)} ({Path(archive).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
