"""Tải và giải nén dữ liệu thô vào data/raw/, có kiểm tra SHA-256.

Chạy từ gốc repo:
    python scripts/download_data.py                 # tất cả bộ trong configs/datasets.yaml
    python scripts/download_data.py --datasets kvasir_seg
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medseg.config import load_dataset_specs  # noqa: E402
from medseg.paths import from_root  # noqa: E402


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.rename(dest)


def extract(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(out_dir)
        return
    # .rar: thử lần lượt bsdtar (macOS), unrar (Colab/Linux), 7z
    for tool, cmd in [
        ("bsdtar", ["bsdtar", "-xf", str(archive), "-C", str(out_dir)]),
        ("unrar", ["unrar", "x", "-o+", str(archive), str(out_dir) + "/"]),
        ("7z", ["7z", "x", "-y", f"-o{out_dir}", str(archive)]),
    ]:
        if shutil.which(tool):
            subprocess.run(cmd, check=True)
            return
    raise RuntimeError("Cần bsdtar, unrar hoặc 7z để giải nén .rar (vd: apt install unrar)")


def main() -> None:
    specs = load_dataset_specs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(specs), choices=list(specs))
    parser.add_argument("--force", action="store_true", help="tải và giải nén lại từ đầu")
    args = parser.parse_args()

    for name in args.datasets:
        spec = specs[name]
        archive = from_root(spec["archive"])
        print(f"[{name}]")
        if args.force or not archive.exists():
            print(f"  tải {spec['url']}")
            download(spec["url"], archive)
        digest = sha256sum(archive)
        if digest != spec["sha256"]:
            raise ValueError(f"  SHA-256 không khớp cho {archive.name}: {digest}")
        print("  SHA-256 khớp")

        img_dir, msk_dir = from_root(spec["raw_images"]), from_root(spec["raw_masks"])
        if args.force or not (img_dir.exists() and msk_dir.exists()):
            print(f"  giải nén vào {spec['extract_to']}")
            extract(archive, from_root(spec["extract_to"]))
        n_img, n_msk = len(list(img_dir.iterdir())), len(list(msk_dir.iterdir()))
        print(f"  {n_img} ảnh, {n_msk} mặt nạ")


if __name__ == "__main__":
    main()
