"""Tải checkpoint SAM vào checkpoints/.

    python scripts/download_sam_weights.py --models vit_b vit_h
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medseg.sam.loader import checkpoint_path, sam_config  # noqa: E402
from medseg.utils.download import download, sha256sum  # noqa: E402
from medseg.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger()


def main() -> None:
    models = sam_config()["models"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["vit_b"], choices=list(models))
    args = parser.parse_args()
    setup_logging("download_sam_weights", args)

    for name in args.models:
        spec, path = models[name], checkpoint_path(name)
        if path.exists() and path.stat().st_size == spec["bytes"]:
            log.info(f"[{name}] đã có {path.name}")
        else:
            log.info(f"[{name}] tải {spec['url']}")
            download(spec["url"], path)
        if path.stat().st_size != spec["bytes"]:
            raise ValueError(f"{path.name}: sai dung lượng")
        if spec.get("sha256") and sha256sum(path) != spec["sha256"]:
            raise ValueError(f"{path.name}: SHA-256 không khớp")
        log.info(f"  OK ({spec['bytes'] / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
