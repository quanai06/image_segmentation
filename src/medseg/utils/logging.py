"""Ghi log mỗi lần chạy script vào logs/<tên>_<ngày-giờ>.log (màn hình: INFO, file: DEBUG)."""

import logging
import platform
import subprocess
import sys
from datetime import datetime

from medseg.paths import ROOT

LOGS = ROOT / "logs"
LOGGER_NAME = "medseg"


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + (" (có thay đổi chưa commit)" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "không rõ"


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def setup_logging(name: str, args=None) -> logging.Logger:
    LOGS.mkdir(exist_ok=True)
    path = LOGS / f"{name}_{datetime.now():%Y%m%d-%H%M%S}.log"

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S"))
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))

    logger = get_logger()
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(console)

    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.handlers.clear()
    warnings_logger.addHandler(file_handler)
    sys.excepthook = lambda t, v, tb: logger.critical("Lỗi, dừng chương trình", exc_info=(t, v, tb))

    logger.info(f"Log: {path.relative_to(ROOT)}")
    logger.debug(f"Lệnh: {' '.join(sys.argv)}")
    logger.debug(f"Git: {git_commit()} | Python {platform.python_version()} | {platform.platform()}")
    if args is not None:
        logger.debug(f"Tham số: {vars(args)}")
    return logger
