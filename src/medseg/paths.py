"""Đường dẫn gốc của repo. Mọi đường dẫn khác tính tương đối từ ROOT."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
REPORT = ROOT / "report"
OUTPUTS = ROOT / "outputs"


def from_root(path: str | Path) -> Path:
    """Đổi đường dẫn tương đối (theo gốc repo) thành tuyệt đối."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def to_root(path: str | Path) -> str:
    """Đổi đường dẫn tuyệt đối thành chuỗi tương đối theo gốc repo."""
    return Path(path).resolve().relative_to(ROOT).as_posix()
