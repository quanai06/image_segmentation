"""Đọc ảnh và mặt nạ thô (JPG của Kvasir-SEG, TIFF của CVC-ClinicDB)."""

from pathlib import Path

import cv2
import numpy as np
import tifffile

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
TIFF_EXTS = {".tif", ".tiff"}


def _imread(path: Path) -> np.ndarray:
    """Đọc thô, trả về RGB (H, W, C) hoặc xám (H, W).

    TIFF của CVC-ClinicDB khai báo photometric=MINISBLACK nhưng có 3 kênh,
    OpenCV chỉ đọc 1 kênh -> dùng tifffile (giống skimage trong FCBFormer).
    """
    if path.suffix.lower() in TIFF_EXTS:
        return tifffile.imread(path)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Không đọc được: {path}")
    if img.ndim == 3:
        code = cv2.COLOR_BGRA2RGBA if img.shape[2] == 4 else cv2.COLOR_BGR2RGB
        img = cv2.cvtColor(img, code)
    return img


def list_images(folder: str | Path) -> list[Path]:
    """Liệt kê file ảnh, sắp xếp theo tên (giống sorted(glob) của FCBFormer)."""
    files = [p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=lambda p: p.name)


def read_image(path: str | Path) -> np.ndarray:
    """Trả về ảnh RGB uint8 (H, W, 3)."""
    img = _imread(Path(path))
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        return np.repeat(img[..., None], 3, axis=2)
    return np.ascontiguousarray(img[..., :3])


def read_mask(path: str | Path, threshold: int = 127) -> np.ndarray:
    """Trả về mặt nạ nhị phân uint8 {0, 1} (H, W)."""
    m = _imread(Path(path))
    if m.ndim == 3:
        m = m[..., :3].max(axis=2)  # mặt nạ lưu 3 kênh: lấy kênh lớn nhất
    if m.dtype != np.uint8:
        m = cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return (m > threshold).astype(np.uint8)


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Resize mặt nạ 0/1 về size x size: INTER_AREA rồi ngưỡng 0.5."""
    m = cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    return (m >= 0.5).astype(np.uint8)


def resize_pair(image: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Resize ảnh (INTER_AREA) và mặt nạ về size x size."""
    img = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return img, resize_mask(mask, size)
