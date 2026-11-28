"""Sinh gợi ý cho SAM từ một mặt nạ nhị phân (nhãn hoặc dự đoán của mô hình khác).

Tọa độ theo quy ước của SAM: điểm (x, y), hộp (x0, y0, x1, y1), đơn vị pixel.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Region:
    mask: np.ndarray  # bool (H, W)
    box: np.ndarray   # (4,) x0, y0, x1, y1


def regions(mask: np.ndarray, min_area: float = 0.001) -> list[Region]:
    """Các vùng liên thông có diện tích >= min_area (tỷ lệ ảnh); nếu không vùng nào đạt, giữ vùng lớn nhất."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return []
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i in np.flatnonzero(areas >= min_area * mask.size)]
    if not keep:
        keep = [int(np.argmax(areas)) + 1]
    out = []
    for i in keep:
        x, y, w, h = stats[i, :4]
        out.append(Region(labels == i, np.array([x, y, x + w - 1, y + h - 1], dtype=np.float32)))
    return out


def center_point(mask: np.ndarray) -> np.ndarray:
    """Điểm xa đường biên nhất bên trong mặt nạ (cực đại của distance transform)."""
    padded = np.pad(mask.astype(np.uint8), 1)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return np.array([[x, y]], dtype=np.float32)


def sample_points(candidates: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    ys, xs = np.nonzero(candidates)
    if len(xs) == 0 or n <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    idx = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
    return np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)


def expand_box(box: np.ndarray, margin: float, shape: tuple[int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    dx, dy = margin * (x1 - x0 + 1), margin * (y1 - y0 + 1)
    h, w = shape
    return np.array([max(0, x0 - dx), max(0, y0 - dy), min(w - 1, x1 + dx), min(h - 1, y1 + dy)], dtype=np.float32)


def point_set(region: Region, full_mask: np.ndarray, n_pos: int, n_neg: int, margin: float,
              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Điểm tâm + (n_pos - 1) điểm dương ngẫu nhiên; n_neg điểm âm ngoài mọi tổn thương, gần vùng này."""
    center = center_point(region.mask)
    inside = region.mask.copy()
    inside[int(center[0, 1]), int(center[0, 0])] = False
    pos = np.concatenate([center, sample_points(inside, n_pos - 1, rng)])

    x0, y0, x1, y1 = expand_box(region.box, margin, full_mask.shape).astype(int)
    near = np.zeros_like(full_mask, dtype=bool)
    near[y0:y1 + 1, x0:x1 + 1] = True
    outside = ~full_mask.astype(bool)
    neg = sample_points(near & outside, n_neg, rng)
    if len(neg) < n_neg:
        neg = np.concatenate([neg, sample_points(outside & ~near, n_neg - len(neg), rng)])

    coords = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.int64)
    return coords, labels


def jitter_box(box: np.ndarray, shape: tuple[int, int], rng: np.random.Generator,
               max_px: float | None = None, max_rel: float | None = None) -> np.ndarray:
    """Lệch ngẫu nhiên mỗi cạnh hộp trong [-d, d]; d cố định (pixel) hoặc theo tỷ lệ kích thước hộp."""
    x0, y0, x1, y1 = box
    if max_px is not None:
        dx = dy = float(max_px)
    elif max_rel is not None:
        dx, dy = max_rel * (x1 - x0 + 1), max_rel * (y1 - y0 + 1)
    else:
        raise ValueError("cần max_px hoặc max_rel")
    noise = rng.uniform(-1, 1, 4) * np.array([dx, dy, dx, dy])
    h, w = shape
    nx0, ny0, nx1, ny1 = box + noise
    nx0, nx1 = np.clip(sorted([nx0, nx1]), 0, w - 1)
    ny0, ny1 = np.clip(sorted([ny0, ny1]), 0, h - 1)
    return np.array([nx0, ny0, max(nx1, nx0 + 1), max(ny1, ny0 + 1)], dtype=np.float32)
