"""Metric phân đoạn nhị phân, tính trên từng ảnh."""

import numpy as np
from scipy import ndimage


def _boundary(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, border_value=0)


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """Khoảng cách Hausdorff phân vị 95 (pixel). NaN nếu chỉ một trong hai mặt nạ rỗng."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return float("nan")
    bp, bg = _boundary(pred), _boundary(gt)
    d_to_gt = ndimage.distance_transform_edt(~bg)[bp]
    d_to_pred = ndimage.distance_transform_edt(~bp)[bg]
    return float(np.percentile(np.concatenate([d_to_gt, d_to_pred]), 95))


def segmentation_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Dice, IoU, precision, recall, HD95. Hai mặt nạ cùng rỗng được tính là khớp hoàn toàn."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    empty = tp + fp + fn == 0
    return {
        "dice": 1.0 if empty else 2 * tp / (2 * tp + fp + fn),
        "iou": 1.0 if empty else tp / (tp + fp + fn),
        "precision": 1.0 if empty else (tp / (tp + fp) if tp + fp else 0.0),
        "recall": 1.0 if empty else (tp / (tp + fn) if tp + fn else 0.0),
        "hd95": hd95(pred, gt),
        "empty_pred": bool(not pred.any()),
    }
