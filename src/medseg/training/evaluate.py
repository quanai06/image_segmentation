"""Đánh giá theo metric gốc của SegFormer (mIoU của mmseg), kèm Dice/IoU lớp tổn thương.

mmseg cộng dồn diện tích giao/hợp của từng lớp trên toàn bộ tập rồi mới chia (không lấy
trung bình theo ảnh). Bộ metric đầy đủ của nhóm (Dice, IoU, HD95, precision, recall theo ảnh)
nằm ở module metric chung; module này chỉ phục vụ chọn checkpoint giống bản gốc.
"""

import torch

from medseg.training.losses import mask_to_label


def intersect_and_union(pred: torch.Tensor, label: torch.Tensor, num_classes: int,
                        ignore_index: int = 255) -> dict[str, torch.Tensor]:
    """Diện tích giao, hợp, dự đoán, nhãn của từng lớp cho một batch (như mmseg.intersect_and_union)."""
    valid = label != ignore_index
    pred, label = pred[valid], label[valid]
    intersect = pred[pred == label]
    area_intersect = torch.bincount(intersect, minlength=num_classes)[:num_classes]
    area_pred = torch.bincount(pred, minlength=num_classes)[:num_classes]
    area_label = torch.bincount(label, minlength=num_classes)[:num_classes]
    area_union = area_pred + area_label - area_intersect
    return {"intersect": area_intersect, "union": area_union, "pred": area_pred, "label": area_label}


def summarize(totals: dict[str, torch.Tensor]) -> dict[str, float]:
    """mIoU, aAcc, IoU từng lớp và Dice/IoU lớp tổn thương (lớp 1) từ diện tích cộng dồn."""
    inter, union = totals["intersect"].double(), totals["union"].double()
    pred, label = totals["pred"].double(), totals["label"].double()
    iou = inter / union  # lớp không xuất hiện -> nan, bị bỏ qua khi lấy trung bình như np.nanmean
    out = {
        "aAcc": float(inter.sum() / label.sum()),
        "mIoU": float(torch.nanmean(iou)),
    }
    for c, v in enumerate(iou.tolist()):
        out[f"IoU.{c}"] = v
    if len(iou) == 2:
        out["fg_IoU"] = float(iou[1])
        out["fg_Dice"] = float(2 * inter[1] / (pred[1] + label[1]))
    return out


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device, num_classes: int,
             ignore_index: int = 255) -> dict[str, float]:
    """Chạy suy luận 'whole' trên cả loader và trả về metric kiểu mmseg."""
    model.eval()
    totals = None
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        label = mask_to_label(batch["mask"]).to(device)
        pred = model.predict(image)
        areas = intersect_and_union(pred, label, num_classes, ignore_index)
        totals = areas if totals is None else {k: totals[k] + areas[k] for k in totals}
    if totals is None:
        raise ValueError("Loader đánh giá rỗng")
    return summarize({k: v.cpu() for k, v in totals.items()})
