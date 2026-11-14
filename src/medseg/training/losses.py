"""Hàm loss của SegFormer gốc: CrossEntropyLoss (use_sigmoid=False) trên logits đã upsample.

Tái tạo BaseDecodeHead.losses + mmseg cross_entropy:
logits (B, C, h, w) được resize bilinear về kích thước nhãn, F.cross_entropy(reduction='none',
ignore_index=255) rồi lấy trung bình trên MỌI pixel (pixel bị bỏ qua vẫn tính vào mẫu số).
"""

import torch
import torch.nn.functional as F


def mask_to_label(mask: torch.Tensor) -> torch.Tensor:
    """Mặt nạ của DataLoader chung (B, 1, H, W) giá trị {0, 1} -> nhãn lớp (B, H, W) kiểu long."""
    if mask.ndim == 4:
        mask = mask[:, 0]
    return mask.round().long()


def segformer_ce_loss(logits: torch.Tensor, mask: torch.Tensor, align_corners: bool = False,
                      ignore_index: int = 255, loss_weight: float = 1.0) -> torch.Tensor:
    label = mask_to_label(mask)
    if logits.shape[2:] != label.shape[1:]:
        logits = F.interpolate(logits, size=label.shape[1:], mode="bilinear", align_corners=align_corners)
    loss = F.cross_entropy(logits, label, reduction="none", ignore_index=ignore_index)
    return loss_weight * loss.mean()
