"""Optimizer và lịch learning rate, tái tạo hành vi mmcv 1.x mà SegFormer dùng.

- `build_param_groups` <- DefaultOptimizerConstructor với paramwise_cfg.custom_keys:
  khóa dài hơn được ưu tiên, cùng độ dài thì theo thứ tự chữ cái, chỉ áp dụng khóa khớp đầu tiên
  (so khớp chuỗi con trong tên tham số).
- `poly_warmup_lr` <- PolyLrUpdaterHook + warmup 'linear' (by_epoch=False, cập nhật mỗi iteration).
"""

import torch
from torch.optim.lr_scheduler import LambdaLR


def build_param_groups(model: torch.nn.Module, lr: float, weight_decay: float,
                       custom_keys: dict | None = None) -> list[dict]:
    """Mỗi tham số một nhóm (như mmcv), lr/weight_decay nhân theo khóa khớp đầu tiên."""
    custom_keys = custom_keys or {}
    sorted_keys = sorted(sorted(custom_keys), key=len, reverse=True)
    groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group = {"params": [param], "name": name, "lr": lr, "weight_decay": weight_decay}
        for key in sorted_keys:
            if key in name:
                group["lr"] = lr * custom_keys[key].get("lr_mult", 1.0)
                group["weight_decay"] = weight_decay * custom_keys[key].get("decay_mult", 1.0)
                group["custom_key"] = key
                break
        groups.append(group)
    return groups


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """AdamW với paramwise_cfg, từ mục `optimizer` trong config.json."""
    opt = cfg["optimizer"]
    if opt["type"] != "AdamW":
        raise ValueError(f"Chỉ hỗ trợ AdamW, nhận {opt['type']}")
    groups = build_param_groups(model, opt["lr"], opt["weight_decay"], opt.get("custom_keys"))
    return torch.optim.AdamW(groups, lr=opt["lr"], betas=tuple(opt["betas"]), weight_decay=opt["weight_decay"])


def poly_warmup_lr(base_lr: float, it: int, max_iters: int, power: float = 1.0, min_lr: float = 0.0,
                   warmup_iters: int = 0, warmup_ratio: float = 0.1) -> float:
    """Learning rate tại iteration `it` (đếm từ 0).

    poly:   lr = (base_lr - min_lr) * (1 - it / max_iters) ** power + min_lr
    warmup: khi it < warmup_iters, nhân thêm 1 - (1 - it / warmup_iters) * (1 - warmup_ratio)
    """
    coeff = (1 - it / max_iters) ** power
    lr = (base_lr - min_lr) * coeff + min_lr
    if warmup_iters and it < warmup_iters:
        k = (1 - it / warmup_iters) * (1 - warmup_ratio)
        lr = lr * (1 - k)
    return lr


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict) -> LambdaLR:
    """LambdaLR theo iteration; gọi scheduler.step() sau mỗi optimizer.step()."""
    sch = cfg["lr_schedule"]
    if sch["policy"] != "poly" or sch.get("by_epoch", False):
        raise ValueError("Chỉ hỗ trợ policy 'poly' cập nhật theo iteration")
    if sch.get("warmup") not in (None, "linear"):
        raise ValueError(f"Chỉ hỗ trợ warmup 'linear', nhận {sch['warmup']}")
    max_iters = cfg["runner"]["max_iters"]
    warmup_iters = sch["warmup_iters"] if sch.get("warmup") else 0

    def make(base_lr: float):
        def factor(it: int) -> float:
            it = min(it, max_iters)
            return poly_warmup_lr(base_lr, it, max_iters, sch["power"], sch["min_lr"],
                                  warmup_iters, sch["warmup_ratio"]) / base_lr
        return factor

    return LambdaLR(optimizer, [make(g["lr"]) for g in optimizer.param_groups])
