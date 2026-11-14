"""Vòng huấn luyện theo iteration, tái tạo IterBasedRunner + các hook của mmcv mà SegFormer dùng.

- Lặp DataLoader vô hạn (hết một vòng thì bắt đầu vòng mới), tổng `max_iters` iteration.
- LR cập nhật mỗi iteration (poly + warmup tuyến tính).
- Log mỗi `log_interval`, lưu checkpoint mỗi `checkpoint_interval`, đánh giá mIoU trên val
  mỗi `eval_interval` iteration.
- Bổ sung so với bản gốc (không đổi siêu tham số): lưu thêm `best.pth` theo mIoU trên val,
  log dạng JSON lines, và tiếp tục huấn luyện từ checkpoint (`resume`).
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from medseg.training.evaluate import evaluate
from medseg.training.losses import segformer_ce_loss
from medseg.training.optim import build_optimizer, build_scheduler


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def infinite_loader(loader):
    """Lặp DataLoader không dừng, như IterLoader của mmcv."""
    while True:
        yield from loader


def save_checkpoint(path: Path, model, optimizer, scheduler, it: int, meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "iter": it,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "meta": meta or {},
    }, path)


def load_checkpoint(path: str | Path, model, optimizer=None, scheduler=None, map_location="cpu") -> int:
    """Nạp checkpoint do `save_checkpoint` tạo; trả về số iteration đã chạy."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt.get("iter", 0))


def train(model: torch.nn.Module, train_loader, val_loader, cfg: dict, work_dir: str | Path,
          device: torch.device, resume: str | Path | None = None, log=print) -> dict:
    """Huấn luyện theo cấu hình `load_model_config(...)`. Trả về metric tốt nhất trên val."""
    common, runner, head = cfg["common"], cfg["runner"], cfg["decode_head"]
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda" and runner.get("cudnn_benchmark") and not common.get("deterministic"):
        torch.backends.cudnn.benchmark = True

    model.to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    start = load_checkpoint(resume, model, optimizer, scheduler, map_location=device) if resume else 0

    max_iters = runner["max_iters"]
    loss_cfg = cfg["loss"]
    log_path = work_dir / "log.jsonl"
    key = runner.get("eval_metric", "mIoU")
    best = {"metric": key, "score": float("-inf"), "iter": None}
    data = infinite_loader(train_loader)
    window_loss, t0 = [], time.time()

    model.train()
    for it in range(start, max_iters):
        batch = next(data)
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        logits = model(image)
        loss = segformer_ce_loss(logits, mask, align_corners=head["align_corners"],
                                 ignore_index=common["ignore_index"], loss_weight=loss_cfg["loss_weight"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        lr = optimizer.param_groups[0]["lr"]  # mmcv log lr của nhóm đầu (backbone)
        scheduler.step()
        window_loss.append(loss.item())
        done = it + 1

        if done % runner["log_interval"] == 0 or done == max_iters:
            rec = {"iter": done, "lr": lr, "loss": float(np.mean(window_loss)),
                   "sec_per_iter": (time.time() - t0) / len(window_loss)}
            log(f"iter {done}/{max_iters}  lr {lr:.3e}  loss {rec['loss']:.4f}  {rec['sec_per_iter']:.3f}s/it")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            window_loss, t0 = [], time.time()

        if done % runner["checkpoint_interval"] == 0 or done == max_iters:
            save_checkpoint(work_dir / f"iter_{done}.pth", model, optimizer, scheduler, done)
            save_checkpoint(work_dir / "latest.pth", model, optimizer, scheduler, done)

        if val_loader is not None and (done % runner["eval_interval"] == 0 or done == max_iters):
            metrics = evaluate(model, val_loader, device, common["num_classes"], common["ignore_index"])
            rec = {"iter": done, "split": "val", **metrics}
            log("  val " + "  ".join(f"{k} {v:.4f}" for k, v in metrics.items()))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            if metrics[key] > best["score"]:
                best = {"metric": key, "score": metrics[key], "iter": done, **metrics}
                save_checkpoint(work_dir / "best.pth", model, optimizer, scheduler, done, meta=best)
            model.train()

    return best
