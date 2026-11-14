"""Huấn luyện và đánh giá SegFormer-B2 với siêu tham số gốc (configs/config.json).

Chạy từ gốc repo:
    python scripts/train_segformer.py                          # train Kvasir-SEG, 160k iter như bản gốc
    python scripts/train_segformer.py --dataset cvc_clinicdb
    python scripts/train_segformer.py --resume outputs/segformer_b2/kvasir_seg/latest.pth
    python scripts/train_segformer.py --eval-only outputs/segformer_b2/kvasir_seg/best.pth

Tùy chọn ghi đè (--max-iters, --batch-size, --no-pretrained) chỉ dùng để chạy thử nhanh;
mặc định giữ nguyên cấu hình gốc của SegFormer.

Cần trọng số ImageNet `pretrained/mit_b2.pth` (link trong README của SegFormer).
"""

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medseg.config import load_model_config  # noqa: E402
from medseg.data import build_dataloader  # noqa: E402
from medseg.models import build_segformer, load_mit_pretrained  # noqa: E402
from medseg.paths import from_root, to_root  # noqa: E402
from medseg.training import evaluate, load_checkpoint, resolve_device, set_seed, train  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.json")
    p.add_argument("--model", default="segformer_b2")
    p.add_argument("--dataset", default=None, help="mặc định: common.train_dataset")
    p.add_argument("--work-dir", default=None, help="mặc định: <output_dir>/<model>/<dataset>")
    p.add_argument("--resume", default=None)
    p.add_argument("--eval-only", default=None, metavar="CKPT", help="chỉ đánh giá checkpoint trên test và chuyển miền")
    p.add_argument("--max-iters", type=int, default=None, help="ghi đè runner.max_iters (chạy thử)")
    p.add_argument("--batch-size", type=int, default=None, help="ghi đè runner.batch_size (chạy thử)")
    p.add_argument("--no-pretrained", action="store_true", help="không nạp mit_b2.pth (chạy thử)")
    return p.parse_args()


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Áp các ghi đè từ dòng lệnh lên bản sao cấu hình; in ra để không ai nhầm với cấu hình gốc."""
    cfg = copy.deepcopy(cfg)
    runner = cfg["runner"]
    if args.max_iters is not None:
        runner["max_iters"] = args.max_iters
        for k in ("log_interval", "checkpoint_interval", "eval_interval"):
            runner[k] = min(runner[k], args.max_iters)
        print(f"[ghi đè] max_iters={args.max_iters}")
    if args.batch_size is not None:
        runner["batch_size"] = args.batch_size
        print(f"[ghi đè] batch_size={args.batch_size}")
    return cfg


def make_loader(common: dict, dataset: str, split: str, batch_size: int):
    return build_dataloader(
        dataset, split,
        batch_size=batch_size,
        num_workers=common["num_workers"],
        seed=common["seed"],
        image_size=common["image_size"],
        augment=common["augment"],
        normalize=common["normalize"],
        preprocess=common.get("preprocess", "none"),
    )


def final_eval(model, cfg: dict, dataset: str, device, work_dir: Path) -> dict:
    """Đánh giá trên tập test cùng bộ và trên bộ chuyển miền; ghi results.json."""
    common, bs = cfg["common"], cfg["runner"]["batch_size"]
    targets = {f"{dataset}/{common['test_split']}": (dataset, common["test_split"])}
    cross = common.get("cross_domain")
    if cross and cross["dataset"] != dataset:
        targets[f"{cross['dataset']}/{cross['split']}"] = (cross["dataset"], cross["split"])
    results = {}
    for tag, (ds, split) in targets.items():
        loader = make_loader(common, ds, split, bs)
        results[tag] = evaluate(model, loader, device, common["num_classes"], common["ignore_index"])
        print(f"[{tag}] " + "  ".join(f"{k} {v:.4f}" for k, v in results[tag].items()))
    with open(work_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_model_config(args.model, args.config), args)
    common, runner = cfg["common"], cfg["runner"]
    dataset = args.dataset or common["train_dataset"]
    work_dir = from_root(args.work_dir) if args.work_dir else from_root(common["output_dir"]) / args.model / dataset
    work_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(common["device"])
    set_seed(common["seed"], common["deterministic"])
    print(f"model={args.model} dataset={dataset} device={device} work_dir={to_root(work_dir)}")

    model = build_segformer(cfg)
    if args.eval_only:
        load_checkpoint(from_root(args.eval_only), model, map_location="cpu")
        model.to(device)
        final_eval(model, cfg, dataset, device, work_dir)
        return

    if not args.no_pretrained and not args.resume:
        info = load_mit_pretrained(model.backbone, from_root(cfg["backbone"]["pretrained"]))
        print(f"đã nạp {info['loaded']} tensor pretrained vào backbone")
    with open(work_dir / "config_used.json", "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset, **cfg}, f, indent=2)

    train_loader = make_loader(common, dataset, "train", runner["batch_size"])
    val_loader = make_loader(common, dataset, common["val_split"], runner["batch_size"])
    best = train(model, train_loader, val_loader, cfg, work_dir, device,
                 resume=from_root(args.resume) if args.resume else None)
    print(f"tốt nhất trên val: {best['metric']} {best['score']:.4f} tại iter {best['iter']}")

    load_checkpoint(work_dir / "best.pth", model, map_location="cpu")
    model.to(device)
    final_eval(model, cfg, dataset, device, work_dir)


if __name__ == "__main__":
    main()
