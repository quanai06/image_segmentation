import torch
from segment_anything import sam_model_registry
from segment_anything.modeling import Sam

from medseg.config import load_yaml
from medseg.paths import from_root


def sam_config() -> dict:
    return load_yaml("sam.yaml")


def checkpoint_path(model_type: str):
    cfg = sam_config()
    return from_root(cfg["checkpoints_dir"]) / cfg["models"][model_type]["file"]


def pick_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_sam(model_type: str = "vit_b", device: str | None = None) -> Sam:
    path = checkpoint_path(model_type)
    if not path.exists():
        raise FileNotFoundError(f"Chưa có {path.name}. Chạy: python scripts/download_sam_weights.py --models {model_type}")
    sam = sam_model_registry[model_type](checkpoint=str(path))
    return sam.to(pick_device(device)).eval()
