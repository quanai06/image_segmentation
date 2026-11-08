"""Dataset và DataLoader dùng chung cho mọi mô hình."""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from medseg.config import load_data_config
from medseg.data.splits import load_split
from medseg.data.transforms import build_transforms
from medseg.paths import from_root


def processed_root(dataset: str, image_size: int, processed_dir: str = "data/processed") -> Path:
    return from_root(processed_dir) / dataset / str(image_size)


class SegmentationDataset(Dataset):
    """Đọc ảnh/mặt nạ đã chuẩn hóa trong data/processed/<dataset>/<size>/.

    Mỗi mẫu trả về dict:
        image: float32 (3, H, W), đã chuẩn hóa
        mask:  float32 (1, H, W), giá trị {0, 1}
        name:  tên mẫu
    """

    def __init__(
        self,
        dataset: str,
        split: str,
        train: bool | None = None,
        image_size: int | None = None,
        augment: str | None = None,
        normalize: str | None = None,
        names: list[str] | None = None,
        seed: int | None = None,
    ):
        cfg = load_data_config()
        self.dataset = dataset
        self.split = split
        self.image_size = image_size or cfg["image_size"]
        self.root = processed_root(dataset, self.image_size, cfg["processed_dir"])
        if not self.root.exists():
            raise FileNotFoundError(f"Chưa có {self.root}. Chạy scripts/prepare_data.py trước.")
        self.names = names if names is not None else load_split(dataset, split)
        train = (split == "train") if train is None else train
        self.transform = build_transforms(
            train=train,
            image_size=self.image_size,
            augment=augment or cfg["augment"],
            normalize=normalize or cfg["normalize"],
        )
        self.set_seed(cfg["loader"]["seed"] if seed is None else seed)

    def set_seed(self, seed: int) -> None:
        """Seed riêng cho augmentation (albumentations không dùng seed toàn cục)."""
        self.transform.set_random_seed(int(seed) % 2**32)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> dict:
        name = self.names[idx]
        img = cv2.cvtColor(cv2.imread(str(self.root / "images" / f"{name}.png")), cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(str(self.root / "masks" / f"{name}.png"), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        out = self.transform(image=img, mask=mask)
        image = out["image"]
        mask = out["mask"]
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask)
        return {"image": image.float(), "mask": mask.unsqueeze(0).float(), "name": name}


def _seed_worker(worker_id: int) -> None:
    # Mỗi worker có seed riêng, suy ra từ generator của DataLoader nên tái lập được
    cv2.setNumThreads(0)  # tránh OpenCV tranh CPU với các worker khác
    info = torch.utils.data.get_worker_info()
    seed = info.seed % 2**32
    np.random.seed(seed)
    random.seed(seed)
    info.dataset.set_seed(seed)


def build_dataloader(
    dataset: str,
    split: str,
    batch_size: int | None = None,
    shuffle: bool | None = None,
    num_workers: int | None = None,
    seed: int | None = None,
    **dataset_kwargs,
) -> DataLoader:
    """Tạo DataLoader; tập train mặc định shuffle và drop_last."""
    cfg = load_data_config()["loader"]
    seed = cfg["seed"] if seed is None else seed
    ds = SegmentationDataset(dataset, split, seed=seed, **dataset_kwargs)
    is_train = split == "train"
    generator = torch.Generator().manual_seed(seed)
    workers = cfg["num_workers"] if num_workers is None else num_workers
    return DataLoader(
        ds,
        batch_size=batch_size or cfg["batch_size"],
        shuffle=is_train if shuffle is None else shuffle,
        drop_last=is_train,
        num_workers=workers,
        worker_init_fn=_seed_worker,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,  # không khởi động lại worker mỗi epoch
    )
