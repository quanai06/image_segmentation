"""Augmentation và chuẩn hóa (albumentations).

Preset "fcbformer" bám theo Data/dataloaders.py và Data/dataset.py của FCBFormer:
GaussianBlur + ColorJitter (chỉ ảnh), lật ngang/dọc, affine (xoay, tịnh tiến, co giãn, cắt xiên).
"""

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

MEAN_STD = {
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "fcbformer": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "none": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),  # chỉ đưa về [0, 1], dùng để vẽ hình
}


def fcbformer_augment() -> list:
    return [
        A.GaussianBlur(blur_limit=(25, 25), sigma_limit=(0.001, 2.0), p=1.0),
        A.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            rotate=(-180, 180),
            translate_percent=(-0.125, 0.125),
            scale=(0.5, 1.5),
            shear=(-22.5, 22.5),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=1.0,
        ),
    ]


def build_transforms(train: bool, image_size: int, augment: str = "fcbformer", normalize: str = "imagenet") -> A.Compose:
    if normalize not in MEAN_STD:
        raise ValueError(f"normalize phải thuộc {list(MEAN_STD)}")
    mean, std = MEAN_STD[normalize]
    ops = [A.Resize(image_size, image_size, interpolation=cv2.INTER_AREA, mask_interpolation=cv2.INTER_NEAREST)]
    if train and augment == "fcbformer":
        ops += fcbformer_augment()
    elif train and augment != "none":
        raise ValueError(f"augment không hợp lệ: {augment}")
    ops += [A.Normalize(mean=mean, std=std, max_pixel_value=255.0), ToTensorV2()]
    return A.Compose(ops)
