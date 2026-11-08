"""Đọc file cấu hình YAML trong configs/."""

from pathlib import Path

import yaml

from medseg.paths import CONFIGS


def load_yaml(name: str | Path) -> dict:
    path = Path(name)
    if not path.is_absolute():
        path = CONFIGS / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data_config() -> dict:
    return load_yaml("data.yaml")


def load_dataset_specs() -> dict:
    return load_yaml("datasets.yaml")
