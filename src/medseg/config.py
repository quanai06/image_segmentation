"""Đọc file cấu hình trong configs/: YAML cho pipeline dữ liệu, JSON cho huấn luyện."""

import json
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


def strip_notes(obj):
    """Bỏ các khóa chú thích (bắt đầu bằng '_') ở mọi cấp của dict/list đọc từ JSON."""
    if isinstance(obj, dict):
        return {k: strip_notes(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [strip_notes(v) for v in obj]
    return obj


def load_json(name: str | Path = "config.json") -> dict:
    """Đọc file JSON trong configs/ (hoặc đường dẫn tuyệt đối), đã bỏ chú thích."""
    path = Path(name)
    if not path.is_absolute():
        path = CONFIGS / path
    with open(path, encoding="utf-8") as f:
        return strip_notes(json.load(f))


def load_model_config(model: str, name: str | Path = "config.json") -> dict:
    """Cấu hình của một mô hình kèm khóa 'common' chứa biến dùng chung."""
    cfg = load_json(name)
    if model not in cfg:
        models = [k for k in cfg if k != "common"]
        raise KeyError(f"Không có mô hình '{model}' trong {name}; có: {models}")
    return {"common": cfg["common"], **cfg[model]}
