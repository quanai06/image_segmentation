# Phân đoạn ảnh y tế: U-Net vs Transformer

U-Net tự cài đặt: xem [kiến trúc và cách sử dụng](docs/unet.md).

Đề tài 2 — phân đoạn vùng tổn thương ở mức điểm ảnh, so sánh encoder–decoder kiểu CNN (U-Net) với kiến trúc dựa trên Transformer (SegFormer), đánh giá bằng Dice và IoU.

Paper nền để tái tạo giao thức: **FCBFormer** — Sanderson & Matuszewski, *FCN-Transformer Feature Fusion for Polyp Segmentation*, MIUA 2022 ([arXiv](https://arxiv.org/abs/2208.08352), [code](https://github.com/ESandML/FCBFormer)).

## Cấu trúc repo

```
image_segmentation/
├── configs/          # cấu hình YAML (datasets.yaml, data.yaml)
├── data/             # raw/, processed/ (không lên Git), splits/, metadata/ (có trên Git)
├── src/medseg/       # package Python: paths, config, data/
├── scripts/          # tải, chuẩn hóa, chia tập, kiểm tra dữ liệu
├── tests/            # pytest
├── docs/             # tài liệu kỹ thuật cho nhóm
├── report/           # báo cáo, hình, bảng
├── notebooks/        # thử nghiệm nhanh
└── outputs/          # checkpoint, log (không lên Git)
```

Mọi đường dẫn tính từ gốc repo (`medseg.paths.ROOT`), không dùng đường dẫn tuyệt đối của máy cá nhân.

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                   # cài package medseg từ src/
```

## Chuẩn bị dữ liệu

```bash
python scripts/download_data.py    # tải Kvasir-SEG, CVC-ClinicDB + kiểm tra SHA-256
python scripts/prepare_data.py     # chuẩn hóa -> data/processed/<dataset>/256/
python scripts/make_splits.py      # chia 80/10/10 theo FCBFormer -> data/splits/
python scripts/check_data.py       # kiểm tra + thống kê -> report/
pytest -q                          # 19 test
```

Giải nén `.rar` của CVC-ClinicDB cần `bsdtar` (có sẵn trên macOS) hoặc `unrar` (Colab: `apt install unrar`).

## Dùng dữ liệu trong code

```python
from medseg.data import build_dataloader

train_loader = build_dataloader("kvasir_seg", "train")       # có augmentation
val_loader = build_dataloader("kvasir_seg", "val")
cross_loader = build_dataloader("cvc_clinicdb", "all")       # test chuyển miền
```

Mỗi batch: `image` (B, 3, 256, 256) đã chuẩn hóa ImageNet, `mask` (B, 1, 256, 256) giá trị {0, 1}, `name`.

Chi tiết: [docs/data_pipeline.md](docs/data_pipeline.md) · Báo cáo phần dữ liệu: [report/report.md](report/report.md)
