# Phân đoạn ảnh y tế: U-Net vs Transformer

U-Net tự cài đặt: xem [kiến trúc và cách sử dụng](docs/unet.md).

U-Net với encoder ResNet34 pretrained: xem [kiến trúc và cách sử dụng](docs/unet_resnet34.md).

Hai thí nghiệm đã chạy trên Kaggle: xem [hướng dẫn và kết quả](docs/unet_experiments.md)
và [phân tích điều kiện đúng/sai](docs/unet_error_analysis.md).

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
pytest -q                          # kiểm tra pipeline và mô hình
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

## Thí nghiệm U-Net trên Kaggle

Đặt hai notebook tại `notebooks/Unet.ipynb` và `notebooks/Unet_ResNet34.ipynb`.
Kết quả test đã lưu:

| Mô hình | Global Dice | Global IoU | Mean image Dice | Mean image IoU |
|---|---:|---:|---:|---:|
| U-Net từ đầu | 0.7680 | 0.6233 | 0.7636 | 0.6580 |
| U-Net ResNet34 pretrained | 0.8912 | 0.8038 | 0.8968 | 0.8246 |

Giao thức trong notebook dùng split 80/20 rồi 50/50 với seed 42, khác split
FCBFormer của pipeline phía trên. Bản tái tạo nằm trong
`data/splits/kvasir_seg_notebook/`; trạng thái đối chiếu ID gốc được ghi trong
[`experiments.json`](report/results/experiments.json). Cấu hình U-Net dùng ảnh `[0,1]`, còn ResNet34 dùng chuẩn hóa
ImageNet. Bảng so sánh phản ánh hai cấu hình hoàn chỉnh ở một seed.

Toàn bộ cấu hình, số liệu và nguồn trích xuất của hai run được gộp trong
`report/results/experiments.json`. Tạo lại bảng và biểu đồ bằng một lệnh:

```bash
python scripts/compare_experiments.py --plot
```

Chỉ bước vẽ cần matplotlib. Lịch sử/bảng nhóm trích từ output giữ độ chính
xác đã in. Để so sánh đủ 100 ảnh, tải `test_per_image.csv` và `split_manifest.csv`
gốc của mỗi run vào hai thư mục trong `outputs/`, rồi truyền chúng vào cùng
script bằng `--unet-dir` và `--resnet34-dir`. Xem
[hướng dẫn xuất kết quả và kiểm tra split](docs/unet_experiments.md).

Đưa kết quả nhỏ lên Git trong `report/results/`. Dataset và trọng số lớn
được lưu ngoài Git; `.gitignore` đã bỏ qua `data/raw/`, `data/processed/`,
`outputs/`, `*.pt` và `*.pth`.
