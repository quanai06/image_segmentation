# Phân đoạn ảnh y tế: U-Net vs Transformer

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
python scripts/analyze_artifacts.py   # khung, ô ScopeGuide, phản xạ, màu -> report/ (mục 7.7)
pytest -q
```

Tiền xử lý ảnh (tùy chọn, mặc định tắt để giữ giao thức FCBFormer):

```bash
python scripts/prepare_data.py --preprocess clean         # bỏ khung/ô ScopeGuide, xóa phản xạ
python scripts/prepare_data.py --preprocess clean_color   # thêm chuẩn hóa màu
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

## Huấn luyện

Biến dùng chung và siêu tham số từng mô hình nằm trong `configs/config.json`.

```bash
python scripts/train_segformer.py   # SegFormer-B2, cần pretrained/mit_b2.pth
```

Kế hoạch và đối chiếu với code gốc: [docs/segformer_plan.md](docs/segformer_plan.md)

## Tiến độ: 3 commit gần nhất

Tóm tắt nhanh:

| Commit | Nội dung chính | Quy mô |
|---|---|---|
| `ec0fd75` Setup data + trainer | Port SegFormer-B2 sang PyTorch thuần, vòng huấn luyện, optimizer/LR, loss, đánh giá mIoU; thống kê dữ liệu phục vụ huấn luyện | 16 file, +1342 / −22 |
| `6cfd19e` Plan | Kế hoạch áp dụng SegFormer-B2 (`docs/segformer_plan.md`), mục Huấn luyện trong README | 2 file, +95 |
| `6266c1f` preprocess, data analysis | Phân tích artifact ảnh nội soi, tiền xử lý tùy chọn `clean` / `clean_color`, đo khoảng cách miền; script tạo `mit_b2.pth` từ Hugging Face và đóng gói cho Kaggle | 20 file, +2586 / −17 |

### 1. `ec0fd75` — Setup data + trainer

**Mục tiêu:** có một bản SegFormer-B2 chạy được với PyTorch hiện tại, giữ đúng kiến trúc và siêu tham số của code chính thức (NVIDIA, dựa trên mmseg 0.x / mmcv 1.x vốn không cài được nữa).

**Mô hình** — [src/medseg/models/segformer.py](src/medseg/models/segformer.py):
- Backbone `MixVisionTransformer` (MiT-B2) cùng các khối con `OverlapPatchEmbed`, `Attention` (efficient self-attention có `sr_ratio`), `Mlp` + `DWConv`, `Block`, `DropPath`.
- Decoder `SegFormerHead`: chiếu 4 tầng đặc trưng về `embed_dim` 768, upsample về 1/4, nối lại, `linear_fuse` (Conv + BN + ReLU), dropout 0.1, `linear_pred`.
- `SegFormer.forward` trả logits ở 1/4 độ phân giải (64 × 64 với ảnh 256); `predict_logits` upsample bilinear về kích thước ảnh; `predict` trả nhãn bằng argmax (tương đương `whole_inference` của mmseg).
- `to_binary_logits`: đổi logits 2 lớp thành 1 kênh `(B, 1, H, W)` bằng `l1 − l0`, để dùng chung module metric với U-Net (`sigmoid(l1 − l0) == softmax[:, 1]`).
- `build_segformer(cfg)` dựng mô hình từ cấu hình; `load_mit_pretrained` nạp `mit_b2.pth` vào backbone, báo lỗi rõ nếu thiếu tham số. Tên khóa `state_dict` giữ nguyên như bản gốc nên checkpoint gốc nạp thẳng được.

**Huấn luyện** — [src/medseg/training/](src/medseg/training/):
- [losses.py](src/medseg/training/losses.py): `segformer_ce_loss`, CrossEntropy 2 lớp, upsample logits về kích thước mặt nạ trước khi tính, `ignore_index` 255.
- [optim.py](src/medseg/training/optim.py): `build_param_groups` tái tạo `paramwise_cfg` của mmcv (decoder lr ×10; `norm`, `pos_block` không weight decay); AdamW lr 6e-5; `poly_warmup_lr` / `build_scheduler` = poly power 1.0 + warmup tuyến tính 1500 iter, cập nhật theo iteration.
- [evaluate.py](src/medseg/training/evaluate.py): `intersect_and_union` + `summarize` tính mIoU kiểu mmseg (cộng dồn giao/hợp trên toàn tập rồi mới chia), kèm Dice/IoU của lớp tổn thương.
- [trainer.py](src/medseg/training/trainer.py): vòng lặp theo iteration thay cho `IterBasedRunner` của mmcv: DataLoader lặp vô hạn, log mỗi 50 iter ra `log.jsonl` (có `sec_per_iter`), checkpoint mỗi 4000 iter (`iter_N.pth`, `latest.pth`), đánh giá val mỗi 16000 iter và lưu `best.pth` theo mIoU, hỗ trợ `resume`. Có thêm `set_seed`, `resolve_device`.

**Cấu hình** — [configs/config.json](configs/config.json) và [src/medseg/config.py](src/medseg/config.py):
- Mục `common`: seed 42, ảnh 256, 2 lớp, chuẩn hóa ImageNet, augmentation `fcbformer`, train trên Kvasir-SEG, test chuyển miền trên toàn bộ CVC-ClinicDB.
- Mục `segformer_b2`: toàn bộ siêu tham số gốc (backbone, decode head, loss, optimizer, lịch LR, runner 160k iter × batch 16). Khóa bắt đầu bằng `_` là chú thích, bị `strip_notes` bỏ khi đọc; `load_model_config` gộp `common` với mục của mô hình.

**Script** — [scripts/train_segformer.py](scripts/train_segformer.py): train trên tập train, chọn checkpoint theo val, rồi đánh giá trên test cùng bộ và trên bộ chuyển miền, ghi `outputs/segformer_b2/<dataset>/results.json` và `config_used.json`. Tùy chọn `--dataset`, `--resume`, `--eval-only`, và các ghi đè chỉ để chạy thử `--max-iters`, `--batch-size`, `--no-pretrained` (được in ra để không nhầm với cấu hình gốc).

**Thống kê dữ liệu cho huấn luyện** — [scripts/check_data.py](scripts/check_data.py), [report/report.md](report/report.md) mục 7.3–7.4:
- `modeling_stats`: lệch lớp (nền : tổn thương = 5.5 : 1 ở Kvasir, 9.9 : 1 ở CVC), kích thước tổn thương, số ảnh có tổn thương dưới 50 ô trên lưới logit 64 × 64 (10 ảnh Kvasir, 22 ảnh CVC), méo tỷ lệ khi resize vuông, ảnh trùng SHA-256 (không có) → `report/tables/modeling_stats.csv`.
- `frame_adjacency`: với CVC (tên ảnh là số thứ tự khung hình), 96.7% ảnh val và 91.8% ảnh test có khung hình kề (±1) nằm trong train → xác nhận rò rỉ, val CVC cũng không đáng tin để chọn checkpoint → `report/tables/frame_adjacency.csv`.
- Cờ `--manifest-only` để tính các bảng trên chỉ từ manifest, không cần tải ảnh.

**Kiểm thử** — [tests/test_segformer.py](tests/test_segformer.py) (14 test): cấu hình giữ đúng siêu tham số gốc, kích thước đầu ra từng tầng, số tham số, tên khóa decoder khớp mmseg, `to_binary_logits` đúng bằng softmax lớp tổn thương, param group / LR / loss / metric / trainer, và chạy chính file `mix_transformer.py` gốc (thay mmcv/timm bằng stub) để so `state_dict` và đầu ra số học (sai khác < 1e-5).

### 2. `6cfd19e` — Plan

Thêm [docs/segformer_plan.md](docs/segformer_plan.md) và mục **Huấn luyện** ở trên. Tài liệu gồm:
- **Bảng đối chiếu** từng file của code gốc (mmseg/mmcv) với file tương ứng trong repo.
- **Giai đoạn 1 (hiện tại):** giữ nguyên siêu tham số gốc của SegFormer-B2; chỉ phần dữ liệu (ảnh 256 × 256, augmentation `fcbformer`, chia 80/10/10, 2 lớp) theo pipeline chung của nhóm để so sánh công bằng với U-Net.
- **Thí nghiệm:** E1 Kvasir → Kvasir test (chính), E2 Kvasir → toàn bộ CVC (chuyển miền, đáng tin nhất), E3 CVC → CVC test (chỉ để so với FCBFormer, có cảnh báo rò rỉ).
- **Lưu ý trước khi chạy:** 160k iter trên 800 ảnh ≈ 3200 epoch, nên đo tốc độ trước bằng `--max-iters 200`; tổn thương nhỏ ở lưới 64 × 64; lệch lớp; cách lấy `mit_b2.pth`.
- **Giai đoạn 2 (dự kiến):** đổi từng yếu tố một so với baseline — loss BCE + Dice, số iteration theo epoch + dừng sớm, ảnh 352 × 352, AMP.

### 3. `6266c1f` — preprocess, data analysis

**Phân tích artifact** — [scripts/analyze_artifacts.py](scripts/analyze_artifacts.py), [report/report.md](report/report.md) mục 7.7:
- Khung ngoài vùng nhìn thấy (FOV): khung Kvasir có giá trị 0, khung CVC 11–24; FOV chiếm 83% ảnh Kvasir, 72% ảnh CVC.
- Ô ScopeGuide ở góc dưới trái: có ở **55.4%** ảnh Kvasir (13 ảnh ô xanh, còn lại ô đen), không có ở CVC.
- Phản xạ sáng: dày hơn trong tổn thương so với nền (1.2 lần ở Kvasir, 2.3 lần ở CVC), nên có thể là tín hiệu có ích, xóa đi chưa chắc có lợi.
- Màu: Kvasir sáng hơn CVC rõ (L trung vị 125 so với 107).
- Kiểm tra mặt nạ: pixel tổn thương "ngoài FOV" ở 24 ảnh CVC thực chất là nhãn tràn ra khung đen, không phải bộ dò cắt mất tổn thương.
- **Khoảng cách miền:** hồi quy logistic đoán ảnh thuộc Kvasir hay CVC (5-fold, độ chính xác cân bằng). Hình học khung cho 99.6% với mọi preset — đây là dấu hiệu nguồn ảnh mạnh nhất và tiền xử lý không xóa được; chuẩn hóa màu giảm độ phân biệt bằng màu trong FOV từ 0.92 xuống 0.79.
- Kết quả: `report/tables/artifact_stats.csv`, `domain_gap.csv`, `artifacts_per_image.csv` (từng ảnh), hình `report/figures/preprocess_examples.png`, `color_shift.png`.

**Tiền xử lý tùy chọn** — [src/medseg/data/preprocess.py](src/medseg/data/preprocess.py), mặc định **tắt** để giữ đúng giao thức FCBFormer:
- `mask_outside`: dựng FOV bằng bao lồi các mảng mô lớn, dò ô ScopeGuide (đen tuyệt đối hoặc xanh, nối với góc dưới trái), tô đen mọi thứ ngoài FOV và trong ô.
- `inpaint_highlights`: xóa phản xạ sáng và chữ trắng nhỏ (V ≥ 220, S ≤ 40, vùng ≤ 0.5% ảnh, nới 2%) bằng inpaint TELEA; vùng sáng lớn như polyp nhạt màu được giữ.
- `normalize_color`: chuẩn hóa LAB trong FOV về trung vị tập train Kvasir-SEG (chỉ lấy từ train nên không rò rỉ).
- Preset trong [configs/data.yaml](configs/data.yaml): `none`, `clean` (2 bước đầu), `clean_color` (cả 3 bước); tham số ngưỡng ở `preprocess_params`.
- Tích hợp: `prepare_data.py --preprocess <preset>` ghi ra `data/processed/<ds>/256_<preset>/` kèm `manifest.csv` riêng, không đụng bản gốc; `processed_root` và `SegmentationDataset` nhận tham số `preprocess`; `configs/config.json` có `common.preprocess` (mặc định `"none"`) và `train_segformer.py` truyền nó vào DataLoader.
- Hạn chế: chữ in đè chỉ bị xóa một phần; chưa có kết quả huấn luyện chứng minh preset nào tốt hơn `none` — để dành so sánh ở giai đoạn 2.

**Trọng số pretrained và chạy trên Kaggle:**
- [scripts/convert_hf_mit.py](scripts/convert_hf_mit.py): tải `nvidia/mit-b2` từ Hugging Face (cùng checkpoint ImageNet-1K với bản gốc), đổi tên khóa về kiểu `mix_transformer.py`, ghép `key` + `value` thành `attn.kv`, kiểm tra khớp toàn bộ backbone rồi ghi `pretrained/mit_b2.pth`. Không còn phải tải thủ công từ Google Drive/OneDrive.
- [scripts/build_kaggle.py](scripts/build_kaggle.py): đóng gói code, cấu hình, file chia tập, metadata và `mit_b2.pth` thành `Kaggle/medseg-code/` + file zip để upload làm Kaggle Dataset (ảnh không đóng gói; notebook lấy dữ liệu từ dataset có sẵn trên Kaggle rồi chạy `prepare_data.py`).
- `.gitignore` chặn `pretrained/`, `base/*`, `Kaggle/*`.

**Kiểm thử** — [tests/test_preprocess.py](tests/test_preprocess.py) (13 test): khung được tô về 0, phát hiện ô ScopeGuide đen và xanh, lòng ruột tối không bị nhận nhầm là ô, phản xạ nhỏ bị xóa còn vùng sáng lớn được giữ, chuẩn hóa màu đạt đích và giữ màu trắng trung tính, preset `none` không đổi ảnh, preset trong cấu hình hợp lệ, thư mục theo preset đúng; thêm các ca trên ảnh thật (tự bỏ qua nếu chưa tải dữ liệu). Toàn bộ `pytest -q` theo report: 45 đạt, 1 bỏ qua.

**Tài liệu cập nhật:** [docs/data_pipeline.md](docs/data_pipeline.md) mục 3.1, [data/README.md](data/README.md) (cấu trúc thư mục theo preset), [report/report.md](report/report.md) mục 7.7.

```bash
python scripts/convert_hf_mit.py                        # tạo pretrained/mit_b2.pth
python scripts/analyze_artifacts.py                     # số liệu + hình mục 7.7
python scripts/prepare_data.py --preprocess clean       # tùy chọn: dữ liệu đã tiền xử lý
python scripts/build_kaggle.py --kaggle-user <tên>      # gói upload lên Kaggle
```
