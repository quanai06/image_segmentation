# Pipeline dữ liệu

Tài liệu cho cả nhóm: dữ liệu đi từ file tải về đến batch tensor đưa vào mô hình như thế nào.

## 1. Tổng quan

```
download_data.py   tải + kiểm tra SHA-256 + giải nén      -> data/raw/
prepare_data.py    đọc, nhị phân hóa mặt nạ, resize 256   -> data/processed/<ds>/256/
make_splits.py     chia 80/10/10 theo FCBFormer           -> data/splits/<ds>/*.txt
check_data.py      kiểm tra + thống kê + hình minh họa    -> report/tables, report/figures
SegmentationDataset / build_dataloader                    -> batch cho mô hình
```

## 2. Bộ dữ liệu

| Tên trong repo | Bộ dữ liệu | Số ảnh | Định dạng gốc | Nguồn |
|---|---|---|---|---|
| `kvasir_seg` | Kvasir-SEG | 1000 | JPG, kích thước thay đổi; mặt nạ JPG | datasets.simula.no |
| `cvc_clinicdb` | CVC-ClinicDB | 612 | TIFF 384 x 288; mặt nạ TIFF | link Dropbox trong README của FCBFormer |

URL, SHA-256 và đường dẫn nằm trong `configs/datasets.yaml`. ISIC 2018 và PH2 sẽ thêm sau bằng cách bổ sung một mục vào file này.

## 3. Chuẩn hóa (`prepare_data.py`)

| Bước | Ảnh | Mặt nạ |
|---|---|---|
| Đọc | JPG/PNG bằng OpenCV, TIFF bằng `tifffile`; chuyển về RGB uint8 | lấy kênh lớn nhất nếu có 3 kênh |
| Nhị phân hóa | | pixel > 127 là tổn thương (mặt nạ JPG của Kvasir có nhiễu nén) |
| Resize về 256 x 256 | `INTER_AREA` | `INTER_AREA` trên mặt nạ 0/1 rồi ngưỡng 0.5 |
| Lưu | PNG RGB | PNG xám, giá trị {0, 255} |

**Lưu ý TIFF của CVC-ClinicDB:** file khai báo `photometric = MINISBLACK` (ảnh xám) nhưng thực tế có 3 kênh màu. OpenCV chỉ đọc ra 1 kênh nên ảnh bị mất màu; vì vậy repo đọc TIFF bằng `tifffile` (FCBFormer cũng đọc qua `skimage`, vốn dùng `tifffile`). Test `test_cvc_tiff_read_as_color` chặn lỗi này.

Ảnh bị resize thẳng về hình vuông (không giữ tỷ lệ), giống FCBFormer (họ dùng 352 x 352). Muốn kích thước khác: `python scripts/prepare_data.py --size 352`, dữ liệu sẽ nằm ở `data/processed/<ds>/352/`.

## 4. Chia tập (`make_splits.py`)

Giao thức của FCBFormer, tái tạo đúng hàm `split_ids()` trong `Data/dataloaders.py` của họ:

1. Sắp xếp file ảnh theo tên (như `sorted(glob(...))`).
2. `train_test_split(chỉ_số, test_size=round(0.1 n), random_state=42)` lấy tập test.
3. Lặp lại trên phần còn lại để lấy tập val cùng kích thước.

| Bộ | train | val | test | all |
|---|---|---|---|---|
| Kvasir-SEG | 800 | 100 | 100 | 1000 |
| CVC-ClinicDB | 490 | 61 | 61 | 612 |

`all.txt` dùng cho thí nghiệm chuyển miền (train trên bộ này, test toàn bộ bộ kia). Lưu ý: khi tái tạo FCBFormer, họ test chuyển miền trên **tập test** của bộ kia, không phải toàn bộ.

## 5. Augmentation và chuẩn hóa (`src/medseg/data/transforms.py`)

Preset `fcbformer` (chỉ áp dụng cho tập train):

| Phép biến đổi | Tham số | Áp dụng cho |
|---|---|---|
| GaussianBlur | kernel 25, sigma 0.001–2 | ảnh |
| ColorJitter | độ sáng 0.4, tương phản 0.5, bão hòa 0.25, sắc độ 0.01 | ảnh |
| Lật ngang, lật dọc | xác suất 0.5 mỗi loại | ảnh + mặt nạ |
| Affine | xoay ±180°, tịnh tiến ±12.5%, co giãn 0.5–1.5, cắt xiên ±22.5°, viền đen | ảnh + mặt nạ (mặt nạ nội suy nearest) |

Chuẩn hóa (`normalize` trong `configs/data.yaml`):
- `imagenet` (mặc định): mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225), hợp với encoder pretrained (ResNet34, MiT của SegFormer).
- `fcbformer`: mean = std = 0.5, dùng khi cần tái tạo đúng FCBFormer.

## 6. Dùng trong code huấn luyện

```python
from medseg.data import build_dataloader

train_loader = build_dataloader("kvasir_seg", "train", batch_size=16)
val_loader = build_dataloader("kvasir_seg", "val", batch_size=16)
cross_loader = build_dataloader("cvc_clinicdb", "all", batch_size=16)  # test chuyển miền

batch = next(iter(train_loader))
batch["image"]  # float32 (B, 3, 256, 256), đã chuẩn hóa
batch["mask"]   # float32 (B, 1, 256, 256), giá trị {0, 1}
batch["name"]   # list tên mẫu
```

Quy ước chung cho mọi mô hình: nhận `(B, 3, 256, 256)`, trả logits `(B, 1, 256, 256)`.

Tập train mặc định có shuffle và `drop_last=True`; val/test không shuffle. Seed của DataLoader lấy từ `configs/data.yaml` (`loader.seed`).

**Tái lập augmentation:** albumentations 2.x không dùng seed toàn cục của NumPy/PyTorch. Repo tự seed bộ biến đổi của từng worker từ seed của DataLoader (`_seed_worker`), nên cùng seed thì ra cùng batch, còn các worker khác nhau thì không bị trùng augmentation (`tests/test_dataloader.py`).

**Số worker:** DataLoader dùng `persistent_workers=True`, nên worker chỉ khởi động một lần. Đo trên MacBook (CPU), một epoch train Kvasir-SEG (800 ảnh): 2.4s với 0 worker, 1.3s với 2 worker, 0.7s với 4 worker (không tính epoch đầu khởi động worker). Trên macOS, code tạo DataLoader có worker phải nằm trong file `.py` có `if __name__ == "__main__":`.

## 7. Đường dẫn

Mọi đường dẫn tính từ gốc repo qua `medseg.paths.ROOT`, nên repo chạy được ở bất kỳ thư mục nào (máy cá nhân, Colab, Kaggle). Không ghi đường dẫn tuyệt đối vào code hay config.
