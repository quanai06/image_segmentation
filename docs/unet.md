# U-Net tự cài đặt

Mô hình nằm ở `src/medseg/models/unet.py`, chỉ dùng các lớp cơ bản của
PyTorch, không dùng mô hình segmentation có sẵn hoặc trọng số pretrained.

Encoder có bốn mức MaxPool, số kênh 64, 128, 256, 512; bottleneck có
1024 kênh. Mỗi khối dùng hai Conv 3×3 + ReLU. Decoder dùng
ConvTranspose2d 2×2 để tăng kích thước, ghép đặc trưng encoder tương ứng
theo chiều kênh, rồi áp dụng hai Conv 3×3 + ReLU. Lớp cuối là Conv 1×1.

Đây là biến thể U-Net dùng same-padding để giữ kích thước đầu ra,
khác bản gốc dùng valid convolution và crop. Không dùng BatchNorm.
Với kích thước lẻ, decoder được padding trước khi ghép skip connection.

## Dùng với pipeline hiện tại

Sau khi cài package bằng `pip install -e .`:

```python
import torch
from medseg.data import build_dataloader
from medseg.models import UNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    loader = build_dataloader("kvasir_seg", "train", batch_size=2)
    batch = next(iter(loader))
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)

    logits = model(images)  # (B, 1, 256, 256)
    loss = torch.nn.BCEWithLogitsLoss()(logits, masks)
    loss.backward()

    model.eval()
    with torch.no_grad():
        probabilities = model(images).sigmoid()
        predictions = probabilities >= 0.5
    print(logits.shape, loss.item(), predictions.shape)


if __name__ == "__main__":  # cần thiết cho DataLoader nhiều worker trên Windows
    main()
```

Đây là ví dụ kiểm tra một batch, chưa phải vòng lặp huấn luyện đầy đủ.
`forward` trả logits: không sigmoid trước `BCEWithLogitsLoss`.
Chỉ dùng sigmoid để lấy xác suất hoặc tính soft Dice loss.

Mặc định `UNet(in_channels=3, out_channels=1, base_channels=64)`.
Có thể giảm `base_channels` xuống 32 khi thiếu bộ nhớ; cần ghi lại thay đổi
này trong cấu hình thí nghiệm. Chiều cao và rộng đầu vào phải ít nhất 16.

## Kiểm tra mô hình

Không cần tải dữ liệu, nhưng môi trường Python phải có PyTorch và import
được package `medseg` nằm trong `src/`. Chạy tại thư mục gốc dự án, dùng
cùng Python đã cài các thư viện:

```bash
python -m pip install -e .
python -m unittest discover -s tests -p test_unet.py -v
```

Lệnh cài editable chỉ cần chạy một lần cho mỗi môi trường Python. Sau đó
có thể chạy lại lệnh test khi sửa mô hình.

Nếu chỉ muốn kiểm tra mô hình mà chưa cài package, có thể khai báo đường
dẫn `src` cho phiên PowerShell hiện tại:

```powershell
Set-Location D:\image_segmentation
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
python -m unittest discover -s tests -p test_unet.py -v
```

Nếu thấy `ModuleNotFoundError: No module named 'medseg'` và
`Ran 1 test ... FAILED (errors=1)`, unittest đang lỗi khi import tệp test,
chưa chạy được năm bài kiểm tra mô hình. Dùng một trong hai cách trên để
Python tìm thấy package. Nếu lỗi là `No module named 'torch'`, cần cài
PyTorch trong đúng môi trường Python đang dùng.

Các kiểm tra cũng chạy được bằng `pytest tests/test_unet.py` nếu đã cài pytest.
Bao gồm đầu ra ở độ phân giải 256×256, ảnh lẻ không vuông, kênh tùy chọn,
gradient, logits chưa qua sigmoid và đầu vào không hợp lệ.
