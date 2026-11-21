# U-Net với encoder ResNet34 pretrained

Kết quả sau huấn luyện và điều kiện dự đoán đúng/sai được giải thích trong
[phân tích lỗi hai mô hình](unet_error_analysis.md).

Mô hình ở `src/medseg/models/unet_resnet34.py`, import bằng:

```python
from medseg.models import UNetResNet34

model = UNetResNet34()  # mặc định pretrained=True
```

## Phần dùng thư viện và phần tự cài

- Encoder dùng `torchvision.models.resnet34` với trọng số
  `ResNet34_Weights.IMAGENET1K_V1` đã học trên ImageNet-1K.
- Bỏ global average pooling và lớp phân loại; lấy đặc trưng từ stem và
  `layer1` đến `layer4` làm đầu vào cho decoder.
- Decoder tự cài bằng ConvTranspose2d, ghép skip theo chiều kênh và
  `DoubleConv` dùng chung với U-Net từ đầu: hai Conv 3×3 + ReLU.
- Encoder có BatchNorm theo ResNet34; decoder không thêm BatchNorm.
- Decoder và head khởi tạo ngẫu nhiên. Encoder giữ nguyên trọng số đã nạp
  và được phép cập nhật khi fine-tune; không tự đóng băng tầng nào.

Tham khảo API và nguồn trọng số: [torchvision ResNet34](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet34.html).
Trong báo cáo cần phân biệt encoder lấy từ torchvision với decoder nhóm
tự triển khai; không mô tả toàn bộ ResNet34 là do nhóm tự cài.

## Kích thước tensor

Với ảnh RGB 256×256, không ghi chiều batch:

| Tầng | Kích thước C×H×W |
|---|---|
| Input | 3×256×256 |
| Stem: conv1 → bn1 → relu | 64×128×128 |
| maxpool → layer1 | 64×64×64 |
| layer2 | 128×32×32 |
| layer3 | 256×16×16 |
| layer4 | 512×8×8 |
| dec4, skip layer3 | 256×16×16 |
| dec3, skip layer2 | 128×32×32 |
| dec2, skip layer1 | 64×64×64 |
| dec1, skip stem | 32×128×128 |
| dec0, không có skip | 16×256×256 |
| Conv 1×1, head | 1×256×256 |

ResNet34 giảm kích thước 32 lần, vì vậy decoder có năm bước upsampling.
Với ảnh lẻ, decoder nội suy bilinear sau upsampling nếu kích thước chưa
khớp với skip hoặc ảnh gốc. Đầu ra luôn cùng H, W với đầu vào; H, W phải
ít nhất 32. Khi train, BatchNorm cần hơn một giá trị mỗi kênh: nếu ảnh chỉ
32×32 thì dùng batch ít nhất hai ảnh. Khi suy luận dùng `model.eval()`.

## Dùng với pipeline dữ liệu

Giữ `normalize: imagenet` trong `configs/data.yaml`: ảnh RGB được chia 255,
rồi chuẩn hóa bằng mean `(0.485, 0.456, 0.406)` và std
`(0.229, 0.224, 0.225)`. Không áp dụng transform phân loại của torchvision
để center-crop riêng ảnh; cần giữ ảnh và mask thẳng hàng như pipeline hiện tại.

Ví dụ một bước học với dữ liệu hiện có, chạy sau khi cài package
(`python -m pip install -e .`) và chuẩn bị dữ liệu:

```python
import torch
from medseg.data import build_dataloader
from medseg.models import UNetResNet34


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetResNet34(pretrained=True).to(device)
    loader = build_dataloader("kvasir_seg", "train", batch_size=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    batch = next(iter(loader))
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = torch.nn.BCEWithLogitsLoss()(logits, masks)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        probabilities = model(images).sigmoid()
        predictions = probabilities >= 0.5
    print(logits.shape, loss.item(), predictions.shape)


if __name__ == "__main__":
    main()
```

Đây là ví dụ một bước học, chưa phải script huấn luyện đầy đủ hoặc kết quả
thí nghiệm. Forward trả logits, dùng trực tiếp với BCEWithLogitsLoss.
Learning rate trong ví dụ là giá trị khởi đầu, chưa được tinh chỉnh.
Hai mô hình phải dùng chung train/val/test và giao thức đánh giá.

## Tải trọng số và kiểm thử

Lần đầu khởi tạo `pretrained=True` cần Internet để tải khoảng 83 MB.
Torchvision lưu vào cache của PyTorch, các lần sau dùng lại. Có thể đặt
`TORCH_HOME` vào `outputs/torch_cache` nếu muốn cache nằm trong dự án;
`outputs/` đã được bỏ qua trong Git.

Các unit test dùng `pretrained=False` hoặc mock thao tác nạp encoder nên
không cần Internet. Chúng kiểm tra kiến trúc, gradient và cấu hình nạp
trọng số, không thay thế việc thử tải trọng số thật.

Chạy ở thư mục gốc bằng PowerShell, kể cả khi chưa cài package editable:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
python -m unittest discover -s tests -p "test_unet*.py" -v
```

Muốn chạy riêng mô hình mới, đổi pattern thành `test_unet_resnet34.py`.
Để kiểm tra encoder ngẫu nhiên hoặc làm ablation dùng
`UNetResNet34(pretrained=False)`; cấu hình này không phải mô hình pretrained.
