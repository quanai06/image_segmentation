"""U-Net với encoder ResNet34 từ torchvision và decoder tự cài đặt."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet34_Weights, resnet34

from medseg.models.unet import DoubleConv


class _DecoderBlock(nn.Module):
    """Upsampling, ghép skip nếu có, rồi hai Conv 3x3 + ReLU."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        output_size: tuple[int, int],
        skip: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.up(x)
        # ResNet làm tròn lên khi giảm kích thước: ảnh lẻ có thể lệch 1 pixel.
        if x.shape[-2:] != output_size:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetResNet34(nn.Module):
    """Nhận ảnh RGB (B, 3, H, W), trả logits (B, out_channels, H, W).

    Mặc định encoder nạp ResNet34_Weights.IMAGENET1K_V1, decoder khởi tạo
    ngẫu nhiên; toàn bộ encoder được phép học tiếp. Lần đầu cần tải trọng
    số nếu cache chưa có. pretrained=False không tải trọng số, dùng cho
    kiểm thử hoặc ablation encoder khởi tạo ngẫu nhiên.

    Đầu vào cần chuẩn hóa ImageNet; H và W >= 32. Với batch một ảnh rất
    nhỏ, BatchNorm của encoder cần model.eval() khi suy luận.
    """

    def __init__(self, pretrained: bool = True, out_channels: int = 1):
        super().__init__()
        if not isinstance(pretrained, bool):
            raise ValueError("pretrained phải là bool")
        if not isinstance(out_channels, int) or isinstance(out_channels, bool) or out_channels < 1:
            raise ValueError("out_channels phải là số nguyên dương")

        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = resnet34(weights=weights)
        # Chỉ dùng các tầng đặc trưng, bỏ head phân loại ImageNet.
        self.encoder.avgpool = nn.Identity()
        self.encoder.fc = nn.Identity()

        self.dec4 = _DecoderBlock(512, 256, 256)
        self.dec3 = _DecoderBlock(256, 128, 128)
        self.dec2 = _DecoderBlock(128, 64, 64)
        self.dec1 = _DecoderBlock(64, 64, 32)
        # Stem đã giảm ảnh xuống H/2: cần thêm một bước về kích thước gốc.
        self.dec0 = _DecoderBlock(32, 0, 16)
        self.head = nn.Conv2d(16, out_channels, kernel_size=1)
        # Không khởi tạo lại toàn mạng để giữ nguyên trọng số pretrained.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Đầu vào phải có dạng (B, 3, H, W)")
        if min(x.shape[-2:]) < 32:
            raise ValueError("H và W phải >= 32 cho encoder ResNet34")
        output_size = x.shape[-2:]
        enc = self.encoder
        e0 = enc.relu(enc.bn1(enc.conv1(x)))
        e1 = enc.layer1(enc.maxpool(e0))
        e2 = enc.layer2(e1)
        e3 = enc.layer3(e2)
        e4 = enc.layer4(e3)

        x = self.dec4(e4, e3.shape[-2:], e3)
        x = self.dec3(x, e2.shape[-2:], e2)
        x = self.dec2(x, e1.shape[-2:], e1)
        x = self.dec1(x, e0.shape[-2:], e0)
        x = self.dec0(x, output_size)
        return self.head(x)
