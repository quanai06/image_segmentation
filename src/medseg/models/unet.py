"""U-Net tự cài bằng PyTorch, toàn bộ trọng số khởi tạo ngẫu nhiên.

Dùng convolution có padding để giữ kích thước đầu ra bằng ảnh đầu vào.
Đây là biến thể same-padding, khác bản gốc dùng valid convolution và crop.
"""

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """Hai lần Conv 3x3 + ReLU, giữ nguyên kích thước không gian."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UpBlock(nn.Module):
    """Tăng kích thước, ghép skip theo chiều kênh, rồi học đặc trưng mới."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pooling làm tròn xuống với ảnh lẻ; bù lại trước khi ghép skip.
        dh = skip.shape[-2] - x.shape[-2]
        dw = skip.shape[-1] - x.shape[-1]
        x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """U-Net bốn mức pooling, trả logits (B, out_channels, H, W).

    Mặc định: ảnh RGB, phân đoạn nhị phân, số kênh encoder
    64 -> 128 -> 256 -> 512 -> 1024. H và W phải >= 16.
    Không sigmoid trong forward: dùng trực tiếp với BCEWithLogitsLoss;
    khi suy luận dùng logits.sigmoid() để lấy xác suất.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 64):
        super().__init__()
        for name, value in (
            ("in_channels", in_channels),
            ("out_channels", out_channels),
            ("base_channels", base_channels),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} phải là số nguyên dương")
        self.in_channels = in_channels
        c = base_channels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.enc4 = DoubleConv(c * 4, c * 8)
        self.bottleneck = DoubleConv(c * 8, c * 16)
        self.dec4 = UpBlock(c * 16, c * 8)
        self.dec3 = UpBlock(c * 8, c * 4)
        self.dec2 = UpBlock(c * 4, c * 2)
        self.dec1 = UpBlock(c * 2, c)
        self.head = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Đầu vào phải có dạng (B, {self.in_channels}, H, W)")
        if min(x.shape[-2:]) < 16:
            raise ValueError("H và W phải >= 16 để qua bốn mức pooling")
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        x = self.bottleneck(self.pool(e4))
        x = self.dec4(x, e4)
        x = self.dec3(x, e3)
        x = self.dec2(x, e2)
        x = self.dec1(x, e1)
        return self.head(x)
