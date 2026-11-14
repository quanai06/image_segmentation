"""SegFormer (Xie et al., NeurIPS 2021): encoder Mix Transformer (MiT) + decoder All-MLP.

Port từ code gốc base/SegFormer-master sang PyTorch thuần, không cần mmcv/mmseg/timm:
- MixVisionTransformer, Block, Attention, Mlp, DWConv, OverlapPatchEmbed
  <- mmseg/models/backbones/mix_transformer.py
- SegFormerHead <- mmseg/models/decode_heads/segformer_head.py
- SegFormer (EncoderDecoder) <- mmseg/models/segmentors/encoder_decoder.py

Tên tham số giữ đúng như bản gốc, nên checkpoint ImageNet `mit_b2.pth` của tác giả
nạp thẳng vào backbone mà không cần đổi tên khóa.

Khác biệt có chủ ý so với bản gốc:
- SyncBN trong decoder -> BatchNorm2d (huấn luyện 1 GPU, tương đương).
- Bỏ `conv_seg` thừa của BaseDecodeHead (khai báo nhưng SegFormerHead không dùng).
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- tiện ích

def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Stochastic depth theo từng mẫu (giống timm.models.layers.drop_path)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


def init_mit_weights(m: nn.Module) -> None:
    """Khởi tạo trọng số như `_init_weights` của MiT gốc."""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()


def resize(x: torch.Tensor, size, align_corners: bool = False) -> torch.Tensor:
    """Nội suy bilinear như mmseg.ops.resize."""
    return F.interpolate(x, size=size, mode="bilinear", align_corners=align_corners)


# ----------------------------------------------------------------------------- encoder MiT

class DWConv(nn.Module):
    """Conv 3x3 depthwise trong Mix-FFN, thay cho positional encoding."""

    def __init__(self, dim: int = 768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    """Mix-FFN: Linear -> DWConv 3x3 -> GELU -> Linear."""

    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None,
                 act_layer=nn.GELU, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Attention(nn.Module):
    """Efficient self-attention: K, V được thu nhỏ không gian theo tỷ lệ sr_ratio."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, qk_scale: float | None = None,
                 attn_drop: float = 0.0, proj_drop: float = 0.0, sr_ratio: int = 1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} phải chia hết cho num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)  # eps mặc định 1e-5, đúng như bản gốc

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class Block(nn.Module):
    """Một khối Transformer: LN -> Efficient Attention -> LN -> Mix-FFN, có residual + DropPath."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = False,
                 qk_scale: float | None = None, drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0,
                 act_layer=nn.GELU, norm_eps: float = 1e-6, sr_ratio: int = 1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        return x + self.drop_path(self.mlp(self.norm2(x), H, W))


class OverlapPatchEmbed(nn.Module):
    """Patch embedding chồng lấn: Conv (kernel > stride) + LayerNorm."""

    def __init__(self, patch_size: int = 7, stride: int = 4, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)  # eps mặc định 1e-5, đúng như bản gốc

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), H, W


class MixVisionTransformer(nn.Module):
    """Encoder phân cấp 4 tầng, trả về đặc trưng ở stride 4, 8, 16, 32."""

    def __init__(self, in_chans: int = 3, embed_dims=(64, 128, 256, 512), num_heads=(1, 2, 4, 8),
                 mlp_ratios=(4, 4, 4, 4), qkv_bias: bool = False, qk_scale: float | None = None,
                 drop_rate: float = 0.0, attn_drop_rate: float = 0.0, drop_path_rate: float = 0.0,
                 norm_eps: float = 1e-6, depths=(3, 4, 6, 3), sr_ratios=(8, 4, 2, 1)):
        super().__init__()
        self.depths = list(depths)
        self.embed_dims = list(embed_dims)

        self.patch_embed1 = OverlapPatchEmbed(7, 4, in_chans, embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(3, 2, embed_dims[0], embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(3, 2, embed_dims[1], embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(3, 2, embed_dims[2], embed_dims[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # tăng tuyến tính theo độ sâu
        cur = 0
        for i in range(4):
            blocks = nn.ModuleList([
                Block(dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                      qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                      norm_eps=norm_eps, sr_ratio=sr_ratios[i])
                for j in range(depths[i])
            ])
            setattr(self, f"block{i + 1}", blocks)
            setattr(self, f"norm{i + 1}", nn.LayerNorm(embed_dims[i], eps=norm_eps))
            cur += depths[i]

        self.apply(init_mit_weights)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        B = x.shape[0]
        outs = []
        for i in range(1, 5):
            x, H, W = getattr(self, f"patch_embed{i}")(x)
            for blk in getattr(self, f"block{i}"):
                x = blk(x, H, W)
            x = getattr(self, f"norm{i}")(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs


# ----------------------------------------------------------------------------- decoder All-MLP

class MLP(nn.Module):
    """Linear embedding: (B, C, H, W) -> (B, H*W, embed_dim)."""

    def __init__(self, input_dim: int = 2048, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(2).transpose(1, 2))


class ConvBNReLU(nn.Module):
    """Tương đương mmcv ConvModule(kernel_size=1, norm=BN, act=ReLU); tên con `conv`/`bn` giữ như mmcv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)  # bias='auto' khi có norm
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.ReLU(inplace=True)
        # khởi tạo như ConvModule.init_weights: kaiming fan_out cho conv, BN weight=1 bias=0
        nn.init.kaiming_normal_(self.conv.weight, a=0, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))


class SegFormerHead(nn.Module):
    """Decoder All-MLP: chiếu 4 tầng về embed_dim, upsample về 1/4, nối, fuse, dự đoán."""

    def __init__(self, in_channels=(64, 128, 320, 512), in_index=(0, 1, 2, 3), feature_strides=(4, 8, 16, 32),
                 embed_dim: int = 768, num_classes: int = 2, dropout_ratio: float = 0.1, align_corners: bool = False):
        super().__init__()
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.in_channels = list(in_channels)
        self.in_index = list(in_index)
        self.feature_strides = list(feature_strides)
        self.num_classes = num_classes
        self.align_corners = align_corners

        c1, c2, c3, c4 = self.in_channels
        self.linear_c4 = MLP(input_dim=c4, embed_dim=embed_dim)
        self.linear_c3 = MLP(input_dim=c3, embed_dim=embed_dim)
        self.linear_c2 = MLP(input_dim=c2, embed_dim=embed_dim)
        self.linear_c1 = MLP(input_dim=c1, embed_dim=embed_dim)
        self.linear_fuse = ConvBNReLU(embed_dim * 4, embed_dim)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        c1, c2, c3, c4 = [inputs[i] for i in self.in_index]
        n = c4.shape[0]
        size = c1.shape[2:]
        # các bước resize trong bản gốc cố định align_corners=False
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size, align_corners=False)
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size, align_corners=False)
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size, align_corners=False)
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        x = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(x)
        return self.linear_pred(x)


# ----------------------------------------------------------------------------- mô hình đầy đủ

class SegFormer(nn.Module):
    """EncoderDecoder: backbone MiT + SegFormerHead.

    forward(x) trả logits ở 1/4 độ phân giải đầu vào, (B, num_classes, H/4, W/4), như decode head gốc.
    Dùng `predict_logits` để có logits ở đúng kích thước ảnh (encode_decode / whole_inference).
    """

    def __init__(self, backbone: MixVisionTransformer, decode_head: SegFormerHead):
        super().__init__()
        self.backbone = backbone
        self.decode_head = decode_head
        self.align_corners = decode_head.align_corners
        self.num_classes = decode_head.num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode_head(self.backbone(x))

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Logits đã upsample về kích thước ảnh vào: (B, num_classes, H, W)."""
        return resize(self(x), x.shape[2:], align_corners=self.align_corners)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Nhãn dự đoán (B, H, W) theo test_mode 'whole': softmax rồi argmax."""
        return self.predict_logits(x).softmax(dim=1).argmax(dim=1)


def to_binary_logits(logits: torch.Tensor, size=None, align_corners: bool = False) -> torch.Tensor:
    """Đổi logits 2 lớp (B, 2, h, w) thành logit nhị phân (B, 1, H, W) theo quy ước chung của nhóm.

    softmax([l0, l1])[1] == sigmoid(l1 - l0), nên dùng được với các hàm metric nhận logit 1 kênh.
    """
    if logits.shape[1] != 2:
        raise ValueError(f"Cần logits 2 lớp, nhận {logits.shape[1]} lớp")
    if size is not None and tuple(logits.shape[2:]) != tuple(size):
        logits = resize(logits, size, align_corners=align_corners)
    return logits[:, 1:2] - logits[:, 0:1]


def build_segformer(cfg: dict) -> SegFormer:
    """Dựng SegFormer từ cấu hình `load_model_config('segformer_b2')`. Không nạp trọng số pretrained."""
    common, bb, head = cfg["common"], cfg["backbone"], cfg["decode_head"]
    backbone = MixVisionTransformer(
        in_chans=common["in_channels"],
        embed_dims=bb["embed_dims"],
        num_heads=bb["num_heads"],
        mlp_ratios=bb["mlp_ratios"],
        qkv_bias=bb["qkv_bias"],
        qk_scale=bb["qk_scale"],
        drop_rate=bb["drop_rate"],
        attn_drop_rate=bb["attn_drop_rate"],
        drop_path_rate=bb["drop_path_rate"],
        norm_eps=bb["norm_eps"],
        depths=bb["depths"],
        sr_ratios=bb["sr_ratios"],
    )
    decode_head = SegFormerHead(
        in_channels=head["in_channels"],
        in_index=head["in_index"],
        feature_strides=head["feature_strides"],
        embed_dim=head["embed_dim"],
        num_classes=common["num_classes"],
        dropout_ratio=head["dropout_ratio"],
        align_corners=head["align_corners"],
    )
    return SegFormer(backbone, decode_head)


def load_mit_pretrained(backbone: MixVisionTransformer, path: str | Path) -> dict:
    """Nạp checkpoint ImageNet của MiT (vd. `mit_b2.pth` do tác giả SegFormer phát hành).

    Bỏ qua khóa `head.*` (lớp phân loại ImageNet), chấp nhận checkpoint có bọc `state_dict`
    hoặc tiền tố `backbone.`/`module.`. Báo lỗi nếu backbone thiếu bất kỳ tham số nào
    (bản gốc dùng strict=False và im lặng; ở đây chặn để tránh chạy nhầm với backbone ngẫu nhiên).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Không thấy {path}. Tải mit_b2.pth (ImageNet-1K) từ link trong README của SegFormer "
            "(Google Drive/OneDrive) rồi đặt vào thư mục pretrained/."
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt.get("state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
    clean = {}
    for k, v in state.items():
        for prefix in ("module.", "backbone."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        if not k.startswith("head."):
            clean[k] = v
    result = backbone.load_state_dict(clean, strict=False)
    if result.missing_keys:
        raise RuntimeError(f"Checkpoint thiếu {len(result.missing_keys)} tham số backbone, vd. {result.missing_keys[:5]}")
    return {"missing_keys": result.missing_keys, "unexpected_keys": result.unexpected_keys, "loaded": len(clean)}
