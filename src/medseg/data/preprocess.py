"""Tiền xử lý ảnh nội soi (tùy chọn, mặc định tắt), áp trên ảnh gốc trước khi resize.

Lý do và số liệu: report/report.md mục 7.7, scripts/analyze_artifacts.py.

Các bước, theo thứ tự:
- mask_outside: tô đen mọi pixel ngoài vùng nhìn thấy của ống soi (FOV) và trong ô
  ScopeGuide ở góc dưới trái của Kvasir-SEG. Khung của CVC-ClinicDB có giá trị 11-24 còn
  của Kvasir-SEG là 0, ô ScopeGuide lúc đen lúc xanh: đều là dấu hiệu nhận ra nguồn ảnh,
  không liên quan tới tổn thương.
- inpaint_highlights: xóa phản xạ sáng nhỏ và chữ trắng in đè lên mô (inpaint TELEA).
  Vùng sáng lớn (vd. polyp màu nhạt) được giữ nguyên nhờ giới hạn diện tích.
- normalize_color: chuẩn hóa màu trong FOV theo LAB (độ sáng L: trung bình và độ lệch
  chuẩn; a, b: trung bình) về một giá trị đích cố định. Kvasir-SEG sáng hơn CVC-ClinicDB rõ.

Mặt nạ không bị thay đổi. Preset "none" giữ nguyên ảnh (giao thức FCBFormer).
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

STEPS = ("mask_outside", "inpaint_highlights", "normalize_color")


@dataclass
class PreprocessParams:
    dark_thr: int = 30  # pixel có kênh lớn nhất <= ngưỡng là nền tối (khung CVC tới 24)
    green_margin: int = 20  # G > R + margin: màu xanh của ô ScopeGuide, mô nội soi luôn đỏ trội
    min_part: float = 0.01  # mảnh mô nhỏ hơn tỷ lệ này của ảnh không dùng để dựng FOV
    overlay_dark: int = 8  # overlay của Kvasir-SEG đen tuyệt đối; lòng ruột và khung CVC (11-24) sáng hơn
    overlay_min: float = 0.02  # overlay chiếm >= tỷ lệ này của ảnh bên trong FOV: coi là có ô ScopeGuide
    highlight_v: int = 220  # phản xạ sáng: V >= highlight_v và S <= highlight_s (HSV, 0-255)
    highlight_s: int = 40
    highlight_dilate: float = 0.02  # nới mặt nạ phản xạ (tỷ lệ cạnh ngắn của ảnh)
    highlight_max_area: float = 0.005  # bỏ vùng sáng lớn hơn tỷ lệ này của ảnh
    # trung vị tập train Kvasir-SEG (LAB thang OpenCV), xem configs/data.yaml
    color_target: dict = field(default_factory=lambda: {"L_mean": 124.0, "L_std": 45.6, "a_mean": 154.5, "b_mean": 155.0})

    @classmethod
    def from_config(cls, cfg: dict | None) -> "PreprocessParams":
        return cls(**(cfg or {}))


def _odd(n: float) -> int:
    n = max(3, int(round(n)))
    return n if n % 2 else n + 1


def _green(img: np.ndarray, p: PreprocessParams) -> np.ndarray:
    r, g = img[..., 0].astype(np.int16), img[..., 1].astype(np.int16)
    return g > r + p.green_margin


def tissue_mask(img: np.ndarray, p: PreprocessParams = PreprocessParams()) -> np.ndarray:
    """Pixel thuộc mô: không tối và không xanh (ô ScopeGuide). bool (H, W)."""
    return (img.max(axis=2) > p.dark_thr) & ~_green(img, p)


def fov_mask(img: np.ndarray, p: PreprocessParams = PreprocessParams()) -> np.ndarray:
    """Vùng nhìn thấy của ống soi: bao lồi các mảng mô lớn. bool (H, W).

    Dùng bao lồi để lòng ruột tối hay chữ đè lên mô không làm thủng FOV.
    """
    h, w = img.shape[:2]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(0.01 * min(h, w)),) * 2)
    t = cv2.morphologyEx(tissue_mask(img, p).astype(np.uint8), cv2.MORPH_OPEN, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(t)
    big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= p.min_part * h * w]
    if not big:
        return np.ones((h, w), bool)
    pts = cv2.findNonZero(np.isin(lab, big).astype(np.uint8))
    fov = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(fov, cv2.convexHull(pts), 1)
    return fov.astype(bool)


def overlay_region(img: np.ndarray, p: PreprocessParams = PreprocessParams()) -> np.ndarray:
    """Vùng đen tuyệt đối hoặc xanh nối với góc dưới trái, trong góc phần tư dưới trái. bool (H, W).

    Bắt được ô ScopeGuide của Kvasir-SEG dù ô đen áp sát góc, ô đen hình bậc thang hay ô
    xanh có viền đen và chữ bên trong (đóng hình thái học + lấp lỗ). Góc cắt chéo của khung
    bát giác cũng rơi vào đây, nhưng phần đó vốn nằm ngoài FOV.
    """
    h, w = img.shape[:2]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(0.01 * min(h, w)),) * 2)
    nt = (img.max(axis=2) <= p.overlay_dark) | _green(img, p)
    nt = cv2.morphologyEx(nt.astype(np.uint8), cv2.MORPH_CLOSE, k)
    nt[: int(0.45 * h)] = 0
    nt[:, int(0.55 * w):] = 0
    _, lab = cv2.connectedComponents(nt)
    if lab[h - 1, 0] == 0:
        return np.zeros((h, w), bool)
    comp = (lab == lab[h - 1, 0]).astype(np.uint8)
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(comp, contours, -1, 1, thickness=cv2.FILLED)  # lấp lỗ (chữ trong ô xanh)
    return comp.astype(bool)


def valid_mask(img: np.ndarray, p: PreprocessParams = PreprocessParams()) -> tuple[np.ndarray, float]:
    """FOV trừ vùng overlay. Trả về (mask bool, tỷ lệ diện tích ảnh overlay chiếm bên trong FOV).

    Tỷ lệ overlay >= p.overlay_min là dấu hiệu ảnh có ô ScopeGuide.
    """
    fov = fov_mask(img, p)
    ov = overlay_region(img, p) & fov
    return fov & ~ov, float(ov.mean())


def highlight_mask(img: np.ndarray, region: np.ndarray, p: PreprocessParams = PreprocessParams()) -> np.ndarray:
    """Phản xạ sáng và chữ trắng: pixel rất sáng, ít bão hòa, thuộc vùng nhỏ. bool (H, W).

    Vùng sáng lớn hơn highlight_max_area được giữ nguyên (vd. polyp màu nhạt). Mặt nạ được
    nới rộng (highlight_dilate) qua khỏi quầng sáng đã mất màu do nén JPEG; nếu nới ít,
    inpaint lấy màu từ quầng đó và để lại đốm xám.
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 2] >= p.highlight_v) & (hsv[..., 1] <= p.highlight_s) & region).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = stats[:, cv2.CC_STAT_AREA] <= p.highlight_max_area * h * w
    keep[0] = False
    m = keep[lab].astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(p.highlight_dilate * min(h, w)),) * 2)
    return cv2.dilate(m, k).astype(bool) & region


def inpaint_highlights(img: np.ndarray, region: np.ndarray, p: PreprocessParams = PreprocessParams()) -> tuple[np.ndarray, np.ndarray]:
    m = highlight_mask(img, region, p)
    if not m.any():
        return img, m
    radius = max(3, int(round(0.01 * min(img.shape[:2]))))
    return cv2.inpaint(img, m.astype(np.uint8), radius, cv2.INPAINT_TELEA), m


def lab_stats(img: np.ndarray, region: np.ndarray) -> dict:
    """Thống kê màu LAB (thang uint8 của OpenCV) trong vùng."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[region].astype(np.float32)
    return {"L_mean": float(lab[:, 0].mean()), "L_std": float(lab[:, 0].std()),
            "a_mean": float(lab[:, 1].mean()), "b_mean": float(lab[:, 2].mean())}


def _chroma_scale(mean: float, target: float) -> float:
    """Hệ số co giãn sắc độ quanh điểm trung tính 128, giới hạn trong [0.5, 2]."""
    if abs(mean - 128) < 1:
        return 1.0
    return float(np.clip((target - 128) / (mean - 128), 0.5, 2.0))


def normalize_color(img: np.ndarray, region: np.ndarray, p: PreprocessParams = PreprocessParams()) -> np.ndarray:
    """Chuẩn hóa màu trong vùng về color_target (LAB thang OpenCV).

    L: đưa trung bình và độ lệch chuẩn về đích. a, b: co giãn quanh 128 (điểm trung tính)
    thay vì dịch, để vùng trắng (polyp nhạt màu, phản xạ còn sót) không bị nhuộm màu.
    """
    if region.sum() < 100:
        return img
    t = p.color_target
    s = lab_stats(img, region)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] = (lab[..., 0] - s["L_mean"]) / max(s["L_std"], 1.0) * t["L_std"] + t["L_mean"]
    lab[..., 1] = 128 + (lab[..., 1] - 128) * _chroma_scale(s["a_mean"], t["a_mean"])
    lab[..., 2] = 128 + (lab[..., 2] - 128) * _chroma_scale(s["b_mean"], t["b_mean"])
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    out[~region] = img[~region]
    return out


def preprocess(img: np.ndarray, steps: list[str] | tuple[str, ...], p: PreprocessParams = PreprocessParams()) -> tuple[np.ndarray, dict]:
    """Áp các bước theo thứ tự cố định STEPS. Trả về (ảnh RGB uint8, thông tin cho manifest)."""
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise ValueError(f"bước tiền xử lý không hợp lệ: {sorted(unknown)}; có: {STEPS}")
    if not steps:
        return img, {}
    valid, ov = valid_mask(img, p)
    info = {"valid_frac": round(float(valid.mean()), 6), "overlay_frac": round(ov, 6),
            "overlay_box": ov >= p.overlay_min, "highlight_frac": 0.0}
    out = img.copy()
    if "mask_outside" in steps:
        out[~valid] = 0
    if "inpaint_highlights" in steps:
        out, hl = inpaint_highlights(out, valid, p)
        info["highlight_frac"] = round(float(hl.mean()), 6)
    if "normalize_color" in steps:
        out = normalize_color(out, valid, p)
    return out, info


def resolve_preset(name: str, presets: dict) -> list[str]:
    """Tên preset -> danh sách bước. 'none' luôn là danh sách rỗng."""
    if name == "none":
        return []
    if name not in presets:
        raise ValueError(f"preset tiền xử lý không có: {name}; có: {['none', *presets]}")
    return list(presets[name])
