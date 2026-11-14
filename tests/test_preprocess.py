"""Kiểm thử tiền xử lý ảnh nội soi (src/medseg/data/preprocess.py). Chạy: pytest -q"""

import cv2
import numpy as np
import pytest

from medseg.config import load_data_config
from medseg.data.dataset import processed_root
from medseg.data.io import read_image
from medseg.data.preprocess import (
    PreprocessParams,
    STEPS,
    highlight_mask,
    lab_stats,
    normalize_color,
    preprocess,
    resolve_preset,
    valid_mask,
)
from medseg.paths import ROOT

H, W = 300, 400
TISSUE = (190, 110, 95)  # hồng đỏ, giống mô nội soi


def octagon(cut: int = 50) -> np.ndarray:
    pts = np.array([[cut, 0], [W - cut, 0], [W, cut], [W, H - cut], [W - cut, H], [cut, H], [0, H - cut], [0, cut]])
    m = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(m, pts.astype(np.int32), 1)
    return m.astype(bool)


def endoscope(frame: int = 0, box: str | None = None, seed: int = 0) -> np.ndarray:
    """Ảnh giả: mô trong khung bát giác, khung có mức xám `frame`, có thể có ô ScopeGuide."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), frame, np.uint8)
    noise = rng.normal(0, 6, (H, W, 3))
    tissue = np.clip(np.array(TISSUE) + noise, 0, 255).astype(np.uint8)
    fov = octagon()
    img[fov] = tissue[fov]
    if box == "black":
        img[H - 90:, :120] = 0
    elif box == "green":
        img[H - 100:, :130] = 0  # viền đen
        img[H - 90:H - 10, 10:120] = (20, 160, 140)  # nền xanh
        cv2.putText(img, "Fh:A1", (15, H - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img


def test_frame_is_outside_fov_and_zeroed():
    img = endoscope(frame=15)
    valid, ov = valid_mask(img)
    fov = octagon()
    inner = cv2.erode(fov.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    assert valid[inner].all()
    assert not valid[~fov].any()
    out, info = preprocess(img, ["mask_outside"])
    assert (out[~fov] == 0).all()  # khung CVC (15) và Kvasir (0) về cùng giá trị
    assert not info["overlay_box"]


@pytest.mark.parametrize("box", ["black", "green"])
def test_scopeguide_box_detected(box):
    img = endoscope(box=box)
    valid, ov = valid_mask(img)
    assert ov >= PreprocessParams().overlay_min
    assert not valid[H - 60:H - 20, 30:100].any()  # lòng ô bị loại, kể cả chữ trong ô xanh
    out, _ = preprocess(img, ["mask_outside"])
    assert (out[H - 60:H - 20, 30:100] == 0).all()


def test_dark_lumen_is_not_overlay():
    # lòng ruột tối (giá trị 20) sát góc dưới trái: không phải ô ScopeGuide
    img = endoscope(frame=15)
    cv2.circle(img, (70, H - 70), 45, (20, 18, 16), -1)
    _, ov = valid_mask(img)
    assert ov < PreprocessParams().overlay_min


def test_small_highlights_removed_large_bright_region_kept():
    img = endoscope()
    for x, y in [(150, 100), (220, 140), (260, 200)]:
        cv2.circle(img, (x, y), 3, (255, 255, 255), -1)
    img[80:150, 280:350] = (240, 236, 232)  # polyp nhạt màu, 4% ảnh > highlight_max_area
    valid, _ = valid_mask(img)
    hl = highlight_mask(img, valid)
    assert hl[100, 150] and hl[140, 220] and hl[200, 260]
    assert not hl[115, 315]
    out, info = preprocess(img, ["mask_outside", "inpaint_highlights"])
    assert out[100, 150].min() < 230  # đã inpaint bằng màu mô
    assert np.array_equal(out[90:140, 290:340], img[90:140, 290:340])
    assert 0 < info["highlight_frac"] < 0.05


def test_normalize_color_hits_target_and_keeps_white_neutral():
    p = PreprocessParams()
    img = endoscope()
    # độ sáng thay đổi theo chiều ngang như ảnh thật (L có độ lệch chuẩn đáng kể)
    shade = np.linspace(0.45, 1.0, W)[None, :, None]
    img = (img.astype(np.float32) * shade).astype(np.uint8)
    valid, _ = valid_mask(img)
    dark = (img.astype(np.float32) * 0.7).astype(np.uint8)  # ảnh tối hơn kiểu CVC
    s = lab_stats(normalize_color(dark, valid, p), valid)
    assert abs(s["L_mean"] - p.color_target["L_mean"]) < 3
    assert abs(s["L_std"] - p.color_target["L_std"]) < 3
    dark[120:160, 180:220] = (245, 245, 245)  # vùng trắng (polyp nhạt / phản xạ còn sót)
    out = normalize_color(dark, valid, p)
    lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)[140, 200].astype(int)
    assert abs(lab[1] - 128) <= 4 and abs(lab[2] - 128) <= 4  # trắng vẫn trắng


def test_preprocess_none_is_identity_and_rejects_unknown():
    img = endoscope(frame=15)
    out, info = preprocess(img, [])
    assert out is img and info == {}
    with pytest.raises(ValueError):
        preprocess(img, ["sharpen"])
    assert resolve_preset("none", {}) == []
    with pytest.raises(ValueError):
        resolve_preset("khong_co", {"clean": ["mask_outside"]})


def test_config_presets_are_valid():
    cfg = load_data_config()
    assert cfg["preprocess"] == "none"  # mặc định giữ giao thức FCBFormer
    for name, steps in cfg["preprocess_presets"].items():
        assert set(steps) <= set(STEPS), name
    PreprocessParams.from_config(cfg["preprocess_params"])  # không có khóa lạ


def test_processed_root_by_preset():
    assert processed_root("kvasir_seg", 256).name == "256"
    assert processed_root("kvasir_seg", 256, preprocess="clean").name == "256_clean"


@pytest.mark.parametrize("rel,expect_box", [
    ("data/raw/kvasir_seg/Kvasir-SEG/images/cjyzjzssvd8pq0838f4nolj5l.jpg", True),   # ô xanh
    ("data/raw/kvasir_seg/Kvasir-SEG/images/cju6xa0qmvzun0818xjukgncj.jpg", True),   # ô đen hình bậc thang
    ("data/raw/kvasir_seg/Kvasir-SEG/images/cju5xjn5mm78b09871spyqhhr.jpg", False),
    ("data/raw/cvc_clinicdb/CVC-ClinicDB/Original/302.tif", False),                 # lòng ruột tối ở góc
])
def test_real_overlay_detection(rel, expect_box):
    path = ROOT / rel
    if not path.exists():
        pytest.skip("chưa tải dữ liệu")
    _, ov = valid_mask(read_image(path))
    assert (ov >= PreprocessParams().overlay_min) == expect_box
