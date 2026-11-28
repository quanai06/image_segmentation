"""Chạy SAM zero-shot với các kiểu gợi ý P0–P4 trên một ảnh."""

import copy

import numpy as np
from segment_anything import SamAutomaticMaskGenerator, SamPredictor
from segment_anything.modeling import Sam

from medseg.sam.prompts import center_point, jitter_box, point_set, regions

PROTOCOLS = (
    "auto_oracle",     # P0: không gợi ý, chọn mặt nạ khớp nhãn nhất
    "auto_top",        # P0: không gợi ý, chọn theo IoU SAM tự dự đoán
    "point_top",       # P1: 1 điểm tâm, chọn theo IoU dự đoán
    "point_oracle",    # P1: 1 điểm tâm, chọn mặt nạ khớp nhãn nhất
    "points",          # P2: điểm dương + âm
    "box",             # P3: hộp chính xác
    "box_jitter_px",   # P4a: hộp lệch tối đa N pixel
    "box_jitter_rel",  # P4b: hộp lệch tối đa N% kích thước
)
AUTO = {"auto_oracle", "auto_top"}


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


class SamZeroShot:
    def __init__(self, sam: Sam, cfg: dict):
        self.sam = sam
        self.prompt_cfg = cfg["prompts"]
        self.auto_cfg = cfg["automatic"]
        self.predictor = SamPredictor(sam)
        self._generator = None

    @property
    def generator(self) -> SamAutomaticMaskGenerator:
        if self._generator is None:
            sam = self.sam
            if sam.device.type == "mps":  # mask generator dùng float64, MPS không hỗ trợ
                sam = copy.deepcopy(sam).to("cpu")
            self._generator = SamAutomaticMaskGenerator(sam, **self.auto_cfg)
        return self._generator

    def predict(self, image: np.ndarray, prompt_mask: np.ndarray, protocols=PROTOCOLS, seed: int = 0) -> dict[str, np.ndarray]:
        """image: RGB uint8 (H, W, 3); prompt_mask: mặt nạ (H, W) dùng để sinh gợi ý.

        Trả về {kiểu gợi ý: mặt nạ bool (H, W)}. Ảnh nhiều tổn thương: mỗi vùng một gợi ý, hợp kết quả.
        """
        unknown = set(protocols) - set(PROTOCOLS)
        if unknown:
            raise ValueError(f"Kiểu gợi ý không hợp lệ: {sorted(unknown)}")
        gt = prompt_mask.astype(bool)
        out = {}
        prompted = [p for p in protocols if p not in AUTO]
        if prompted:
            self.predictor.set_image(image)
            out.update(self._prompted(gt, prompted, seed))
        if AUTO & set(protocols):
            out.update(self._automatic(image, gt, protocols))
        return {p: out[p] for p in protocols}

    def _single(self, **prompt) -> np.ndarray:
        masks, _, _ = self.predictor.predict(multimask_output=False, **prompt)
        return masks[0]

    def _prompted(self, gt: np.ndarray, protocols: list[str], seed: int) -> dict[str, np.ndarray]:
        cfg = self.prompt_cfg
        rng = {p: np.random.default_rng([seed, PROTOCOLS.index(p)]) for p in protocols}
        out = {p: np.zeros_like(gt) for p in protocols}
        for r in regions(gt, cfg["min_region_area"]):
            if {"point_top", "point_oracle"} & set(protocols):
                masks, scores, _ = self.predictor.predict(
                    point_coords=center_point(r.mask), point_labels=np.array([1]), multimask_output=True
                )
                if "point_top" in out:
                    out["point_top"] |= masks[np.argmax(scores)]
                if "point_oracle" in out:
                    out["point_oracle"] |= masks[np.argmax([iou(m, r.mask) for m in masks])]
            if "points" in out:
                coords, labels = point_set(r, gt, cfg["n_pos"], cfg["n_neg"], cfg["neg_margin"], rng["points"])
                out["points"] |= self._single(point_coords=coords, point_labels=labels)
            if "box" in out:
                out["box"] |= self._single(box=r.box)
            if "box_jitter_px" in out:
                box = jitter_box(r.box, gt.shape, rng["box_jitter_px"], max_px=cfg["jitter_px"])
                out["box_jitter_px"] |= self._single(box=box)
            if "box_jitter_rel" in out:
                box = jitter_box(r.box, gt.shape, rng["box_jitter_rel"], max_rel=cfg["jitter_rel"])
                out["box_jitter_rel"] |= self._single(box=box)
        return out

    def _automatic(self, image: np.ndarray, gt: np.ndarray, protocols) -> dict[str, np.ndarray]:
        anns = self.generator.generate(image)
        empty = np.zeros_like(gt)
        if not anns:
            return {p: empty for p in AUTO & set(protocols)}
        masks = [a["segmentation"] for a in anns]
        return {
            "auto_oracle": masks[int(np.argmax([iou(m, gt) for m in masks]))],
            "auto_top": masks[int(np.argmax([a["predicted_iou"] for a in anns]))],
        }
