import math

import numpy as np
import pytest

from medseg.evaluation import hd95, segmentation_metrics


def square(size=20, y=5, x=5, s=8):
    m = np.zeros((size, size), bool)
    m[y:y + s, x:x + s] = True
    return m


def test_perfect_match():
    m = segmentation_metrics(square(), square())
    assert m["dice"] == m["iou"] == m["precision"] == m["recall"] == 1.0
    assert m["hd95"] == 0.0 and not m["empty_pred"]


def test_both_empty_counts_as_match():
    z = np.zeros((10, 10), bool)
    m = segmentation_metrics(z, z)
    assert m["dice"] == 1.0 and m["hd95"] == 0.0 and m["empty_pred"]


def test_empty_prediction():
    m = segmentation_metrics(np.zeros((20, 20), bool), square())
    assert m["dice"] == 0.0 and m["recall"] == 0.0 and math.isnan(m["hd95"])


def test_half_overlap():
    gt = np.zeros((10, 10), bool)
    gt[:, :4] = True
    pred = np.zeros((10, 10), bool)
    pred[:, 2:6] = True
    m = segmentation_metrics(pred, gt)
    assert m["dice"] == pytest.approx(0.5)
    assert m["iou"] == pytest.approx(1 / 3)


def test_hd95_of_shifted_square():
    assert hd95(square(x=8), square(x=5)) == pytest.approx(3.0)
