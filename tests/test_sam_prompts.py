import numpy as np

from medseg.sam.prompts import center_point, jitter_box, point_set, regions


def blob_mask():
    m = np.zeros((100, 120), np.uint8)
    m[20:60, 30:80] = 1      # tổn thương chính
    m[80:83, 100:103] = 1    # chấm nhiễu 9 px < 0.1% của 12000 px
    return m


def test_regions_drop_specks():
    regs = regions(blob_mask(), min_area=0.001)
    assert len(regs) == 1
    assert regs[0].box.tolist() == [30, 20, 79, 59]


def test_regions_keep_largest_if_all_small():
    m = np.zeros((100, 100), np.uint8)
    m[0:2, 0:2] = 1
    m[50:53, 50:53] = 1
    regs = regions(m, min_area=0.5)
    assert len(regs) == 1 and regs[0].mask.sum() == 9


def test_regions_empty_mask():
    assert regions(np.zeros((10, 10), np.uint8)) == []


def test_center_point_inside_and_central():
    (x, y), = center_point(blob_mask() == 1).astype(int)
    assert blob_mask()[y, x] == 1
    assert 45 <= x <= 64 and 30 <= y <= 49


def test_point_set_labels_and_locations():
    m = blob_mask()
    reg = regions(m)[0]
    coords, labels = point_set(reg, m, n_pos=3, n_neg=3, margin=0.5, rng=np.random.default_rng(0))
    assert labels.tolist() == [1, 1, 1, 0, 0, 0]
    xs, ys = coords[:, 0].astype(int), coords[:, 1].astype(int)
    assert (m[ys[labels == 1], xs[labels == 1]] == 1).all()
    assert (m[ys[labels == 0], xs[labels == 0]] == 0).all()


def test_point_set_is_deterministic():
    m = blob_mask()
    reg = regions(m)[0]
    a = point_set(reg, m, 3, 3, 0.5, np.random.default_rng(7))
    b = point_set(reg, m, 3, 3, 0.5, np.random.default_rng(7))
    assert np.array_equal(a[0], b[0])


def test_jitter_box_bounds():
    box = np.array([30, 20, 79, 59], np.float32)
    rng = np.random.default_rng(0)
    for _ in range(200):
        j = jitter_box(box, (100, 120), rng, max_px=20)
        assert np.all(np.abs(j - box) <= 20 + 1e-4)
        assert 0 <= j[0] < j[2] <= 119 and 0 <= j[1] < j[3] <= 99


def test_jitter_box_relative():
    box = np.array([30, 20, 79, 59], np.float32)
    j = jitter_box(box, (100, 120), np.random.default_rng(1), max_rel=0.1)
    assert np.all(np.abs(j - box) <= np.array([5, 4, 5, 4]) + 1e-4)
