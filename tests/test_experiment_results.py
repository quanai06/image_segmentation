"""Kiểm tra ghép ảnh và ID chia tập, không cần dữ liệu ảnh hoặc weights."""

import csv
from pathlib import Path
import tempfile
import unittest

from scripts.compare_experiments import compare_splits, paired_results, read_splits


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class TestExperimentResults(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[1] / "outputs"
        base.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.folder = Path(self.temp.name).resolve()
        assert self.folder.is_relative_to(base.resolve())
        self.addCleanup(self.temp.cleanup)

    def test_pairing_uses_names_instead_of_row_positions(self):
        a, b = self.folder / "a.csv", self.folder / "b.csv"
        write_rows(a, [{"name": "a", "dice": .5, "iou": .3, "gt_pixels": 12},
                       {"name": "b", "dice": .9, "iou": .8, "gt_pixels": 14}])
        write_rows(b, [{"name": "b", "dice": .8, "iou": .7, "gt_pixels": 14},
                       {"name": "a", "dice": .7, "iou": .5, "gt_pixels": 12}])
        rows = paired_results(a, b, 2)
        self.assertEqual([r["name"] for r in rows], ["a", "b"])
        self.assertAlmostEqual(rows[0]["delta_dice"], .2)
        self.assertAlmostEqual(rows[1]["delta_dice"], -.1)
        with self.assertRaises(ValueError):
            paired_results(a, b, 100)

    def test_pairing_rejects_missing_ids_and_different_masks(self):
        a, b = self.folder / "a.csv", self.folder / "b.csv"
        write_rows(a, [{"name": "a", "dice": .5, "iou": .3, "gt_pixels": 12}])
        write_rows(b, [{"name": "b", "dice": .7, "iou": .5, "gt_pixels": 12}])
        with self.assertRaises(ValueError):
            paired_results(a, b, 1)
        write_rows(b, [{"name": "a", "dice": .7, "iou": .5, "gt_pixels": 13}])
        with self.assertRaises(ValueError):
            paired_results(a, b, 1)

    def test_split_comparison_checks_membership_not_only_counts(self):
        a, b = self.folder / "a.csv", self.folder / "b.csv"
        rows = [{"name": "a", "split": "train"}, {"name": "b", "split": "val"},
                {"name": "c", "split": "test"}]
        write_rows(a, rows)
        write_rows(b, [rows[2], {"name": "b", "split": "validation"}, rows[0]])
        self.assertTrue(compare_splits(a, b)["match"])
        write_rows(b, [{"name": "a", "split": "validation"},
                       {"name": "b", "split": "train"}, rows[2]])
        result = compare_splits(a, b)
        self.assertFalse(result["match"])
        self.assertEqual(result["changed"], ["a", "b"])

    def test_split_reader_rejects_duplicate_ids(self):
        path = self.folder / "bad.csv"
        write_rows(path, [{"name": "a", "split": "train"}, {"name": "a", "split": "test"}])
        with self.assertRaises(ValueError):
            read_splits(path)


if __name__ == "__main__":
    unittest.main()
