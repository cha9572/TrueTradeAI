"""Metrics are correct on absent classes/types and threshold edge cases."""

import unittest
import numpy as np
from src.utils import binary_metrics, choose_threshold


class MetricTests(unittest.TestCase):
    def test_type_recall_and_matrix(self):
        result = binary_metrics([0, 1, 1, 1], [.1, .9, .2, .8], .5, ["real", "A", "B", "C"])
        self.assertEqual(result["confusion_matrix"], [[1, 0], [1, 2]])
        self.assertEqual(result["fake_type_recall"]["A"]["recall"], 1)
        self.assertEqual(result["fake_type_recall"]["B"]["recall"], 0)
        self.assertAlmostEqual(result["f1"], .8)

    def test_single_class_auc_and_absent_type(self):
        result = binary_metrics([0, 0], [.1, .8], .5, ["real", "real"])
        self.assertIsNone(result["roc_auc"])
        self.assertIsNone(result["fake_type_recall"]["A"]["recall"])
        self.assertEqual(result["confusion_matrix"], [[1, 1], [0, 0]])

    def test_validation_threshold_exact_optimum(self):
        labels = [0, 1, 0, 1, 1]
        scores = [.9, .8, .1, .2, .8]
        chosen = choose_threshold(labels, scores)
        f1 = binary_metrics(labels, scores, chosen)["f1"]
        expected = max(binary_metrics(labels, scores, float(t))["f1"] for t in np.unique([*scores, .5]))
        self.assertEqual(f1, expected)

    def test_threshold_tie_prefers_half(self):
        self.assertEqual(choose_threshold([0, 1], [.1, .9]), .5)

    def test_threshold_needs_both_classes(self):
        with self.assertRaisesRegex(ValueError, "both"):
            choose_threshold([1, 1], [.1, .9])


if __name__ == "__main__":
    unittest.main()
