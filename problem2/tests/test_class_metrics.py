import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from class_metrics import metrics_by_label


class LabelMetricsTests(unittest.TestCase):
    def test_one_vs_rest_and_true_label_mae(self):
        predictions = {
            "true_class": [0, 0, 1, 2],
            "class": [0, 1, 1, 2],
            "true_score": [-1.0, -2.0, 0.0, 1.0],
            "score": [-1.2, -1.0, 0.3, 0.8],
        }
        rows = metrics_by_label(predictions, "test")
        negative = rows[0]
        self.assertEqual(negative["n"], 2)
        self.assertAlmostEqual(negative["class_accuracy_recall"], 0.5)
        self.assertAlmostEqual(negative["one_vs_rest_accuracy"], 0.75)
        self.assertAlmostEqual(negative["class_f1"], 2 / 3)
        self.assertAlmostEqual(negative["mae_true_class"], 0.6)


if __name__ == "__main__":
    unittest.main()
