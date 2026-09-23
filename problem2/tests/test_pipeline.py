import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment import block_availability, metric_values
from model import RobustFusion


class Problem2Tests(unittest.TestCase):
    def test_contiguous_mask_never_hides_padding_or_quality_failures(self):
        batch = {
            "padding": torch.ones(1, 50, 3, dtype=torch.bool),
            "quality": torch.ones(1, 50, 3, dtype=torch.bool),
            "availability": torch.ones(1, 50, 3, dtype=torch.bool),
        }
        batch["padding"][:, 20:] = False
        batch["quality"][:, 5, 1] = False
        changed = block_availability(batch, .3, "middle", "A", np.random.default_rng(1))
        hidden = batch["availability"] & ~changed
        self.assertTrue(torch.all(hidden <= (batch["padding"] & batch["quality"])))
        self.assertEqual(int(hidden[:, :, 0].sum()), 0)
        self.assertEqual(int(hidden[:, :, 2].sum()), 0)
        indices = torch.where(hidden[0, :, 1])[0]
        self.assertEqual(int(indices[-1] - indices[0] + 1), len(indices))

    def test_full_quality_failure_produces_finite_predictions(self):
        model = RobustFusion(hidden=32, heads=4)
        batch = {
            "text": torch.randn(2, 50, 768),
            "audio": torch.randn(2, 50, 74),
            "vision": torch.zeros(2, 50, 35),
            "padding": torch.ones(2, 50, 3, dtype=torch.bool),
            "quality": torch.ones(2, 50, 3, dtype=torch.bool),
            "availability": torch.ones(2, 50, 3, dtype=torch.bool),
        }
        batch["padding"][:, 15:] = False
        batch["quality"][:, :, 2] = False
        output = model(batch)
        self.assertTrue(torch.isfinite(output["logits"]).all())
        self.assertTrue(torch.isfinite(output["score"]).all())
        self.assertTrue(torch.all(output["score"].abs() <= 3))

    def test_metrics_three_class_and_regression(self):
        out = metric_values(np.array([0, 1, 2]), np.array([-1., 0., 1.]),
                            np.array([0, 1, 2]), np.array([-1., 0., 1.]))
        self.assertEqual(out["macro_f1"], 1.0)
        self.assertEqual(out["mae"], 0.0)
        self.assertAlmostEqual(out["pearson"], 1.0)


if __name__ == "__main__":
    unittest.main()
