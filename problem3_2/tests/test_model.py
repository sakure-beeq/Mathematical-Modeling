import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from explain import exact_shapley  # noqa: E402
from model import InteractionModel  # noqa: E402


class InteractionModelTests(unittest.TestCase):
    def test_shapley_efficiency_with_interactions(self):
        values = np.array([.2, .6, .1, 1.8, .3, .8, .7, 2.5])
        contributions = exact_shapley(values)
        self.assertAlmostEqual(float(contributions.sum()), values[7] - values[0])

    def test_hidden_modality_cannot_affect_prediction(self):
        torch.manual_seed(1)
        model = InteractionModel(hidden=16, dropout=0).eval()
        batch = {"text": torch.randn(2, 50, 768),
                 "audio": torch.randn(2, 50, 74),
                 "vision": torch.randn(2, 50, 35),
                 "padding": torch.ones(2, 50, 3, dtype=torch.bool),
                 "quality": torch.ones(2, 50, 3, dtype=torch.bool),
                 "availability": torch.ones(2, 50, 3, dtype=torch.bool)}
        mask = batch["availability"].clone()
        mask[:, :, 2] = False
        first = model(batch, mask)
        batch["vision"] = torch.randn_like(batch["vision"]) * 100
        second = model(batch, mask)
        torch.testing.assert_close(first["logits"], second["logits"])
        torch.testing.assert_close(first["score"], second["score"])

    def test_all_missing_predictions_are_finite(self):
        model = InteractionModel(hidden=16, dropout=0).eval()
        batch = {"text": torch.zeros(1, 50, 768),
                 "audio": torch.zeros(1, 50, 74),
                 "vision": torch.zeros(1, 50, 35),
                 "padding": torch.zeros(1, 50, 3, dtype=torch.bool),
                 "quality": torch.zeros(1, 50, 3, dtype=torch.bool),
                 "availability": torch.zeros(1, 50, 3, dtype=torch.bool)}
        out = model(batch)
        self.assertTrue(torch.isfinite(out["logits"]).all())
        self.assertTrue(torch.isfinite(out["score"]).all())

    def test_one_call_returns_all_required_problem3_outputs(self):
        torch.manual_seed(3)
        model = InteractionModel(hidden=16, dropout=0).eval()
        active = torch.zeros(50, 3, dtype=torch.bool)
        active[1:3] = True
        sample = {"id": "example", "text": torch.randn(50, 768),
                  "audio": torch.randn(50, 74), "vision": torch.randn(50, 35),
                  "padding": active, "quality": active,
                  "availability": active}
        times = [None] * 50
        times[1:3] = [[0.0, 0.5], [0.5, 1.0]]
        offsets = [[0, 0]] * 50
        offsets[1:3] = [[0, 5], [6, 11]]
        timeline = {"raw_text": "hello world", "token_times": times,
                    "token_offsets": offsets}
        result = model.predict_with_explanation(sample, timeline)
        self.assertIn(result["pred_class"], ("Negative", "Neutral", "Positive"))
        self.assertTrue(-3 <= result["pred_score"] <= 3)
        self.assertIn(result["main_modality"], ("text", "audio", "vision"))
        self.assertEqual(set(result["shapley_margin"]), {"text", "audio", "vision"})
        self.assertAlmostEqual(sum(result["importance_share"].values()), 1.0)
        self.assertEqual(result["windows"]["text"]["location"]["text"], "hello world")
        self.assertEqual(result["key_text"], "hello world")
        self.assertEqual(result["key_audio_start_sec"], 0.0)
        self.assertEqual(result["key_vision_frame_sec"], 0.5)
        self.assertEqual(result["windows"]["audio"]["location"]["start_sec"], 0.0)
        self.assertEqual(result["windows"]["vision"]["location"]["end_sec"], 1.0)


if __name__ == "__main__":
    unittest.main()
