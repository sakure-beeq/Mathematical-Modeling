from __future__ import annotations

import unittest

import numpy as np

from mmer_preprocess.pipeline import DatasetRole, preprocess_aligned_split
from mmer_preprocess.stats import fit_feature_stats, normalize_features


def synthetic_source() -> dict[str, np.ndarray]:
    text_bert = np.zeros((1, 3, 6), dtype=np.float32)
    text_bert[0, 0] = [101, 10, 11, 12, 102, 0]
    text_bert[0, 1] = [1, 1, 1, 1, 1, 0]

    audio = np.zeros((1, 6, 2), dtype=np.float64)
    audio[0, 1] = [1, 2]
    audio[0, 3] = [3, 4]

    vision = np.zeros((1, 6, 2), dtype=np.float64)
    vision[0, 1] = [10, 20]
    vision[0, 2] = [30, 40]
    vision[0, 3] = [50, 60]
    return {
        "id": np.asarray(["video$_$clip"]),
        "raw_text": np.asarray(["small synthetic sample"]),
        "text_bert": text_bert,
        "audio": audio,
        "vision": vision,
        "classification_labels": np.asarray([2.0]),
        "regression_labels": np.asarray([0.5]),
    }


class PreprocessTests(unittest.TestCase):
    def test_standard_zero_row_is_quality_failure(self) -> None:
        result = preprocess_aligned_split(synthetic_source(), DatasetRole.STANDARD)
        audio_masks = result["masks"]["audio"]
        np.testing.assert_array_equal(
            audio_masks["padding"][0], [False, True, True, True, False, False]
        )
        self.assertTrue(audio_masks["availability"][0, 2])
        self.assertFalse(audio_masks["quality"][0, 2])
        self.assertFalse(audio_masks["effective"][0, 2])
        self.assertEqual(result["text_bert"].dtype, np.int64)
        self.assertEqual(result["audio"].dtype, np.float32)
        self.assertEqual(result["classification_labels"].dtype, np.int64)
        self.assertEqual(result["regression_labels"].dtype, np.float32)

    def test_missing_test_zero_row_is_unavailable(self) -> None:
        result = preprocess_aligned_split(synthetic_source(), DatasetRole.MISSING_TEST)
        audio_masks = result["masks"]["audio"]
        self.assertFalse(audio_masks["availability"][0, 2])
        self.assertTrue(audio_masks["quality"][0, 2])
        self.assertFalse(audio_masks["effective"][0, 2])

    def test_training_stats_ignore_masked_rows_and_invalid_rows_stay_zero(self) -> None:
        source = synthetic_source()
        unscaled = preprocess_aligned_split(source, DatasetRole.STANDARD)
        mask = unscaled["masks"]["audio"]["effective"]
        stats = fit_feature_stats(unscaled["audio"], mask)
        np.testing.assert_allclose(stats.mean, [2.0, 3.0])
        normalized = normalize_features(unscaled["audio"], mask, stats)
        np.testing.assert_array_equal(normalized[0, 2], [0.0, 0.0])
        np.testing.assert_allclose(normalized[0, 1], [-1.0, -1.0])
        np.testing.assert_allclose(normalized[0, 3], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()

