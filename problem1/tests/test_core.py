from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from problem1.alignment import is_digital_silence, parse_textgrid, safe_sample_name, uniform_word_intervals
from problem1.core import WordInterval, align_and_pad, interpolate_short_gaps, quality_weighted_pool
from problem1.audio_features import AUDIO_FEATURE_NAMES
from problem1.visual_features import VISUAL_FEATURE_NAMES


class CoreTests(unittest.TestCase):
    def test_overlap_and_quality_weighted_pool(self) -> None:
        times = np.asarray([0.25, 0.75, 1.25])
        frames = np.asarray([[1.0], [3.0], [100.0]], dtype=np.float32)
        quality = np.asarray([1.0, 0.5, 0.0], dtype=np.float32)
        words = [WordInterval("hello", 0.0, 1.0), WordInterval("world", 1.0, 1.5)]
        pooled, available, scores = quality_weighted_pool(times, frames, words, quality)
        self.assertAlmostEqual(float(pooled[0, 0]), (1.0 + 1.5) / 1.5, places=5)
        self.assertTrue(available[0])
        self.assertFalse(available[1])
        self.assertAlmostEqual(float(scores[0]), 0.75)
        self.assertEqual(float(pooled[1, 0]), 0.0)

    def test_only_short_bounded_gap_is_interpolated(self) -> None:
        times = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        values = np.asarray([[0.0], [1.0], [np.nan], [3.0], [np.nan], [np.nan]])
        quality = np.asarray([1, 1, 0, 1, 0, 0], dtype=np.float32)
        filled, q = interpolate_short_gaps(times, values, quality, max_gap_seconds=0.25)
        self.assertAlmostEqual(float(filled[2, 0]), 2.0)
        self.assertGreater(float(q[2]), 0.0)
        self.assertEqual(float(q[4]), 0.0)
        self.assertEqual(float(q[5]), 0.0)

    def test_nonfinite_frame_cannot_keep_positive_quality(self) -> None:
        times = np.asarray([0.0, 0.1])
        values = np.asarray([[np.nan], [1.0]], dtype=np.float32)
        _, q = interpolate_short_gaps(times, values, np.ones(2), max_gap_seconds=0.25)
        self.assertEqual(float(q[0]), 0.0)

    def test_contiguous_aggregation_preserves_end_of_sequence(self) -> None:
        words = [WordInterval(f"w{i}", float(i), float(i + 1)) for i in range(7)]
        text = np.arange(7, dtype=np.float32)[:, None]
        audio = text + 10
        vision = text + 20
        availability = np.ones((7, 3), dtype=bool)
        quality = np.ones((7, 3), dtype=np.float32)
        result = align_and_pad(words, text, audio, vision, availability, quality, target_length=3)
        self.assertEqual(result["valid_length"], 3)
        self.assertEqual(result["words"], ["w0 w1 w2", "w3 w4", "w5 w6"])
        np.testing.assert_allclose(result["intervals"], [[0, 3], [3, 5], [5, 7]])
        self.assertAlmostEqual(float(result["text"][-1, 0]), 5.5)
        self.assertTrue(result["was_aggregated"])
        self.assertFalse(np.asarray(result["padding_mask"]).any())

    def test_right_padding_masks_and_zeros(self) -> None:
        words = [WordInterval("a", 0.0, 0.5), WordInterval("b", 0.5, 1.0)]
        values = np.ones((2, 2), dtype=np.float32)
        result = align_and_pad(
            words, values, values, values, np.ones((2, 3), bool), np.ones((2, 3)), target_length=4
        )
        np.testing.assert_array_equal(result["padding_mask"], [False, False, True, True])
        np.testing.assert_array_equal(result["valid_mask"], [True, True, False, False])
        np.testing.assert_array_equal(result["audio"][2:], 0.0)

    def test_textgrid_parser(self) -> None:
        content = '''File type = "ooTextFile"
Object class = "TextGrid"
item [1]:
    class = "IntervalTier"
    name = "words"
    intervals [1]:
        xmin = 0
        xmax = 0.4
        text = "hello"
    intervals [2]:
        xmin = 0.4
        xmax = 0.6
        text = ""
    intervals [3]:
        xmin = 0.6
        xmax = 1.0
        text = "world"
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.TextGrid"
            path.write_text(content, encoding="utf-8")
            words = parse_textgrid(path)
        self.assertEqual([word.word for word in words], ["hello", "world"])
        self.assertEqual(safe_sample_name("-abc", "2"), "-abc__2")

    def test_uniform_fallback_covers_full_duration(self) -> None:
        words = uniform_word_intervals("One two, three!", 3.0)
        self.assertEqual([item.word for item in words], ["One", "two", "three"])
        self.assertEqual(words[0].start, 0.0)
        self.assertEqual(words[-1].end, 3.0)

    def test_digital_silence_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\x00\x00" * 10)
            self.assertTrue(is_digital_silence(path))
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\x00\x00" * 9 + b"\x01\x00")
            self.assertFalse(is_digital_silence(path))

    def test_visual_dimension_contract(self) -> None:
        self.assertEqual(len(AUDIO_FEATURE_NAMES), 74)
        self.assertEqual(len(VISUAL_FEATURE_NAMES), 35)


if __name__ == "__main__":
    unittest.main()
