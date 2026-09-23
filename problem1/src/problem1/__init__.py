"""Problem 1: word-level multimodal feature extraction and alignment."""

from .core import WordInterval, align_and_pad, quality_weighted_pool

__all__ = ["WordInterval", "align_and_pad", "quality_weighted_pool"]
__version__ = "0.1.0"

