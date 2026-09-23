"""Multimodal emotion recognition preprocessing utilities."""

from .pipeline import DatasetRole, preprocess_aligned_split
from .stats import FeatureStats, fit_feature_stats, normalize_features

__all__ = [
    "DatasetRole",
    "FeatureStats",
    "fit_feature_stats",
    "normalize_features",
    "preprocess_aligned_split",
]

