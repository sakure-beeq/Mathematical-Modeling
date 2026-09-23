"""Training-only feature statistics and mask-aware normalization."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FeatureStats:
    mean: np.ndarray
    std: np.ndarray
    count: int

    def to_jsonable(self) -> dict[str, object]:
        return {
            "mean": np.asarray(self.mean, dtype=np.float64).tolist(),
            "std": np.asarray(self.std, dtype=np.float64).tolist(),
            "count": int(self.count),
        }

    @classmethod
    def from_jsonable(cls, value: dict[str, object]) -> "FeatureStats":
        return cls(
            mean=np.asarray(value["mean"], dtype=np.float32),
            std=np.asarray(value["std"], dtype=np.float32),
            count=int(value["count"]),
        )


def fit_feature_stats(
    features: np.ndarray,
    effective_mask: np.ndarray,
    epsilon: float = 1e-6,
) -> FeatureStats:
    """Fit per-dimension mean/std using only effective training rows."""

    values = np.asarray(features, dtype=np.float64)
    mask = np.asarray(effective_mask, dtype=bool)
    if values.ndim != 3 or mask.shape != values.shape[:2]:
        raise ValueError(
            f"expected features (N,T,D) and mask (N,T); got {values.shape}, {mask.shape}"
        )
    selected = values[mask]
    if selected.shape[0] == 0:
        raise ValueError("cannot fit statistics without effective feature rows")
    finite_rows = np.all(np.isfinite(selected), axis=1)
    selected = selected[finite_rows]
    if selected.shape[0] == 0:
        raise ValueError("all effective feature rows contain NaN or Inf")

    mean = selected.mean(axis=0)
    std = selected.std(axis=0)
    std = np.where(std < epsilon, 1.0, std)
    return FeatureStats(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        count=int(selected.shape[0]),
    )


def normalize_features(
    features: np.ndarray,
    effective_mask: np.ndarray,
    stats: FeatureStats,
    clip_sigma: float | None = 5.0,
) -> np.ndarray:
    """Normalize effective rows and keep every invalid row exactly zero."""

    values = np.asarray(features, dtype=np.float32)
    mask = np.asarray(effective_mask, dtype=bool)
    if values.ndim != 3 or mask.shape != values.shape[:2]:
        raise ValueError(
            f"expected features (N,T,D) and mask (N,T); got {values.shape}, {mask.shape}"
        )
    if values.shape[-1] != stats.mean.shape[0]:
        raise ValueError(
            f"feature dimension {values.shape[-1]} does not match stats {stats.mean.shape[0]}"
        )

    output = np.zeros_like(values, dtype=np.float32)
    normalized = (values[mask] - stats.mean) / stats.std
    if clip_sigma is not None:
        if clip_sigma <= 0:
            raise ValueError("clip_sigma must be positive or None")
        normalized = np.clip(normalized, -clip_sigma, clip_sigma)
    output[mask] = normalized.astype(np.float32, copy=False)
    return output
