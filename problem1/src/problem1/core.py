"""Dependency-light alignment and sequence-shaping primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class WordInterval:
    """One aligned, non-silence word interval in seconds."""

    word: str
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid interval {self.word!r}: [{self.start}, {self.end}]")


def _frame_bounds(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Infer frame support intervals from monotonically increasing centre times."""

    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("frame_times must be a non-empty 1-D array")
    if np.any(np.diff(times) <= 0):
        raise ValueError("frame_times must be strictly increasing")
    if times.size == 1:
        half = 0.005
        return np.maximum(0.0, times - half), times + half
    mid = (times[:-1] + times[1:]) / 2.0
    left = np.r_[times[0] - (mid[0] - times[0]), mid]
    right = np.r_[mid, times[-1] + (times[-1] - mid[-1])]
    return np.maximum(0.0, left), right


def quality_weighted_pool(
    frame_times: np.ndarray,
    frame_features: np.ndarray,
    intervals: Sequence[WordInterval],
    quality: np.ndarray | None = None,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool frames into word intervals using overlap duration times frame quality.

    Returns ``(features, availability, quality_score)``. A frame contributes by
    the duration of overlap between its inferred support and the word interval.
    Non-finite frames and frames with zero quality never contribute.
    """

    times = np.asarray(frame_times, dtype=np.float64)
    features = np.asarray(frame_features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != times.size:
        raise ValueError("frame_features must have shape (number_of_times, feature_dim)")
    if quality is None:
        q = np.ones(times.size, dtype=np.float32)
    else:
        q = np.asarray(quality, dtype=np.float32)
        if q.shape != times.shape:
            raise ValueError("quality must have the same shape as frame_times")
    finite = np.isfinite(features).all(axis=1) & np.isfinite(q)
    q = np.where(finite, np.clip(q, 0.0, 1.0), 0.0)
    clean = np.where(np.isfinite(features), features, 0.0)
    left, right = _frame_bounds(times)

    pooled = np.zeros((len(intervals), features.shape[1]), dtype=np.float32)
    available = np.zeros(len(intervals), dtype=bool)
    scores = np.zeros(len(intervals), dtype=np.float32)
    for i, interval in enumerate(intervals):
        overlap = np.maximum(
            0.0,
            np.minimum(right, interval.end) - np.maximum(left, interval.start),
        ).astype(np.float32)
        raw_overlap = float(overlap.sum())
        weights = overlap * q
        denominator = float(weights.sum())
        if denominator > eps:
            pooled[i] = (weights[:, None] * clean).sum(axis=0) / (denominator + eps)
            available[i] = True
        scores[i] = 0.0 if raw_overlap <= eps else denominator / raw_overlap
    return pooled, available, np.clip(scores, 0.0, 1.0)


def interpolate_short_gaps(
    times: np.ndarray,
    features: np.ndarray,
    quality: np.ndarray,
    max_gap_seconds: float,
    interpolated_quality: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly fill only bounded short invalid runs; preserve long gaps."""

    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(features, dtype=np.float32).copy()
    q = np.asarray(quality, dtype=np.float32).copy()
    if values.ndim != 2 or values.shape[0] != times.size or q.shape != times.shape:
        raise ValueError("incompatible times/features/quality shapes")
    valid = (q > 0) & np.isfinite(values).all(axis=1)
    q[~valid] = 0.0
    i = 0
    while i < len(valid):
        if valid[i]:
            i += 1
            continue
        start = i
        while i < len(valid) and not valid[i]:
            i += 1
        stop = i
        bounded = start > 0 and stop < len(valid)
        gap = times[stop] - times[start - 1] if bounded else np.inf
        if bounded and gap <= max_gap_seconds:
            alpha = (times[start:stop] - times[start - 1]) / gap
            values[start:stop] = (
                values[start - 1][None, :] * (1.0 - alpha[:, None])
                + values[stop][None, :] * alpha[:, None]
            )
            q[start:stop] = np.minimum(q[start - 1], q[stop]) * interpolated_quality
            valid[start:stop] = True
    values[~np.isfinite(values)] = 0.0
    return values, np.clip(q, 0.0, 1.0)


def _groups(length: int, target: int) -> list[np.ndarray]:
    if length <= target:
        return [np.asarray([i], dtype=np.int64) for i in range(length)]
    # Every group is non-empty and contiguous; no suffix of the utterance is lost.
    return [g.astype(np.int64) for g in np.array_split(np.arange(length), target)]


def align_and_pad(
    words: Sequence[WordInterval],
    text: np.ndarray,
    audio: np.ndarray,
    vision: np.ndarray,
    availability: np.ndarray,
    quality_scores: np.ndarray,
    target_length: int = 50,
    quality_threshold: float = 0.25,
) -> dict[str, object]:
    """Aggregate overlong sequences into contiguous regions, then right-pad.

    ``availability`` and ``quality_scores`` use modality order text/audio/vision.
    Padding positions are zeros. ``padding_mask=True`` explicitly means padding.
    """

    if target_length <= 0 or len(words) == 0:
        raise ValueError("target_length and word count must be positive")
    arrays = [np.asarray(text), np.asarray(audio), np.asarray(vision)]
    n = len(words)
    if any(a.ndim != 2 or a.shape[0] != n for a in arrays):
        raise ValueError("all modality arrays must have shape (word_count, dim)")
    avail = np.asarray(availability, dtype=bool)
    scores = np.asarray(quality_scores, dtype=np.float32)
    if avail.shape != (n, 3) or scores.shape != (n, 3):
        raise ValueError("availability and quality_scores must have shape (word_count, 3)")

    groups = _groups(n, target_length)
    out_words: list[str] = []
    intervals = np.zeros((target_length, 2), dtype=np.float32)
    out_features = [np.zeros((target_length, a.shape[1]), dtype=np.float32) for a in arrays]
    out_avail = np.zeros((target_length, 3), dtype=bool)
    out_scores = np.zeros((target_length, 3), dtype=np.float32)

    for row, group in enumerate(groups):
        out_words.append(" ".join(words[i].word for i in group))
        intervals[row] = (words[group[0]].start, words[group[-1]].end)
        durations = np.asarray([words[i].end - words[i].start for i in group], dtype=np.float32)
        for modality, source in enumerate(arrays):
            valid = avail[group, modality]
            weights = durations * scores[group, modality] * valid
            if modality == 0 and not weights.any():
                weights = durations  # Text is available for every aligned word by construction.
            if weights.sum() > 0:
                out_features[modality][row] = np.average(source[group], axis=0, weights=weights)
                out_avail[row, modality] = True
                out_scores[row, modality] = np.average(scores[group, modality], weights=durations)

    valid_length = len(groups)
    valid_mask = np.zeros(target_length, dtype=bool)
    valid_mask[:valid_length] = True
    padding_mask = ~valid_mask
    quality_mask = (out_scores >= quality_threshold) & out_avail & valid_mask[:, None]
    return {
        "words": out_words + [""] * (target_length - valid_length),
        "intervals": intervals,
        "text": out_features[0],
        "audio": out_features[1],
        "vision": out_features[2],
        "valid_length": valid_length,
        "valid_mask": valid_mask,
        "padding_mask": padding_mask,
        "modality_availability_mask": out_avail,
        "quality_scores": out_scores,
        "quality_mask": quality_mask,
        "was_aggregated": n > target_length,
        "original_word_count": n,
        "aggregation_ratio": float(n / valid_length),
    }
