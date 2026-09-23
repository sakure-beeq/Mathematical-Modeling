"""Mask construction for aligned text, audio, and vision sequences.

The three masks have deliberately different meanings:

* padding: the timestep belongs to the sample rather than right padding;
* availability: the modality was observed at that timestep;
* quality: the observed feature passed basic extraction-quality checks.

The effective mask is their logical conjunction. Keeping these masks separate
prevents padding, artificial missing blocks, and feature-extraction failures
from being treated as the same phenomenon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MaskSet:
    """Masks for one modality, each with shape ``(N, T)``."""

    padding: np.ndarray
    availability: np.ndarray
    quality: np.ndarray

    def __post_init__(self) -> None:
        shapes = {np.asarray(mask).shape for mask in self.as_dict().values()}
        if len(shapes) != 1:
            raise ValueError(f"mask shapes do not match: {sorted(shapes)}")

    @property
    def effective(self) -> np.ndarray:
        return self.padding & self.availability & self.quality

    def as_dict(self, include_effective: bool = False) -> dict[str, np.ndarray]:
        result = {
            "padding": np.asarray(self.padding, dtype=bool),
            "availability": np.asarray(self.availability, dtype=bool),
            "quality": np.asarray(self.quality, dtype=bool),
        }
        if include_effective:
            result["effective"] = self.effective
        return result


def ensure_batched(array: np.ndarray, ndim: int, name: str) -> np.ndarray:
    """Add a sample dimension to a single sample and validate rank."""

    value = np.asarray(array)
    if value.ndim == ndim - 1:
        value = value[None, ...]
    if value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}; got shape {value.shape}")
    return value


def row_nonzero(features: np.ndarray) -> np.ndarray:
    """Return whether each feature row contains at least one nonzero value."""

    values = ensure_batched(features, 3, "features")
    finite = np.all(np.isfinite(values), axis=-1)
    nonzero = np.any(values != 0, axis=-1)
    return finite & nonzero


def attention_envelope(attention_mask: np.ndarray) -> np.ndarray:
    """Infer the non-padding envelope while preserving possible internal gaps.

    For a missing-test sample, an internal zero in the attention mask may mean
    that text is unavailable rather than padded. The interval from the first to
    the last active token is therefore treated as the sample envelope.
    """

    mask = np.asarray(attention_mask, dtype=bool)
    if mask.ndim == 1:
        mask = mask[None, :]
    envelope = np.zeros_like(mask, dtype=bool)
    for index, row in enumerate(mask):
        active = np.flatnonzero(row)
        if active.size:
            envelope[index, active[0] : active[-1] + 1] = True
    return envelope


def content_padding_from_attention(attention_mask: np.ndarray) -> np.ndarray:
    """Build aligned audio/vision padding masks from BERT token masks.

    The first and last valid BERT positions are CLS and SEP. Aligned audio and
    vision features have no corresponding content at those two positions.
    """

    envelope = attention_envelope(attention_mask)
    content = envelope.copy()
    for index, row in enumerate(envelope):
        active = np.flatnonzero(row)
        if active.size:
            content[index, active[0]] = False
        if active.size > 1:
            content[index, active[-1]] = False
    return content


def make_text_masks(attention_mask: np.ndarray, missing_test: bool) -> MaskSet:
    observed = np.asarray(attention_mask) > 0
    if observed.ndim == 1:
        observed = observed[None, :]
    if missing_test:
        padding = attention_envelope(observed)
        availability = observed & padding
    else:
        padding = observed.copy()
        availability = padding.copy()
    quality = padding.copy()
    return MaskSet(padding=padding, availability=availability, quality=quality)


def make_feature_masks(
    features: np.ndarray,
    padding: np.ndarray,
    missing_test: bool,
) -> MaskSet:
    """Construct masks for audio or vision aligned to BERT positions.

    Exact-zero rows inside ordinary train/validation/test data are interpreted
    as extraction-quality failures. In the dedicated missing-modality test set,
    the same pattern is interpreted as unavailable input because the benchmark
    explicitly creates zero-valued missing blocks.
    """

    observed = row_nonzero(features)
    padding_mask = np.asarray(padding, dtype=bool)
    if observed.shape != padding_mask.shape:
        raise ValueError(
            f"feature and padding shapes differ: {observed.shape} vs {padding_mask.shape}"
        )

    if missing_test:
        availability = observed & padding_mask
        quality = padding_mask.copy()
    else:
        availability = padding_mask.copy()
        quality = observed & padding_mask

    return MaskSet(
        padding=padding_mask,
        availability=availability,
        quality=quality,
    )
