"""Aligned-data preprocessing pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

import numpy as np

from .bert import FrozenBertEmbedder
from .masks import (
    MaskSet,
    content_padding_from_attention,
    ensure_batched,
    make_feature_masks,
    make_text_masks,
)
from .stats import FeatureStats, normalize_features


class DatasetRole(str, Enum):
    STANDARD = "standard"
    MISSING_TEST = "missing_test"
    EXPLAIN_TEST = "explain_test"


def _optional_array(
    source: Mapping[str, Any],
    key: str,
    dtype: np.dtype[Any] | type,
    sample_count: int,
) -> np.ndarray | None:
    if key not in source:
        return None
    value = np.asarray(source[key], dtype=dtype)
    if value.ndim == 0:
        value = value[None]
    if value.shape[0] != sample_count:
        raise ValueError(
            f"field {key!r} has {value.shape[0]} samples; expected {sample_count}"
        )
    return value


def _serialize_masks(masks: Mapping[str, MaskSet]) -> dict[str, dict[str, np.ndarray]]:
    return {name: value.as_dict(include_effective=True) for name, value in masks.items()}


def preprocess_aligned_split(
    source: Mapping[str, Any],
    role: DatasetRole | str,
    stats: Mapping[str, FeatureStats] | None = None,
    clip_sigma: float | None = 5.0,
    bert_embedder: FrozenBertEmbedder | None = None,
) -> dict[str, Any]:
    """Preprocess one aligned split or a batch assembled from sample files."""

    dataset_role = DatasetRole(role)
    if "text_bert" not in source or "audio" not in source or "vision" not in source:
        missing = {"text_bert", "audio", "vision"} - set(source)
        raise KeyError(f"missing required aligned fields: {sorted(missing)}")

    text_bert = ensure_batched(np.asarray(source["text_bert"]), 3, "text_bert")
    if text_bert.shape[1] != 3:
        raise ValueError(f"text_bert must have shape (N,3,T); got {text_bert.shape}")
    text_bert = text_bert.astype(np.int64, copy=False)

    audio = ensure_batched(np.asarray(source["audio"]), 3, "audio").astype(
        np.float32, copy=False
    )
    vision = ensure_batched(np.asarray(source["vision"]), 3, "vision").astype(
        np.float32, copy=False
    )
    sample_count, _, time_steps = text_bert.shape
    if audio.shape[:2] != (sample_count, time_steps):
        raise ValueError(f"audio shape {audio.shape} is not aligned with {text_bert.shape}")
    if vision.shape[:2] != (sample_count, time_steps):
        raise ValueError(f"vision shape {vision.shape} is not aligned with {text_bert.shape}")

    missing_test = dataset_role is DatasetRole.MISSING_TEST
    attention = text_bert[:, 1, :]
    text_masks = make_text_masks(attention, missing_test=missing_test)
    content_padding = content_padding_from_attention(attention)
    audio_masks = make_feature_masks(audio, content_padding, missing_test=missing_test)
    vision_masks = make_feature_masks(vision, content_padding, missing_test=missing_test)
    masks = {"text": text_masks, "audio": audio_masks, "vision": vision_masks}

    if stats is not None:
        missing_stats = {"audio", "vision"} - set(stats)
        if missing_stats:
            raise KeyError(f"normalization stats missing: {sorted(missing_stats)}")
        audio = normalize_features(audio, audio_masks.effective, stats["audio"], clip_sigma)
        vision = normalize_features(vision, vision_masks.effective, stats["vision"], clip_sigma)
    else:
        audio = np.where(audio_masks.effective[..., None], audio, 0.0).astype(np.float32)
        vision = np.where(vision_masks.effective[..., None], vision, 0.0).astype(np.float32)

    result: dict[str, Any] = {
        "text_bert": text_bert,
        "audio": audio,
        "vision": vision,
        "masks": _serialize_masks(masks),
        "metadata": {
            "dataset_role": dataset_role.value,
            "sample_count": sample_count,
            "time_steps": time_steps,
            "audio_dim": audio.shape[-1],
            "vision_dim": vision.shape[-1],
            "clip_sigma": clip_sigma,
        },
    }

    ids = _optional_array(source, "id", str, sample_count)
    raw_text = _optional_array(source, "raw_text", str, sample_count)
    classification = _optional_array(source, "classification_labels", np.int64, sample_count)
    regression = _optional_array(source, "regression_labels", np.float32, sample_count)
    if ids is not None:
        result["id"] = ids
    if raw_text is not None:
        result["raw_text"] = raw_text
    if classification is not None:
        result["classification_labels"] = classification
    if regression is not None:
        result["regression_labels"] = regression

    if bert_embedder is not None:
        result["text"] = bert_embedder.encode(text_bert)
        result["text"] = np.where(
            text_masks.effective[..., None], result["text"], 0.0
        ).astype(np.float32, copy=False)

    return result


def batch_sample_dicts(samples: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack per-sample dictionaries into the split-oriented input format."""

    if not samples:
        raise ValueError("no samples to batch")
    required = ("text_bert", "audio", "vision")
    batched: dict[str, Any] = {}
    for key in required:
        arrays = [np.asarray(sample[key]) for sample in samples]
        arrays = [array[0] if array.ndim == 3 and array.shape[0] == 1 else array for array in arrays]
        batched[key] = np.stack(arrays, axis=0)

    for key in ("id", "raw_text", "classification_labels", "regression_labels"):
        if all(key in sample for sample in samples):
            values = [np.asarray(sample[key]).reshape(-1)[0] for sample in samples]
            batched[key] = np.asarray(values)
    return batched

