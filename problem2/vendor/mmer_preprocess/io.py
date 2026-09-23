"""I/O helpers for the competition pickle layouts."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .pipeline import batch_sample_dicts
from .stats import FeatureStats


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_pickle(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def save_stats(stats: Mapping[str, FeatureStats], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: value.to_jsonable() for name, value in stats.items()}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_stats(path: Path) -> dict[str, FeatureStats]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {name: FeatureStats.from_jsonable(value) for name, value in payload.items()}


def load_attachment3_aligned(directory: Path) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    files = sorted(directory.glob("*.pkl"))
    for path in files:
        value = load_pickle(path)
        source = value["test"] if isinstance(value, dict) and "test" in value else value
        sample = dict(source)
        sample["id"] = np.asarray([path.stem])
        samples.append(sample)
    return batch_sample_dicts(samples)


def load_attachment4_aligned(directory: Path) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.pkl")):
        value = dict(load_pickle(path))
        if "id" not in value:
            value["id"] = path.stem
        samples.append(value)
    return batch_sample_dicts(samples)
