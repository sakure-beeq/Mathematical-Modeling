"""Descriptive validation error slices, without treating correlations as causes."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from experiment import CLASSES, metric_values


def analyze(data_path: Path, predictions_path: Path, output: Path) -> None:
    with np.load(data_path, allow_pickle=False) as data:
        ids = list(map(str, data["id"]))
        padding = data["padding"]
        quality = data["quality"]
        availability = data["availability"]
    with predictions_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if ids != [row["id"] for row in rows]:
        raise ValueError("prediction rows must match prepared data IDs")
    pred = np.asarray([CLASSES.index(row["pred_class"]) for row in rows])
    true = np.asarray([CLASSES.index(row["true_class"]) for row in rows])
    score = np.asarray([float(row["pred_score"]) for row in rows])
    target = np.asarray([float(row["true_score"]) for row in rows])
    content_tokens = (padding[:, :, 0].sum(1) - 2).clip(min=0)
    vision_rows = (padding[:, :, 2] & quality[:, :, 2] & availability[:, :, 2]).sum(1)
    vision_total = padding[:, :, 2].sum(1).clip(min=1)
    groups = {
        "short_text_le_10": content_tokens <= 10,
        "medium_text_11_to_30": (content_tokens > 10) & (content_tokens <= 30),
        "long_text_gt_30": content_tokens > 30,
        "vision_entirely_unavailable": vision_rows == 0,
        "vision_partly_unavailable": (vision_rows > 0) & (vision_rows < vision_total),
        "vision_all_content_valid": vision_rows == vision_total,
    }
    result = {}
    for name, select in groups.items():
        if select.any():
            result[name] = {"n": int(select.sum()),
                            **metric_values(pred[select], score[select],
                                            true[select], target[select])}
    errors = np.flatnonzero(pred != true)
    ranked = sorted(errors, key=lambda i: abs(score[i] - target[i]), reverse=True)[:20]
    result["largest_misclassified_score_errors"] = [
        {"id": ids[i], "true_class": CLASSES[true[i]], "pred_class": CLASSES[pred[i]],
         "true_score": float(target[i]), "pred_score": float(score[i])}
        for i in ranked
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {output}")
