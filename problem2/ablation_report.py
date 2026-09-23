"""Summarize checkpoints selected by the same validation criterion."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

VARIANTS = ("none", "no_block", "no_reconstruction", "fixed_gate", "no_consistency")


def summarize(root: Path, output: Path) -> None:
    rows = []
    for variant in VARIANTS:
        folder = root / "outputs" if variant == "none" else root / "ablations" / variant
        checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
        best_epoch = int(checkpoint["epoch"])
        history = json.loads((folder / "training_history.json").read_text(encoding="utf-8"))
        selected = next(row for row in history if row["epoch"] == best_epoch)
        clean = json.loads((folder / "valid_metrics.json").read_text(encoding="utf-8"))
        missing = selected["valid_missing_30pct"]
        rows.append({
            "variant": variant, "best_epoch": best_epoch,
            "valid_accuracy": clean["accuracy"],
            "valid_macro_f1": clean["macro_f1"],
            "valid_mae": clean["mae"],
            "missing_30pct_macro_f1": missing["macro_f1"],
            "missing_30pct_mae": missing["mae"],
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")
