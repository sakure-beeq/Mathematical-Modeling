"""Select the final architecture using validation data only."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


def select(root: Path) -> dict:
    with (root / "outputs/ablation_summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 5:
        raise ValueError("expected the full model and four ablations")
    for row in rows:
        row["selection_score"] = (
            0.5 * float(row["valid_macro_f1"])
            + 0.5 * float(row["missing_30pct_macro_f1"])
            - 0.05 * float(row["missing_30pct_mae"])
        )
    winner = max(rows, key=lambda row: row["selection_score"])
    source = (root / "outputs" if winner["variant"] == "none" else
              root / "ablations" / winner["variant"])
    destination = root / "outputs"
    shutil.copy2(source / "best.pt", destination / "final_model.pt")
    shutil.copy2(source / "training_config.json", destination / "final_training_config.json")
    report = {
        "selection_rule": "0.5*valid_clean_macro_f1 + 0.5*valid_missing_30pct_macro_f1 - 0.05*valid_missing_30pct_mae",
        "selected_variant": winner["variant"],
        "selected_score": winner["selection_score"],
        "candidates": rows,
        "test_used_for_selection": False,
    }
    (destination / "model_selection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"selected {winner['variant']} with validation score {winner['selection_score']:.6f}")
    return report
