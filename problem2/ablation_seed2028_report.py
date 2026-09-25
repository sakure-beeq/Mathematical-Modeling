"""Re-evaluate five matched seed-2028 architecture variants on validation data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import PreparedDataset
from experiment import load_model, metrics_from_predictions, predict_loader


VARIANTS = ("none", "no_block", "no_reconstruction", "fixed_gate", "no_consistency")


def generate(data_dir: Path, runs_dir: Path, fixed_gate_dir: Path, output: Path) -> None:
    device = torch.device("cpu")
    loader = DataLoader(PreparedDataset(data_dir / "valid.npz"), batch_size=64)
    rows = []
    for variant in VARIANTS:
        folder = fixed_gate_dir if variant == "fixed_gate" else runs_dir / variant
        checkpoint_path = folder / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        expected = {"seed": 2028, "hidden": 128, "heads": 4, "dropout": .3,
                    "lr": .0002, "batch_size": 32, "epochs_requested": 12,
                    "patience": 4, "class_counts": [967, 758, 1670], "limit": None,
                    "lr_drop_epoch": 6, "lr_drop_factor": .25,
                    "ablation": variant, "ema_start_epoch": None}
        for key, value in expected.items():
            if config[key] != value:
                raise ValueError(f"{variant} {key}: expected {value}, found {config[key]}")
        history = json.loads((folder / "training_history.json").read_text(encoding="utf-8"))
        best_epoch = int(checkpoint["epoch"])
        selected = next(row for row in history if row["epoch"] == best_epoch)
        if selected["selection"] < max(row["selection"] for row in history) - 1e-8:
            raise ValueError(f"{variant}: selected checkpoint is not best in history")
        model = load_model(checkpoint_path, device)
        clean = metrics_from_predictions(predict_loader(model, loader, device))
        missing = metrics_from_predictions(
            predict_loader(model, loader, device, (.3, "random", "TAV", 2028)))
        for condition, measured in (("valid_clean", clean),
                                    ("valid_missing_30pct", missing)):
            for key in ("accuracy", "macro_f1", "mae"):
                if not np.isclose(measured[key], selected[condition][key], atol=1e-8):
                    raise ValueError(f"{variant} {condition} {key} disagrees with history")
        selection = .5 * clean["macro_f1"] + .5 * missing["macro_f1"] - .05 * missing["mae"]
        rows.append({
            "variant": variant, "training_seed": 2028, "mask_seed": 2028,
            "dropout": .3, "best_epoch": best_epoch,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "valid_accuracy": clean["accuracy"],
            "valid_macro_f1": clean["macro_f1"],
            "valid_mae": clean["mae"],
            "missing_30pct_accuracy": missing["accuracy"],
            "missing_30pct_macro_f1": missing["macro_f1"],
            "missing_30pct_mae": missing["mae"],
            "selection_score": selection,
            "checkpoint": str(checkpoint_path),
        })
        print(f"{variant}: epoch {best_epoch}, clean F1={clean['macro_f1']:.4f}, "
              f"30% missing F1={missing['macro_f1']:.4f}, score={selection:.5f}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"training_seed": 2028, "mask_seed": 2028,
               "selection_rule": "0.5*clean_macro_f1 + 0.5*missing_macro_f1 - 0.05*missing_mae",
               "highest_scoring_variant": max(rows, key=lambda row: row["selection_score"])["variant"],
               "fixed_gate_retained_as_user_selected_final_model": True,
               "test_used_for_ablation_selection": False}
    (output.parent / "ablation_summary_seed2028.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "cache")
    parser.add_argument("--runs", type=Path,
                        default=root / "outputs/seed2028_reanalysis/ablations")
    parser.add_argument("--fixed-gate", type=Path,
                        default=root / "outputs/dropout03_lr_drop/seed2028")
    parser.add_argument("--output", type=Path,
                        default=root / "outputs/seed2028_reanalysis/ablation_summary_seed2028.csv")
    args = parser.parse_args()
    generate(args.data, args.runs, args.fixed_gate, args.output)


if __name__ == "__main__":
    main()
