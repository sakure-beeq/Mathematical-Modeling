"""Verify a metrics replay against an original run and plot its train curve."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from dropout_report import plot_primary


def generate(reference_dir: Path, replay_dir: Path, output_dir: Path) -> None:
    original = json.loads((reference_dir / "training_history.json").read_text(encoding="utf-8"))
    replay = json.loads((replay_dir / "training_history.json").read_text(encoding="utf-8"))
    if len(original) != len(replay):
        raise ValueError("replay and original have different epoch counts")
    for old, new in zip(original, replay):
        if old["epoch"] != new["epoch"]:
            raise ValueError("epoch order changed")
        for key in ("train_loss", "learning_rate", "selection"):
            if not np.isclose(old[key], new[key], atol=1e-6, rtol=0):
                raise ValueError(f"epoch {old['epoch']} {key} changed")
        for condition in ("valid_clean", "valid_missing_30pct"):
            for metric in ("accuracy", "macro_f1", "mae"):
                if not np.isclose(old[condition][metric], new[condition][metric],
                                  atol=1e-6, rtol=0):
                    raise ValueError(f"epoch {old['epoch']} {condition} {metric} changed")
        if "train_clean" not in new:
            raise ValueError(f"epoch {new['epoch']} lacks train metrics")
    old_checkpoint = torch.load(reference_dir / "best.pt", map_location="cpu",
                                weights_only=False)
    new_checkpoint = torch.load(replay_dir / "best.pt", map_location="cpu",
                                weights_only=False)
    if old_checkpoint["epoch"] != new_checkpoint["epoch"]:
        raise ValueError("best epoch changed")
    if any(not torch.equal(old_checkpoint["model"][key], new_checkpoint["model"][key])
           for key in old_checkpoint["model"]):
        raise ValueError("best model weights changed")
    seed = int(new_checkpoint["config"]["seed"])
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "savefig.bbox": "tight"})
    plot_primary(replay, int(new_checkpoint["epoch"]), seed, output_dir)
    rows = []
    for entry in replay:
        row = {"epoch": entry["epoch"], "learning_rate": entry["learning_rate"],
               "train_joint_loss": entry["train_loss"],
               "train_eval_task_loss": entry["train_eval_task_loss"],
               "valid_eval_task_loss": entry["valid_eval_task_loss"]}
        for condition, prefix in (("train_clean", "train"), ("valid_clean", "valid")):
            for metric in ("accuracy", "macro_f1", "mae"):
                row[f"{prefix}_{metric}"] = entry[condition][metric]
        rows.append(row)
    with (output_dir / f"seed{seed}_epoch_metrics.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Verified {len(rows)} epochs and identical best checkpoint at epoch "
          f"{new_checkpoint['epoch']}; wrote seed {seed} curve")


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path,
                        default=root / "outputs/dropout03_lr_drop/seed2028")
    parser.add_argument("--replay", type=Path,
                        default=root / "outputs/dropout03_lr_drop/seed2028_metrics_replay")
    parser.add_argument("--out", type=Path,
                        default=root / "outputs/dropout03_lr_drop/report")
    args = parser.parse_args()
    generate(args.reference, args.replay, args.out)


if __name__ == "__main__":
    main()
