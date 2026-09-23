"""Verify a training replay and export per-epoch train/validation metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


def generate(reference_path: Path, replay_path: Path, output_dir: Path,
             selected_epoch: int = 5) -> None:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    if len(reference) != len(replay):
        raise ValueError("replay and original training history have different lengths")
    for old, new in zip(reference, replay):
        if old["epoch"] != new["epoch"]:
            raise ValueError("replay epoch order differs from original")
        for key in ("train_loss", "selection"):
            if not np.isclose(old[key], new[key], rtol=0, atol=1e-6):
                raise ValueError(f"epoch {old['epoch']} {key} differs from original")
        for condition in ("valid_clean", "valid_missing_30pct"):
            for key in ("accuracy", "macro_f1", "mae"):
                if not np.isclose(old[condition][key], new[condition][key],
                                  rtol=0, atol=1e-6):
                    raise ValueError(f"epoch {old['epoch']} {condition} {key} differs")
        if "train_clean" not in new:
            raise ValueError(f"epoch {new['epoch']} has no train metrics")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in replay:
        row = {"epoch": entry["epoch"], "train_loss": entry["train_loss"]}
        for condition, prefix in (("train_clean", "train"),
                                  ("valid_clean", "valid_complete"),
                                  ("valid_missing_30pct", "valid_missing_30pct")):
            for metric in ("accuracy", "macro_f1", "mae"):
                row[f"{prefix}_{metric}"] = entry[condition][metric]
        rows.append(row)
    with (output_dir / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.6), constrained_layout=True)
    epochs = [row["epoch"] for row in rows]
    axes[0, 0].plot(epochs, [row["train_loss"] for row in rows],
                    color="#0072B2", marker="o", linewidth=2)
    axes[0, 0].set(title="Optimization", ylabel="Joint training loss")
    for ax, metric, title in ((axes[0, 1], "accuracy", "Classification accuracy"),
                              (axes[1, 0], "macro_f1", "Classification Macro F1"),
                              (axes[1, 1], "mae", "Intensity MAE")):
        for prefix, label, color, marker in (
                ("train", "Train, complete", "#0072B2", "o"),
                ("valid_complete", "Valid, complete", "#D55E00", "s"),
                ("valid_missing_30pct", "Valid, 30% missing", "#CC79A7", "^")):
            ax.plot(epochs, [row[f"{prefix}_{metric}"] for row in rows],
                    label=label, color=color, marker=marker, linewidth=1.8)
        ax.set(title=title, ylabel=metric.upper() if metric == "mae" else metric.replace("_", " ").title())
    for ax in axes.flat:
        ax.axvline(selected_epoch, color="#009E73", linestyle="--", linewidth=1.3)
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.18, linewidth=0.6)
    axes[0, 1].legend(fontsize=8, loc="lower right")
    fig.savefig(output_dir / "training_curves_with_train.png", dpi=300)
    fig.savefig(output_dir / "training_curves_with_train.pdf")
    plt.close(fig)
    print(f"Verified {len(rows)} epochs and wrote {output_dir / 'epoch_metrics.csv'}")


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path,
                        default=root / "ablations/fixed_gate/training_history.json")
    parser.add_argument("--replay", type=Path,
                        default=root / "outputs/train_metrics_replay/training_history.json")
    parser.add_argument("--out", type=Path, default=root / "outputs/final")
    parser.add_argument("--selected-epoch", type=int, default=5)
    args = parser.parse_args()
    generate(args.reference, args.replay, args.out, args.selected_epoch)


if __name__ == "__main__":
    main()
