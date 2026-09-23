"""Plot the selected model's recorded training loss and validation curves."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def generate(history_path: Path, best_epoch: int, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_stem.parent / ".matplotlib"))
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    history = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = [row["epoch"] for row in history]
    if not history or best_epoch not in epochs:
        raise ValueError("history is empty or best epoch is absent")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelsize": 10, "axes.titlesize": 11,
        "legend.frameon": False, "savefig.bbox": "tight",
    })
    blue, orange, green = "#0072B2", "#E69F00", "#009E73"
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.2), constrained_layout=True)
    axes[0].plot(epochs, [row["train_loss"] for row in history],
                 color=blue, marker="o", linewidth=2, label="Train loss")
    axes[0].set(title="Optimization", ylabel="Joint training loss")
    axes[1].plot(epochs, [row["valid_clean"]["macro_f1"] for row in history],
                 color=blue, marker="o", linewidth=2, label="Valid, complete")
    axes[1].plot(epochs, [row["valid_missing_30pct"]["macro_f1"] for row in history],
                 color=orange, marker="s", linewidth=2, label="Valid, 30% missing")
    axes[1].set(title="Classification", ylabel="Macro F1")
    axes[2].plot(epochs, [row["valid_clean"]["mae"] for row in history],
                 color=blue, marker="o", linewidth=2, label="Valid, complete")
    axes[2].plot(epochs, [row["valid_missing_30pct"]["mae"] for row in history],
                 color=orange, marker="s", linewidth=2, label="Valid, 30% missing")
    axes[2].set(title="Intensity regression", ylabel="MAE")
    for ax in axes:
        ax.axvline(best_epoch, color=green, linestyle="--", linewidth=1.4,
                   label=f"Selected epoch {best_epoch}")
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=.18, linewidth=.6)
    axes[1].legend(loc="lower right", fontsize=8)
    axes[2].legend(loc="upper right", fontsize=8)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path,
                        default=root / "ablations/fixed_gate/training_history.json")
    parser.add_argument("--checkpoint", type=Path,
                        default=root / "outputs/final_model.pt")
    parser.add_argument("--best-epoch", type=int)
    parser.add_argument("--output-stem", type=Path,
                        default=root / "outputs/final/training_curves")
    args = parser.parse_args()
    if args.best_epoch is None:
        import torch
        args.best_epoch = int(torch.load(args.checkpoint, map_location="cpu",
                                         weights_only=False)["epoch"])
    generate(args.history, args.best_epoch, args.output_stem)


if __name__ == "__main__":
    main()
