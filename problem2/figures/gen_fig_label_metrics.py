"""Compare classwise metrics across train, validation, and test splits."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def generate(input_csv: Path, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_stem.parent / ".matplotlib"))
    import matplotlib.pyplot as plt
    import numpy as np

    with input_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 9:
        raise ValueError("expected 3 splits x 3 emotion labels")
    lookup = {(row["split"], row["label"]): row for row in rows}
    labels = ("Negative", "Neutral", "Positive")
    splits = ("train", "valid", "test")
    colors = {"train": "#0072B2", "valid": "#E69F00", "test": "#009E73"}
    panels = (
        ("one_vs_rest_accuracy", "One-vs-rest accuracy", "Accuracy"),
        ("one_vs_rest_macro_f1", "One-vs-rest Macro F1", "Macro F1"),
        ("mae_true_class", "Intensity MAE by true class", "MAE"),
    )
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelsize": 9, "axes.titlesize": 10,
        "legend.frameon": False, "savefig.bbox": "tight",
    })
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.2), constrained_layout=True)
    x = np.arange(3)
    width = .23
    for ax, (field, title, ylabel) in zip(axes, panels):
        for j, split in enumerate(splits):
            values = [float(lookup[(split, label)][field]) for label in labels]
            ax.bar(x + (j - 1) * width, values, width, color=colors[split], label=split)
        ax.set_xticks(x, labels)
        ax.set(title=title, ylabel=ylabel)
        ax.set_ylim(0, 1 if field != "mae_true_class" else .9)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    axes[0].legend(loc="upper center", ncol=3, fontsize=8)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=root / "outputs/final/label_metrics.csv")
    parser.add_argument("--output-stem", type=Path,
                        default=root / "outputs/final/label_metrics_chart")
    args = parser.parse_args()
    generate(args.input, args.output_stem)


if __name__ == "__main__":
    main()
