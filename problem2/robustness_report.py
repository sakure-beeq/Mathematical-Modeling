"""Summarize modality/rate robustness and the original architecture ablations."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


MODALITIES = ("T", "A", "V", "TA", "TV", "AV", "TAV")
RATIOS = (0.1, 0.2, 0.3, 0.4, 0.5)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_random(rows: list[dict], baseline: dict) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if row["position"] == "random":
            grouped[row["modality"], float(row["ratio"])].append(row)
    summary = []
    for modality in MODALITIES:
        for ratio in RATIOS:
            group = grouped[modality, ratio]
            if len(group) != 5:
                raise ValueError(f"expected five random positions for {modality} {ratio}")
            mean = lambda key: float(np.mean([float(row[key]) for row in group]))
            std = lambda key: float(np.std([float(row[key]) for row in group], ddof=1))
            f1 = mean("macro_f1")
            mae = mean("mae")
            summary.append({"modality": modality, "target_missing_ratio": ratio,
                            "actual_missing_ratio": mean("actual_missing_ratio"),
                            "n_seeds": len(group), "accuracy_mean": mean("accuracy"),
                            "macro_f1_mean": f1, "macro_f1_sd": std("macro_f1"),
                            "mae_mean": mae, "mae_sd": std("mae"),
                            "delta_macro_f1": f1 - baseline["macro_f1"],
                            "delta_mae": mae - baseline["mae"]})
    return summary


def summarize_positions(rows: list[dict], baseline: dict) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["modality"], row["position"]].append(row)
    summary = []
    for modality in MODALITIES:
        for position in ("start", "middle", "end", "random"):
            group = grouped[modality, position]
            # Equal weight to each target ratio, and to each seed within a ratio.
            by_ratio = defaultdict(list)
            for row in group:
                by_ratio[float(row["ratio"])].append(row)
            f1 = float(np.mean([np.mean([float(row["macro_f1"]) for row in by_ratio[r]])
                                for r in RATIOS]))
            mae = float(np.mean([np.mean([float(row["mae"]) for row in by_ratio[r]])
                                 for r in RATIOS]))
            summary.append({"modality": modality, "position": position,
                            "macro_f1_mean": f1, "mae_mean": mae,
                            "delta_macro_f1": f1 - baseline["macro_f1"],
                            "delta_mae": mae - baseline["mae"]})
    return summary


def plot_robustness(summary: list[dict], baseline: dict, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    f1_grid = np.array([[next(row["delta_macro_f1"] for row in summary
                              if row["modality"] == m and row["target_missing_ratio"] == r)
                         for r in RATIOS] for m in MODALITIES])
    mae_grid = np.array([[next(row["delta_mae"] for row in summary
                               if row["modality"] == m and row["target_missing_ratio"] == r)
                          for r in RATIOS] for m in MODALITIES])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for ax, grid, title in ((axes[0], f1_grid, "Change in Macro F1"),
                            (axes[1], mae_grid, "Change in MAE")):
        span = max(abs(grid.min()), abs(grid.max()), 1e-4)
        im = ax.imshow(grid, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0,
                       vmin=-span, vmax=span), aspect="auto")
        ax.set(title=title, xticks=np.arange(5), yticks=np.arange(7),
               xticklabels=[f"{int(r*100)}%" for r in RATIOS], yticklabels=MODALITIES,
               xlabel="Target missing ratio", ylabel="Masked modality")
        for i in range(7):
            for j in range(5):
                ax.text(j, i, f"{grid[i,j]:+.3f}", ha="center", va="center",
                        color="white" if abs(grid[i,j]) > span * .65 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=.78)
    fig.savefig(output_dir / "modality_rate_heatmaps.png", dpi=300)
    fig.savefig(output_dir / "modality_rate_heatmaps.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), constrained_layout=True)
    palette = plt.cm.tab10(np.linspace(0, 1, len(MODALITIES)))
    for m, color in zip(MODALITIES, palette):
        subset = [row for row in summary if row["modality"] == m]
        axes[0].plot(RATIOS, [row["macro_f1_mean"] for row in subset],
                     marker="o", label=m, color=color)
        axes[1].plot(RATIOS, [row["mae_mean"] for row in subset],
                     marker="o", label=m, color=color)
    for ax, baseline_value, title, ylabel in (
            (axes[0], baseline["macro_f1"], "Macro F1", "Macro F1"),
            (axes[1], baseline["mae"], "Intensity MAE", "MAE")):
        ax.axhline(baseline_value, color="black", linestyle="--", linewidth=1,
                   label="Complete input")
        ax.set(title=title, xlabel="Target missing ratio", ylabel=ylabel,
               xticks=RATIOS, xticklabels=[f"{int(r*100)}%" for r in RATIOS])
        ax.grid(alpha=.2)
    axes[0].legend(ncol=4, fontsize=8)
    fig.savefig(output_dir / "modality_rate_curves.png", dpi=300)
    fig.savefig(output_dir / "modality_rate_curves.pdf")
    plt.close(fig)


def plot_ablations(ablations: list[dict], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    names = [row["variant"] for row in ablations]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), constrained_layout=True)
    for ax, columns, title in (
            (axes[0], ("valid_macro_f1", "missing_30pct_macro_f1"), "Macro F1"),
            (axes[1], ("valid_mae", "missing_30pct_mae"), "Intensity MAE")):
        for offset, key, label, color in ((-.18, columns[0], "Complete", "#0072B2"),
                                          (.18, columns[1], "30% missing", "#D55E00")):
            values = [float(row[key]) for row in ablations]
            bars = ax.bar(x + offset, values, width=.35, label=label, color=color)
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
        ax.set(title=title, xticks=x, xticklabels=names)
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(float(row[key]) for row in ablations for key in columns) * 1.18)
    axes[0].legend(fontsize=8)
    fig.savefig(output_dir / "architecture_ablations_original_lr.png", dpi=300)
    fig.savefig(output_dir / "architecture_ablations_original_lr.pdf")
    plt.close(fig)


def generate(robustness_path: Path, baseline_path: Path,
             ablation_path: Path, output_dir: Path) -> None:
    rows = read_csv(robustness_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    ablations = read_csv(ablation_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "savefig.bbox": "tight"})
    random_summary = summarize_random(rows, baseline)
    positions = summarize_positions(rows, baseline)
    write_csv(output_dir / "modality_rate_summary_random5.csv", random_summary)
    write_csv(output_dir / "position_summary.csv", positions)
    plot_robustness(random_summary, baseline, output_dir)
    plot_ablations(ablations, output_dir)
    print(f"complete baseline: F1={baseline['macro_f1']:.4f}, MAE={baseline['mae']:.4f}")
    print("30% random position, five-seed means:")
    for row in random_summary:
        if row["target_missing_ratio"] == 0.3:
            print(row["modality"], f"F1={row['macro_f1_mean']:.4f}",
                  f"MAE={row['mae_mean']:.4f}",
                  f"actual={row['actual_missing_ratio']:.3f}")


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robustness", type=Path,
                        default=root / "outputs/lr_drop_epoch6/report/valid_robustness.csv")
    parser.add_argument("--baseline", type=Path,
                        default=root / "outputs/lr_drop_epoch6/report/valid_robustness_baseline.json")
    parser.add_argument("--ablations", type=Path,
                        default=root / "outputs/ablation_summary.csv")
    parser.add_argument("--out", type=Path,
                        default=root / "outputs/lr_drop_epoch6/report")
    args = parser.parse_args()
    generate(args.robustness, args.baseline, args.ablations, args.out)


if __name__ == "__main__":
    main()
