"""Compare the hidden-size-96 experiment with matched 128-dimensional runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import PreparedDataset
from experiment import load_model, metrics_from_predictions, predict_loader


METRICS = ("accuracy", "macro_f1", "mae")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_primary(history: list[dict], best_epoch: int, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(epochs, [r["train_loss"] for r in history], "o-", color="#0072B2",
            label="Train joint loss")
    ax.plot(epochs, [r["train_eval_task_loss"] for r in history], "s--",
            color="#009E73", label="Train task loss")
    ax.plot(epochs, [r["valid_eval_task_loss"] for r in history], "^--",
            color="#D55E00", label="Valid task loss")
    ax.set(title="Loss", ylabel="Loss")
    ax.legend(fontsize=8)
    for ax, metric, title in zip(
            (axes[0, 1], axes[1, 0], axes[1, 1]), METRICS,
            ("Classification accuracy", "Classification Macro F1", "Intensity MAE")):
        ax.plot(epochs, [r["train_clean"][metric] for r in history], "o-",
                color="#0072B2", label="Train, complete")
        ax.plot(epochs, [r["valid_clean"][metric] for r in history], "s-",
                color="#D55E00", label="Valid, complete")
        ax.set(title=title, ylabel=title.split()[-1])
    for ax in axes.flat:
        ax.axvline(5.5, color="#777777", linestyle=":", linewidth=1.2,
                   label="LR reduced at epoch 6")
        ax.axvline(best_epoch, color="#009E73", linestyle="--", linewidth=1.2,
                   label=f"Selected epoch {best_epoch}")
        ax.set_xlabel("Epoch")
        ax.set_xticks(epochs)
        ax.grid(alpha=.18)
    axes[0, 1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Hidden size 96, seed 2026", fontsize=13)
    fig.savefig(output_dir / "seed2026_training_curves.png", dpi=300)
    fig.savefig(output_dir / "seed2026_training_curves.pdf")
    plt.close(fig)


def plot_validation_comparison(new_histories: dict[int, list[dict]],
                               baseline_histories: dict[int, list[dict]],
                               output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), constrained_layout=True)
    colors = ("#0072B2", "#D55E00", "#009E73")
    for ax, metric, title in zip(
            axes, METRICS,
            ("Validation accuracy", "Validation Macro F1", "Validation MAE")):
        for (seed, history), color in zip(sorted(new_histories.items()), colors):
            baseline = baseline_histories[seed]
            ax.plot([r["epoch"] for r in baseline],
                    [r["valid_clean"][metric] for r in baseline], ":",
                    color=color, alpha=.65, linewidth=1.8)
            ax.plot([r["epoch"] for r in history],
                    [r["valid_clean"][metric] for r in history], "o-",
                    color=color, markersize=3.5, linewidth=1.8, label=f"Seed {seed}")
        ax.set(title=title, xlabel="Epoch", ylabel=title.split()[-1])
        ax.grid(alpha=.18)
    axes[0].legend(fontsize=7)
    fig.suptitle("Solid: hidden size 96    Dotted: hidden size 128", fontsize=11)
    fig.savefig(output_dir / "three_seed_validation_comparison.png", dpi=300)
    fig.savefig(output_dir / "three_seed_validation_comparison.pdf")
    plt.close(fig)


def plot_selected_comparison(rows: list[dict], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), constrained_layout=True)
    x = np.arange(len(rows))
    for ax, metric, title in zip(
            axes, METRICS,
            ("Validation accuracy", "Validation Macro F1", "Validation MAE")):
        old = [r[f"baseline_valid_{metric}"] for r in rows]
        new = [r[f"valid_{metric}"] for r in rows]
        bars1 = ax.bar(x - .18, old, width=.35, color="#D55E00", label="Hidden 128")
        bars2 = ax.bar(x + .18, new, width=.35, color="#0072B2", label="Hidden 96")
        ax.bar_label(bars1, fmt="%.3f", fontsize=8, padding=2)
        ax.bar_label(bars2, fmt="%.3f", fontsize=8, padding=2)
        ax.set(title=title, xlabel="Seed", xticks=x,
               xticklabels=[str(r["seed"]) for r in rows])
        ax.set_ylim(0, max(old + new) * 1.15)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=8, loc="lower center")
    fig.savefig(output_dir / "selected_model_comparison.png", dpi=300)
    fig.savefig(output_dir / "selected_model_comparison.pdf")
    plt.close(fig)


def generate(data_dir: Path, runs_dir: Path, baseline_dir: Path,
             output_dir: Path, seeds: tuple[int, ...]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "savefig.bbox": "tight"})
    new_histories, baseline_histories = {}, {}
    summary_rows, epoch_rows = [], []
    for seed in seeds:
        run_dir = runs_dir / f"seed{seed}"
        history = json.loads((run_dir / "training_history.json").read_text(encoding="utf-8"))
        baseline_path = (baseline_dir / "lr_drop_epoch6/training_history.json" if seed == 2026
                         else baseline_dir / f"ema_lr_drop/seed{seed}/training_history.json")
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        if (config["hidden"] != 96 or config["seed"] != seed or
                config["lr_drop_epoch"] != 6 or config["ema_start_epoch"] is not None):
            raise ValueError(f"unexpected settings for seed {seed}")
        best_epoch = int(checkpoint["epoch"])
        best_row = next(r for r in history if r["epoch"] == best_epoch)
        if not np.isclose(best_row["selection"], max(r["selection"] for r in history),
                          rtol=0, atol=1e-6):
            raise ValueError(f"seed {seed} checkpoint does not match best epoch")
        baseline_best = max(baseline, key=lambda r: r.get("raw_selection", r["selection"]))
        model = load_model(run_dir / "best.pt", torch.device("cpu"))
        split_metrics = {}
        for split in ("train", "valid"):
            loader = DataLoader(PreparedDataset(data_dir / f"{split}.npz"), batch_size=64)
            split_metrics[split] = metrics_from_predictions(
                predict_loader(model, loader, torch.device("cpu")))
            expected = best_row.get("train_clean") if split == "train" else best_row["valid_clean"]
            if expected is not None:
                for metric in METRICS:
                    if not np.isclose(split_metrics[split][metric], expected[metric],
                                      rtol=0, atol=1e-8):
                        raise ValueError(f"seed {seed} {split} {metric} mismatch")
        summary_rows.append({
            "seed": seed, "best_epoch": best_epoch, "selection": best_row["selection"],
            **{f"{split}_{metric}": split_metrics[split][metric]
               for split in ("train", "valid") for metric in METRICS},
            "baseline_best_epoch": baseline_best["epoch"],
            **{f"baseline_valid_{metric}": baseline_best["valid_clean"][metric]
               for metric in METRICS},
        })
        for row in history:
            flat = {"seed": seed, "epoch": row["epoch"],
                    "learning_rate": row["learning_rate"],
                    "train_joint_loss": row["train_loss"],
                    "train_eval_task_loss": row.get("train_eval_task_loss", ""),
                    "valid_eval_task_loss": row.get("valid_eval_task_loss", "")}
            for split, field in (("train", "train_clean"), ("valid", "valid_clean")):
                for metric in METRICS:
                    flat[f"{split}_{metric}"] = (row[field][metric] if field in row else "")
            epoch_rows.append(flat)
        new_histories[seed], baseline_histories[seed] = history, baseline
        if all("train_clean" in row for row in history):
            plot_primary(history, best_epoch, output_dir)
    write_csv(output_dir / "seed_summary.csv", summary_rows)
    write_csv(output_dir / "epoch_metrics.csv", epoch_rows)
    plot_validation_comparison(new_histories, baseline_histories, output_dir)
    plot_selected_comparison(summary_rows, output_dir)
    aggregate = {}
    for metric in METRICS:
        current = np.asarray([r[f"valid_{metric}"] for r in summary_rows], dtype=float)
        baseline = np.asarray([r[f"baseline_valid_{metric}"] for r in summary_rows], dtype=float)
        aggregate[f"valid_{metric}_mean"] = float(current.mean())
        aggregate[f"valid_{metric}_sd"] = float(current.std(ddof=1))
        aggregate[f"baseline_valid_{metric}_mean"] = float(baseline.mean())
        aggregate[f"delta_{metric}_mean"] = float((current - baseline).mean())
    result = {"runs": summary_rows, "aggregate": aggregate}
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "cache")
    parser.add_argument("--runs", type=Path, default=root / "outputs/hidden96_lr_drop")
    parser.add_argument("--baseline", type=Path, default=root / "outputs")
    parser.add_argument("--out", type=Path, default=root / "outputs/hidden96_lr_drop/report")
    parser.add_argument("--seeds", nargs="+", type=int, default=(2026, 2027, 2028))
    args = parser.parse_args()
    generate(args.data, args.runs, args.baseline, args.out, tuple(args.seeds))


if __name__ == "__main__":
    main()
