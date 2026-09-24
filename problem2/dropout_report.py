"""Compare dropout 0.3 with dropout 0.2 on matched training seeds."""

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


def plot_primary(history: list[dict], best_epoch: int, seed: int,
                 output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7), constrained_layout=True)
    ax = axes[0, 0]
    for key, label, style, color in (
            ("train_loss", "Train joint loss", "o-", "#0072B2"),
            ("train_eval_task_loss", "Train complete-input task loss", "s--", "#009E73"),
            ("valid_eval_task_loss", "Valid complete-input task loss", "^--", "#D55E00")):
        ax.plot(epochs, [r[key] for r in history], style,
                color=color, linewidth=1.8, label=label)
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
        ax.axvline(5.5, color="#CC79A7", linestyle=":", linewidth=1.4,
                   label="LR reduced at epoch 6")
        ax.axvline(best_epoch, color="#009E73", linestyle="--", linewidth=1.2,
                   label=f"Selected epoch {best_epoch}")
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=.18)
    axes[0, 1].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Dropout 0.3, seed {seed}", fontsize=13)
    fig.savefig(output_dir / f"seed{seed}_training_curves.png", dpi=300)
    fig.savefig(output_dir / f"seed{seed}_training_curves.pdf")
    plt.close(fig)


def plot_validation_curves(histories: dict[int, list[dict]],
                           baselines: dict[int, list[dict]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4), constrained_layout=True)
    for ax, metric, title in zip(
            axes, METRICS,
            ("Validation accuracy", "Validation Macro F1", "Validation MAE")):
        for (seed, history), color in zip(sorted(histories.items()),
                                          ("#0072B2", "#D55E00", "#009E73")):
            baseline = baselines[seed]
            ax.plot([r["epoch"] for r in baseline],
                    [r["valid_clean"][metric] for r in baseline], ":",
                    color=color, alpha=.65, linewidth=1.7)
            ax.plot([r["epoch"] for r in history],
                    [r["valid_clean"][metric] for r in history], "o-",
                    color=color, linewidth=1.7, markersize=3.5, label=str(seed))
        ax.axvline(5.5, color="#777777", linestyle="--", alpha=.5)
        ax.set(title=title, xlabel="Epoch", ylabel=title.split()[-1])
        ax.grid(alpha=.18)
    axes[0].legend(title="Seed", fontsize=8)
    fig.suptitle("Solid: dropout 0.3    Dotted: dropout 0.2", fontsize=11)
    fig.savefig(output_dir / "three_seed_validation_curves.png", dpi=300)
    fig.savefig(output_dir / "three_seed_validation_curves.pdf")
    plt.close(fig)


def plot_selected_comparison(rows: list[dict], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9), constrained_layout=True)
    x = np.arange(len(rows))
    for ax, metric, title in zip(
            axes, METRICS,
            ("Validation accuracy", "Validation Macro F1", "Validation MAE")):
        baseline = [r[f"baseline_{metric}"] for r in rows]
        new = [r[f"dropout03_{metric}"] for r in rows]
        a = ax.bar(x - .18, baseline, width=.35, color="#D55E00", label="Dropout 0.2")
        b = ax.bar(x + .18, new, width=.35, color="#0072B2", label="Dropout 0.3")
        ax.bar_label(a, fmt="%.3f", fontsize=8, padding=2)
        ax.bar_label(b, fmt="%.3f", fontsize=8, padding=2)
        ax.set(title=title, xticks=x, xticklabels=[str(r["seed"]) for r in rows],
               xlabel="Seed", ylim=(0, max(baseline + new) * 1.15))
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=8, loc="lower center")
    fig.savefig(output_dir / "selected_model_comparison.png", dpi=300)
    fig.savefig(output_dir / "selected_model_comparison.pdf")
    plt.close(fig)


def generate(data_dir: Path, run_dir: Path, baseline_dir: Path,
             output_dir: Path, seeds: tuple[int, ...]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "savefig.bbox": "tight"})
    histories, baselines, epoch_rows, summary_rows = {}, {}, [], []
    device = torch.device("cpu")
    for seed in seeds:
        current = run_dir / f"seed{seed}"
        history = json.loads((current / "training_history.json").read_text(encoding="utf-8"))
        baseline = json.loads((baseline_dir / f"seed{seed}" / "training_history.json")
                              .read_text(encoding="utf-8"))
        checkpoint = torch.load(current / "best.pt", map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        if (config["seed"] != seed or config["dropout"] != .3 or
                config["lr_drop_epoch"] != 6 or config["ema_start_epoch"] is not None):
            raise ValueError(f"unexpected training config for seed {seed}")
        best = next(r for r in history if r["epoch"] == checkpoint["epoch"])
        if not np.isclose(best["selection"], max(r["selection"] for r in history),
                          atol=1e-6, rtol=0):
            raise ValueError(f"checkpoint does not match seed {seed} log")
        raw_baseline_best = max(baseline, key=lambda r: r["raw_selection"])
        model = load_model(current / "best.pt", device)
        selected_metrics = {}
        for split in ("train", "valid"):
            loader = DataLoader(PreparedDataset(data_dir / f"{split}.npz"), batch_size=64)
            selected_metrics[split] = metrics_from_predictions(
                predict_loader(model, loader, device))
            expected = (best.get("train_clean") if split == "train" else best["valid_clean"])
            if expected is not None:
                for key in METRICS:
                    if not np.isclose(selected_metrics[split][key], expected[key],
                                      rtol=0, atol=1e-8):
                        raise ValueError(f"seed {seed} {split} {key} checkpoint mismatch")
        summary_rows.append({
            "seed": seed, "best_epoch": best["epoch"],
            "baseline_best_epoch": raw_baseline_best["epoch"],
            **{f"dropout03_{split}_{key}": selected_metrics[split][key]
               for split in ("train", "valid") for key in METRICS},
            **{f"baseline_{key}": raw_baseline_best["valid_clean"][key]
               for key in METRICS},
            **{f"dropout03_{key}": selected_metrics["valid"][key]
               for key in METRICS},
        })
        for row in history:
            flat = {"seed": seed, "epoch": row["epoch"],
                    "learning_rate": row["learning_rate"],
                    "train_joint_loss": row["train_loss"],
                    "train_eval_task_loss": row.get("train_eval_task_loss", ""),
                    "valid_eval_task_loss": row.get("valid_eval_task_loss", "")}
            for name, prefix in (("train_clean", "train"), ("valid_clean", "valid")):
                for metric in METRICS:
                    flat[f"{prefix}_{metric}"] = (row[name][metric] if name in row else "")
            epoch_rows.append(flat)
        histories[seed], baselines[seed] = history, baseline
        if all("train_clean" in r for r in history):
            plot_primary(history, best["epoch"], seed, output_dir)
    write_csv(output_dir / "epoch_metrics.csv", epoch_rows)
    write_csv(output_dir / "seed_summary.csv", summary_rows)
    plot_validation_curves(histories, baselines, output_dir)
    plot_selected_comparison(summary_rows, output_dir)
    aggregate = {}
    for metric in METRICS:
        new = np.asarray([r[f"dropout03_{metric}"] for r in summary_rows])
        old = np.asarray([r[f"baseline_{metric}"] for r in summary_rows])
        aggregate[f"dropout03_{metric}_mean"] = float(new.mean())
        aggregate[f"dropout03_{metric}_sd"] = float(new.std(ddof=1))
        aggregate[f"baseline_{metric}_mean"] = float(old.mean())
        aggregate[f"delta_{metric}_mean"] = float((new - old).mean())
    result = {"runs": summary_rows, "aggregate": aggregate}
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "cache")
    parser.add_argument("--runs", type=Path, default=root / "outputs/dropout03_lr_drop")
    parser.add_argument("--baseline", type=Path, default=root / "outputs/ema_lr_drop")
    parser.add_argument("--out", type=Path,
                        default=root / "outputs/dropout03_lr_drop/report")
    parser.add_argument("--seeds", type=int, nargs="+", default=(2026, 2027, 2028))
    args = parser.parse_args()
    generate(args.data, args.runs, args.baseline, args.out, tuple(args.seeds))


if __name__ == "__main__":
    main()
