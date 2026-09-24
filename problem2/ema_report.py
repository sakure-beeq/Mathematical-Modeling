"""Summarize three EMA training runs and draw their learning curves."""

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


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_single(history: list[dict], best_epoch: int, seed: int,
                output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    epochs = [r["epoch"] for r in history]
    ema_epochs = [r["epoch"] for r in history if "ema_valid_clean" in r]
    ema_rows = [r for r in history if "ema_valid_clean" in r]
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.0), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(epochs, [r["train_loss"] for r in history], "o-", color="#0072B2",
            label="Train joint loss")
    ax.plot(epochs, [r["train_eval_task_loss"] for r in history], "s--",
            color="#009E73", label="Train task loss, raw")
    ax.plot(epochs, [r["valid_eval_task_loss"] for r in history], "^--",
            color="#D55E00", label="Valid task loss, raw")
    ax.plot(ema_epochs, [r["ema_valid_eval_task_loss"] for r in ema_rows], "d-",
            color="#CC79A7", label="Valid task loss, EMA")
    ax.set(title="Loss", ylabel="Loss")
    ax.legend(fontsize=7, loc="upper right")
    for ax, metric, title, ylabel in (
            (axes[0, 1], "accuracy", "Classification accuracy", "Accuracy"),
            (axes[1, 0], "macro_f1", "Classification Macro F1", "Macro F1"),
            (axes[1, 1], "mae", "Intensity MAE", "MAE")):
        for condition, rows, xs, label, color, style in (
                ("train_clean", history, epochs, "Train, raw", "#0072B2", "o-"),
                ("valid_clean", history, epochs, "Valid, raw", "#D55E00", "s-"),
                ("ema_train_clean", ema_rows, ema_epochs,
                 "Train, EMA", "#009E73", "^--"),
                ("ema_valid_clean", ema_rows, ema_epochs,
                 "Valid, EMA", "#CC79A7", "d--")):
            ax.plot(xs, [r[condition][metric] for r in rows], style,
                    color=color, linewidth=1.6, label=label)
        ax.set(title=title, ylabel=ylabel)
    for ax in axes.flat:
        ax.axvline(5.5, color="#777777", linestyle=":", linewidth=1.2,
                   label="LR drops at epoch 6")
        ax.axvline(best_epoch, color="#009E73", linestyle="--", linewidth=1.0,
                   label=f"Selected epoch {best_epoch}")
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.18)
    axes[0, 1].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"EMA experiment, seed {seed}", fontsize=13)
    fig.savefig(output_dir / f"seed{seed}_training_curves.png", dpi=300)
    fig.savefig(output_dir / f"seed{seed}_training_curves.pdf")
    plt.close(fig)


def plot_seed_comparison(runs: dict[int, list[dict]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9), constrained_layout=True)
    colors = ("#0072B2", "#D55E00", "#009E73")
    for ax, metric, title in zip(
            axes, ("accuracy", "macro_f1", "mae"),
            ("Validation accuracy", "Validation Macro F1", "Validation MAE")):
        for (seed, history), color in zip(sorted(runs.items()), colors):
            raw = [r for r in history if "ema_valid_clean" in r]
            xs = [r["epoch"] for r in raw]
            ax.plot(xs, [r["valid_clean"][metric] for r in raw], ":",
                    color=color, alpha=.55, linewidth=1.5)
            ax.plot(xs, [r["ema_valid_clean"][metric] for r in raw], "o-",
                    color=color, markersize=3.5, linewidth=1.8, label=f"Seed {seed}, EMA")
        ax.set(title=title, xlabel="Epoch", ylabel=metric.upper() if metric == "mae" else title.split()[-1])
        ax.grid(alpha=.18)
    axes[0].legend(fontsize=7)
    fig.suptitle("Solid: EMA weights    Dotted: current weights", fontsize=11)
    fig.savefig(output_dir / "three_seed_validation_curves.png", dpi=300)
    fig.savefig(output_dir / "three_seed_validation_curves.pdf")
    plt.close(fig)


def plot_selected_comparison(summary_rows: list[dict], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9), constrained_layout=True)
    x = np.arange(len(summary_rows))
    for ax, metric, title in zip(
            axes, ("accuracy", "macro_f1", "mae"),
            ("Validation accuracy", "Validation Macro F1", "Validation MAE")):
        raw = [row[f"raw_best_valid_{metric}"] for row in summary_rows]
        ema = [row[f"valid_{metric}"] for row in summary_rows]
        left = ax.bar(x - .18, raw, width=.35, color="#D55E00", label="Current weights")
        right = ax.bar(x + .18, ema, width=.35, color="#0072B2", label="EMA weights")
        ax.bar_label(left, fmt="%.3f", fontsize=8, padding=2)
        ax.bar_label(right, fmt="%.3f", fontsize=8, padding=2)
        ax.set(title=title, xticks=x,
               xticklabels=[str(row["seed"]) for row in summary_rows], xlabel="Seed")
        ax.set_ylim(0, max(raw + ema) * 1.15)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=8, loc="lower center")
    fig.savefig(output_dir / "selected_model_comparison.png", dpi=300)
    fig.savefig(output_dir / "selected_model_comparison.pdf")
    plt.close(fig)


def generate(data_dir: Path, runs_dir: Path, output_dir: Path,
             seeds: tuple[int, ...]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "savefig.bbox": "tight"})

    histories = {}
    epoch_rows = []
    summary_rows = []
    device = torch.device("cpu")
    for seed in seeds:
        run_dir = runs_dir / f"seed{seed}"
        history = json.loads((run_dir / "training_history.json").read_text(encoding="utf-8"))
        checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        if config.get("seed") != seed or config.get("ema_start_epoch") != 5:
            raise ValueError(f"unexpected config in seed {seed} run")
        best_epoch = int(checkpoint["epoch"])
        kind = checkpoint["weights_kind"]
        best_row = next(r for r in history if r["epoch"] == best_epoch)
        if not np.isclose(best_row["selection"], max(r["selection"] for r in history),
                          rtol=0, atol=1e-6):
            raise ValueError(f"seed {seed} checkpoint does not match best score")
        expected_field = "ema_valid_clean" if kind == "ema" else "valid_clean"
        raw_best = max(history, key=lambda r: r["raw_selection"])
        post_drop = [r for r in history if r["epoch"] >= 6 and "ema_valid_clean" in r]
        model = load_model(run_dir / "best.pt", device)
        metrics = {}
        for split in ("train", "valid"):
            loader = DataLoader(PreparedDataset(data_dir / f"{split}.npz"), batch_size=64)
            predictions = predict_loader(model, loader, device)
            metrics[split] = metrics_from_predictions(predictions)
            expected = (best_row[expected_field] if split == "valid" else
                        best_row.get("ema_train_clean" if kind == "ema" else "train_clean"))
            if expected is not None:
                for key in ("accuracy", "macro_f1", "mae"):
                    if not np.isclose(metrics[split][key], expected[key], rtol=0, atol=1e-8):
                        raise ValueError(f"seed {seed} {split} {key} checkpoint mismatch")
        summary_rows.append({
            "seed": seed, "best_epoch": best_epoch, "weights_kind": kind,
            "selection": best_row["selection"],
            **{f"{split}_{key}": metrics[split][key]
               for split in ("train", "valid") for key in ("accuracy", "macro_f1", "mae")},
            "raw_best_epoch": raw_best["epoch"],
            "raw_best_selection": raw_best["raw_selection"],
            **{f"raw_best_valid_{key}": raw_best["valid_clean"][key]
               for key in ("accuracy", "macro_f1", "mae")},
            "raw_post_drop_f1_sd": float(np.std(
                [r["valid_clean"]["macro_f1"] for r in post_drop], ddof=1)),
            "ema_post_drop_f1_sd": float(np.std(
                [r["ema_valid_clean"]["macro_f1"] for r in post_drop], ddof=1)),
        })
        for row in history:
            flat = {"seed": seed, "epoch": row["epoch"],
                    "learning_rate": row["learning_rate"],
                    "train_joint_loss": row["train_loss"],
                    "train_eval_task_loss_raw": row.get("train_eval_task_loss", ""),
                    "valid_eval_task_loss_raw": row.get("valid_eval_task_loss", ""),
                    "valid_eval_task_loss_ema": row.get("ema_valid_eval_task_loss", "")}
            for name, prefix in (("train_clean", "train_raw"),
                                 ("valid_clean", "valid_raw"),
                                 ("ema_train_clean", "train_ema"),
                                 ("ema_valid_clean", "valid_ema")):
                for metric in ("accuracy", "macro_f1", "mae"):
                    flat[f"{prefix}_{metric}"] = (row[name][metric] if name in row else "")
            epoch_rows.append(flat)
        histories[seed] = history
        if all("train_clean" in row for row in history):
            plot_single(history, best_epoch, seed, output_dir)
    write_csv(output_dir / "epoch_metrics.csv", epoch_rows)
    write_csv(output_dir / "seed_summary.csv", summary_rows)
    plot_seed_comparison(histories, output_dir)
    plot_selected_comparison(summary_rows, output_dir)
    aggregate = {}
    for metric in ("accuracy", "macro_f1", "mae"):
        values = np.asarray([r[f"valid_{metric}"] for r in summary_rows], dtype=float)
        raw_values = np.asarray([r[f"raw_best_valid_{metric}"] for r in summary_rows],
                                dtype=float)
        aggregate[f"valid_{metric}_mean"] = float(values.mean())
        aggregate[f"valid_{metric}_sd"] = float(values.std(ddof=1))
        aggregate[f"raw_best_valid_{metric}_mean"] = float(raw_values.mean())
        aggregate[f"ema_minus_raw_{metric}_mean"] = float((values - raw_values).mean())
    result = {"runs": summary_rows, "aggregate": aggregate}
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "cache")
    parser.add_argument("--runs", type=Path, default=root / "outputs/ema_lr_drop")
    parser.add_argument("--out", type=Path, default=root / "outputs/ema_lr_drop/report")
    parser.add_argument("--seeds", nargs="+", type=int, default=(2026, 2027, 2028))
    args = parser.parse_args()
    generate(args.data, args.runs, args.out, tuple(args.seeds))


if __name__ == "__main__":
    main()
