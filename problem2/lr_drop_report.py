"""Report the epoch-6 learning-rate experiment without replacing the original model."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from class_metrics import metrics_by_label
from data import PreparedDataset
from experiment import load_model, metrics_from_predictions, predict_loader


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_unchanged_start(history: list[dict], baseline_path: Path, drop_epoch: int) -> None:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    for old, new in zip(baseline[:drop_epoch - 1], history[:drop_epoch - 1]):
        if old["epoch"] != new["epoch"]:
            raise ValueError("epoch order differs before learning-rate drop")
        for key in ("train_loss", "selection"):
            if not np.isclose(old[key], new[key], rtol=0, atol=1e-6):
                raise ValueError(f"epoch {old['epoch']} {key} differs before LR drop")
        for condition in ("valid_clean", "valid_missing_30pct"):
            for key in ("accuracy", "macro_f1", "mae"):
                if not np.isclose(old[condition][key], new[condition][key],
                                  rtol=0, atol=1e-6):
                    raise ValueError(f"epoch {old['epoch']} {condition} {key} differs")


def plot_epoch_metrics(history: list[dict], best_epoch: int, drop_epoch: int,
                       output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(11.3, 7), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(epochs, [r["train_loss"] for r in history], "o-", color="#0072B2",
            label="Train, joint objective")
    ax.plot(epochs, [r["train_eval_task_loss"] for r in history], "s--",
            color="#009E73", label="Train, complete-input task loss")
    ax.plot(epochs, [r["valid_eval_task_loss"] for r in history], "^--",
            color="#D55E00", label="Valid, complete-input task loss")
    ax.set(title="Loss", ylabel="Loss")
    ax.legend(fontsize=8)
    for ax, key, title, ylabel in (
            (axes[0, 1], "accuracy", "Classification accuracy", "Accuracy"),
            (axes[1, 0], "macro_f1", "Classification Macro F1", "Macro F1"),
            (axes[1, 1], "mae", "Intensity regression MAE", "MAE")):
        ax.plot(epochs, [r["train_clean"][key] for r in history], "o-",
                color="#0072B2", label="Train, complete")
        ax.plot(epochs, [r["valid_clean"][key] for r in history], "s-",
                color="#D55E00", label="Valid, complete")
        ax.set(title=title, ylabel=ylabel)
    for ax in axes.flat:
        ax.axvline(drop_epoch - 0.5, color="#CC79A7", linestyle=":", linewidth=1.6,
                   label=f"LR reduced at epoch {drop_epoch}")
        ax.axvline(best_epoch, color="#009E73", linestyle="--", linewidth=1.2,
                   label=f"Selected epoch {best_epoch}")
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.2)
    axes[0, 1].legend(fontsize=8, loc="lower right")
    fig.savefig(output_dir / "training_curves.png", dpi=300)
    fig.savefig(output_dir / "training_curves.pdf")
    plt.close(fig)


def plot_label_metrics(class_rows: list[dict], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    labels = ("Negative", "Neutral", "Positive")
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9), constrained_layout=True)
    x = np.arange(3)
    for ax, key, title in (
            (axes[0], "one_vs_rest_accuracy", "Accuracy (one vs rest)"),
            (axes[1], "one_vs_rest_macro_f1", "Macro F1 (one vs rest)"),
            (axes[2], "mae_true_class", "Intensity MAE (true class)")):
        for offset, split, color in ((-0.18, "train", "#0072B2"),
                                     (0.18, "valid", "#D55E00")):
            values = [next(row[key] for row in class_rows
                           if row["split"] == split and row["label"] == label)
                      for label in labels]
            bars = ax.bar(x + offset, values, width=0.35, color=color, label=split.title())
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
        ax.set(title=title, xticks=x, xticklabels=labels)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(row[key] for row in class_rows) * 1.17)
    axes[0].legend(fontsize=8)
    fig.savefig(output_dir / "label_metrics.png", dpi=300)
    fig.savefig(output_dir / "label_metrics.pdf")
    plt.close(fig)


def generate(data_dir: Path, run_dir: Path, baseline_history: Path,
             output_dir: Path, batch_size: int = 64) -> None:
    history = json.loads((run_dir / "training_history.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "training_config.json").read_text(encoding="utf-8"))
    drop_epoch = config["lr_drop_epoch"]
    if drop_epoch is None or any("train_clean" not in row for row in history):
        raise ValueError("run must have an LR drop and per-epoch training metrics")
    verify_unchanged_start(history, baseline_history, drop_epoch)
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    best_epoch = int(checkpoint["epoch"])
    if best_epoch != max(history, key=lambda row: row["selection"])["epoch"]:
        raise ValueError("checkpoint epoch disagrees with selection history")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "savefig.bbox": "tight"})

    epoch_rows = []
    for row in history:
        output = {"epoch": row["epoch"], "learning_rate": row["learning_rate"],
                  "train_joint_loss": row["train_loss"],
                  "train_eval_task_loss": row["train_eval_task_loss"],
                  "valid_eval_task_loss": row["valid_eval_task_loss"]}
        for condition, prefix in (("train_clean", "train"),
                                  ("valid_clean", "valid"),
                                  ("valid_missing_30pct", "valid_missing_30pct")):
            for key in ("accuracy", "macro_f1", "mae"):
                output[f"{prefix}_{key}"] = row[condition][key]
        epoch_rows.append(output)
    write_csv(output_dir / "epoch_metrics.csv", epoch_rows)

    model = load_model(run_dir / "best.pt", torch.device("cpu"))
    overall_rows, class_rows = [], []
    for split in ("train", "valid"):
        loader = DataLoader(PreparedDataset(data_dir / f"{split}.npz"),
                            batch_size=batch_size, shuffle=False)
        predictions = predict_loader(model, loader, torch.device("cpu"))
        metrics = metrics_from_predictions(predictions)
        overall_rows.append({"split": split, "n": len(predictions["id"]),
                             **{key: metrics[key] for key in ("accuracy", "macro_f1", "mae")}})
        class_rows.extend(metrics_by_label(predictions, split))
        if split == "train":
            logged = history[best_epoch - 1]["train_clean"]
        else:
            logged = history[best_epoch - 1]["valid_clean"]
        for key in ("accuracy", "macro_f1", "mae"):
            if not np.isclose(metrics[key], logged[key], atol=1e-8, rtol=0):
                raise ValueError(f"checkpoint {split} {key} differs from epoch log")
    write_csv(output_dir / "overall_metrics.csv", overall_rows)
    write_csv(output_dir / "label_metrics.csv", class_rows)
    plot_epoch_metrics(history, best_epoch, drop_epoch, output_dir)
    plot_label_metrics(class_rows, output_dir)
    summary = {"best_epoch": best_epoch, "lr_drop_epoch": drop_epoch,
               "initial_lr": config["lr"],
               "reduced_lr": config["lr"] * config["lr_drop_factor"],
               "best_selection_score": history[best_epoch - 1]["selection"],
               "overall": overall_rows}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "cache")
    parser.add_argument("--run", type=Path, default=root / "outputs/lr_drop_epoch6")
    parser.add_argument("--baseline", type=Path,
                        default=root / "ablations/fixed_gate/training_history.json")
    parser.add_argument("--out", type=Path, default=root / "outputs/lr_drop_epoch6/report")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    generate(args.data, args.run, args.baseline, args.out, args.batch_size)


if __name__ == "__main__":
    main()
