"""Validate and present all unlabeled attachment 3 predictions."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


LABELS = ("Negative", "Neutral", "Positive")
COLORS = {"Negative": "#0072B2", "Neutral": "#E69F00", "Positive": "#D55E00"}


def generate(predictions_path: Path, output_dir: Path,
             previous_path: Path | None = None) -> None:
    with predictions_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["id"] for row in rows]
    if len(rows) != 30 or len(set(ids)) != 30:
        raise ValueError("expected 30 unique attachment 3 samples")
    probs = np.array([[float(row[f"p_{name.lower()}"]) for name in LABELS]
                      for row in rows])
    scores = np.array([float(row["pred_score"]) for row in rows])
    missing = np.array([float(row["missing_ratio"]) for row in rows])
    labels = [row["pred_class"] for row in rows]
    if (not np.isfinite(probs).all() or not np.isfinite(scores).all()
            or np.any(np.abs(probs.sum(axis=1) - 1) > 2e-5)
            or np.any(probs < 0) or np.any(probs > 1)
            or np.any(scores < -3) or np.any(scores > 3)
            or np.any(missing < 0) or np.any(missing > 1)
            or any(LABELS[i] != label for i, label in zip(probs.argmax(axis=1), labels))):
        raise ValueError("predictions contain invalid scores, probabilities, or labels")

    confidence = probs.max(axis=1)
    margin = np.sort(probs, axis=1)[:, -1] - np.sort(probs, axis=1)[:, -2]
    changed = []
    if previous_path is not None and previous_path.exists():
        with previous_path.open(newline="", encoding="utf-8") as handle:
            previous = {row["id"]: row for row in csv.DictReader(handle)}
        if set(previous) != set(ids):
            raise ValueError("previous prediction IDs do not match")
        changed = [{"id": row["id"], "previous": previous[row["id"]]["pred_class"],
                    "current": row["pred_class"]} for row in rows
                   if previous[row["id"]]["pred_class"] != row["pred_class"]]

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": "fixed_gate, lr reduced from 2e-4 to 5e-5 at epoch 6, selected epoch 9",
        "n": len(rows),
        "class_counts": dict(Counter(labels)),
        "mean_score_by_predicted_class": {
            label: float(np.mean(scores[np.array(labels) == label])) for label in LABELS
        },
        "missing_type_counts": dict(Counter(row["missing_type"] for row in rows)),
        "mean_missing_ratio": float(missing.mean()),
        "mean_missing_ratio_nonzero": float(missing[missing > 0].mean()),
        "max_missing_ratio": float(missing.max()),
        "score_mean": float(scores.mean()),
        "score_median": float(np.median(scores)),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "mean_max_softmax_probability": float(confidence.mean()),
        "low_max_probability_below_0_55": [ids[i] for i in np.flatnonzero(confidence < .55)],
        "top_two_probability_margin_below_0_05": [ids[i] for i in np.flatnonzero(margin < .05)],
        "class_score_sign_disagreement": [ids[i] for i, label in enumerate(labels)
                                          if (label == "Negative" and scores[i] > 0)
                                          or (label == "Positive" and scores[i] < 0)],
        "changed_class_from_original_model": changed,
        "ground_truth_available": False,
    }
    (output_dir / "attachment3_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# 附件 3 全量预测结果",
        "",
        "模型：第 6 轮学习率从 2e-4 降至 5e-5，验证集选中的第 9 轮 fixed_gate 权重。",
        "附件 3 共 30 条、无真实标签；无法计算 Accuracy、F1、MAE 或 Pearson。",
        "类别概率为模型 softmax 输出，未经概率校准。缺失类型表示输入样本原有缺失，缺失比例是原本有效位置中的比例。",
        "",
        "| ID | 预测类别 | 强度 [-3,3] | P(Negative) | P(Neutral) | P(Positive) | 缺失类型 | 缺失比例 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row['id']} | {row['pred_class']} | {float(row['pred_score']):+.4f} | "
                     f"{float(row['p_negative']):.4f} | {float(row['p_neutral']):.4f} | "
                     f"{float(row['p_positive']):.4f} | {row['missing_type']} | "
                     f"{100 * float(row['missing_ratio']):.1f}% |")
    (output_dir / "attachment3_predictions.md").write_text("\n".join(lines) + "\n",
                                                        encoding="utf-8")

    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.bbox": "tight"})
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 11.5), sharey=True,
                             gridspec_kw={"width_ratios": [1.5, 2.2, 1.5]},
                             constrained_layout=True)
    y = np.arange(len(rows))
    image = axes[0].imshow(probs, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axes[0].set(title="Class probabilities", xticks=np.arange(3),
                xticklabels=("Neg", "Neu", "Pos"), yticks=y,
                yticklabels=[f"{i+1:02d}" for i in range(len(rows))],
                ylabel="Attachment 3 sample")
    for i in range(len(rows)):
        for j in range(3):
            axes[0].text(j, i, f"{probs[i,j]:.2f}", ha="center", va="center",
                         color="white" if probs[i,j] > .55 else "black", fontsize=7)
    fig.colorbar(image, ax=axes[0], fraction=.05, pad=.02)
    axes[1].barh(y, scores, color=[COLORS[label] for label in labels], height=.7)
    axes[1].axvline(0, color="black", linewidth=.8)
    axes[1].set(title="Predicted intensity", xlim=(-2, 2), xlabel="Score [-3, 3]")
    axes[1].grid(axis="x", alpha=.18)
    axes[2].barh(y, missing * 100, color=["#BDBDBD" if row["missing_type"] == "none"
                                           else "#CC79A7" for row in rows], height=.7)
    axes[2].set(title="Original missing ratio", xlim=(0, 36), xlabel="Percent")
    axes[2].grid(axis="x", alpha=.18)
    fig.savefig(output_dir / "attachment3_predictions_chart.png", dpi=300)
    fig.savefig(output_dir / "attachment3_predictions_chart.pdf")
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path,
                        default=root / "outputs/lr_drop_epoch6/report/attachment3_predictions.csv")
    parser.add_argument("--out", type=Path,
                        default=root / "outputs/lr_drop_epoch6/report")
    parser.add_argument("--previous", type=Path,
                        default=root / "outputs/final/attachment3_predictions.csv")
    args = parser.parse_args()
    generate(args.predictions, args.out, args.previous)


if __name__ == "__main__":
    main()
