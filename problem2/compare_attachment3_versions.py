"""Compare class predictions for the paired aligned and unaligned attachment 3 files."""

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


def read_predictions(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {row["id"]: row for row in rows}
    if len(rows) != 30 or len(result) != 30:
        raise ValueError(f"expected 30 unique predictions in {path}")
    return result


def generate(aligned_path: Path, unaligned_path: Path, output_dir: Path) -> None:
    aligned = read_predictions(aligned_path)
    unaligned = read_predictions(unaligned_path)
    if set(aligned) != set(unaligned):
        raise ValueError("aligned and unaligned sample IDs differ")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample_id in sorted(aligned):
        a, u = aligned[sample_id], unaligned[sample_id]
        a_prob = max(float(a[f"p_{label.lower()}"]) for label in LABELS)
        u_prob = max(float(u[f"p_{label.lower()}"]) for label in LABELS)
        row = {"id": sample_id,
               "aligned_class": a["pred_class"], "aligned_score": float(a["pred_score"]),
               "aligned_max_probability": a_prob,
               "unaligned_class": u["pred_class"], "unaligned_score": float(u["pred_score"]),
               "unaligned_max_probability": u_prob,
               "class_agrees": a["pred_class"] == u["pred_class"],
               "score_difference_unaligned_minus_aligned": (
                   float(u["pred_score"]) - float(a["pred_score"])),
               "aligned_missing_type": a["missing_type"],
               "aligned_missing_ratio": float(a["missing_ratio"]),
               "unaligned_missing_type": u["missing_type"],
               "unaligned_missing_ratio": float(u["missing_ratio"])}
        rows.append(row)
    with (output_dir / "attachment3_both_versions.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    agreed = sum(row["class_agrees"] for row in rows)
    summary = {"n": len(rows), "class_agreements": agreed,
               "class_disagreements": len(rows) - agreed,
               "agreement_rate": agreed / len(rows),
               "aligned_class_counts": dict(Counter(row["aligned_class"] for row in rows)),
               "unaligned_class_counts": dict(Counter(row["unaligned_class"] for row in rows)),
               "mean_absolute_score_difference": float(np.mean([
                   abs(row["score_difference_unaligned_minus_aligned"]) for row in rows])),
               "disagreement_ids": [row["id"] for row in rows if not row["class_agrees"]],
               "ground_truth_available": False}
    (output_dir / "attachment3_both_versions_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = ["# 附件 3 对齐版与未对齐版预测类别", "",
                "两个版本均有 30 条无标签样本。对齐版和未对齐版使用各自训练、验证选权重的模型。",
                "未对齐音频与视觉按各自时间轴汇总为最多 50 个有序位置；文本从未对齐版 raw_text 使用相同冻结 BERT 编码。",
                "", "| ID | 对齐版类别 | 强度 | 未对齐版类别 | 强度 | 类别一致 |",
                "| --- | --- | ---: | --- | ---: | --- |"]
    for row in rows:
        markdown.append(f"| {row['id']} | {row['aligned_class']} | "
                        f"{row['aligned_score']:+.4f} | {row['unaligned_class']} | "
                        f"{row['unaligned_score']:+.4f} | "
                        f"{'是' if row['class_agrees'] else '否'} |")
    (output_dir / "attachment3_both_versions.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8")

    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(7.8, 11.5), constrained_layout=True)
    for i, row in enumerate(rows):
        for j, prefix in enumerate(("aligned", "unaligned")):
            label = row[f"{prefix}_class"]
            score = row[f"{prefix}_score"]
            ax.add_patch(Rectangle((j - .48, i - .46), .96, .92,
                                   facecolor=COLORS[label], edgecolor="white"))
            ax.text(j, i, f"{label}\n{score:+.2f}", ha="center", va="center",
                    fontsize=7.5, color="white" if label != "Neutral" else "black")
        if not row["class_agrees"]:
            ax.text(1.59, i, "*", ha="center", va="center", fontsize=11, color="#C00000")
    ax.set_xlim(-.5, 1.72)
    ax.set_ylim(len(rows) - .5, -.5)
    ax.set_xticks((0, 1), ("Aligned", "Unaligned"))
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(rows)), [f"{i+1:02d}" for i in range(len(rows))])
    ax.set_ylabel("Attachment 3 sample")
    ax.tick_params(length=0)
    fig.savefig(output_dir / "attachment3_both_versions.png", dpi=300)
    fig.savefig(output_dir / "attachment3_both_versions.pdf")
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned", type=Path,
                        default=root / "outputs/lr_drop_epoch6/report/attachment3_predictions.csv")
    parser.add_argument("--unaligned", type=Path,
                        default=root / "outputs/unaligned_lr_drop_epoch6/attachment3_predictions.csv")
    parser.add_argument("--out", type=Path,
                        default=root / "outputs/attachment3_both_versions")
    args = parser.parse_args()
    generate(args.aligned, args.unaligned, args.out)


if __name__ == "__main__":
    main()
