"""Split and sentiment-label metrics for the selected problem 2 model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import PreparedDataset
from experiment import CLASSES, load_model, metrics_from_predictions, predict_loader


def metrics_by_label(predictions: dict, split: str) -> list[dict]:
    true = np.asarray(predictions["true_class"], dtype=int)
    pred = np.asarray(predictions["class"], dtype=int)
    score = np.asarray(predictions["score"], dtype=float)
    target = np.asarray(predictions["true_score"], dtype=float)
    rows = []
    for class_index, label in enumerate(CLASSES):
        positive = true == class_index
        predicted_positive = pred == class_index
        tp = int(np.sum(positive & predicted_positive))
        fn = int(np.sum(positive & ~predicted_positive))
        fp = int(np.sum(~positive & predicted_positive))
        tn = int(np.sum(~positive & ~predicted_positive))
        support = tp + fn
        class_f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        rest_f1 = 2 * tn / max(2 * tn + fp + fn, 1)
        rows.append({
            "split": split,
            "label": label,
            "n": support,
            # Accuracy restricted to samples truly in this class equals recall.
            "class_accuracy_recall": tp / support if support else float("nan"),
            "one_vs_rest_accuracy": (tp + tn) / len(true),
            "class_f1": class_f1,
            "one_vs_rest_macro_f1": (class_f1 + rest_f1) / 2,
            "mae_true_class": float(np.mean(np.abs(score[positive] - target[positive])))
            if support else float("nan"),
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, overall_rows: list[dict], class_rows: list[dict]) -> None:
    lines = [
        "# 最终模型按数据划分与情感标签的指标",
        "",
        "每个标签的 Accuracy 采用该标签对其余标签的一对其余准确率；",
        "Macro F1 采用该标签对其余标签的二分类宏平均 F1。",
        "同时列出该标签的类内准确率（等于召回率）和该类 F1，便于识别一对其余指标受其他类样本影响的情况。",
        "MAE 仅在真实标签属于该类的样本上计算情感强度绝对误差。",
        "总体 Macro F1 则按三个情感类的 F1 取平均。",
        "",
        "## 总体指标",
        "",
        "| 划分 | N | Accuracy | Macro F1 | MAE | Pearson r |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in overall_rows:
        lines.append(f"| {row['split']} | {row['n']} | {row['accuracy']:.4f} | "
                     f"{row['macro_f1']:.4f} | {row['mae']:.4f} | {row['pearson']:.4f} |")
    lines.extend([
        "", "## 分标签指标", "",
        "| 划分 | 标签 | N | Accuracy 一对其余 | Macro F1 一对其余 | 类内准确率 召回率 | 该类 F1 | 强度 MAE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in class_rows:
        lines.append(f"| {row['split']} | {row['label']} | {row['n']} | "
                     f"{row['one_vs_rest_accuracy']:.4f} | "
                     f"{row['one_vs_rest_macro_f1']:.4f} | "
                     f"{row['class_accuracy_recall']:.4f} | "
                     f"{row['class_f1']:.4f} | {row['mae_true_class']:.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate(data_dir: Path, checkpoint: Path, output_dir: Path,
             batch_size: int = 64, device_name: str = "cpu") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    model = load_model(checkpoint, device)
    class_rows = []
    overall_rows = []
    for split in ("train", "valid", "test"):
        loader = DataLoader(PreparedDataset(data_dir / f"{split}.npz"),
                            batch_size=batch_size)
        predictions = predict_loader(model, loader, device)
        overall = metrics_from_predictions(predictions)
        class_rows.extend(metrics_by_label(predictions, split))
        overall_rows.append({"split": split, "n": len(predictions["id"]),
                             **{key: overall[key] for key in
                                ("accuracy", "macro_f1", "weighted_f1", "mae", "pearson")}})
        if split in ("valid", "test"):
            existing = json.loads((output_dir / f"{split}_metrics.json").read_text())
            for key in ("accuracy", "macro_f1", "weighted_f1", "mae", "pearson"):
                if not np.isclose(overall[key], existing[key], rtol=0, atol=1e-8):
                    raise ValueError(f"{split} {key} differs from saved final metrics")
        else:
            (output_dir / "train_metrics.json").write_text(
                json.dumps(overall, indent=2), encoding="utf-8")
            with (output_dir / "train_predictions.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(("id", "true_class", "pred_class", "true_score",
                                 "pred_score", "p_negative", "p_neutral", "p_positive"))
                for i, sample_id in enumerate(predictions["id"]):
                    writer.writerow((sample_id,
                                     CLASSES[int(predictions["true_class"][i])],
                                     CLASSES[int(predictions["class"][i])],
                                     predictions["true_score"][i],
                                     predictions["score"][i],
                                     *predictions["probabilities"][i]))
        print(f"{split}: n={len(predictions['id'])}, macro_f1={overall['macro_f1']:.4f}", flush=True)
    write_csv(output_dir / "label_metrics.csv", class_rows)
    write_csv(output_dir / "overall_split_metrics.csv", overall_rows)
    write_markdown(output_dir / "label_metrics.md", overall_rows, class_rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "problem2/cache")
    parser.add_argument("--checkpoint", type=Path,
                        default=root / "problem2/outputs/final_model.pt")
    parser.add_argument("--out", type=Path, default=root / "problem2/outputs/final")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    generate(args.data, args.checkpoint, args.out, args.batch_size, args.device)


if __name__ == "__main__":
    main()
