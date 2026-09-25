"""Build charts and concise interpretation notes from problem 3 outputs."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "outputs/.matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
MODALITIES = ("text", "audio", "vision")
COLORS = {"text": "#4263a9", "audio": "#d28540", "vision": "#55a386"}


def main() -> None:
    cards = json.loads((OUT / "attachment4_explanation_cards.json").read_text())
    valid = list(csv.DictReader((OUT / "valid_predictions.csv").open(encoding="utf-8-sig")))
    names = [card["id"] for card in cards]
    x = np.arange(len(cards))
    fig, ax = plt.subplots(figsize=(11, 4))
    bottom = np.zeros(len(cards))
    for name in MODALITIES:
        values = np.array([card["importance_share"][name] for card in cards])
        ax.bar(x, values, bottom=bottom, width=.76, color=COLORS[name], label=name)
        bottom += values
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Absolute Shapley share")
    ax.set_xlabel("Attachment 4 sample")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.15))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "modality_comparison.png", dpi=180)
    plt.close(fig)

    plot_dir = OUT / "importance_plots"
    plot_dir.mkdir(exist_ok=True)
    for card in cards:
        name = card["main_modality"]
        if name not in MODALITIES:
            continue
        scores = np.asarray(card["position_importance"][name])
        fig, ax = plt.subplots(figsize=(9, 2.6))
        ax.bar(np.arange(50), scores, color=COLORS[name], width=.85)
        window = card["windows"][name]
        if window:
            ax.axvspan(window["start"] - .5, window["end"] - .5,
                       color="#e8b861", alpha=.35, label="selected evidence")
        ax.set(xlabel="Aligned sequence position", ylabel="Mean positive deletion drop",
               xlim=(-.5, 49.5))
        ax.spines[["top", "right"]].set_visible(False)
        if window:
            ax.legend(loc="upper right", frameon=False)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{card['id']}.png", dpi=180)
        plt.close(fig)

    counts = Counter((row["true_class_index"], row["pred_class_index"]) for row in valid)
    class_names = ("Negative", "Neutral", "Positive")
    confusion = [[counts[str(i), str(j)] for j in range(3)] for i in range(3)]
    errors = [row for row in valid if row["true_class_index"] != row["pred_class_index"]]
    residuals = sorted(valid, key=lambda row:
                       abs(float(row["pred_score"]) - float(row["true_score"])),
                       reverse=True)
    analysis = {"n": len(valid), "wrong_classifications": len(errors),
                "confusion_matrix": confusion,
                "neutral_recall": confusion[1][1] / max(sum(confusion[1]), 1),
                "largest_score_errors": [
                    {"id": row["id"], "true_class": class_names[int(row["true_class_index"])],
                     "pred_class": row["pred_class"], "true_score": float(row["true_score"]),
                     "pred_score": float(row["pred_score"])} for row in residuals[:10]],
                "note": "Error categories are descriptive; no attachment 4 labels were used."}
    (OUT / "valid_error_analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    selected = []
    for name in MODALITIES:
        subset = [card for card in cards if card["main_modality"] == name]
        if subset:
            selected.append(max(subset, key=lambda card: card["shapley_margin"][name]))
    if not selected:
        selected = cards[:3]
    lines = ["# 问题 3 典型样本解释卡", ""]
    for card in selected:
        lines += [f"## 样本 {card['id']}", "",
                  f"原文：{card['raw_text']}", "",
                  f"预测：{card['pred_class']}，强度 {card['pred_score']:.3f}；"
                  f"主要模态：{card['main_modality']}。", "",
                  "| 模态 | 带符号 Shapley 贡献 | 绝对贡献占比 | 关键片段 | 遮挡后类别差值下降 |",
                  "| --- | ---: | ---: | --- | ---: |"]
        for name in MODALITIES:
            win = card["windows"][name]
            if win is None:
                location = "无有效特征"
                drop = "—"
            else:
                loc = win["location"]
                location = (f"{loc.get('text', '')}；" if name == "text" else "")
                location += f"{loc.get('start_sec')}–{loc.get('end_sec')} s"
                drop = f"{win['margin_drop']:.3f}"
            lines.append(f"| {name} | {card['shapley_margin'][name]:.3f} | "
                         f"{card['importance_share'][name]:.1%} | {location} | {drop} |")
        lines += ["", "解释反映模型的决策依据；时间位置可在附件 4 原视频中回看。", ""]
    (OUT / "typical_explanation_cards.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"built comparison chart, {len(cards)} position charts and typical cards")


if __name__ == "__main__":
    main()
