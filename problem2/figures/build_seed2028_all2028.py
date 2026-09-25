"""Build problem-2 figures with seed-2028 model, masks, and matched ablations."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/deliverable_seed2028_all2028"
CHARTS = OUT / "charts"
RUN = ROOT / "outputs/dropout03_lr_drop/seed2028"
REPLAY = ROOT / "outputs/dropout03_lr_drop/seed2028_metrics_replay"
LABELS = ("Negative", "Neutral", "Positive")
SPLITS = ("train", "valid", "test")
MODS = ("T", "A", "V", "TA", "TV", "AV", "TAV")
RATIOS = (0.1, 0.2, 0.3, 0.4, 0.5)
COLORS = {"train": "#0072B2", "valid": "#D55E00", "test": "#009E73"}
ABLATION_CSV = OUT / "ablation_summary_seed2028.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric(row: dict, key: str) -> float:
    return float(row[key])


def setup() -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    (OUT / ".cache").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(OUT / ".cache"))
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.frameon": False, "savefig.bbox": "tight",
    })


def save(fig, stem: str, pdf, titles: list[tuple[str, str]], description: str) -> None:
    import matplotlib.pyplot as plt

    fig.savefig(CHARTS / f"{stem}.png", dpi=300)
    fig.savefig(CHARTS / f"{stem}.pdf")
    pdf.savefig(fig)
    titles.append((stem, description))
    plt.close(fig)


def model_flow(pdf, titles) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(13.1, 5.0), constrained_layout=True)
    ax.set_xlim(0, 13.1)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, line1, line2, color):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                               linewidth=1.1, edgecolor=color, facecolor=color + "18")
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * .62, line1, ha="center", va="center",
                fontsize=9.2, weight="bold")
        ax.text(x + w / 2, y + h * .3, line2, ha="center", va="center",
                fontsize=7.6)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=12, linewidth=1.25, color="#555555"))

    for y, name, dim, color in ((3.63, "Text", "50 × 768", "#0072B2"),
                                (2.26, "Audio", "50 × 74", "#E69F00"),
                                (.89, "Vision", "50 × 35", "#009E73")):
        box(.16, y, 1.55, .96, name, dim, color)
        box(2.15, y, 2.16, .96, "Projection + temporal", "128; 2 layers; 4 heads", color)
        arrow(1.78, y + .48, 2.07, y + .48)
        arrow(4.39, y + .48, 4.8, 2.73)
    box(4.9, 2.08, 2.03, 1.3, "Cross-modal encoder", "150 tokens; 1 layer", "#7B61A8")
    box(7.4, 2.08, 1.9, 1.3, "Missing recovery", "Observed / reconstructed", "#7B61A8")
    box(9.76, 2.08, 1.4, 1.3, "Fusion", "Equal non-pad weights", "#CC79A7")
    box(11.56, 2.08, 1.35, 1.3, "Time pool", "Attention", "#CC79A7")
    arrow(7.03, 2.73, 7.29, 2.73)
    arrow(9.39, 2.73, 9.65, 2.73)
    arrow(11.25, 2.73, 11.47, 2.73)
    box(10.78, 3.82, 2.0, .72, "Class head", "3-way softmax", "#D55E00")
    box(10.78, .95, 2.0, .72, "Intensity head", "3 × tanh, [-3, 3]", "#0072B2")
    arrow(12.21, 3.49, 12.0, 3.72)
    arrow(12.21, 1.99, 12.0, 1.78)
    ax.text(6.8, .45, "P × A × Q masks  ·  contiguous missing blocks during training  ·  dropout 0.3  ·  seed 2028",
            ha="center", fontsize=9, color="#444444")
    save(fig, "01_model_flow", pdf, titles, "模型结构与数据流（种子 2028、等权融合）")


def training_curves(history: list[dict], best_epoch: int, pdf, titles) -> None:
    import matplotlib.pyplot as plt

    epochs = np.array([r["epoch"] for r in history])
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 7.1), constrained_layout=True)
    ax = axes[0, 0]
    for key, label, color, style in (("train_loss", "Train joint loss", "#0072B2", "o-"),
                                     ("train_eval_task_loss", "Train task loss", "#009E73", "s--"),
                                     ("valid_eval_task_loss", "Valid task loss", "#D55E00", "^--")):
        ax.plot(epochs, [r[key] for r in history], style, color=color, label=label)
    ax.set(title="Loss: joint train / clean-input task", ylabel="Loss")
    ax.legend(fontsize=8)
    for ax, key, label in ((axes[0, 1], "accuracy", "Accuracy"),
                           (axes[1, 0], "macro_f1", "Macro F1"),
                           (axes[1, 1], "mae", "MAE")):
        ax.plot(epochs, [r["train_clean"][key] for r in history], "o-",
                color=COLORS["train"], label="Train, complete")
        ax.plot(epochs, [r["valid_clean"][key] for r in history], "s-",
                color=COLORS["valid"], label="Valid, complete")
        ax.plot(epochs, [r["valid_missing_30pct"][key] for r in history], "^-",
                color="#E69F00", label="Valid, TAV 30% missing")
        ax.set(title=label, ylabel=label)
    for ax in axes.flat:
        ax.axvline(5.5, color="#9B59B6", linestyle=":", linewidth=1.8,
                   label="LR reduced at epoch 6")
        ax.axvline(best_epoch, color="#009E73", linestyle="--", linewidth=1.3,
                   label=f"Selected epoch {best_epoch}")
        ax.set(xlabel="Epoch", xticks=epochs)
        ax.grid(alpha=.18)
    axes[1, 0].legend(fontsize=7, loc="lower right")
    save(fig, "02_training_curves", pdf, titles,
         "整合逐轮联合训练损失、训练集完整输入和验证集完整/TAV 30% 缺失输入的 Accuracy、Macro F1、MAE；紫色虚线为第 6 轮降学习率")


def missing_training_curves(history: list[dict], best_epoch: int, pdf, titles) -> None:
    """The run logged a fixed 30% random TAV validation mask at every epoch."""
    import matplotlib.pyplot as plt

    epochs = np.array([r["epoch"] for r in history])
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 7.1), constrained_layout=True)
    axes[0, 0].plot(epochs, [r["train_loss"] for r in history], "o-",
                    color="#0072B2", label="Joint training loss")
    axes[0, 0].set(title="Training objective with missing-block augmentation", ylabel="Joint loss")
    for ax, key, title in ((axes[0, 1], "accuracy", "Accuracy"),
                           (axes[1, 0], "macro_f1", "Macro F1"),
                           (axes[1, 1], "mae", "Intensity MAE")):
        ax.plot(epochs, [r["valid_clean"][key] for r in history], "o-",
                color="#0072B2", label="Valid · complete")
        ax.plot(epochs, [r["valid_missing_30pct"][key] for r in history], "s-",
                color="#E69F00", label="Valid · TAV 30% missing")
        ax.set(title=title, ylabel=title)
    for ax in axes.flat:
        ax.axvline(5.5, color="#9B59B6", linestyle=":", linewidth=1.7,
                   label="LR reduced at epoch 6")
        ax.axvline(best_epoch, color="#009E73", linestyle="--", linewidth=1.3,
                   label=f"Selected epoch {best_epoch}")
        ax.set(xlabel="Epoch", xticks=epochs)
        ax.grid(alpha=.18)
    axes[0, 1].legend(fontsize=8, loc="lower right")
    save(fig, "02b_missing_training_curves", pdf, titles,
         "逐轮联合训练损失，以及验证集完整输入与固定随机 30% 三模态缺失输入的 Accuracy、Macro F1、MAE")


def split_metrics(overall, metrics_json, pdf, titles) -> None:
    import matplotlib.pyplot as plt

    lookup = {r["split"]: r for r in overall}
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.9), constrained_layout=True)
    for ax, key, title in zip(axes.flat, ("accuracy", "macro_f1", "mae", "pearson"),
                              ("Accuracy", "Macro F1", "Intensity MAE", "Pearson r")):
        x = np.arange(3)
        values = [metric(lookup[s], key) for s in SPLITS]
        bars = ax.bar(x, values, color=[COLORS[s] for s in SPLITS], width=.58)
        ax.set(title=title, xticks=x, xticklabels=[s.title() for s in SPLITS])
        ax.set_ylim(0, min(1.0, max(values) * 1.28))
        ax.text(0, values[0] + .018, f"{values[0]:.3f}", ha="center", va="bottom", fontsize=9)
        for j, split in enumerate(SPLITS[1:], 1):
            lo, hi = metrics_json[split]["bootstrap_95pct_ci"][key]
            ax.errorbar(j, values[j], yerr=[[values[j] - lo], [hi - values[j]]],
                        fmt="none", ecolor="#222222", capsize=4, linewidth=1)
            ax.text(j, hi + .018, f"{values[j]:.3f}", ha="center", va="bottom", fontsize=9)
        ax.grid(axis="y", alpha=.17)
        ax.set_axisbelow(True)
    save(fig, "03_split_metrics", pdf, titles, "训练/验证/测试总体指标与验证、测试 95% bootstrap 区间")


def class_metrics(rows, pdf, titles) -> None:
    import matplotlib.pyplot as plt

    lookup = {(r["split"], r["label"]): r for r in rows}
    fields = (("one_vs_rest_accuracy", "One-vs-rest accuracy"),
              ("one_vs_rest_macro_f1", "One-vs-rest Macro F1"),
              ("mae_true_class", "Intensity MAE · true class"))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
    x = np.arange(3)
    for ax, (key, title) in zip(axes, fields):
        for j, split in enumerate(SPLITS):
            vals = [metric(lookup[(split, label)], key) for label in LABELS]
            bars = ax.bar(x + (j - 1) * .23, vals, width=.22,
                          color=COLORS[split], label=split.title())
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=6.5, rotation=90)
        ax.set(title=title, xticks=x, xticklabels=LABELS,
               ylim=(0, 1.07 if key != "mae_true_class" else 1.03))
        ax.grid(axis="y", alpha=.17)
        ax.set_axisbelow(True)
    axes[0].legend(ncol=3, fontsize=8, loc="lower center")
    save(fig, "04_label_metrics", pdf, titles, "Negative/Neutral/Positive 的一对其余 Accuracy、Macro F1 和类内 MAE")


def confusion_matrices(metrics_json, pdf, titles) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.7), constrained_layout=True)
    for ax, split in zip(axes, SPLITS):
        cm = np.array(metrics_json[split]["confusion_matrix"], dtype=int)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center",
                        fontsize=8, color="white" if norm[i, j] > .52 else "black")
        ax.set(title=split.title(), xticks=range(3), yticks=range(3),
               xticklabels=("Neg", "Neu", "Pos"), yticklabels=("Neg", "Neu", "Pos"),
               xlabel="Predicted", ylabel="True")
    fig.colorbar(im, ax=axes, shrink=.72, label="Row-normalized proportion")
    save(fig, "05_confusion_matrices", pdf, titles, "三个数据划分的混淆矩阵（原始计数和按真实类别归一化比例）")


def regression_scatter(metrics_json, pdf, titles) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.75), constrained_layout=True)
    for ax, split in zip(axes, SPLITS):
        rows = read_csv(OUT / f"{split}_predictions.csv")
        truth = np.array([metric(r, "true_score") for r in rows])
        pred = np.array([metric(r, "pred_score") for r in rows])
        ax.scatter(truth, pred, s=8, alpha=.18, color=COLORS[split], rasterized=True)
        ax.plot((-3, 3), (-3, 3), "k--", linewidth=.8)
        ax.set(xlim=(-3.05, 3.05), ylim=(-3.05, 3.05), xlabel="True intensity",
               ylabel="Predicted intensity", title=f"{split.title()} · n={len(rows)}")
        ax.text(.04, .95, f"MAE {metrics_json[split]['mae']:.3f}\nr {metrics_json[split]['pearson']:.3f}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
        ax.grid(alpha=.13)
    save(fig, "06_regression_scatter", pdf, titles, "训练/验证/测试真实情感强度与预测强度散点图")


def error_slices(pdf, titles) -> None:
    import matplotlib.pyplot as plt

    data = json.loads((OUT / "valid_error_analysis.json").read_text(encoding="utf-8"))
    groups = (("short_text_le_10", "medium_text_11_to_30", "long_text_gt_30"),
              ("vision_entirely_unavailable", "vision_partly_unavailable", "vision_all_content_valid"))
    labels = (("Short", "Medium", "Long"), ("Vision absent", "Vision partly invalid", "Vision valid"))
    fig, axes = plt.subplots(1, 2, figsize=(10.9, 3.7), constrained_layout=True)
    for ax, keys, texts, title in zip(axes, groups, labels,
                                       ("By text length", "By original visual quality")):
        vals = [data[k]["macro_f1"] for k in keys]
        bars = ax.bar(np.arange(3), vals, color="#5B8FB9")
        ax.set(title=title, xticks=np.arange(3), xticklabels=texts,
               ylabel="Validation Macro F1", ylim=(0, .85))
        ax.bar_label(bars, labels=[f"{v:.3f}\nn={data[k]['n']}" for k, v in zip(keys, vals)],
                     padding=3, fontsize=8)
        ax.tick_params(axis="x", labelrotation=11)
        ax.grid(axis="y", alpha=.17)
        ax.set_axisbelow(True)
    save(fig, "07_validation_error_slices", pdf, titles, "验证集文本长度与视觉质量切片的描述性误差分析")


def robustness_charts(rows, baseline, pdf, titles) -> tuple[list[dict], list[dict]]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    random_group = defaultdict(list)
    position_group = defaultdict(list)
    for r in rows:
        mod, ratio, position = r["modality"], float(r["ratio"]), r["position"]
        position_group[(mod, position, ratio)].append(r)
        if position == "random":
            random_group[(mod, ratio)].append(r)
    random_summary = []
    for mod in MODS:
        for ratio in RATIOS:
            group = random_group[(mod, ratio)]
            if len(group) != 5:
                raise ValueError(f"expected five random placements for {mod} {ratio}")
            random_summary.append({
                "modality": mod, "target_missing_ratio": ratio,
                "actual_missing_ratio_mean": np.mean([metric(r, "actual_missing_ratio") for r in group]),
                "accuracy_mean": np.mean([metric(r, "accuracy") for r in group]),
                "macro_f1_mean": np.mean([metric(r, "macro_f1") for r in group]),
                "macro_f1_sd": np.std([metric(r, "macro_f1") for r in group], ddof=1),
                "mae_mean": np.mean([metric(r, "mae") for r in group]),
                "mae_sd": np.std([metric(r, "mae") for r in group], ddof=1),
            })
    write_csv(OUT / "missing_random5_summary.csv", random_summary)
    lookup = {(r["modality"], r["target_missing_ratio"]): r for r in random_summary}
    palette = plt.cm.tab10(np.linspace(0, 1, len(MODS)))
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3), constrained_layout=True)
    for ax, key, baseline_key, title in ((axes[0], "macro_f1", "macro_f1", "Macro F1"),
                                          (axes[1], "mae", "mae", "Intensity MAE")):
        for mod, color in zip(MODS, palette):
            values = np.array([lookup[(mod, ratio)][f"{key}_mean"] for ratio in RATIOS])
            sd = np.array([lookup[(mod, ratio)][f"{key}_sd"] for ratio in RATIOS])
            ax.plot(RATIOS, values, "o-", color=color, label=mod, linewidth=1.65)
            ax.fill_between(RATIOS, values - sd, values + sd, color=color, alpha=.11)
        ax.axhline(baseline[baseline_key], color="#333333", linestyle="--",
                   linewidth=1, label="Complete input")
        ax.set(title=title, xlabel="Target missing ratio", ylabel=title,
               xticks=RATIOS, xticklabels=[f"{r:.0%}" for r in RATIOS])
        ax.grid(alpha=.16)
    axes[0].legend(ncol=4, fontsize=8)
    save(fig, "08_missing_rate_curves", pdf, titles, "七种模态缺失组合在 10%–50% 缺失比例下的验证集 Macro F1、MAE 均值及 5 次位置重复的标准差")

    seed2028_rows = [r for r in rows if int(r["seed"]) == 2028]
    if len(seed2028_rows) != 140:
        raise ValueError("expected 140 conditions with mask seed 2028")
    write_csv(OUT / "valid_robustness_maskseed2028_only.csv", seed2028_rows)
    seed2028_random = {(r["modality"], float(r["ratio"])): r
                       for r in seed2028_rows if r["position"] == "random"}
    if len(seed2028_random) != 35:
        raise ValueError("expected 35 random-position conditions with mask seed 2028")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3), constrained_layout=True)
    for ax, key, title in ((axes[0], "macro_f1", "Macro F1"),
                           (axes[1], "mae", "Intensity MAE")):
        for mod, color in zip(MODS, palette):
            values = [metric(seed2028_random[(mod, ratio)], key) for ratio in RATIOS]
            ax.plot(RATIOS, values, "o-", color=color, label=mod, linewidth=1.65)
        ax.axhline(baseline[key], color="#333333", linestyle="--", linewidth=1,
                   label="Complete input")
        ax.set(title=title, xlabel="Target missing ratio", ylabel=title,
               xticks=RATIOS, xticklabels=[f"{r:.0%}" for r in RATIOS])
        ax.grid(alpha=.16)
    axes[0].legend(ncol=4, fontsize=8)
    fig.suptitle("Single random placement · model seed 2028 · mask seed 2028", fontsize=11)
    save(fig, "08b_missing_rate_maskseed2028", pdf, titles,
         "严格使用遮挡随机种子 2028 的七种模态×五个比例预测性能曲线")

    grids = []
    for key in ("macro_f1", "mae"):
        grids.append(np.array([[lookup[(m, r)][f"{key}_mean"] - baseline[key]
                                for r in RATIOS] for m in MODS]))
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5), constrained_layout=True)
    for ax, grid, title in zip(axes, grids, ("Change in Macro F1", "Change in MAE")):
        span = max(np.abs(grid).max(), .01)
        im = ax.imshow(grid, aspect="auto", cmap="RdBu_r",
                       norm=TwoSlopeNorm(vcenter=0, vmin=-span, vmax=span))
        ax.set(title=title, xticks=range(5), yticks=range(7),
               xticklabels=[f"{r:.0%}" for r in RATIOS], yticklabels=MODS,
               xlabel="Target missing ratio", ylabel="Masked modality")
        for i in range(7):
            for j in range(5):
                ax.text(j, i, f"{grid[i, j]:+.3f}", ha="center", va="center",
                        fontsize=7.8, color="white" if abs(grid[i, j]) > span * .62 else "black")
        fig.colorbar(im, ax=ax, shrink=.77)
    save(fig, "09_missing_rate_heatmaps", pdf, titles, "七种模态×五个缺失比例的 Macro F1、MAE 相对完整输入变化热力图")

    position_summary = []
    for mod in MODS:
        for pos in ("start", "middle", "end", "random"):
            f1 = np.mean([np.mean([metric(r, "macro_f1") for r in position_group[(mod, pos, ratio)]])
                          for ratio in RATIOS])
            mae = np.mean([np.mean([metric(r, "mae") for r in position_group[(mod, pos, ratio)]])
                           for ratio in RATIOS])
            position_summary.append({"modality": mod, "position": pos,
                                     "macro_f1_mean": f1, "mae_mean": mae,
                                     "delta_macro_f1": f1 - baseline["macro_f1"],
                                     "delta_mae": mae - baseline["mae"]})
    write_csv(OUT / "missing_position_summary.csv", position_summary)
    positions = ("start", "middle", "end", "random")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=True)
    for ax, key, title in zip(axes, ("delta_macro_f1", "delta_mae"),
                               ("Change in Macro F1", "Change in MAE")):
        grid = np.array([[next(r[key] for r in position_summary if
                               r["modality"] == m and r["position"] == p)
                          for p in positions] for m in MODS])
        span = max(np.abs(grid).max(), .01)
        im = ax.imshow(grid, aspect="auto", cmap="RdBu_r",
                       norm=TwoSlopeNorm(vcenter=0, vmin=-span, vmax=span))
        ax.set(title=title, xticks=range(4), yticks=range(7),
               xticklabels=[p.title() for p in positions], yticklabels=MODS,
               xlabel="Missing-block position", ylabel="Masked modality")
        for i in range(7):
            for j in range(4):
                ax.text(j, i, f"{grid[i, j]:+.3f}", ha="center", va="center",
                        fontsize=8, color="white" if abs(grid[i, j]) > span * .62 else "black")
        fig.colorbar(im, ax=ax, shrink=.77)
    save(fig, "10_missing_position_heatmaps", pdf, titles, "缺失位置效应，五个目标缺失比例等权平均")

    selected = [r for r in rows if r["modality"] == "TAV" and float(r["ratio"]) == .3]
    selected.sort(key=lambda r: (positions.index(r["position"]), int(r["seed"])))
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.2), constrained_layout=True)
    y = np.arange(len(selected))
    names = [r["position"].title() if r["position"] != "random"
             else f"Placement {int(r['seed']) - 2027}"
             for r in selected]
    for ax, key, title in zip(axes, ("macro_f1", "mae"),
                               ("Change in Macro F1", "Change in MAE")):
        vals = np.array([metric(r, f"delta_{key}") for r in selected])
        lo = np.array([metric(r, f"delta_{key}_ci_low") for r in selected])
        hi = np.array([metric(r, f"delta_{key}_ci_high") for r in selected])
        ax.errorbar(vals, y, xerr=[vals - lo, hi - vals], fmt="o", color="#0072B2",
                    ecolor="#666666", capsize=3)
        ax.axvline(0, color="#333333", linestyle="--", linewidth=1)
        ax.set(title=title, yticks=y, yticklabels=names, xlabel="Damaged minus complete")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.16)
    save(fig, "11_missing_30pct_paired_ci", pdf, titles, "TAV 缺失 30% 时各位置及五次随机位置的配对 bootstrap 差值区间")
    return random_summary, position_summary


def attachment3(pdf, titles) -> dict:
    import matplotlib.pyplot as plt

    rows = read_csv(OUT / "attachment3_predictions.csv")
    if len(rows) != 30 or len({r["id"] for r in rows}) != 30:
        raise ValueError("attachment 3 must have 30 unique samples")
    probs = np.array([[metric(r, f"p_{label.lower()}") for label in LABELS]
                      for r in rows])
    scores = np.array([metric(r, "pred_score") for r in rows])
    missing = np.array([metric(r, "missing_ratio") for r in rows])
    if not np.allclose(probs.sum(axis=1), 1, atol=2e-5):
        raise ValueError("attachment 3 probabilities do not sum to 1")
    count = Counter(r["pred_class"] for r in rows)
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 10.4), sharey=True,
                             gridspec_kw={"width_ratios": [1.8, 2, 1.45]},
                             constrained_layout=True)
    y = np.arange(30)
    im = axes[0].imshow(probs, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axes[0].set(title="Class probabilities", xticks=range(3),
                xticklabels=("Neg", "Neu", "Pos"), yticks=y,
                yticklabels=[r["id"].rsplit("_", 1)[-1] for r in rows],
                ylabel="Attachment 3 sample number")
    for i in range(30):
        for j in range(3):
            axes[0].text(j, i, f"{probs[i, j]:.2f}", ha="center", va="center",
                         fontsize=6.7, color="white" if probs[i, j] > .55 else "black")
    fig.colorbar(im, ax=axes[0], shrink=.42, pad=.02)
    class_colors = {"Negative": "#0072B2", "Neutral": "#E69F00", "Positive": "#D55E00"}
    axes[1].barh(y, scores, color=[class_colors[r["pred_class"]] for r in rows], height=.7)
    axes[1].axvline(0, color="#333333", linewidth=.8)
    axes[1].set(title="Predicted intensity", xlabel="Score [-3, 3]", xlim=(-3, 3))
    axes[2].barh(y, missing * 100, color="#CC79A7", height=.7)
    axes[2].set(title="Original missing ratio", xlabel="Percent", xlim=(0, 40))
    for ax in axes[1:]:
        ax.grid(axis="x", alpha=.15)
    save(fig, "12_attachment3_predictions", pdf, titles, "附件 3 全部 30 条样本的类别概率、预测强度和原始缺失比例")

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), constrained_layout=True)
    bars = axes[0].bar(LABELS, [count.get(label, 0) for label in LABELS],
                       color=[class_colors[label] for label in LABELS])
    axes[0].bar_label(bars, padding=3)
    axes[0].set(title="Predicted class distribution", ylabel="Number of samples")
    axes[1].hist(scores, bins=np.linspace(-3, 3, 13), color="#5B8FB9", edgecolor="white")
    axes[1].set(title="Predicted intensity distribution", xlabel="Score [-3, 3]",
                ylabel="Number of samples")
    save(fig, "13_attachment3_distribution", pdf, titles, "附件 3 无标签样本的预测类别数与预测强度分布")
    summary = {"n": 30, "class_counts": {label: count.get(label, 0) for label in LABELS},
               "score_mean": float(scores.mean()), "mean_missing_ratio": float(missing.mean()),
               "ground_truth_available": False}
    (OUT / "attachment3_summary_seed2028.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def matched_ablation(pdf, titles) -> None:
    import matplotlib.pyplot as plt

    rows = read_csv(ABLATION_CSV)
    names = ("Full dynamic", "No blocks", "No recon loss", "Equal fusion", "No consistency")
    if len(rows) != len(names):
        raise ValueError("expected five matched seed-2028 ablation runs")
    if any(int(r["training_seed"]) != 2028 or int(r["mask_seed"]) != 2028 or
           float(r["dropout"]) != .3 for r in rows):
        raise ValueError("ablation table includes a different training or mask configuration")
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.1), constrained_layout=True)
    for ax, keys, title in (
            (axes[0], ("valid_accuracy", "missing_30pct_accuracy"), "Accuracy"),
            (axes[1], ("valid_macro_f1", "missing_30pct_macro_f1"), "Macro F1"),
            (axes[2], ("valid_mae", "missing_30pct_mae"), "Intensity MAE")):
        for offset, key, label, color in ((-.18, keys[0], "Complete", "#0072B2"),
                                          (.18, keys[1], "30% missing", "#D55E00")):
            vals = [metric(r, key) for r in rows]
            bars = ax.bar(x + offset, vals, width=.35, color=color, label=label)
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)
        ax.set(title=title, xticks=x, xticklabels=names,
               ylim=(0, max(metric(r, k) for r in rows for k in keys) * 1.17))
        ax.tick_params(axis="x", labelrotation=17)
        ax.grid(axis="y", alpha=.17)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=8)
    fig.suptitle("Matched ablations · training and selection mask seed 2028 · dropout 0.3",
                 fontsize=11)
    save(fig, "14_architecture_ablation_seed2028", pdf, titles,
         "同一数据划分、训练种子 2028、dropout 0.3 和学习率日程下的五组架构消融")


def main() -> None:
    import torch

    setup()
    from matplotlib.backends.backend_pdf import PdfPages
    checkpoint = torch.load(RUN / "best.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if (config["seed"] != 2028 or config["dropout"] != .3 or
            config["ablation"] != "fixed_gate" or checkpoint["epoch"] != 9):
        raise ValueError("selected checkpoint identity changed")
    history = json.loads((REPLAY / "training_history.json").read_text(encoding="utf-8"))
    chosen = next(r for r in history if r["epoch"] == checkpoint["epoch"])
    if chosen["selection"] != max(r["selection"] for r in history):
        raise ValueError("checkpoint is not the best validation epoch")
    overall = read_csv(OUT / "overall_split_metrics.csv")
    by_split = {r["split"]: r for r in overall}
    for split, field in (("train", "train_clean"), ("valid", "valid_clean")):
        for key in ("accuracy", "macro_f1", "mae"):
            if not np.isclose(metric(by_split[split], key), chosen[field][key], atol=1e-8):
                raise ValueError(f"selected {split} metric does not match epoch 9")
    metrics_json = {s: json.loads((OUT / f"{s}_metrics.json").read_text(encoding="utf-8"))
                    for s in SPLITS}
    class_rows = read_csv(OUT / "label_metrics.csv")
    robustness_rows = read_csv(OUT / "valid_robustness.csv")
    if len(robustness_rows) != 280:
        raise ValueError("expected 280 robustness conditions")
    random_mask_seeds = {int(r["seed"]) for r in robustness_rows if r["position"] == "random"}
    fixed_mask_seeds = {int(r["seed"]) for r in robustness_rows if r["position"] != "random"}
    if random_mask_seeds != set(range(2028, 2033)) or fixed_mask_seeds != {2028}:
        raise ValueError("robustness masks must use base seed 2028 and five repetitions")
    selection_mask_row = next(r for r in robustness_rows if r["modality"] == "TAV" and
                              float(r["ratio"]) == .3 and r["position"] == "random" and
                              int(r["seed"]) == 2028)
    for key in ("accuracy", "macro_f1", "mae"):
        if not np.isclose(metric(selection_mask_row, key),
                          chosen["valid_missing_30pct"][key], atol=1e-8):
            raise ValueError("seed-2028 mask condition differs from selected epoch log")
    robustness_base = json.loads((OUT / "valid_robustness_baseline.json").read_text(encoding="utf-8"))
    for key in ("accuracy", "macro_f1", "mae"):
        if not np.isclose(robustness_base[key], metric(by_split["valid"], key), atol=1e-8):
            raise ValueError("robustness baseline differs from selected checkpoint")
    titles: list[tuple[str, str]] = []
    with PdfPages(OUT / "问题2_种子2028_全部图表.pdf") as pdf:
        model_flow(pdf, titles)
        training_curves(history, checkpoint["epoch"], pdf, titles)
        missing_training_curves(history, checkpoint["epoch"], pdf, titles)
        split_metrics(overall, metrics_json, pdf, titles)
        class_metrics(class_rows, pdf, titles)
        confusion_matrices(metrics_json, pdf, titles)
        regression_scatter(metrics_json, pdf, titles)
        error_slices(pdf, titles)
        missing_random_summary, missing_position_summary = robustness_charts(
            robustness_rows, robustness_base, pdf, titles)
        attachment_summary = attachment3(pdf, titles)
        matched_ablation(pdf, titles)
    shutil.copyfile(ROOT / "outputs/dropout03_lr_drop/report/seed2028_epoch_metrics.csv",
                    OUT / "seed2028_epoch_metrics.csv")
    missing_epochs = []
    for row in history:
        missing_epochs.append({
            "epoch": row["epoch"], "learning_rate": row["learning_rate"],
            "train_joint_loss": row["train_loss"], "selection_score": row["selection"],
            **{f"valid_clean_{key}": row["valid_clean"][key]
               for key in ("accuracy", "macro_f1", "mae")},
            **{f"valid_tav30_missing_{key}": row["valid_missing_30pct"][key]
               for key in ("accuracy", "macro_f1", "mae")},
        })
    write_csv(OUT / "missing_training_epoch_metrics.csv", missing_epochs)
    source_hash = hashlib.sha256((RUN / "best.pt").read_bytes()).hexdigest()
    manifest = {"checkpoint": str(RUN / "best.pt"), "sha256": source_hash,
                "seed": 2028, "selected_epoch": checkpoint["epoch"],
                "model_config": config, "figure_count": len(titles),
                "mask_seed_base": 2028, "random_mask_seeds": sorted(random_mask_seeds),
                "matched_ablation_figure": "14_architecture_ablation_seed2028",
                "attachment3": attachment_summary}
    (OUT / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result_lines = ["# 问题 2 结果汇总：随机种子 2028", "",
                    "训练所用 dropout 0.3，等权融合，第 6 轮降学习率；选中第 9 轮。",
                    "逐轮缺失验证使用固定随机位置的三模态 30% 连续块遮挡；"
                    "第 9 轮缺失输入 Accuracy、Macro F1、MAE 分别为 "
                    f"{chosen['valid_missing_30pct']['accuracy']:.4f}、"
                    f"{chosen['valid_missing_30pct']['macro_f1']:.4f}、"
                    f"{chosen['valid_missing_30pct']['mae']:.4f}。",
                    "", "## 总体指标", "",
                    "| 划分 | N | Accuracy | Macro F1 | MAE | Pearson r |",
                    "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in overall:
        result_lines.append(f"| {row['split']} | {row['n']} | {metric(row, 'accuracy'):.4f} | "
                            f"{metric(row, 'macro_f1'):.4f} | {metric(row, 'mae'):.4f} | "
                            f"{metric(row, 'pearson'):.4f} |")
    result_lines.extend(["", "## 逐标签指标", "",
                         "单标签 Accuracy 和 Macro F1 均按该类对其余两类的二分类口径；强度 MAE 在真实标签为该类的样本上计算。",
                         "", "| 划分 | 标签 | N | Accuracy | Macro F1 | MAE |",
                         "| --- | --- | ---: | ---: | ---: | ---: |"])
    for row in class_rows:
        result_lines.append(f"| {row['split']} | {row['label']} | {row['n']} | "
                            f"{metric(row, 'one_vs_rest_accuracy'):.4f} | "
                            f"{metric(row, 'one_vs_rest_macro_f1'):.4f} | "
                            f"{metric(row, 'mae_true_class'):.4f} |")
    result_lines.extend(["", "## 验证集随机位置缺失", "",
                         "下表为遮挡种子 2028–2032 的五次随机位置重复均值；"
                         "全部使用训练种子 2028 的同一最终模型。",
                         "", "| 缺失模态 | 目标比例 | 实际比例 | Macro F1 | 相对完整变化 | MAE | 相对完整变化 |",
                         "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in missing_random_summary:
        if row["target_missing_ratio"] not in (.3, .5):
            continue
        result_lines.append(f"| {row['modality']} | {row['target_missing_ratio']:.0%} | "
                            f"{row['actual_missing_ratio_mean']:.1%} | "
                            f"{row['macro_f1_mean']:.4f} | "
                            f"{row['macro_f1_mean'] - robustness_base['macro_f1']:+.4f} | "
                            f"{row['mae_mean']:.4f} | "
                            f"{row['mae_mean'] - robustness_base['mae']:+.4f} |")
    seed2028_only = read_csv(OUT / "valid_robustness_maskseed2028_only.csv")
    result_lines.extend(["", "### 严格使用遮挡种子 2028 的 50% 缺失结果", "",
                         "| 缺失模态 | Macro F1 | 相对完整变化 | MAE | 相对完整变化 |",
                         "| --- | ---: | ---: | ---: | ---: |"])
    for mod in MODS:
        row = next(r for r in seed2028_only if r["modality"] == mod and
                   r["position"] == "random" and float(r["ratio"]) == .5)
        result_lines.append(f"| {mod} | {metric(row, 'macro_f1'):.4f} | "
                            f"{metric(row, 'delta_macro_f1'):+.4f} | "
                            f"{metric(row, 'mae'):.4f} | {metric(row, 'delta_mae'):+.4f} |")
    fifty = [row for row in missing_random_summary if row["target_missing_ratio"] == .5]
    text_f1 = np.mean([row["macro_f1_mean"] for row in fifty if "T" in row["modality"]])
    nontext_f1 = np.mean([row["macro_f1_mean"] for row in fifty if "T" not in row["modality"]])
    tav = {row["target_missing_ratio"]: row for row in missing_random_summary
           if row["modality"] == "TAV"}
    tav_positions = [row for row in missing_position_summary if row["modality"] == "TAV"]
    most_sensitive = min(tav_positions, key=lambda row: row["macro_f1_mean"])
    result_lines.extend(["", "## 缺失规律（验证集描述性比较）", "",
                         f"目标缺失率 50% 时，涉及文本的四种组合平均 Macro F1 为 {text_f1:.4f}，"
                         f"不涉及文本的三种组合为 {nontext_f1:.4f}。",
                         f"TAV 目标缺失率 30% 时，单次遮挡种子 2028 的 Macro F1 为 "
                         f"{metric(selection_mask_row, 'macro_f1'):.4f}，"
                         f"五次位置重复均值为 {tav[.3]['macro_f1_mean']:.4f}，"
                         "说明单次遮挡位置会影响估计值。",
                         f"三模态 TAV 的随机位置 Macro F1 从目标缺失率 10% 的 "
                         f"{tav[.1]['macro_f1_mean']:.4f} 降至 50% 的 "
                         f"{tav[.5]['macro_f1_mean']:.4f}。",
                         f"TAV 各位置跨五个目标比例等权平均后，Macro F1 最低的是 "
                         f"{most_sensitive['position']}（{most_sensitive['macro_f1_mean']:.4f}）。",
                         "上述差异是当前模型与验证样本上的描述性结果，不代表因果效应。"])
    ablation_rows = read_csv(ABLATION_CSV)
    result_lines.extend(["", "## 同种子架构消融", "",
                         "五组模型均用训练种子 2028、dropout 0.3、第 6 轮降学习率；"
                         "30% 三模态缺失的选模遮挡种子为 2028。仅用验证集评价。",
                         "", "| 变体 | 最佳轮次 | 完整 Acc | 完整 Macro F1 | 完整 MAE | 缺失30% Acc | 缺失30% Macro F1 | 缺失30% MAE | 选模分数 |",
                         "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in ablation_rows:
        result_lines.append(f"| {row['variant']} | {row['best_epoch']} | "
                            f"{metric(row, 'valid_accuracy'):.4f} | "
                            f"{metric(row, 'valid_macro_f1'):.4f} | "
                            f"{metric(row, 'valid_mae'):.4f} | "
                            f"{metric(row, 'missing_30pct_accuracy'):.4f} | "
                            f"{metric(row, 'missing_30pct_macro_f1'):.4f} | "
                            f"{metric(row, 'missing_30pct_mae'):.4f} | "
                            f"{metric(row, 'selection_score'):.5f} |")
    result_lines.extend(["", "最终提交模型仍为用户已选定的种子 2028 等权融合检查点。",
                         "", "## 附件 3", "",
                         f"30 条无标签样本预测：Negative {attachment_summary['class_counts']['Negative']} 条、"
                         f"Neutral {attachment_summary['class_counts']['Neutral']} 条、"
                         f"Positive {attachment_summary['class_counts']['Positive']} 条。",
                         "附件 3 没有真实标签，不能计算 Accuracy、F1 或 MAE。", ""])
    (OUT / "结果汇总.md").write_text("\n".join(result_lines), encoding="utf-8")
    lines = ["# 问题 2 图表交付目录：随机种子 2028", "",
             "主结果使用 `dropout=0.3`、`fixed_gate`、第 6 轮起学习率 `5e-5`、第 9 轮最佳权重。",
             "训练集 3395 条、验证集 728 条、测试集 727 条；附件 3 为 30 条无标签样本。",
             "[复现说明](复现说明.md)列出同种子缺失实验、消融训练及附件 3 推理命令。",
             "", "## 主图", ""]
    for stem, description in titles:
        lines.append(f"- [{stem}](charts/{stem}.png) · [PDF](charts/{stem}.pdf)：{description}")
    lines += ["", "## 数值表与提交文件", "",
              "- `overall_split_metrics.csv`：训练/验证/测试总体指标。",
              "- `结果汇总.md`：总体、分标签、缺失实验关键数值表。",
              "- `seed2028_epoch_metrics.csv`：逐轮训练指标原始数值。",
              "- `missing_training_epoch_metrics.csv`：逐轮完整/三模态 30% 缺失验证指标。",
              "- `label_metrics.csv`：每个情感标签的 Accuracy、Macro F1、MAE。",
              "- `missing_random5_summary.csv`、`missing_position_summary.csv`：缺失实验汇总。",
              "- `valid_robustness.csv`：280 个缺失条件及配对 bootstrap 区间。",
              "- `valid_robustness_maskseed2028_only.csv`：严格采用遮挡种子 2028 的 140 个条件。",
              "- `attachment3_predictions.csv`：附件 3 的 30 条最终预测。",
              "- `best_seed2028.pt`：用户选定的第 9 轮等权融合模型权重。",
              "- `ablation_summary_seed2028.csv`：同种子、同配置的五组架构消融数值。",
              "- `*_predictions.csv`、`*_metrics.json`：有真值数据的逐条预测及指标。",
              "", "## 指标口径", "",
              "单标签 Accuracy 和 Macro F1 是该类对其余两类的二分类指标；MAE 在真实标签为该类的样本上计算。",
              "总体 Macro F1 是三类 F1 的平均。附件 3 无真实标签，因此只提供预测，不计算准确率或 MAE。",
              "缺失曲线阴影为五次随机位置实验的 ±1 个标准差；TAV 30% 图的横线为同一样本配对 bootstrap 95% 区间。",
              "其中 Placement 1–5 使用遮挡种子 2028–2032；模型训练种子始终为 2028。",
              "02 与 02b 图的缺失验证在每轮使用同一遮挡随机种子；逐轮没有单独记录‘缺失验证集损失’或‘缺失训练集损失’，故不绘制虚构曲线。",
              "验证和测试总体指标的区间是样本 bootstrap 95% 区间。",
              "", "## 全部图表", "",
              "[多页 PDF](问题2_种子2028_全部图表.pdf)；PNG/PDF 单图位于 `charts/`。",
              "", "## 可复现性", "",
              f"最佳权重 SHA-256：`{source_hash}`。逐轮训练数据来自经权重逐项核对的种子 2028 回放日志。",
              "图表使用 `python problem2/figures/build_seed2028_all2028.py` 重新生成。",
              ""]
    (OUT / "图表目录.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(titles)} figures, figure manifest and index to {OUT}")


if __name__ == "__main__":
    main()
