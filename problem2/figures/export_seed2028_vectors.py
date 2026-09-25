"""Export the selected seed-2028 figures as native vector SVG and PDF.

Reads the verified result tables without changing the existing chart package.
The four matrix plots use vector cells and the regression scatter uses vector
markers, so neither SVG nor PDF needs an embedded raster image.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

_output_root = Path(__file__).resolve().parents[1] / "outputs" / "problem2_seed2028_vector_figures"
(_output_root / ".matplotlib").mkdir(parents=True, exist_ok=True)
(_output_root / ".cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_output_root / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_output_root / ".cache"))

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

import build_seed2028_all2028 as builder


SOURCE = builder.OUT
DEST = builder.ROOT / "outputs" / "problem2_seed2028_vector_figures"
FIGURES = DEST / "figures"
INPUTS = (
    "overall_split_metrics.csv",
    "label_metrics.csv",
    "valid_robustness.csv",
    "valid_robustness_baseline.json",
    "attachment3_predictions.csv",
    "valid_error_analysis.json",
    "ablation_summary_seed2028.csv",
    *[f"{split}_metrics.json" for split in builder.SPLITS],
    *[f"{split}_predictions.csv" for split in builder.SPLITS],
)


_original_imshow = Axes.imshow
_original_scatter = Axes.scatter
_original_colorbar = Figure.colorbar


def vector_imshow(self: Axes, values, *args, **kwargs):
    """Draw small numerical matrices as individual vector quadrilaterals."""
    array = np.asarray(values)
    if array.ndim != 2 or args:
        return _original_imshow(self, values, *args, **kwargs)
    aspect = kwargs.pop("aspect", None)
    if aspect is not None:
        self.set_aspect(aspect)
    else:
        self.set_aspect("equal")
    rows, cols = array.shape
    edges_x = np.arange(cols + 1) - 0.5
    edges_y = np.arange(rows + 1) - 0.5
    artist = self.pcolormesh(
        edges_x, edges_y, array, shading="flat", edgecolors="none", **kwargs
    )
    self.set_xlim(-0.5, cols - 0.5)
    self.set_ylim(rows - 0.5, -0.5)
    return artist


def vector_scatter(self: Axes, *args, **kwargs):
    kwargs.pop("rasterized", None)
    return _original_scatter(self, *args, **kwargs)


def vector_colorbar(self: Figure, *args, **kwargs):
    colorbar = _original_colorbar(self, *args, **kwargs)
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    return colorbar


def vector_save(fig, stem: str, pdf, titles: list[tuple[str, str]],
                description: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.svg")
    fig.savefig(FIGURES / f"{stem}.pdf")
    pdf.savefig(fig)
    titles.append((stem, description))
    plt.close(fig)


def supplementary_robustness() -> None:
    with (SOURCE / "valid_robustness.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    baseline = json.loads(
        (SOURCE / "valid_robustness_baseline.json").read_text(encoding="utf-8")
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, position in zip(axes.flat, ("start", "middle", "end", "random")):
        for modality in builder.MODS:
            x = builder.RATIOS
            y = [
                np.mean([
                    float(row["macro_f1"]) for row in rows
                    if row["modality"] == modality
                    and row["position"] == position
                    and float(row["ratio"]) == ratio
                ])
                for ratio in x
            ]
            ax.plot(x, y, marker="o", label=modality)
        ax.axhline(baseline["macro_f1"], color="black", linestyle="--", linewidth=1)
        ax.set(title=position.title(), xlabel="Target missing ratio", ylabel="Macro F1")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "15_valid_robustness_by_position.svg")
    fig.savefig(FIGURES / "15_valid_robustness_by_position.pdf")
    plt.close(fig)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for filename in INPUTS:
        shutil.copyfile(SOURCE / filename, DEST / filename)

    Axes.imshow = vector_imshow
    Axes.scatter = vector_scatter
    Figure.colorbar = vector_colorbar
    builder.OUT = DEST
    builder.CHARTS = FIGURES
    builder.ABLATION_CSV = DEST / "ablation_summary_seed2028.csv"
    builder.save = vector_save
    builder.setup()
    history = json.loads(
        (builder.REPLAY / "training_history.json").read_text(encoding="utf-8")
    )
    overall = builder.read_csv(DEST / "overall_split_metrics.csv")
    metrics_json = {
        split: json.loads((DEST / f"{split}_metrics.json").read_text(encoding="utf-8"))
        for split in builder.SPLITS
    }
    class_rows = builder.read_csv(DEST / "label_metrics.csv")
    robustness_rows = builder.read_csv(DEST / "valid_robustness.csv")
    robustness_base = json.loads(
        (DEST / "valid_robustness_baseline.json").read_text(encoding="utf-8")
    )
    if len(robustness_rows) != 280:
        raise ValueError("expected 280 verified robustness rows")
    titles: list[tuple[str, str]] = []
    with PdfPages(DEST / "问题2_种子2028_全部图表.pdf") as pdf:
        builder.model_flow(pdf, titles)
        builder.training_curves(history, 9, pdf, titles)
        builder.missing_training_curves(history, 9, pdf, titles)
        builder.split_metrics(overall, metrics_json, pdf, titles)
        builder.class_metrics(class_rows, pdf, titles)
        builder.confusion_matrices(metrics_json, pdf, titles)
        builder.regression_scatter(metrics_json, pdf, titles)
        builder.error_slices(pdf, titles)
        builder.robustness_charts(robustness_rows, robustness_base, pdf, titles)
        builder.attachment3(pdf, titles)
        builder.matched_ablation(pdf, titles)
    if len(titles) != 16:
        raise RuntimeError(f"expected 16 main figures, got {len(titles)}")
    supplementary_robustness()

    pdf_count = len(list(FIGURES.glob("*.pdf")))
    svg_count = len(list(FIGURES.glob("*.svg")))
    if (pdf_count, svg_count) != (17, 17):
        raise RuntimeError(f"expected 17 PDF and SVG figures, got {pdf_count}, {svg_count}")
    (DEST / "矢量图说明.md").write_text(
        "# 问题2随机种子2028矢量图\n\n"
        "本目录包含最终交付的16张主图及1张验证集缺失位置补充图，"
        "每图同时提供可缩放的SVG和PDF。"
        "主图合订文件为“问题2_种子2028_全部图表.pdf”，共16页；"
        "第17张补充图单独提供。\n\n"
        "所有图均从随机种子2028的最终结果表重新绘制。热图使用矢量网格，"
        "散点使用矢量标记，SVG保留可编辑文字。"
        "目录中的CSV和JSON是再现绘图所需的数据，"
        "不代表新增训练或新的实验结果。\n",
        encoding="utf-8",
    )
    print(f"Exported {svg_count} SVG and {pdf_count} PDF figures to {FIGURES}")


if __name__ == "__main__":
    main()
