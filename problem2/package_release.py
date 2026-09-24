"""Package the selected problem-2 seed-2028 model and reviewed figures for GitHub."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "outputs/dropout03_lr_drop/seed2028"
REPLAY = ROOT / "outputs/dropout03_lr_drop/seed2028_metrics_replay"
FIGURES = ROOT / "outputs/deliverable_seed2028"
RELEASE = ROOT / "release/seed2028"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    checkpoint = torch.load(RUN / "best.pt", map_location="cpu", weights_only=False)
    replay_checkpoint = torch.load(REPLAY / "best.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if (config["seed"], config["dropout"], config["ablation"], checkpoint["epoch"]) != (
            2028, 0.3, "fixed_gate", 9):
        raise ValueError("release checkpoint identity is not the selected seed-2028 model")
    if checkpoint["epoch"] != replay_checkpoint["epoch"] or any(
            not torch.equal(weight, replay_checkpoint["model"][name])
            for name, weight in checkpoint["model"].items()):
        raise ValueError("metrics replay differs from selected checkpoint")
    expected_hash = sha256(RUN / "best.pt")
    figure_manifest = json.loads((FIGURES / "figure_manifest.json").read_text(encoding="utf-8"))
    if figure_manifest["sha256"] != expected_hash or figure_manifest["figure_count"] != 15:
        raise ValueError("figure pack does not match selected checkpoint")
    if len(list((FIGURES / "charts").glob("*.png"))) != 15 or len(
            list((FIGURES / "charts").glob("*.pdf"))) != 15:
        raise ValueError("figure pack must contain 15 PNG and 15 PDF charts")
    RELEASE.mkdir(parents=True, exist_ok=True)
    for source in sorted(FIGURES.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(FIGURES)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if relative == Path("valid_robustness_curve.png"):
            continue  # Older auto-generated duplicate; the reviewed charts are under charts/.
        destination = RELEASE / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".csv":
            # csv.writer emits CRLF; keep the tracked release text LF-normalized.
            destination.write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
        else:
            shutil.copy2(source, destination)
    shutil.copy2(RUN / "best.pt", RELEASE / "best.pt")
    shutil.copy2(RUN / "training_config.json", RELEASE / "training_config.json")
    shutil.copy2(RUN / "training_history.json", RELEASE / "training_history.json")
    shutil.copy2(REPLAY / "training_history.json",
                 RELEASE / "training_history_with_train_metrics.json")
    portable_figure_manifest = dict(figure_manifest)
    portable_figure_manifest["checkpoint"] = "best.pt"
    (RELEASE / "figure_manifest.json").write_text(
        json.dumps(portable_figure_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    manifest = {
        "model": "fixed_gate",
        "seed": 2028,
        "dropout": 0.3,
        "selected_epoch": 9,
        "checkpoint": "best.pt",
        "checkpoint_sha256": expected_hash,
        "chart_count": 15,
        "historical_ablation": "A1_historical_architecture_ablation uses seed 2026 and dropout 0.2",
        "raw_datasets_included": False,
    }
    (RELEASE / "release_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 问题 2 最终交付：随机种子 2028", "",
             "本目录包含选中的第 9 轮模型权重、逐轮训练日志、15 张图的 PNG/PDF、"
             "图表合订 PDF、指标与预测 CSV。原始附件数据和预处理缓存不在仓库中。", "",
             "- [图表目录](图表目录.md)", "- [逐图解读](逐图解读.md)",
             "- [结果汇总](结果汇总.md)", "- [全部图表 PDF](问题2_种子2028_全部图表.pdf)",
             "- [附件 3 预测](attachment3_predictions.csv)", "",
             "使用 `torch.load('best.pt', map_location='cpu', weights_only=False)` 读取权重，"
             "由 `problem2.experiment.load_model` 构建网络。模型训练配置见 `training_config.json`。",
             "`training_history_with_train_metrics.json` 是逐轮指标回放，已与原始最佳权重逐参数核对一致。",
             "最后一张架构消融图来自种子 2026、dropout 0.2，是明确标注的历史对照。", ""]
    (RELEASE / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Packaged {len(list(RELEASE.rglob('*')))} entries in {RELEASE}")


if __name__ == "__main__":
    main()
