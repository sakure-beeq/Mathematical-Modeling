"""Command line interface for problem 2."""

from __future__ import annotations

import argparse
from pathlib import Path

from data import prepare
from experiment import evaluate, predict_attachment3, robustness, train
from error_analysis import analyze
from ablation_report import summarize
from select_model import select

ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Problem 2: robust multimodal sentiment prediction")
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--attachment2", type=Path,
                      default=ROOT / "E题数据/附件2-数据集特征文件/aligned_50.pkl")
    prep.add_argument("--attachment3", type=Path,
                      default=ROOT / "E题数据/附件3-模态缺失特征样本/对齐版本")
    prep.add_argument("--bert-model", type=Path,
                      default=ROOT / "problem1/models/bert-base-uncased")
    prep.add_argument("--out", type=Path, default=ROOT / "problem2/cache")
    prep.add_argument("--manifest-output", type=Path,
                      default=ROOT / "problem2/outputs/preprocessing_manifest.json")
    prep.add_argument("--bert-batch-size", type=int, default=32)
    prep.add_argument("--device", default="cpu")
    fit = sub.add_parser("train")
    fit.add_argument("--data", type=Path, default=ROOT / "problem2/cache")
    fit.add_argument("--out", type=Path, default=ROOT / "problem2/outputs")
    fit.add_argument("--epochs", type=int, default=20)
    fit.add_argument("--batch-size", type=int, default=32)
    fit.add_argument("--hidden", type=int, default=128)
    fit.add_argument("--heads", type=int, default=4)
    fit.add_argument("--dropout", type=float, default=0.2)
    fit.add_argument("--lr", type=float, default=2e-4)
    fit.add_argument("--patience", type=int, default=5)
    fit.add_argument("--seed", type=int, default=2026)
    fit.add_argument("--device", default="cpu")
    fit.add_argument("--limit", type=int, help="smoke test only; never use for final results")
    fit.add_argument("--ablation", choices=("none", "no_block", "no_reconstruction",
                                            "fixed_gate", "no_consistency"), default="none")
    fit.add_argument("--record-train-metrics", action="store_true",
                     help="evaluate full-input training set after each epoch")
    fit.add_argument("--lr-drop-epoch", type=int,
                     help="first epoch to use a reduced learning rate")
    fit.add_argument("--lr-drop-factor", type=float, default=0.25)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--data", type=Path, default=ROOT / "problem2/cache")
    ev.add_argument("--checkpoint", type=Path, default=ROOT / "problem2/outputs/best.pt")
    ev.add_argument("--out", type=Path, default=ROOT / "problem2/outputs")
    ev.add_argument("--split", choices=("valid", "test"), default="test")
    ev.add_argument("--batch-size", type=int, default=64)
    ev.add_argument("--device", default="cpu")
    ev.add_argument("--bootstrap", type=int, default=500)
    rob = sub.add_parser("robustness")
    rob.add_argument("--data", type=Path, default=ROOT / "problem2/cache")
    rob.add_argument("--checkpoint", type=Path, default=ROOT / "problem2/outputs/best.pt")
    rob.add_argument("--out", type=Path, default=ROOT / "problem2/outputs")
    rob.add_argument("--split", choices=("valid", "test"), default="valid")
    rob.add_argument("--batch-size", type=int, default=64)
    rob.add_argument("--device", default="cpu")
    rob.add_argument("--seeds", type=int, default=5)
    rob.add_argument("--bootstrap", type=int, default=200)
    pred = sub.add_parser("predict3")
    pred.add_argument("--data", type=Path, default=ROOT / "problem2/cache")
    pred.add_argument("--checkpoint", type=Path, default=ROOT / "problem2/outputs/best.pt")
    pred.add_argument("--output", type=Path,
                      default=ROOT / "problem2/outputs/attachment3_predictions.csv")
    pred.add_argument("--batch-size", type=int, default=32)
    pred.add_argument("--device", default="cpu")
    err = sub.add_parser("errors")
    err.add_argument("--data", type=Path, default=ROOT / "problem2/cache/valid.npz")
    err.add_argument("--predictions", type=Path,
                     default=ROOT / "problem2/outputs/valid_predictions.csv")
    err.add_argument("--output", type=Path,
                     default=ROOT / "problem2/outputs/valid_error_analysis.json")
    abl = sub.add_parser("ablation-summary")
    abl.add_argument("--root", type=Path, default=ROOT / "problem2")
    abl.add_argument("--output", type=Path,
                     default=ROOT / "problem2/outputs/ablation_summary.csv")
    sel = sub.add_parser("select-model")
    sel.add_argument("--root", type=Path, default=ROOT / "problem2")
    return p


def main() -> None:
    a = parser().parse_args()
    if a.command == "prepare":
        prepare(a.attachment2, a.attachment3, a.bert_model, a.out,
                a.bert_batch_size, a.device, a.manifest_output)
    elif a.command == "train":
        train(a.data, a.out, epochs=a.epochs, batch_size=a.batch_size,
              hidden=a.hidden, heads=a.heads, dropout=a.dropout,
              lr=a.lr, patience=a.patience, seed=a.seed,
              device_name=a.device, limit=a.limit, ablation=a.ablation,
              record_train_metrics=a.record_train_metrics,
              lr_drop_epoch=a.lr_drop_epoch, lr_drop_factor=a.lr_drop_factor)
    elif a.command == "evaluate":
        evaluate(a.data, a.checkpoint, a.out, a.split, a.batch_size,
                 a.device, a.bootstrap)
    elif a.command == "robustness":
        robustness(a.data, a.checkpoint, a.out, a.split, a.batch_size,
                   a.device, a.seeds, a.bootstrap)
    elif a.command == "predict3":
        predict_attachment3(a.data, a.checkpoint, a.output, a.batch_size, a.device)
    elif a.command == "errors":
        analyze(a.data, a.predictions, a.output)
    elif a.command == "ablation-summary":
        summarize(a.root, a.output)
    elif a.command == "select-model":
        select(a.root)


if __name__ == "__main__":
    main()
