"""Train and evaluate the independent interaction model for problem 3."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "problem2"))
from data import PreparedDataset, to_device  # noqa: E402
from model import InteractionModel  # noqa: E402

CLASSES = ("Negative", "Neutral", "Positive")


def metric_values(classes: np.ndarray, scores: np.ndarray,
                  truth: np.ndarray, targets: np.ndarray) -> dict:
    matrix = np.zeros((3, 3), dtype=int)
    for actual, predicted in zip(truth, classes):
        matrix[int(actual), int(predicted)] += 1
    support = matrix.sum(axis=1)
    f1 = []
    for k in range(3):
        tp = matrix[k, k]
        denominator = matrix[k].sum() + matrix[:, k].sum()
        f1.append(float(2 * tp / denominator) if denominator else 0.0)
    r = (float(np.corrcoef(targets, scores)[0, 1])
         if len(scores) > 1 and np.std(scores) > 0 and np.std(targets) > 0 else None)
    return {"accuracy": float(np.mean(classes == truth)),
            "macro_f1": float(np.mean(f1)),
            "weighted_f1": float(np.average(f1, weights=support)),
            "mae": float(np.mean(np.abs(scores - targets))), "pearson": r,
            "confusion_matrix": matrix.tolist()}


def bootstrap_ci(predictions: dict, repeats: int, seed: int = 2030) -> dict:
    rng = np.random.default_rng(seed)
    values = {k: [] for k in ("accuracy", "macro_f1", "weighted_f1", "mae", "pearson")}
    vectors = {k: np.asarray(v) for k, v in predictions.items()}
    n = len(vectors["class"])
    for _ in range(repeats):
        pick = rng.integers(0, n, n)
        result = metric_values(vectors["class"][pick], vectors["score"][pick],
                               vectors["true_class"][pick], vectors["true_score"][pick])
        for key in values:
            if result[key] is not None:
                values[key].append(result[key])
    return {k: np.quantile(v, (.025, .975)).tolist() if v else None
            for k, v in values.items()}


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(6, torch.get_num_threads()))


def sampled_availability(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Hide a random subset of observed modalities for faithful interventions."""
    natural = batch["availability"].bool()
    present = (batch["padding"] & batch["quality"] & natural).any(dim=1)
    keep = torch.rand((len(natural), 3), device=natural.device) > 0.25
    # Every example retains at least one genuinely observed modality.
    empty = ~(keep & present).any(dim=1)
    for index in torch.where(empty)[0].tolist():
        choices = torch.where(present[index])[0]
        if len(choices):
            keep[index, choices[torch.randint(len(choices), (1,), device=natural.device)]] = True
    return natural & keep[:, None, :]


@torch.inference_mode()
def infer(model: InteractionModel, loader: DataLoader,
          device: torch.device) -> tuple[list[dict], dict | None]:
    model.eval()
    rows = []
    for raw in loader:
        batch = to_device(raw, device)
        out = model(batch)
        probs = torch.softmax(out["logits"], dim=1).cpu().numpy()
        scores = out["score"].cpu().numpy()
        for i, sample_id in enumerate(raw["id"]):
            label = int(np.argmax(probs[i]))
            row = {"id": sample_id, "pred_class_index": label,
                   "pred_class": CLASSES[label], "pred_score": float(scores[i]),
                   "p_negative": float(probs[i, 0]),
                   "p_neutral": float(probs[i, 1]),
                   "p_positive": float(probs[i, 2])}
            if "classification_labels" in raw:
                row["true_class_index"] = int(raw["classification_labels"][i])
                row["true_score"] = float(raw["regression_labels"][i])
            rows.append(row)
    if "true_class_index" not in rows[0]:
        return rows, None
    metrics = metric_values(np.array([r["pred_class_index"] for r in rows]),
                            np.array([r["pred_score"] for r in rows]),
                            np.array([r["true_class_index"] for r in rows]),
                            np.array([r["true_score"] for r in rows]))
    return rows, metrics


def load_model(checkpoint: Path, device: torch.device) -> InteractionModel:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = InteractionModel(saved["config"]["hidden"], saved["config"]["dropout"])
    model.load_state_dict(saved["model"])
    return model.to(device).eval()


def train(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    device = torch.device(args.device)
    train_set = PreparedDataset(args.data / "train.npz")
    valid_set = PreparedDataset(args.data / "valid.npz")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size)
    counts = np.bincount(train_set.arrays["classification_labels"].astype(int), minlength=3)
    weights = torch.tensor(1 / np.sqrt(counts), device=device, dtype=torch.float32)
    weights = weights / weights.mean()
    model = InteractionModel(args.hidden, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    config = {"model": "TemporalInteractionNet", "hidden": args.hidden,
              "dropout": args.dropout, "batch_size": args.batch_size, "lr": args.lr,
              "seed": args.seed, "epochs_requested": args.epochs,
              "patience": args.patience, "regression_weight": args.regression_weight,
              "subset_weight": args.subset_weight, "class_counts": counts.tolist(),
              "feature_version": "aligned_50", "text_interface": "frozen BERT",
              "selection_rule": "valid macro_f1 + 0.2*pearson - 0.1*mae"}
    args.out.mkdir(parents=True, exist_ok=True)
    save_json(args.out / "training_config.json", config)
    history, best, stale = [], -float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for raw in train_loader:
            batch = to_device(raw, device)
            labels = batch["classification_labels"].long()
            scores = batch["regression_labels"].float()
            full = model(batch)
            full_loss = (F.cross_entropy(full["logits"], labels, weight=weights)
                         + args.regression_weight * F.smooth_l1_loss(full["score"], scores))
            subset = model(batch, sampled_availability(batch))
            subset_loss = (F.cross_entropy(subset["logits"], labels, weight=weights)
                           + args.regression_weight * F.smooth_l1_loss(subset["score"], scores))
            loss = full_loss + args.subset_weight * subset_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        rows, metrics = infer(model, valid_loader, device)
        selection = metrics["macro_f1"] + .2 * (metrics["pearson"] or 0.0) - .1 * metrics["mae"]
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                  "selection": float(selection), **metrics}
        history.append(record)
        print(f"epoch={epoch:02d} loss={record['train_loss']:.4f} "
              f"valid acc={metrics['accuracy']:.4f} f1={metrics['macro_f1']:.4f} "
              f"mae={metrics['mae']:.4f} r={metrics['pearson']:.4f}", flush=True)
        if selection > best + 1e-5:
            best, stale = selection, 0
            torch.save({"model": model.state_dict(), "config": config,
                        "epoch": epoch, "selection": best}, args.out / "best.pt")
            save_json(args.out / "valid_metrics.json", metrics)
            save_csv(args.out / "valid_predictions.csv", rows)
        else:
            stale += 1
            if stale >= args.patience:
                break
    save_json(args.out / "training_history.json", history)
    print(f"selected_epoch={max(history, key=lambda x: x['selection'])['epoch']}", flush=True)


def evaluate(args: argparse.Namespace) -> None:
    torch.set_num_threads(min(6, torch.get_num_threads()))
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    dataset = PreparedDataset(args.data / f"{args.split}.npz")
    rows, metrics = infer(model, DataLoader(dataset, batch_size=args.batch_size), device)
    pred = {"class": [r["pred_class_index"] for r in rows],
            "score": [r["pred_score"] for r in rows],
            "true_class": [r["true_class_index"] for r in rows],
            "true_score": [r["true_score"] for r in rows]}
    metrics["bootstrap_95pct_ci"] = bootstrap_ci(pred, args.bootstrap)
    save_csv(args.out / f"{args.split}_predictions.csv", rows)
    save_json(args.out / f"{args.split}_metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--data", type=Path, default=HERE / "cache")
    fit.add_argument("--out", type=Path, default=HERE / "outputs")
    fit.add_argument("--epochs", type=int, default=25)
    fit.add_argument("--patience", type=int, default=6)
    fit.add_argument("--batch-size", type=int, default=64)
    fit.add_argument("--hidden", type=int, default=96)
    fit.add_argument("--dropout", type=float, default=.25)
    fit.add_argument("--lr", type=float, default=3e-4)
    fit.add_argument("--regression-weight", type=float, default=.65)
    fit.add_argument("--subset-weight", type=float, default=.3)
    fit.add_argument("--seed", type=int, default=2030)
    fit.add_argument("--device", default="cpu")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--data", type=Path, default=HERE / "cache")
    ev.add_argument("--checkpoint", type=Path, default=HERE / "outputs/best.pt")
    ev.add_argument("--out", type=Path, default=HERE / "outputs")
    ev.add_argument("--split", choices=("valid", "test"), default="test")
    ev.add_argument("--batch-size", type=int, default=64)
    ev.add_argument("--bootstrap", type=int, default=500)
    ev.add_argument("--device", default="cpu")
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.command == "train":
        train(arguments)
    else:
        evaluate(arguments)
