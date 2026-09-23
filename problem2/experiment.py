"""Training, validation, paired missingness experiments, and attachment 3 inference."""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import MODALITIES, PreparedDataset, to_device
from model import RobustFusion

CLASSES = ("Negative", "Neutral", "Positive")
MODALITY_SETS = ("T", "A", "V", "TA", "TV", "AV", "TAV")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def block_availability(batch: dict, ratio: float, position: str,
                       modalities: str, rng: np.random.Generator) -> torch.Tensor:
    """Hide contiguous intervals of originally observed positions only."""
    base = batch["availability"].detach().cpu().numpy().copy().astype(bool)
    original = (batch["padding"] & batch["quality"] & batch["availability"])
    original = original.detach().cpu().numpy().astype(bool)
    for i in range(len(base)):
        for m, letter in enumerate("TAV"):
            if letter not in modalities:
                continue
            valid = np.flatnonzero(original[i, :, m])
            # BERT's boundary tokens are structural markers, not local content.
            if m == 0 and len(valid) > 2:
                valid = valid[1:-1]
            if len(valid) < 2:
                continue
            length = max(1, min(len(valid) - 1, round(len(valid) * ratio)))
            if position == "start":
                start = 0
            elif position == "middle":
                start = (len(valid) - length) // 2
            elif position == "end":
                start = len(valid) - length
            elif position == "random":
                start = int(rng.integers(0, len(valid) - length + 1))
            else:
                raise ValueError(f"unknown position {position}")
            base[i, valid[start:start + length], m] = False
    return torch.as_tensor(base, device=batch["padding"].device)


def metric_values(classes: np.ndarray, scores: np.ndarray,
                  true_classes: np.ndarray, true_scores: np.ndarray) -> dict:
    cm = np.zeros((3, 3), dtype=int)
    for true, pred in zip(true_classes, classes):
        cm[int(true), int(pred)] += 1
    support = cm.sum(axis=1)
    f1 = []
    for k in range(3):
        tp = cm[k, k]
        denom = 2 * tp + (cm[:, k].sum() - tp) + (support[k] - tp)
        f1.append(float(2 * tp / denom) if denom else 0.0)
    pearson = float(np.corrcoef(true_scores, scores)[0, 1]) if (
        len(scores) > 1 and np.std(scores) > 0 and np.std(true_scores) > 0
    ) else None
    return {
        "accuracy": float(np.mean(classes == true_classes)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=support)),
        "mae": float(np.mean(np.abs(scores - true_scores))),
        "pearson": pearson,
        "confusion_matrix": cm.tolist(),
    }


@torch.inference_mode()
def predict_loader(model: RobustFusion, loader: DataLoader, device: torch.device,
                   missing: tuple[float, str, str, int] | None = None) -> dict:
    model.eval()
    predictions = {key: [] for key in ("id", "class", "score", "probabilities",
                                     "true_class", "true_score", "missing_ratio")}
    rng = np.random.default_rng(missing[3]) if missing else None
    for raw in loader:
        batch = to_device(raw, device)
        availability = (block_availability(batch, *missing[:3], rng)
                        if missing else None)
        output = model(batch, availability)
        probs = torch.softmax(output["logits"], dim=-1).cpu().numpy()
        predictions["id"].extend(batch["id"])
        predictions["class"].extend(probs.argmax(axis=1).tolist())
        predictions["score"].extend(output["score"].cpu().numpy().tolist())
        predictions["probabilities"].extend(probs.tolist())
        if "classification_labels" in batch:
            predictions["true_class"].extend(batch["classification_labels"].cpu().numpy().tolist())
            predictions["true_score"].extend(batch["regression_labels"].cpu().numpy().tolist())
        if missing:
            orig = (batch["padding"] & batch["quality"] & batch["availability"])
            targeted = torch.tensor([m in missing[2] for m in "TAV"], device=device)
            eligible = orig & targeted[None, None]
            removed = eligible & ~availability
            ratios = removed.sum((1, 2)).float() / eligible.sum((1, 2)).clamp_min(1)
            predictions["missing_ratio"].extend(ratios.cpu().tolist())
    return predictions


def metrics_from_predictions(predictions: dict) -> dict:
    return metric_values(np.asarray(predictions["class"]), np.asarray(predictions["score"]),
                         np.asarray(predictions["true_class"]),
                         np.asarray(predictions["true_score"]))


def task_loss_from_predictions(predictions: dict, class_weights: np.ndarray) -> float:
    """Complete-input weighted CE + Smooth L1, without auxiliary training losses."""
    labels = np.asarray(predictions["true_class"], dtype=int)
    probabilities = np.asarray(predictions["probabilities"], dtype=float)
    weights = np.asarray(class_weights, dtype=float)[labels]
    selected = np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
    classification = -np.sum(weights * np.log(selected)) / np.sum(weights)
    residual = np.abs(np.asarray(predictions["score"], dtype=float)
                      - np.asarray(predictions["true_score"], dtype=float))
    regression = np.mean(np.where(residual < 1.0, 0.5 * residual ** 2,
                                  residual - 0.5))
    return float(classification + regression)


def bootstrap_ci(predictions: dict, repeats: int = 500, seed: int = 2026) -> dict:
    rng = np.random.default_rng(seed)
    n = len(predictions["class"])
    vectors = {k: np.asarray(predictions[k]) for k in
               ("class", "score", "true_class", "true_score")}
    values = {k: [] for k in ("accuracy", "macro_f1", "weighted_f1", "mae", "pearson")}
    for _ in range(repeats):
        pick = rng.integers(0, n, n)
        m = metric_values(vectors["class"][pick], vectors["score"][pick],
                          vectors["true_class"][pick], vectors["true_score"][pick])
        for key in values:
            if m[key] is not None:
                values[key].append(m[key])
    return {key: [float(x) for x in np.quantile(v, [0.025, 0.975])]
            if v else None for key, v in values.items()}


def paired_delta_ci(clean: dict, damaged: dict, repeats: int = 200,
                    seed: int = 2026) -> dict:
    """Sample-bootstrap paired metric changes for exactly the same IDs."""
    if clean["id"] != damaged["id"]:
        raise ValueError("paired robustness predictions must have identical ID order")
    rng = np.random.default_rng(seed)
    n = len(clean["id"])
    vector_keys = ("class", "score", "true_class", "true_score")
    a = {k: np.asarray(clean[k]) for k in vector_keys}
    b = {k: np.asarray(damaged[k]) for k in vector_keys}
    deltas = {"macro_f1": [], "mae": []}
    for _ in range(repeats):
        pick = rng.integers(0, n, n)
        baseline = metric_values(a["class"][pick], a["score"][pick],
                                 a["true_class"][pick], a["true_score"][pick])
        perturbed = metric_values(b["class"][pick], b["score"][pick],
                                  b["true_class"][pick], b["true_score"][pick])
        for key in deltas:
            deltas[key].append(perturbed[key] - baseline[key])
    return {key: np.quantile(values, [0.025, 0.975]).tolist()
            for key, values in deltas.items()}


def train(data_dir: Path, out_dir: Path, *, epochs: int = 20,
          batch_size: int = 32, hidden: int = 128, heads: int = 4,
          dropout: float = 0.2, lr: float = 2e-4, patience: int = 5,
          seed: int = 2026, device_name: str = "cpu", limit: int | None = None,
          ablation: str = "none", record_train_metrics: bool = False,
          lr_drop_epoch: int | None = None, lr_drop_factor: float = 0.25) -> None:
    if lr_drop_epoch is not None and (lr_drop_epoch < 2 or not 0 < lr_drop_factor < 1):
        raise ValueError("lr drop requires epoch >= 2 and factor between 0 and 1")
    set_seed(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device(device_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_data = PreparedDataset(data_dir / "train.npz", limit)
    valid_data = PreparedDataset(data_dir / "valid.npz", limit)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    train_eval_loader = (DataLoader(train_data, batch_size=batch_size)
                         if record_train_metrics else None)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    if ablation not in ("none", "no_block", "no_reconstruction", "fixed_gate", "no_consistency"):
        raise ValueError(f"unknown ablation: {ablation}")
    model = RobustFusion(hidden, heads, dropout, fixed_gate=ablation == "fixed_gate").to(device)
    labels = np.asarray(train_data.arrays["classification_labels"], dtype=int)
    counts = np.bincount(labels, minlength=3)
    class_weights = torch.tensor(len(labels) / (3 * np.maximum(counts, 1)),
                                 dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    rng = np.random.default_rng(seed)
    config = {"hidden": hidden, "heads": heads, "dropout": dropout,
              "lr": lr, "seed": seed, "batch_size": batch_size,
              "epochs_requested": epochs, "patience": patience,
              "class_counts": counts.tolist(), "limit": limit, "ablation": ablation,
              "lr_drop_epoch": lr_drop_epoch, "lr_drop_factor": lr_drop_factor}
    (out_dir / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    best = float("-inf")
    stale = 0
    log = []
    for epoch in range(1, epochs + 1):
        if epoch == lr_drop_epoch:
            for group in optimizer.param_groups:
                group["lr"] *= lr_drop_factor
        model.train()
        total = 0.0
        for raw in train_loader:
            batch = to_device(raw, device)
            if ablation == "no_block":
                clean = model(batch)
                damaged = clean
            else:
                chosen = MODALITY_SETS[int(rng.integers(len(MODALITY_SETS)))]
                ratio = float(rng.uniform(0.1, 0.5))
                position = ("start", "middle", "end", "random")[int(rng.integers(4))]
                available = block_availability(batch, ratio, position, chosen, rng)
                clean = model(batch)
                damaged = model(batch, available)
            y_class = batch["classification_labels"].long()
            y_score = batch["regression_labels"].float().reshape(-1)
            task = (F.cross_entropy(damaged["logits"], y_class, weight=class_weights)
                    + F.smooth_l1_loss(damaged["score"], y_score)
                    + 0.3 * F.cross_entropy(clean["logits"], y_class, weight=class_weights)
                    + 0.3 * F.smooth_l1_loss(clean["score"], y_score))
            artificial = clean["observed"] & ~damaged["observed"]
            reconstruct = F.smooth_l1_loss(
                damaged["reconstruction"][artificial],
                clean["targets"].detach()[artificial]
            ) if artificial.any() else task.new_zeros(())
            consistency = (F.mse_loss(torch.softmax(damaged["logits"], -1),
                                      torch.softmax(clean["logits"].detach(), -1))
                           + F.smooth_l1_loss(damaged["score"], clean["score"].detach()))
            loss = (task + (0 if ablation in ("no_block", "no_reconstruction") else 0.1) * reconstruct
                    + (0 if ablation in ("no_block", "no_consistency") else 0.1) * consistency)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(y_class)
        clean_valid_predictions = predict_loader(model, valid_loader, device)
        clean_valid = metrics_from_predictions(clean_valid_predictions)
        missing_valid = metrics_from_predictions(
            predict_loader(model, valid_loader, device, (0.3, "random", "TAV", seed)))
        selection = (0.5 * clean_valid["macro_f1"] + 0.5 * missing_valid["macro_f1"]
                     - 0.05 * missing_valid["mae"])
        entry = {"epoch": epoch, "train_loss": total / len(train_data),
                 "learning_rate": optimizer.param_groups[0]["lr"],
                 "selection": selection, "valid_clean": clean_valid,
                 "valid_missing_30pct": missing_valid}
        if train_eval_loader is not None:
            # Iterating a DataLoader consumes torch RNG state even without shuffling.
            # Restore it so recording metrics cannot alter later training epochs.
            cuda_devices = ([device.index if device.index is not None else torch.cuda.current_device()]
                            if device.type == "cuda" else [])
            with torch.random.fork_rng(devices=cuda_devices):
                clean_train_predictions = predict_loader(model, train_eval_loader, device)
            entry["train_clean"] = metrics_from_predictions(clean_train_predictions)
            weights = class_weights.detach().cpu().numpy()
            entry["train_eval_task_loss"] = task_loss_from_predictions(
                clean_train_predictions, weights)
            entry["valid_eval_task_loss"] = task_loss_from_predictions(
                clean_valid_predictions, weights)
        log.append(entry)
        print(f"epoch {epoch}: loss={entry['train_loss']:.4f} "
              f"clean F1={clean_valid['macro_f1']:.3f} "
              f"missing F1={missing_valid['macro_f1']:.3f}", flush=True)
        if selection > best + 1e-4:
            best = selection
            stale = 0
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch},
                       out_dir / "best.pt")
        else:
            stale += 1
            if stale >= patience:
                break
    (out_dir / "training_history.json").write_text(json.dumps(log, indent=2), encoding="utf-8")


def load_model(path: Path, device: torch.device) -> RobustFusion:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = RobustFusion(config["hidden"], config["heads"], config["dropout"],
                         fixed_gate=config.get("ablation") == "fixed_gate").to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def evaluate(data_dir: Path, checkpoint: Path, out_dir: Path,
             split: str = "test", batch_size: int = 64,
             device_name: str = "cpu", repeats: int = 500) -> None:
    if split not in ("valid", "test"):
        raise ValueError("evaluation split must be valid or test")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    model = load_model(checkpoint, device)
    loader = DataLoader(PreparedDataset(data_dir / f"{split}.npz"), batch_size=batch_size)
    predictions = predict_loader(model, loader, device)
    result = metrics_from_predictions(predictions)
    result["bootstrap_95pct_ci"] = bootstrap_ci(predictions, repeats)
    (out_dir / f"{split}_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out_dir / f"{split}_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("id", "true_class", "pred_class", "true_score", "pred_score",
                         "p_negative", "p_neutral", "p_positive"))
        for i, sample_id in enumerate(predictions["id"]):
            writer.writerow((sample_id, CLASSES[int(predictions["true_class"][i])],
                             CLASSES[int(predictions["class"][i])],
                             predictions["true_score"][i], predictions["score"][i],
                             *predictions["probabilities"][i]))
    print(json.dumps(result, indent=2), flush=True)


def robustness(data_dir: Path, checkpoint: Path, out_dir: Path,
               split: str = "valid", batch_size: int = 64,
               device_name: str = "cpu", seeds: int = 5,
               bootstrap: int = 200) -> None:
    if split not in ("valid", "test"):
        raise ValueError("robustness split must be valid or test")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    model = load_model(checkpoint, device)
    loader = DataLoader(PreparedDataset(data_dir / f"{split}.npz"), batch_size=batch_size)
    baseline_predictions = predict_loader(model, loader, device)
    baseline = metrics_from_predictions(baseline_predictions)
    rows = []
    for mods in MODALITY_SETS:
        for ratio in (0.1, 0.2, 0.3, 0.4, 0.5):
            for position in ("start", "middle", "end", "random"):
                # Start/middle/end are deterministic; repeat only random placement.
                for seed in range(seeds if position == "random" else 1):
                    pred = predict_loader(model, loader, device,
                                          (ratio, position, mods, 2026 + seed))
                    metrics = metrics_from_predictions(pred)
                    ci = paired_delta_ci(baseline_predictions, pred, bootstrap,
                                         2026 + seed) if bootstrap else None
                    rows.append({"modality": mods, "ratio": ratio,
                                 "position": position, "seed": 2026 + seed,
                                 "actual_missing_ratio": float(np.mean(pred["missing_ratio"])),
                                 **{k: metrics[k] for k in
                                    ("accuracy", "macro_f1", "weighted_f1", "mae", "pearson")},
                                 "delta_macro_f1": metrics["macro_f1"] - baseline["macro_f1"],
                                 "delta_mae": metrics["mae"] - baseline["mae"],
                                 "delta_macro_f1_ci_low": ci["macro_f1"][0] if ci else "",
                                 "delta_macro_f1_ci_high": ci["macro_f1"][1] if ci else "",
                                 "delta_mae_ci_low": ci["mae"][0] if ci else "",
                                 "delta_mae_ci_high": ci["mae"][1] if ci else ""})
                print(f"{mods} {ratio:.1f} {position}", flush=True)
    with (out_dir / f"{split}_robustness.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / f"{split}_robustness_baseline.json").write_text(
        json.dumps(baseline, indent=2), encoding="utf-8")
    # Mean curves summarize the effect of duration separately at each position.
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".matplotlib"))
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        for ax, position in zip(axes.flat, ("start", "middle", "end", "random")):
            for mods in MODALITY_SETS:
                x = [0.1, 0.2, 0.3, 0.4, 0.5]
                y = [float(np.mean([row["macro_f1"] for row in rows
                                    if row["modality"] == mods and row["position"] == position
                                    and row["ratio"] == ratio])) for ratio in x]
                ax.plot(x, y, marker="o", label=mods)
            ax.axhline(baseline["macro_f1"], color="black", linestyle="--", linewidth=1)
            ax.set(title=position, xlabel="Target missing ratio", ylabel="Macro F1")
            ax.grid(alpha=.2)
        axes[0, 0].legend(ncol=4, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{split}_robustness_curve.png", dpi=160)
        plt.close(fig)
    except ImportError:
        pass


def predict_attachment3(data_dir: Path, checkpoint: Path, output: Path,
                        batch_size: int = 32, device_name: str = "cpu") -> None:
    device = torch.device(device_name)
    model = load_model(checkpoint, device)
    dataset = PreparedDataset(data_dir / "attachment3.npz")
    predictions = predict_loader(model, DataLoader(dataset, batch_size=batch_size), device)
    output.parent.mkdir(parents=True, exist_ok=True)
    masks = dataset.arrays
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("id", "pred_class", "pred_score", "p_negative", "p_neutral",
                         "p_positive", "missing_type", "missing_ratio"))
        for i, sample_id in enumerate(predictions["id"]):
            eligible = masks["padding"][i] & masks["quality"][i]
            missing = eligible & ~masks["availability"][i]
            types = "".join(letter for m, letter in enumerate("TAV") if missing[:, m].any())
            ratio = float(missing.sum() / max(eligible.sum(), 1))
            writer.writerow((sample_id, CLASSES[int(predictions["class"][i])],
                             f"{predictions['score'][i]:.6f}",
                             *(f"{v:.6f}" for v in predictions["probabilities"][i]),
                             types or "none", f"{ratio:.6f}"))
    print(f"wrote {len(predictions['id'])} attachment 3 predictions to {output}", flush=True)
