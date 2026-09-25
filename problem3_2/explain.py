"""Exact modality Shapley and verified local evidence for the new model."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "problem2"))
from data import PreparedDataset  # noqa: E402
from timeline import evidence_location  # noqa: E402
from main import CLASSES, load_model, save_csv, save_json  # noqa: E402

MODALITIES = ("text", "audio", "vision")
INPUT_KEYS = ("text", "audio", "vision", "padding", "availability", "quality")


def subset_mask(sample: dict, bits: int) -> np.ndarray:
    natural = sample["availability"].numpy().astype(bool)
    switches = np.array([(bits >> m) & 1 for m in range(3)], dtype=bool)
    return natural & switches[None, :]


@torch.inference_mode()
def forward_masks(model, sample: dict, masks: list[np.ndarray], device: torch.device,
                  batch_size: int = 96) -> tuple[np.ndarray, np.ndarray]:
    logits, scores = [], []
    for first in range(0, len(masks), batch_size):
        part = masks[first:first + batch_size]
        n = len(part)
        batch = {key: sample[key].unsqueeze(0).expand(n, *sample[key].shape).to(device)
                 for key in INPUT_KEYS}
        out = model(batch, torch.as_tensor(np.stack(part), device=device))
        logits.append(out["logits"].cpu().numpy())
        scores.append(out["score"].cpu().numpy())
    return np.concatenate(logits), np.concatenate(scores)


def exact_shapley(values: np.ndarray) -> np.ndarray:
    """Three-player Shapley values; values indexed by bit-set 0..7."""
    if len(values) != 8:
        raise ValueError("expected values for all eight modality subsets")
    phi = np.zeros(3, dtype=float)
    for m in range(3):
        for subset in range(8):
            if subset & (1 << m):
                continue
            count = subset.bit_count()
            weight = (1 / 3) if count in (0, 2) else (1 / 6)
            phi[m] += weight * (values[subset | (1 << m)] - values[subset])
    return phi


def candidate_windows(sample: dict, modality: int,
                      timeline: dict | None) -> list[tuple[int, int]]:
    observed = (sample["padding"][:, modality] & sample["quality"][:, modality]
                & sample["availability"][:, modality]).numpy().astype(bool)
    if modality == 0:
        indices = np.flatnonzero(observed)
        if len(indices) > 2:
            observed[indices[0]] = False
            observed[indices[-1]] = False
    if timeline is not None:
        observed &= np.array([span is not None for span in timeline["token_times"]], dtype=bool)
    windows = [(start, start + size) for size in (2, 3, 4, 5)
               for start in range(51 - size) if observed[start:start + size].all()]
    if not windows:
        windows = [(int(t), int(t + 1)) for t in np.flatnonzero(observed)]
    return windows


def format_location(timeline: dict | None, start: int, end: int) -> dict:
    if timeline is None:
        return {"text": "", "start_sec": None, "end_sec": None, "mid_sec": None}
    return evidence_location(timeline, start, end)


def explain_one(model, sample: dict, device: torch.device,
                timeline: dict | None = None, rng: np.random.Generator | None = None,
                selection_method: str = "occlusion") -> dict:
    if rng is None:
        rng = np.random.default_rng(2030)
    masks = [subset_mask(sample, bits) for bits in range(8)]
    logits, scores = forward_masks(model, sample, masks, device)
    full = logits[7]
    ordering = np.argsort(full)[::-1]
    target, runner = int(ordering[0]), int(ordering[1])
    margin = logits[:, target] - logits[:, runner]
    shap = exact_shapley(margin)
    shap_score = exact_shapley(scores)
    assert np.isclose(shap.sum(), margin[7] - margin[0], atol=1e-5)
    assert np.isclose(shap_score.sum(), scores[7] - scores[0], atol=1e-5)
    probs = torch.softmax(torch.as_tensor(full), dim=0).numpy()
    if selection_method not in ("gradient", "occlusion"):
        raise ValueError(f"unknown selection method: {selection_method}")
    gradient_positions = None
    if selection_method == "gradient":
        # Independent proposal for the validation experiment. Deletion is
        # computed afterwards and did not participate in this selection.
        grad_batch = {key: sample[key].unsqueeze(0).to(device) for key in INPUT_KEYS}
        grad_features = []
        for name in MODALITIES:
            grad_batch[name] = grad_batch[name].clone().detach().requires_grad_(True)
            grad_features.append(grad_batch[name])
        with torch.enable_grad():
            grad_output = model(grad_batch)
            grad_margin = grad_output["logits"][0, target] - grad_output["logits"][0, runner]
            gradients = torch.autograd.grad(grad_margin, grad_features)
        gradient_positions = {
            name: (gradients[m][0] * grad_features[m][0]).sum(dim=-1).detach().cpu().numpy()
            for m, name in enumerate(MODALITIES)
        }
    shares = np.abs(shap) / max(float(np.abs(shap).sum()), 1e-12)
    positive = [m for m in range(3) if shap[m] > 0]
    if positive:
        main = max(positive, key=lambda m: shap[m])
        main_status = "supports_prediction"
    elif np.any(shares):
        main = int(np.argmax(shares))
        main_status = "no_positive_modality_contribution"
    else:
        main = None
        main_status = "no_valid_contribution"
    natural = sample["availability"].numpy().astype(bool)
    windows: dict[str, dict | None] = {}
    position_importance: dict[str, list[float]] = {}
    for m, name in enumerate(MODALITIES):
        candidates = candidate_windows(sample, m, timeline)
        if not candidates:
            windows[name] = None
            position_importance[name] = [0.0] * 50
            continue
        altered_masks = []
        for start, end in candidates:
            mask = natural.copy()
            mask[start:end, m] = False
            altered_masks.append(mask)
        altered_logits, altered_scores = forward_masks(model, sample, altered_masks, device)
        drops = margin[7] - (altered_logits[:, target] - altered_logits[:, runner])
        totals = np.zeros(50, dtype=float)
        counts = np.zeros(50, dtype=int)
        for (a, b), drop in zip(candidates, drops):
            totals[a:b] += max(float(drop), 0.0)
            counts[a:b] += 1
        position_importance[name] = (totals / np.maximum(counts, 1)).tolist()
        if selection_method == "gradient":
            ranking = np.array([float(gradient_positions[name][a:b].sum())
                                for a, b in candidates])
        else:
            ranking = drops
        supportive = np.flatnonzero(ranking > 0)
        chosen = (int(supportive[np.argmax(ranking[supportive])])
                  if len(supportive) else int(np.argmax(np.abs(ranking))))
        start, end = candidates[chosen]
        comparables = [k for k, (a, b) in enumerate(candidates)
                       if b - a == end - start and k != chosen]
        random_index = int(rng.choice(comparables)) if comparables else chosen
        retained = np.zeros_like(natural)
        retained[start:end, m] = natural[start:end, m]
        retained_logits, _ = forward_masks(model, sample, [retained], device)
        retained_margin = retained_logits[0, target] - retained_logits[0, runner]
        location = format_location(timeline, start, end)
        windows[name] = {"start": start, "end": end,
                         "selection_score": float(ranking[chosen]),
                         "margin_drop": float(drops[chosen]),
                         "retention_margin": float(retained_margin),
                         "retention_keeps_class": bool(np.argmax(retained_logits[0]) == target),
                         "score_change": float(scores[7] - altered_scores[chosen]),
                         "random_margin_drop": float(drops[random_index]),
                         "random_start": candidates[random_index][0],
                         "random_end": candidates[random_index][1],
                         "direction": "supports" if drops[chosen] > 0 else "unverified_or_conflicts",
                         "location": location}
    return {"id": str(sample["id"]), "pred_class": CLASSES[target],
            "pred_class_index": target, "pred_score": float(scores[7]),
            "p_negative": float(probs[0]), "p_neutral": float(probs[1]),
            "p_positive": float(probs[2]), "main_modality":
            MODALITIES[main] if main is not None else "none",
            "key_text": (windows["text"]["location"].get("text", "")
                         if windows["text"] else ""),
            "key_audio_start_sec": (windows["audio"]["location"].get("start_sec")
                                    if windows["audio"] else None),
            "key_audio_end_sec": (windows["audio"]["location"].get("end_sec")
                                  if windows["audio"] else None),
            "key_vision_start_sec": (windows["vision"]["location"].get("start_sec")
                                     if windows["vision"] else None),
            "key_vision_end_sec": (windows["vision"]["location"].get("end_sec")
                                   if windows["vision"] else None),
            "key_vision_frame_sec": (windows["vision"]["location"].get("mid_sec")
                                    if windows["vision"] else None),
            "main_status": main_status, "class_margin": float(margin[7]),
            "empty_subset_margin": float(margin[0]),
            "shapley_margin": {name: float(shap[m]) for m, name in enumerate(MODALITIES)},
            "shapley_score": {name: float(shap_score[m]) for m, name in enumerate(MODALITIES)},
            "importance_share": {name: float(shares[m]) for m, name in enumerate(MODALITIES)},
            "windows": windows, "position_importance": position_importance,
            "selection_method": selection_method,
            "quality_flag": {name: bool((sample["padding"][:, m] &
                                          sample["quality"][:, m] &
                                          sample["availability"][:, m]).any())
                             for m, name in enumerate(MODALITIES)}}


def flatten(card: dict, timeline: dict | None = None) -> dict:
    row = {key: card[key] for key in ("id", "pred_class", "pred_score", "p_negative",
                                      "p_neutral", "p_positive", "main_modality", "main_status")}
    for name in MODALITIES:
        row[f"shap_{name}"] = card["shapley_margin"][name]
        row[f"share_{name}"] = card["importance_share"][name]
        window = card["windows"][name]
        row[f"{name}_start_position"] = window["start"] if window else ""
        row[f"{name}_end_position"] = window["end"] if window else ""
        row[f"{name}_start_sec"] = (window["location"].get("start_sec")
                                     if window else "")
        row[f"{name}_end_sec"] = (window["location"].get("end_sec")
                                   if window else "")
        row[f"{name}_deletion_drop"] = window["margin_drop"] if window else ""
        row[f"{name}_retention_margin"] = window["retention_margin"] if window else ""
        row[f"{name}_valid"] = card["quality_flag"][name]
    row["key_text"] = (card["windows"]["text"]["location"].get("text", "")
                       if card["windows"]["text"] else "")
    row["vision_key_frame_sec"] = (card["windows"]["vision"]["location"].get("mid_sec")
                                    if card["windows"]["vision"] else "")
    row["alignment_method"] = timeline["alignment_method"] if timeline else ""
    return row


def extract_key_frame(video: Path, seconds: float, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", f"{seconds:.3f}", "-i", str(video), "-frames:v", "1", str(output)],
                   check=True, capture_output=True)


def explain_attachment4(args: argparse.Namespace) -> None:
    torch.set_num_threads(min(6, torch.get_num_threads()))
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    dataset = PreparedDataset(args.data / "attachment4.npz")
    timelines = {str(x["id"]): x for x in json.loads(
        (args.data / "attachment4_timelines.json").read_text(encoding="utf-8"))}
    cards = []
    rng = np.random.default_rng(2030)
    for i in range(len(dataset)):
        sample = dataset[i]
        timeline = timelines[str(sample["id"])]
        card = model.predict_with_explanation(sample, timeline, rng=rng)
        card["raw_text"] = timeline["raw_text"]
        cards.append(card)
        if card["windows"]["vision"]:
            sec = card["windows"]["vision"]["location"].get("mid_sec")
            if sec is not None:
                video = args.videos / f"{sample['id']}.mp4"
                frame = args.out / "key_frames" / f"{sample['id']}.jpg"
                extract_key_frame(video, float(sec), frame)
        print(f"explained attachment4 {i + 1}/{len(dataset)}: {sample['id']}", flush=True)
    rows = [flatten(card, timelines[card["id"]]) for card in cards]
    for row in rows:
        frame = args.out / "key_frames" / f"{row['id']}.jpg"
        row["vision_key_frame"] = str(frame.relative_to(args.out)) if frame.exists() else ""
    save_csv(args.out / "attachment4_predictions_explanations.csv", rows)
    save_json(args.out / "attachment4_explanation_cards.json", cards)
    position_rows = [{"id": card["id"], "modality": name,
                      "position": position, "importance": float(value)}
                     for card in cards for name in MODALITIES
                     for position, value in enumerate(card["position_importance"][name])]
    save_csv(args.out / "attachment4_position_importance.csv", position_rows)
    save_json(args.out / "attachment4_summary.json", {
        "n": len(cards), "class_counts": {name: sum(c["pred_class"] == name for c in cards)
                                     for name in CLASSES},
        "main_modality_counts": {name: sum(c["main_modality"] == name for c in cards)
                                 for name in MODALITIES},
        "alignment_methods": {method: sum(t["alignment_method"] == method
                                   for t in timelines.values())
                              for method in sorted({t["alignment_method"] for t in timelines.values()})},
        "visual_invalid_ids": [c["id"] for c in cards if not c["quality_flag"]["vision"]],
    })


def validate_explanations(args: argparse.Namespace) -> None:
    torch.set_num_threads(min(6, torch.get_num_threads()))
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    dataset = PreparedDataset(args.data / "valid.npz")
    labels = dataset.arrays["classification_labels"].astype(int)
    rng = np.random.default_rng(2030)
    indices = np.concatenate([rng.choice(np.flatnonzero(labels == label),
                                         size=min(args.per_class, np.sum(labels == label)),
                                         replace=False)
                              for label in range(3)])
    rows = []
    for i, index in enumerate(indices):
        sample = dataset[int(index)]
        card = model.predict_with_explanation(sample, selection_method="gradient", rng=rng)
        main = card["main_modality"]
        window = card["windows"].get(main)
        if window is not None:
            rows.append({"id": card["id"], "true_class": CLASSES[int(labels[index])],
                         "pred_class": card["pred_class"], "main_modality": main,
                         "full_margin": card["class_margin"],
                         "top_deletion_margin_drop": window["margin_drop"],
                         "random_deletion_margin_drop": window["random_margin_drop"],
                         "retention_margin": window["retention_margin"],
                         "retention_keeps_class": window["retention_keeps_class"],
                         "top_start": window["start"], "top_end": window["end"]})
        print(f"validated explanation {i + 1}/{len(indices)}", flush=True)
    save_csv(args.out / "valid_explanation_faithfulness.csv", rows)
    top = np.array([r["top_deletion_margin_drop"] for r in rows])
    random = np.array([r["random_deletion_margin_drop"] for r in rows])
    retained = np.array([r["retention_margin"] for r in rows])
    save_json(args.out / "valid_explanation_report.json", {
        "n": len(rows), "sampling": f"{args.per_class} samples per true class, seed 2030",
        "candidate_selection": "gradient times input; deletion used only for evaluation",
        "mean_top_deletion_margin_drop": float(top.mean()),
        "mean_random_deletion_margin_drop": float(random.mean()),
        "fraction_top_exceeds_random": float(np.mean(top > random)),
        "mean_paired_difference": float(np.mean(top - random)),
        "mean_retention_margin": float(retained.mean()),
        "fraction_retention_keeps_class": float(np.mean(
            [r["retention_keeps_class"] for r in rows]))})


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("predict4", "explain4", "validate"):
        s = sub.add_parser(command)
        s.add_argument("--checkpoint", type=Path, default=HERE / "outputs/best.pt")
        s.add_argument("--data", type=Path,
                       default=HERE / "cache" if command in ("predict4", "explain4")
                       else HERE / "cache")
        s.add_argument("--out", type=Path, default=HERE / "outputs")
        s.add_argument("--device", default="cpu")
        if command in ("predict4", "explain4"):
            s.add_argument("--videos", type=Path,
                           default=ROOT / "E题数据/附件4-可解释专项视频样本与特征文件"
                           / "附件4-可解释专项视频样本与特征文件/对齐版本/videos")
        else:
            s.add_argument("--per-class", type=int, default=20)
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.command in ("predict4", "explain4"):
        explain_attachment4(arguments)
    else:
        validate_explanations(arguments)
