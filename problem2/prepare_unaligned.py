"""Prepare attachment 2/3 unaligned sequences for the 50-step fusion model.

Audio and vision retain their independent clocks until each sequence is pooled
into 50 ordered bins. This is a separately trained extension, not an assertion
that these bins are token-aligned to the text positions.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from data import pack
from vendor.mmer_preprocess.bert import FrozenBertEmbedder


ROOT = Path(__file__).resolve().parents[1]
MODALITIES = ("audio", "vision")
FEATURE_DIMS = {"audio": 74, "vision": 35}


def pool_sequence(values: np.ndarray, declared_length: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Pool valid frames to ordered bins, retaining wholly missing intervals."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != 500:
        raise ValueError(f"expected (500,D), got {values.shape}")
    valid_frame = np.isfinite(values).all(axis=1) & np.any(values != 0, axis=1)
    last_nonzero = int(np.flatnonzero(valid_frame)[-1] + 1) if valid_frame.any() else 0
    declared = max(0, min(500, int(declared_length))) if declared_length is not None else 0
    extent = max(last_nonzero, declared)
    out = np.zeros((50, values.shape[1]), dtype=np.float32)
    padding = np.zeros(50, dtype=bool)
    available = np.zeros(50, dtype=bool)
    if extent:
        count = min(50, extent)
        edges = np.linspace(0, extent, count + 1, dtype=int)
        for j in range(count):
            padding[j] = True
            lo, hi = edges[j], edges[j + 1]
            keep = valid_frame[lo:hi]
            if keep.any():
                out[j] = values[lo:hi][keep].mean(axis=0)
                available[j] = True
    return out, padding, available, declared_length is not None and last_nonzero != declared


def pool_modality(raw: np.ndarray, lengths: np.ndarray | None, dim: int
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    n = len(raw)
    result = np.zeros((n, 50, dim), dtype=np.float32)
    padding = np.zeros((n, 50), dtype=bool)
    available = np.zeros((n, 50), dtype=bool)
    anomaly = 0
    for i in range(n):
        result[i], padding[i], available[i], mismatch = pool_sequence(
            raw[i], int(lengths[i]) if lengths is not None else None)
        anomaly += int(mismatch)
    return result, padding, available, anomaly


def fit_stats(values: np.ndarray, available: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observed = values[available]
    if len(observed) == 0:
        raise ValueError("no observed training frames")
    mean = observed.mean(axis=0, dtype=np.float64)
    std = observed.std(axis=0, dtype=np.float64)
    std = np.where(std > 1e-6, std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def standardize(values: np.ndarray, available: np.ndarray,
                mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values)
    result[available] = np.clip((values[available] - mean) / std, -5, 5)
    return result


def encode_text(raw_text: np.ndarray, tokenizer, embedder: FrozenBertEmbedder) -> tuple[np.ndarray, np.ndarray]:
    encoded = tokenizer(list(map(str, raw_text)), max_length=50, truncation=True,
                        padding="max_length", return_tensors="np")
    bert_input = np.stack((encoded["input_ids"], encoded["attention_mask"],
                           encoded["token_type_ids"]), axis=1).astype(np.int64)
    text = embedder.encode(bert_input)
    mask = encoded["attention_mask"].astype(bool)
    return text, mask


def make_processed(block: dict, ids: np.ndarray, tokenizer,
                   embedder: FrozenBertEmbedder,
                   stats: dict[str, tuple[np.ndarray, np.ndarray]] | None
                   ) -> tuple[dict, dict, dict]:
    text, text_mask = encode_text(np.asarray(block["raw_text"]), tokenizer, embedder)
    processed = {"text": text, "id": ids}
    masks = {"text": {"padding": text_mask, "availability": text_mask,
                      "quality": text_mask}}
    pooled = {}
    diagnostics = {}
    for name in MODALITIES:
        length_key = f"{name}_lengths"
        length = np.asarray(block[length_key]) if length_key in block else None
        values, padding, availability, anomaly = pool_modality(
            np.asarray(block[name]), length, FEATURE_DIMS[name])
        pooled[name] = values
        masks[name] = {"padding": padding, "availability": availability,
                       "quality": padding}
        diagnostics[f"{name}_length_disagreements"] = anomaly
        diagnostics[f"{name}_observed_bin_fraction"] = float(
            availability.sum() / max(padding.sum(), 1))
    if stats is None:
        stats = {name: fit_stats(pooled[name], masks[name]["availability"])
                 for name in MODALITIES}
    for name in MODALITIES:
        processed[name] = standardize(pooled[name], masks[name]["availability"],
                                      *stats[name])
    processed["masks"] = masks
    for key in ("classification_labels", "regression_labels"):
        if key in block:
            dtype = np.int64 if key == "classification_labels" else np.float32
            processed[key] = np.asarray(block[key], dtype=dtype)
    return processed, stats, diagnostics


def prepare(source_path: Path, attachment3_dir: Path, bert_model: Path,
            out_dir: Path, bert_batch_size: int = 32) -> None:
    torch.set_num_threads(min(8, torch.get_num_threads()))
    out_dir.mkdir(parents=True, exist_ok=True)
    with source_path.open("rb") as handle:
        source = pickle.load(handle)
    tokenizer = AutoTokenizer.from_pretrained(str(bert_model), local_files_only=True)
    embedder = FrozenBertEmbedder(str(bert_model), device="cpu",
                                  batch_size=bert_batch_size)
    stats = None
    diagnostics = {}
    for split in ("train", "valid", "test"):
        block = source[split]
        ids = np.asarray(block["id"]).astype(str)
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate IDs in {split}")
        processed, stats, diagnostics[split] = make_processed(
            block, ids, tokenizer, embedder, stats)
        np.savez_compressed(out_dir / f"{split}.npz", **pack(processed))
        print(f"prepared unaligned {split}: {len(ids)}", flush=True)
        del processed

    paths = sorted(attachment3_dir.glob("*.pkl"))
    if len(paths) != 30:
        raise ValueError(f"expected 30 unaligned attachment 3 files, got {len(paths)}")
    ids = []
    texts = []
    audio = []
    vision = []
    for path in paths:
        suffix = path.stem.rsplit("_", 1)[-1]
        sample_id = f"附件3_{suffix}"
        if sample_id in ids:
            raise ValueError(f"duplicate attachment 3 ID {sample_id}")
        with path.open("rb") as handle:
            value = pickle.load(handle)
        block = value["test"] if "test" in value else value
        ids.append(sample_id)
        texts.append(str(np.asarray(block["raw_text"])[0]))
        audio.append(np.asarray(block["audio"])[0])
        vision.append(np.asarray(block["vision"])[0])
    attachment = {"raw_text": np.asarray(texts), "audio": np.asarray(audio),
                  "vision": np.asarray(vision)}
    processed, _, diagnostics["attachment3"] = make_processed(
        attachment, np.asarray(ids), tokenizer, embedder, stats)
    np.savez_compressed(out_dir / "attachment3.npz", **pack(processed))
    print(f"prepared unaligned attachment3: {len(ids)}", flush=True)
    manifest = {"source": str(source_path), "attachment3": str(attachment3_dir),
                "text": "raw_text tokenized with the same frozen local BERT across all splits",
                "audio_vision": "independent raw clocks pooled into at most 50 ordered bins; all-zero bins unavailable",
                "length_rule": "max(declared length, last finite nonzero row) for labeled splits; last nonzero row for attachment3",
                "normalization": "train-only observed-bin mean and std, clipped to [-5,5]",
                "stats": {name: {"mean": mean.tolist(), "std": std.tolist()}
                          for name, (mean, std) in stats.items()},
                "diagnostics": diagnostics}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "E题数据/附件2-数据集特征文件/unaligned_50.pkl")
    parser.add_argument("--attachment3", type=Path,
                        default=ROOT / "E题数据/附件3-模态缺失特征样本/未对齐版本")
    parser.add_argument("--bert-model", type=Path,
                        default=ROOT / "problem1/models/bert-base-uncased")
    parser.add_argument("--out", type=Path, default=ROOT / "problem2/cache_unaligned")
    parser.add_argument("--bert-batch-size", type=int, default=32)
    args = parser.parse_args()
    prepare(args.source, args.attachment3, args.bert_model, args.out, args.bert_batch_size)


if __name__ == "__main__":
    main()
