"""Prepare problem 3 train/valid/test features using attachment 2 only."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "problem2"))
sys.path.insert(0, str(ROOT / "problem2/vendor"))
from data import pack, portable_path, sha256  # noqa: E402
from mmer_preprocess.bert import FrozenBertEmbedder  # noqa: E402
from mmer_preprocess.pipeline import DatasetRole, preprocess_aligned_split  # noqa: E402
from mmer_preprocess.stats import fit_feature_stats  # noqa: E402


def check_splits(source: dict) -> None:
    if not {"train", "valid", "test"}.issubset(source):
        raise ValueError("attachment 2 must contain train, valid and test splits")
    ids = {}
    for split in ("train", "valid", "test"):
        block = source[split]
        names = set(map(str, np.asarray(block["id"])))
        if len(names) != len(block["id"]):
            raise ValueError(f"duplicate IDs in {split}")
        labels = np.asarray(block["classification_labels"])
        scores = np.asarray(block["regression_labels"])
        if not np.isin(labels, (0, 1, 2)).all():
            raise ValueError(f"invalid classification labels in {split}")
        if not np.isfinite(scores).all() or np.any((scores < -3) | (scores > 3)):
            raise ValueError(f"invalid regression labels in {split}")
        ids[split] = names
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if ids[left] & ids[right]:
            raise ValueError(f"sample ID overlap between {left} and {right}")
        left_videos = {name.split("$_$")[0] for name in ids[left]}
        right_videos = {name.split("$_$")[0] for name in ids[right]}
        if left_videos & right_videos:
            raise ValueError(f"video ID overlap between {left} and {right}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attachment2", type=Path,
                        default=ROOT / "E题数据/附件2-数据集特征文件/aligned_50.pkl")
    parser.add_argument("--bert-model", type=Path,
                        default=ROOT / "problem1/models/bert-base-uncased")
    parser.add_argument("--out", type=Path, default=HERE / "cache")
    parser.add_argument("--bert-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with args.attachment2.open("rb") as handle:
        source = pickle.load(handle)
    check_splits(source)
    train_raw = preprocess_aligned_split(source["train"], DatasetRole.STANDARD)
    stats = {name: fit_feature_stats(train_raw[name],
                                    train_raw["masks"][name]["effective"])
             for name in ("audio", "vision")}
    embedder = FrozenBertEmbedder(str(args.bert_model), device=args.device,
                                  batch_size=args.bert_batch_size)
    args.out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        processed = preprocess_aligned_split(source[split], DatasetRole.STANDARD,
                                             stats=stats, bert_embedder=embedder)
        np.savez_compressed(args.out / f"{split}.npz", **pack(processed))
        print(f"prepared {split}: {len(processed['id'])} samples", flush=True)
    manifest = {
        "source": portable_path(args.attachment2),
        "source_sha256": sha256(args.attachment2),
        "bert_model": portable_path(args.bert_model),
        "bert_config_sha256": sha256(args.bert_model / "config.json"),
        "bert_weights_sha256": sha256(args.bert_model / "model.safetensors"),
        "text_interface": "text_bert encoded by the same frozen local BERT for all splits",
        "clip_sigma": 5.0,
        "statistics": {name: value.to_jsonable() for name, value in stats.items()},
        "attachment3_used": False,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                               encoding="utf-8")


if __name__ == "__main__":
    main()
