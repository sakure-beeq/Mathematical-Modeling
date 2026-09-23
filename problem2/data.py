"""Load attachment 2/3 through the shared, mask-aware preprocessing pipeline."""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
from mmer_preprocess.bert import FrozenBertEmbedder  # noqa: E402
from mmer_preprocess.io import load_attachment3_aligned  # noqa: E402
from mmer_preprocess.pipeline import DatasetRole, preprocess_aligned_split  # noqa: E402
from mmer_preprocess.stats import fit_feature_stats  # noqa: E402

MODALITIES = ("text", "audio", "vision")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    """Record reproducible relative inputs without leaking local user paths."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return path.name


def pack(processed: dict) -> dict[str, np.ndarray]:
    if "text" not in processed:
        raise ValueError("frozen BERT text embeddings are required")
    result = {name: np.asarray(processed[name], dtype=np.float32) for name in MODALITIES}
    for mask_name in ("padding", "availability", "quality"):
        result[mask_name] = np.stack(
            [processed["masks"][name][mask_name] for name in MODALITIES], axis=-1
        ).astype(bool)
    for name in ("id", "classification_labels", "regression_labels"):
        if name in processed:
            result[name] = np.asarray(processed[name])
    return result


def prepare(attachment2: Path, attachment3: Path, bert_model: Path,
            out_dir: Path, batch_size: int = 32, device: str = "cpu",
            manifest_output: Path | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with attachment2.open("rb") as handle:
        source = pickle.load(handle)
    if set(("train", "valid", "test")) - set(source):
        raise ValueError("attachment 2 must contain train, valid and test splits")
    split_ids = {}
    for split in ("train", "valid", "test"):
        block = source[split]
        ids = set(map(str, np.asarray(block["id"])))
        if len(ids) != len(block["id"]):
            raise ValueError(f"duplicate sample IDs in {split}")
        classes = np.asarray(block["classification_labels"])
        scores = np.asarray(block["regression_labels"])
        if not np.isin(classes, (0, 1, 2)).all() or not np.isfinite(scores).all():
            raise ValueError(f"invalid labels in {split}")
        if np.any((scores < -3) | (scores > 3)):
            raise ValueError(f"regression labels out of range in {split}")
        split_ids[split] = ids
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if split_ids[a] & split_ids[b]:
            raise ValueError("sample ID leakage across attachment 2 splits")
        videos_a = {sample.split("$_$")[0] for sample in split_ids[a]}
        videos_b = {sample.split("$_$")[0] for sample in split_ids[b]}
        if videos_a & videos_b:
            raise ValueError("video ID leakage across attachment 2 splits")
    train_raw = preprocess_aligned_split(source["train"], DatasetRole.STANDARD)
    stats = {name: fit_feature_stats(train_raw[name],
                                    train_raw["masks"][name]["effective"])
             for name in ("audio", "vision")}
    embedder = FrozenBertEmbedder(str(bert_model), device=device, batch_size=batch_size)
    for split in ("train", "valid", "test"):
        processed = preprocess_aligned_split(source[split], DatasetRole.STANDARD,
                                             stats=stats, bert_embedder=embedder)
        np.savez_compressed(out_dir / f"{split}.npz", **pack(processed))
        print(f"prepared {split}: {len(processed['id'])} samples", flush=True)
    special = preprocess_aligned_split(load_attachment3_aligned(attachment3),
                                       DatasetRole.MISSING_TEST, stats=stats,
                                       bert_embedder=embedder)
    np.savez_compressed(out_dir / "attachment3.npz", **pack(special))
    manifest = {
        "source": portable_path(attachment2), "source_sha256": sha256(attachment2),
        "attachment3": portable_path(attachment3),
        "bert_model": portable_path(bert_model),
        "bert_config_sha256": sha256(bert_model / "config.json"),
        "bert_weights_sha256": sha256(bert_model / "model.safetensors"),
        "text_interface": "text_bert encoded by the same frozen local BERT for all splits",
        "clip_sigma": 5.0,
        "statistics": {name: value.to_jsonable() for name, value in stats.items()},
    }
    manifest_text = json.dumps(manifest, indent=2)
    (out_dir / "manifest.json").write_text(manifest_text, encoding="utf-8")
    if manifest_output is not None:
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        manifest_output.write_text(manifest_text, encoding="utf-8")
    print("prepared attachment3: 30 samples", flush=True)


class PreparedDataset(Dataset):
    def __init__(self, path: Path, limit: int | None = None):
        with np.load(path, allow_pickle=False) as f:
            self.arrays = {key: f[key][:limit] for key in f.files}
        size = len(self.arrays["text"])
        if any(len(value) != size for value in self.arrays.values()):
            raise ValueError("inconsistent sample counts")
        if self.arrays["text"].shape[1:] != (50, 768):
            raise ValueError("expected aligned text shape (N,50,768)")

    def __len__(self) -> int:
        return len(self.arrays["text"])

    def __getitem__(self, index: int) -> dict:
        sample = {}
        for key, value in self.arrays.items():
            if key == "id":
                sample[key] = str(value[index])
            else:
                sample[key] = torch.as_tensor(value[index].copy())
        return sample


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}
