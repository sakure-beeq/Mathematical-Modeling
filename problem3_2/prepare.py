"""Prepare attachment 4 with train-only statistics and align its supplied videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "problem2"))
sys.path.insert(0, str(ROOT / "problem2/vendor"))
from data import pack  # noqa: E402
from mmer_preprocess.bert import FrozenBertEmbedder  # noqa: E402
from mmer_preprocess.io import load_attachment4_aligned  # noqa: E402
from mmer_preprocess.pipeline import DatasetRole, preprocess_aligned_split  # noqa: E402
from mmer_preprocess.stats import FeatureStats  # noqa: E402
from timeline import build_timelines, save_timelines  # noqa: E402


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attachment4", type=Path,
                   default=ROOT / "E题数据/附件4-可解释专项视频样本与特征文件"
                   / "附件4-可解释专项视频样本与特征文件/对齐版本")
    p.add_argument("--training-cache", type=Path, default=HERE / "cache")
    p.add_argument("--bert-model", type=Path, default=ROOT / "problem1/models/bert-base-uncased")
    p.add_argument("--out", type=Path, default=HERE / "cache")
    p.add_argument("--align", action="store_true", help="also make token-to-video timelines")
    p.add_argument("--mfa-work", type=Path, default=HERE / "work")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    files = sorted(args.attachment4.glob("*.pkl"))
    if len(files) != 20:
        raise ValueError(f"expected 20 aligned feature files, found {len(files)}")
    manifest_path = args.training_cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if digest(args.bert_model / "config.json") != manifest["bert_config_sha256"]:
        raise ValueError("frozen BERT config differs from the training interface")
    if digest(args.bert_model / "model.safetensors") != manifest["bert_weights_sha256"]:
        raise ValueError("frozen BERT weights differ from the training interface")
    stats = {name: FeatureStats.from_jsonable(manifest["statistics"][name])
             for name in ("audio", "vision")}
    source = load_attachment4_aligned(args.attachment4)
    ids = list(map(str, source["id"]))
    if ids != [path.stem for path in files]:
        raise ValueError("feature file names and sample IDs disagree")
    embedder = FrozenBertEmbedder(str(args.bert_model), device=args.device, batch_size=16)
    processed = preprocess_aligned_split(source, DatasetRole.EXPLAIN_TEST,
                                          stats=stats, bert_embedder=embedder)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "attachment4.npz", **pack(processed))
    metadata = []
    for i, sample_id in enumerate(ids):
        bert = np.asarray(source["text_bert"][i])
        metadata.append({"id": sample_id, "raw_text": str(source["raw_text"][i]),
                         "input_ids": bert[0].astype(int).tolist(),
                         "valid_tokens": int(bert[1].sum())})
    (args.out / "attachment4_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.align:
        timelines = build_timelines(metadata, args.attachment4 / "videos", args.bert_model,
                                    args.mfa_work)
        save_timelines(timelines, args.out / "attachment4_timelines.json")
    provenance = {"feature_version": "aligned_50", "sample_count": len(ids),
                  "attachment4_files": {file.name: digest(file) for file in files},
                  "train_statistics_manifest_sha256": digest(manifest_path),
                  "bert_config_sha256": digest(args.bert_model / "config.json"),
                  "bert_weights_sha256": digest(args.bert_model / "model.safetensors"),
                  "alignment": "MFA token-to-video mapping" if args.align else "not requested",
                  "prediction_model_checkpoint_used": False}
    (args.out / "preparation_manifest.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"prepared {len(ids)} attachment 4 samples -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
