"""Command-line interface for aligned competition data preprocessing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .bert import FrozenBertEmbedder
from .io import (
    load_attachment3_aligned,
    load_attachment4_aligned,
    load_pickle,
    load_stats,
    save_pickle,
    save_stats,
)
from .pipeline import DatasetRole, preprocess_aligned_split
from .stats import fit_feature_stats


def _add_common_transform_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stats", required=True, type=Path, help="training stats JSON")
    parser.add_argument("--output", required=True, type=Path, help="output pickle")
    parser.add_argument("--clip-sigma", type=float, default=5.0)
    parser.add_argument("--bert-model", help="optional local path or Hugging Face model id")
    parser.add_argument("--bert-device", help="cpu, cuda, or mps")
    parser.add_argument("--bert-batch-size", type=int, default=32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmer-preprocess",
        description="Preprocess aligned multimodal emotion features with explicit masks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    attachment2 = subparsers.add_parser(
        "attachment2", help="fit train statistics and transform train/valid/test"
    )
    attachment2.add_argument("--input", required=True, type=Path)
    attachment2.add_argument("--output-dir", required=True, type=Path)
    attachment2.add_argument("--clip-sigma", type=float, default=5.0)
    attachment2.add_argument("--bert-model")
    attachment2.add_argument("--bert-device")
    attachment2.add_argument("--bert-batch-size", type=int, default=32)

    attachment3 = subparsers.add_parser(
        "attachment3", help="transform aligned missing-modality sample files"
    )
    attachment3.add_argument("--input-dir", required=True, type=Path)
    _add_common_transform_arguments(attachment3)

    attachment4 = subparsers.add_parser(
        "attachment4", help="transform aligned explainability sample files"
    )
    attachment4.add_argument("--input-dir", required=True, type=Path)
    _add_common_transform_arguments(attachment4)

    validate = subparsers.add_parser("validate", help="inspect an aligned attachment2 file")
    validate.add_argument("--input", required=True, type=Path)
    return parser


def _embedder_from_args(args: argparse.Namespace) -> FrozenBertEmbedder | None:
    if not getattr(args, "bert_model", None):
        return None
    return FrozenBertEmbedder(
        model_name=args.bert_model,
        device=args.bert_device,
        batch_size=args.bert_batch_size,
    )


def _mask_summary(processed: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = dict(processed["metadata"])
    for modality, masks in processed["masks"].items():
        summary[modality] = {
            name: int(np.asarray(mask, dtype=bool).sum()) for name, mask in masks.items()
        }
    return summary


def command_attachment2(args: argparse.Namespace) -> None:
    source = load_pickle(args.input)
    required_splits = {"train", "valid", "test"}
    if not required_splits.issubset(source):
        raise KeyError(f"attachment2 is missing splits: {sorted(required_splits - set(source))}")

    train_unscaled = preprocess_aligned_split(source["train"], DatasetRole.STANDARD)
    stats = {
        modality: fit_feature_stats(
            train_unscaled[modality],
            train_unscaled["masks"][modality]["effective"],
        )
        for modality in ("audio", "vision")
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_stats(stats, args.output_dir / "normalization_stats.json")

    embedder = _embedder_from_args(args)
    summaries: dict[str, Any] = {}
    for split_name in ("train", "valid", "test"):
        processed = preprocess_aligned_split(
            source[split_name],
            DatasetRole.STANDARD,
            stats=stats,
            clip_sigma=args.clip_sigma,
            bert_embedder=embedder,
        )
        save_pickle(processed, args.output_dir / f"{split_name}.pkl")
        summaries[split_name] = _mask_summary(processed)
    (args.output_dir / "preprocessing_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )


def _command_special(args: argparse.Namespace, role: DatasetRole) -> None:
    if role is DatasetRole.MISSING_TEST:
        source = load_attachment3_aligned(args.input_dir)
    else:
        source = load_attachment4_aligned(args.input_dir)
    processed = preprocess_aligned_split(
        source,
        role,
        stats=load_stats(args.stats),
        clip_sigma=args.clip_sigma,
        bert_embedder=_embedder_from_args(args),
    )
    save_pickle(processed, args.output)
    print(json.dumps(_mask_summary(processed), indent=2))


def command_validate(args: argparse.Namespace) -> None:
    source = load_pickle(args.input)
    report: dict[str, Any] = {}
    for split_name in ("train", "valid", "test"):
        processed = preprocess_aligned_split(source[split_name], DatasetRole.STANDARD)
        report[split_name] = _mask_summary(processed)
        report[split_name]["ids"] = int(len(np.asarray(source[split_name]["id"])))
        report[split_name]["classification_dtype"] = str(
            processed["classification_labels"].dtype
        )
        report[split_name]["regression_dtype"] = str(processed["regression_labels"].dtype)
    print(json.dumps(report, indent=2))


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "attachment2":
        command_attachment2(args)
    elif args.command == "attachment3":
        _command_special(args, DatasetRole.MISSING_TEST)
    elif args.command == "attachment4":
        _command_special(args, DatasetRole.EXPLAIN_TEST)
    elif args.command == "validate":
        command_validate(args)
    else:
        raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()

