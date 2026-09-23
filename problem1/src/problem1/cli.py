"""Command-line interface."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from .audit import audit_file
from .pipeline import PipelineConfig, run_pipeline
from .plotting import plot_sample


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Problem 1 word-aligned multimodal extraction")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run preparation, alignment, extraction and packaging")
    run.add_argument("--labels", type=Path, required=True)
    run.add_argument("--videos-root", type=Path, required=True)
    run.add_argument("--work-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--bert-model", required=True, help="prefer a pinned local bert-base-uncased directory")
    run.add_argument("--mfa-dictionary", default="english_us_arpa")
    run.add_argument("--mfa-acoustic-model", default="english_us_arpa")
    run.add_argument("--ffmpeg", default="ffmpeg")
    run.add_argument("--ffprobe", default="ffprobe")
    run.add_argument("--mfa", default="mfa")
    run.add_argument("--openface", default="FeatureExtraction")
    run.add_argument("--openface-model", type=Path, help="optional OpenFace landmark model")
    run.add_argument("--openface-face-detector", type=Path, help="optional Haar face detector XML")
    run.add_argument("--device", default="cpu")
    run.add_argument("--target-length", type=int, default=50)
    run.add_argument("--visual-interpolation-gap", type=float, default=0.25)
    run.add_argument("--quality-threshold", type=float, default=0.25)
    run.add_argument("--no-resume", action="store_true")
    run.add_argument("--clean-mfa", action="store_true")

    plot = subparsers.add_parser("plot", help="draw one packaged sample on a common timeline")
    plot.add_argument("--features", type=Path, required=True)
    plot.add_argument("--output", type=Path, required=True)
    plot.add_argument("--sample-index", type=int, default=0)
    plot.add_argument("--ffmpeg", default="ffmpeg")

    audit = subparsers.add_parser("audit", help="verify source coverage and enrich feature provenance")
    audit.add_argument("--features", type=Path, required=True)
    audit.add_argument("--labels", type=Path, required=True)
    audit.add_argument("--videos-root", type=Path, required=True)
    audit.add_argument("--work-dir", type=Path, required=True)
    audit.add_argument("--processing-log", type=Path)
    audit.add_argument("--verify-only", action="store_true")
    audit.add_argument("--refresh-method-assets", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "run":
        config = PipelineConfig(
            labels=args.labels,
            videos_root=args.videos_root,
            work_dir=args.work_dir,
            output=args.output,
            bert_model=args.bert_model,
            mfa_dictionary=args.mfa_dictionary,
            mfa_acoustic_model=args.mfa_acoustic_model,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
            mfa=args.mfa,
            openface=args.openface,
            openface_model=args.openface_model,
            openface_face_detector=args.openface_face_detector,
            device=args.device,
            target_length=args.target_length,
            visual_interpolation_gap=args.visual_interpolation_gap,
            quality_threshold=args.quality_threshold,
            resume=not args.no_resume,
            clean_mfa=args.clean_mfa,
        )
        payload = run_pipeline(config)
        print(f"wrote {len(payload['samples'])} samples to {args.output}")
    elif args.command == "plot":
        with args.features.open("rb") as handle:
            payload = pickle.load(handle)
        samples = payload["samples"]
        if not 0 <= args.sample_index < len(samples):
            raise IndexError(f"sample-index must be in [0, {len(samples) - 1}]")
        plot_sample(samples[args.sample_index], args.output, args.ffmpeg)
        print(f"wrote {args.output}")
    else:
        report = audit_file(
            args.features, args.labels, args.videos_root, args.work_dir,
            args.processing_log, args.verify_only, args.refresh_method_assets,
        )
        print(f"verified {report['feature_count']} samples against {report['raw_video_count']} videos")


if __name__ == "__main__":
    main()
