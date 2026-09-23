"""End-to-end extraction pipeline for Attachment 1."""

from __future__ import annotations

import difflib
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .alignment import (
    is_digital_silence,
    parse_textgrid,
    prepare_mfa_item,
    run_mfa,
    safe_sample_name,
    uniform_word_intervals,
)
from .audio_features import AUDIO_FEATURE_NAMES, extract_audio74
from .audit import audit_payload, write_audit_artifacts
from .core import align_and_pad, quality_weighted_pool
from .io import LabelRecord, atomic_json_dump, read_labels, record_to_jsonable
from .text_features import BertWordEmbedder
from .visual_features import VISUAL_FEATURE_NAMES, load_openface35, run_openface


@dataclass(frozen=True)
class PipelineConfig:
    labels: Path
    videos_root: Path
    work_dir: Path
    output: Path
    bert_model: str
    mfa_dictionary: str = "english_us_arpa"
    mfa_acoustic_model: str = "english_us_arpa"
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    mfa: str = "mfa"
    openface: str = "FeatureExtraction"
    openface_model: Path | None = None
    openface_face_detector: Path | None = None
    device: str = "cpu"
    target_length: int = 50
    visual_interpolation_gap: float = 0.25
    quality_threshold: float = 0.25
    resume: bool = True
    clean_mfa: bool = False


def _version(command: str, arguments: list[str] | None = None) -> str:
    arguments = arguments or ["--version"]
    path = shutil.which(command) or (command if Path(command).exists() else None)
    if path is None:
        return "not found"
    try:
        result = subprocess.run(
            [str(path), *arguments], capture_output=True, text=True, timeout=20, check=False
        )
        line = (result.stdout or result.stderr).strip().splitlines()
        return line[0] if line else "version unavailable"
    except (OSError, subprocess.SubprocessError) as exc:
        return f"version unavailable: {exc}"


def _duration(video: Path, ffprobe: str) -> float:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def _normal_words(text: str) -> list[str]:
    return [token.lower().replace("’", "'") for token in re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*", text)]


def _alignment_confidence(record: LabelRecord, aligned_words: list[str]) -> tuple[float, dict[str, Any]]:
    expected = _normal_words(record.text)
    actual = _normal_words(" ".join(aligned_words))
    lexical_ratio = difflib.SequenceMatcher(a=expected, b=actual, autojunk=False).ratio()
    unknown_rate = sum(word.lower() in {"<unk>", "<oov>", "spn"} for word in aligned_words) / max(1, len(aligned_words))
    score = float(lexical_ratio * (1.0 - unknown_rate))
    return score, {
        "kind": "lexical_coverage_proxy_not_acoustic_posterior",
        "expected_token_count": len(expected),
        "aligned_token_count": len(actual),
        "sequence_match_ratio": lexical_ratio,
        "unknown_rate": unknown_rate,
    }


def prepare_corpus(records: list[LabelRecord], config: PipelineConfig) -> Path:
    corpus = config.work_dir / "mfa_corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    manifest = []
    for record in records:
        name = safe_sample_name(record.video_id, record.clip_id)
        wav = corpus / f"{name}.wav"
        lab = corpus / f"{name}.lab"
        if not (config.resume and wav.exists() and lab.exists()):
            prepare_mfa_item(record.video_path, record.text, corpus, name, config.ffmpeg)
        manifest.append({**record_to_jsonable(record), "working_name": name})
    atomic_json_dump(manifest, config.work_dir / "manifest.json")
    return corpus


def _textgrid_for(alignment_dir: Path, name: str) -> Path:
    matches = list(alignment_dir.rglob(f"{name}.TextGrid"))
    if not matches:
        raise FileNotFoundError(f"no TextGrid for {name}")
    if len(matches) != 1:
        raise ValueError(f"expected one TextGrid for {name}, found {len(matches)}")
    return matches[0]


def _sample(
    record: LabelRecord,
    config: PipelineConfig,
    corpus: Path,
    alignment_dir: Path,
    bert: BertWordEmbedder,
) -> dict[str, Any]:
    name = safe_sample_name(record.video_id, record.clip_id)
    wav = corpus / f"{name}.wav"
    video_duration = _duration(record.video_path, config.ffprobe)
    try:
        words = parse_textgrid(_textgrid_for(alignment_dir, name))
        forced_alignment_available = True
        alignment_method = "mfa_forced_alignment"
    except FileNotFoundError:
        if not is_digital_silence(wav):
            raise RuntimeError(f"MFA TextGrid is missing for non-silent audio: {record.sample_id}")
        words = uniform_word_intervals(record.text, video_duration)
        forced_alignment_available = False
        alignment_method = "uniform_text_timeline_fallback"
    word_strings = [item.word for item in words]
    text = bert.encode(word_strings)

    audio_times, audio_frames, audio_q, audio_meta = extract_audio74(wav)
    audio, audio_available, audio_scores = quality_weighted_pool(
        audio_times, audio_frames, words, audio_q
    )

    face_dir = config.work_dir / "openface" / name
    csv_path = face_dir / f"{record.video_path.stem}.csv"
    if not (config.resume and csv_path.exists()):
        csv_path = run_openface(
            record.video_path,
            face_dir,
            config.openface,
            config.openface_model,
            config.openface_face_detector,
        )
    vision_times, vision_frames, vision_q, vision_meta = load_openface35(
        csv_path, max_interpolation_gap=config.visual_interpolation_gap
    )
    vision, vision_available, vision_scores = quality_weighted_pool(
        vision_times, vision_frames, words, vision_q
    )

    availability = np.column_stack(
        [np.ones(len(words), dtype=bool), audio_available, vision_available]
    )
    quality_scores = np.column_stack(
        [np.ones(len(words), dtype=np.float32), audio_scores, vision_scores]
    )
    fixed = align_and_pad(
        words,
        text,
        audio,
        vision,
        availability,
        quality_scores,
        target_length=config.target_length,
        quality_threshold=config.quality_threshold,
    )
    if forced_alignment_available:
        confidence, confidence_detail = _alignment_confidence(record, word_strings)
    else:
        confidence = 0.0
        confidence_detail = {
            "kind": "forced_alignment_unavailable",
            "reason": "MFA produced no TextGrid; uniform transcript intervals were used",
        }
    fixed.update(
        {
            "id": record.sample_id,
            "video_id": record.video_id,
            "clip_id": record.clip_id,
            "raw_text": record.text,
            "video_path": str(record.video_path),
            "video_duration": video_duration,
            "regression_label": np.float32(record.label),
            "classification_label": np.int64(
                {"negative": 0, "neutral": 1, "positive": 2}.get(record.annotation.lower(), -1)
            ),
            "annotation": record.annotation,
            "alignment_confidence": np.float32(confidence),
            "alignment_confidence_detail": confidence_detail,
            "forced_alignment_available": forced_alignment_available,
            "alignment_method": alignment_method,
            "face_detection_failure_rate": np.float32(vision_meta["face_detection_failure_rate"]),
            "was_truncated": False,
            "truncation_rule": "none; contiguous duration-weighted aggregation is used above 50 words",
            "tool_metadata": {
                "bert": bert.metadata,
                "audio": audio_meta,
                "vision": vision_meta,
                "pooling": "frame_support_overlap * frame_quality",
            },
        }
    )
    return fixed


def _stack(samples: list[dict[str, Any]]) -> dict[str, Any]:
    stack_keys = [
        "intervals", "text", "audio", "vision", "valid_mask", "padding_mask",
        "modality_availability_mask", "quality_scores", "quality_mask",
    ]
    result = {key: np.stack([sample[key] for sample in samples]) for key in stack_keys}
    result.update(
        {
            "id": [sample["id"] for sample in samples],
            "raw_text": [sample["raw_text"] for sample in samples],
            "words": [sample["words"] for sample in samples],
            "valid_length": np.asarray([sample["valid_length"] for sample in samples], dtype=np.int32),
            "video_duration": np.asarray([sample["video_duration"] for sample in samples], dtype=np.float32),
            "regression_labels": np.asarray([sample["regression_label"] for sample in samples], dtype=np.float32),
            "classification_labels": np.asarray([sample["classification_label"] for sample in samples], dtype=np.int64),
            "annotation": [sample["annotation"] for sample in samples],
            "alignment_confidence": np.asarray([sample["alignment_confidence"] for sample in samples], dtype=np.float32),
            "face_detection_failure_rate": np.asarray([sample["face_detection_failure_rate"] for sample in samples], dtype=np.float32),
            "was_aggregated": np.asarray([sample["was_aggregated"] for sample in samples], dtype=bool),
            "was_truncated": np.zeros(len(samples), dtype=bool),
            "forced_alignment_available": np.asarray(
                [sample["forced_alignment_available"] for sample in samples], dtype=bool
            ),
        }
    )
    return result


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    """Run all stages and atomically write one reproducible pickle feature file."""

    records = read_labels(config.labels, config.videos_root)
    corpus = prepare_corpus(records, config)
    alignment_dir = config.work_dir / "mfa_alignment"
    # MFA exports TextGrids only after finishing the alignment command. Some
    # genuinely unalignable clips (e.g. digital-silence audio) legitimately
    # produce no TextGrid and are handled explicitly in _sample.
    missing = [
        safe_sample_name(record.video_id, record.clip_id)
        for record in records
        if not list(alignment_dir.rglob(f"{safe_sample_name(record.video_id, record.clip_id)}.TextGrid"))
    ]
    if not config.resume or any(
        not is_digital_silence(corpus / f"{name}.wav") for name in missing
    ):
        run_mfa(
            corpus,
            alignment_dir,
            config.mfa_dictionary,
            config.mfa_acoustic_model,
            config.mfa,
            config.clean_mfa,
        )
    bert = BertWordEmbedder(config.bert_model, config.device)
    samples = [_sample(record, config, corpus, alignment_dir, bert) for record in records]
    tool_versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "ffmpeg": _version(config.ffmpeg, ["-version"]),
        "ffprobe": _version(config.ffprobe, ["-version"]),
        "mfa": _version(config.mfa, ["version"]),
        "openface": "version not exposed by FeatureExtraction; executable SHA-256 is recorded in audit",
    }
    payload = {
        "format_version": "problem1-word-aligned-v1",
        "metadata": {
            "sample_count": len(samples),
            "target_length": config.target_length,
            "text_dim": int(samples[0]["text"].shape[-1]),
            "audio_dim": 74,
            "vision_dim": 35,
            "audio_feature_names": AUDIO_FEATURE_NAMES,
            "visual_feature_names": VISUAL_FEATURE_NAMES,
            "mask_modality_order": ["text", "audio", "vision"],
            "padding_mask_semantics": "True means padded/ignored position",
            "tool_versions": tool_versions,
            "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(config).items()},
        },
        "samples": samples,
        "stacked": _stack(samples),
    }
    payload, audit_report, audit_manifest = audit_payload(
        payload, config.labels, config.videos_root, config.work_dir
    )
    write_audit_artifacts(config.output, payload, audit_report, audit_manifest)
    summary = {
        **payload["metadata"],
        "output": str(config.output),
        "mean_alignment_confidence": float(np.mean(payload["stacked"]["alignment_confidence"])),
        "mean_face_detection_failure_rate": float(np.mean(payload["stacked"]["face_detection_failure_rate"])),
        "aggregated_sample_count": int(payload["stacked"]["was_aggregated"].sum()),
        "forced_alignment_failure_count": int(
            (~payload["stacked"]["forced_alignment_available"]).sum()
        ),
        "truncated_sample_count": 0,
    }
    atomic_json_dump(summary, config.output.with_suffix(".summary.json"))
    return payload
