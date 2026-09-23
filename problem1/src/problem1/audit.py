"""Source coverage, timeline lineage, and reproducibility audit for Problem 1."""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import shlex
import shutil
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .alignment import is_digital_silence, parse_textgrid, safe_sample_name, uniform_word_intervals
from .io import atomic_json_dump, atomic_pickle_dump, read_labels


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _asset(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"not a regular file: {path}")
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": _sha256(resolved)}


def _tree_asset(path: Path, suffixes: set[str] | None = None) -> dict[str, Any]:
    files = sorted(
        item for item in path.rglob("*")
        if item.is_file() and (suffixes is None or item.suffix in suffixes)
    )
    digest = hashlib.sha256()
    total_bytes = 0
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(bytes.fromhex(_sha256(item)))
        total_bytes += item.stat().st_size
    return {"path": str(path.resolve()), "file_count": len(files), "bytes": total_bytes, "sha256": digest.hexdigest()}


def _method_assets(parameters: dict[str, Any]) -> dict[str, Any]:
    bert = Path(parameters["bert_model"])
    openface = Path(parameters["openface"])
    model = parameters.get("openface_model")
    artifacts: dict[str, Any] = {}
    if bert.is_dir():
        for name in (
            "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt",
            "model.safetensors", "pytorch_model.bin",
        ):
            if (bert / name).is_file():
                artifacts[f"bert_{name}"] = _asset(bert / name)
    if openface.is_file():
        artifacts["openface_executable"] = _asset(openface)
        for name in ("model", "AU_predictors", "classifiers"):
            directory = openface.parent / name
            if directory.is_dir():
                artifacts[f"openface_{name}_tree"] = _tree_asset(directory)
    if model is not None and Path(model).is_file():
        artifacts["openface_landmark_model"] = _asset(Path(model))
    mfa_root = Path.home() / "Documents" / "MFA" / "pretrained_models"
    for kind, extension, option in (
        ("dictionary", ".dict", "mfa_dictionary"),
        ("acoustic", ".zip", "mfa_acoustic_model"),
    ):
        value = Path(parameters[option])
        path = value if value.is_file() else mfa_root / kind / f"{value}{extension}"
        if path.is_file():
            artifacts[f"mfa_{kind}"] = _asset(path)
    code_dir = Path(__file__).parent
    artifacts["audited_source_code"] = _tree_asset(code_dir, {".py"})
    return artifacts


def _label_rows(labels: Path) -> list[int]:
    import openpyxl

    book = openpyxl.load_workbook(labels, read_only=True, data_only=True)
    rows = book.active.iter_rows(values_only=True)
    header = [str(value).strip() for value in next(rows)]
    id_column = header.index("video_id")
    return [number for number, row in enumerate(rows, start=2) if row and row[id_column] is not None]


def _feature_sha256(sample: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in (
        "intervals", "text", "audio", "vision", "valid_mask", "padding_mask",
        "modality_availability_mask", "quality_scores", "quality_mask",
    ):
        array = np.ascontiguousarray(sample[key])
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _check_sequence(sample: dict[str, Any], stacked: dict[str, Any], index: int, words: list[Any]) -> np.ndarray:
    target = int(sample["padding_mask"].shape[0])
    count = len(words)
    groups = (
        [np.asarray([i]) for i in range(count)]
        if count <= target else np.array_split(np.arange(count), target)
    )
    length = len(groups)
    if int(sample["original_word_count"]) != count or int(sample["valid_length"]) != length:
        raise ValueError(f"{sample['id']}: original word count or valid length disagrees with source")
    if bool(sample["was_aggregated"]) != (count > target) or bool(sample["was_truncated"]):
        raise ValueError(f"{sample['id']}: aggregation or truncation marker disagrees with source")
    expected_valid = np.arange(target) < length
    if not np.array_equal(sample["valid_mask"], expected_valid):
        raise ValueError(f"{sample['id']}: valid_mask disagrees with valid_length")
    if not np.array_equal(sample["padding_mask"], ~expected_valid):
        raise ValueError(f"{sample['id']}: padding_mask disagrees with valid_length")
    spans = np.full((target, 2), -1, dtype=np.int32)
    for position, group in enumerate(groups):
        spans[position] = (int(group[0]), int(group[-1]) + 1)
        expected_word = " ".join(words[i].word for i in group)
        expected_interval = (words[int(group[0])].start, words[int(group[-1])].end)
        if sample["words"][position] != expected_word:
            raise ValueError(f"{sample['id']}: word text mismatch at position {position}")
        if not np.allclose(sample["intervals"][position], expected_interval, atol=1e-3):
            raise ValueError(f"{sample['id']}: source time mismatch at position {position}")
    if any(sample["words"][length:]):
        raise ValueError(f"{sample['id']}: padded words are not empty")
    for key in ("intervals", "text", "audio", "vision", "quality_scores"):
        if not np.isfinite(sample[key]).all() or not np.all(sample[key][length:] == 0):
            raise ValueError(f"{sample['id']}: {key} contains non-finite or nonzero padding values")
    for key in ("modality_availability_mask", "quality_mask"):
        if np.any(sample[key][length:]):
            raise ValueError(f"{sample['id']}: {key} has positive padding entries")
    for key in (
        "intervals", "text", "audio", "vision", "valid_mask", "padding_mask",
        "modality_availability_mask", "quality_scores", "quality_mask",
    ):
        if not np.array_equal(sample[key], stacked[key][index]):
            raise ValueError(f"{sample['id']}: stacked {key} differs from sample")
    return spans


def audit_payload(
    payload: dict[str, Any], labels: Path, videos_root: Path, work_dir: Path,
    processing_log: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Validate every source and output row, then add traceability to the pickle."""

    records = read_labels(labels, videos_root)
    rows = _label_rows(labels)
    samples = payload["samples"]
    stacked = payload["stacked"]
    if len(records) != len(samples) or len(rows) != len(records):
        raise ValueError("label and feature sample counts differ")
    ids = [record.sample_id for record in records]
    if len(set(ids)) != len(ids) or [sample["id"] for sample in samples] != ids:
        raise ValueError("feature IDs are duplicated or differ from label row order")
    videos = {path.resolve() for path in videos_root.rglob("*.mp4")}
    expected_videos = {record.video_path.resolve() for record in records}
    if videos != expected_videos:
        raise ValueError(f"raw video coverage mismatch: {len(expected_videos - videos)} missing, {len(videos - expected_videos)} unlabelled")
    if stacked["text"].shape[0] != len(records) or stacked["id"] != ids:
        raise ValueError("stacked output row order differs from labels")

    label_asset = _asset(labels)
    manifest: list[dict[str, Any]] = []
    spans_all: list[np.ndarray] = []
    wav_paths: set[Path] = set()
    lab_paths: set[Path] = set()
    grid_paths: set[Path] = set()
    csv_paths: set[Path] = set()
    fallback_count = 0
    zero_face_count = 0
    max_wav_video_gap = 0.0
    for index, (record, label_row, sample) in enumerate(zip(records, rows, samples)):
        if Path(sample["video_path"]).resolve() != record.video_path.resolve():
            raise ValueError(f"{record.sample_id}: recorded video path differs from label")
        if (sample["raw_text"] != record.text or sample["annotation"] != record.annotation
                or not np.isclose(sample["regression_label"], record.label)):
            raise ValueError(f"{record.sample_id}: label text, annotation, or score differs")
        name = safe_sample_name(record.video_id, record.clip_id)
        wav = work_dir / "mfa_corpus" / f"{name}.wav"
        lab = work_dir / "mfa_corpus" / f"{name}.lab"
        csv_path = work_dir / "openface" / name / f"{record.video_path.stem}.csv"
        if wav.resolve() in wav_paths or csv_path.resolve() in csv_paths:
            raise ValueError(f"{record.sample_id}: modality file reused by another sample")
        wav_paths.add(wav.resolve())
        lab_paths.add(lab.resolve())
        csv_paths.add(csv_path.resolve())
        if lab.read_text(encoding="utf-8") != record.text.strip() + "\n":
            raise ValueError(f"{record.sample_id}: MFA transcript differs from label")
        if Path(sample["tool_metadata"]["vision"]["source_csv"]).resolve() != csv_path.resolve():
            raise ValueError(f"{record.sample_id}: recorded OpenFace source CSV differs")
        with wave.open(str(wav)) as audio_file:
            wav_duration = audio_file.getnframes() / audio_file.getframerate()
            if audio_file.getframerate() != 16000 or audio_file.getnchannels() != 1:
                raise ValueError(f"{record.sample_id}: WAV is not mono 16 kHz")
        wav_video_gap = abs(wav_duration - float(sample["video_duration"]))
        max_wav_video_gap = max(max_wav_video_gap, wav_video_gap)
        if wav_video_gap > 0.25:
            raise ValueError(f"{record.sample_id}: WAV and video durations differ")
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            csv_rows = list(csv.DictReader(handle, skipinitialspace=True))
        if not csv_rows:
            raise ValueError(f"{record.sample_id}: OpenFace CSV has no frames")
        csv_times = np.asarray([float(row["timestamp"]) for row in csv_rows])
        if (not np.isfinite(csv_times).all() or np.any(np.diff(csv_times) <= 0)
                or csv_times[0] < 0 or csv_times[-1] > float(sample["video_duration"]) + 0.2):
            raise ValueError(f"{record.sample_id}: OpenFace frame times disagree with video duration")
        grids = list((work_dir / "mfa_alignment").rglob(f"{name}.TextGrid"))
        forced = bool(sample["forced_alignment_available"])
        expected_method = "mfa_forced_alignment" if forced else "uniform_text_timeline_fallback"
        if sample["alignment_method"] != expected_method:
            raise ValueError(f"{record.sample_id}: alignment method marker disagrees with source")
        if forced and len(grids) != 1:
            raise ValueError(f"{record.sample_id}: MFA TextGrid missing or ambiguous")
        if not forced and grids:
            raise ValueError(f"{record.sample_id}: fallback marked despite existing TextGrid")
        if not forced and not is_digital_silence(wav):
            raise ValueError(f"{record.sample_id}: uniform fallback used for non-silent audio")
        if forced:
            grid_paths.add(grids[0].resolve())
        words = parse_textgrid(grids[0]) if forced else uniform_word_intervals(record.text, float(sample["video_duration"]))
        if any(word.end > float(sample["video_duration"]) + 0.1 for word in words):
            raise ValueError(f"{record.sample_id}: word interval extends beyond source video")
        if not forced:
            if float(sample["alignment_confidence"]) != 0:
                raise ValueError(f"{record.sample_id}: silence fallback confidence must be zero")
            fallback_count += 1
        spans = _check_sequence(sample, stacked, index, words)
        threshold = float(payload["metadata"]["parameters"]["quality_threshold"])
        expected_quality_mask = (
            (sample["quality_scores"] >= threshold)
            & sample["modality_availability_mask"]
            & sample["valid_mask"][:, None]
        )
        if not np.array_equal(sample["quality_mask"], expected_quality_mask):
            raise ValueError(f"{record.sample_id}: quality mask disagrees with threshold")
        if "source_word_span" in sample and not np.array_equal(sample["source_word_span"], spans):
            raise ValueError(f"{record.sample_id}: saved source word spans differ from source")
        if "sequence_position" in sample and not np.array_equal(sample["sequence_position"], np.arange(spans.shape[0])):
            raise ValueError(f"{record.sample_id}: saved sequence positions differ")
        if "sample_index" in sample and int(sample["sample_index"]) != index:
            raise ValueError(f"{record.sample_id}: saved sample index differs")
        if "label_row" in sample and int(sample["label_row"]) != label_row:
            raise ValueError(f"{record.sample_id}: saved label row differs")
        spans_all.append(spans)
        if float(sample["face_detection_failure_rate"]) == 1.0:
            zero_face_count += 1
        files = {
            "video": _asset(record.video_path),
            "wav": _asset(wav),
            "transcript_lab": _asset(lab),
            "textgrid": _asset(grids[0]) if forced else None,
            "openface_csv": _asset(csv_path),
        }
        if "source_files" in sample:
            for modality, current in files.items():
                previous = sample["source_files"].get(modality)
                if modality in sample["source_files"] and (previous or {}).get("sha256") != (current or {}).get("sha256"):
                    raise ValueError(f"{record.sample_id}: {modality} changed since prior audit")
        word_timeline = [
            {"index": i, "word": word.word, "start_s": word.start, "end_s": word.end}
            for i, word in enumerate(words)
        ]
        if "source_word_timeline" in sample and sample["source_word_timeline"] != word_timeline:
            raise ValueError(f"{record.sample_id}: original word timeline changed since prior audit")
        feature_digest = _feature_sha256(sample)
        if "feature_sha256" in sample and sample["feature_sha256"] != feature_digest:
            raise ValueError(f"{record.sample_id}: feature values changed since prior audit")
        sample.update({
            "sample_index": index,
            "label_row": label_row,
            "source_files": files,
            "source_word_timeline": word_timeline,
            "sequence_position": np.arange(spans.shape[0], dtype=np.int32),
            "source_word_span": spans,
            "feature_sha256": feature_digest,
        })
        manifest.append({
            "sample_index": index,
            "label_row": label_row,
            "sample_id": record.sample_id,
            "video_id": record.video_id,
            "clip_id": record.clip_id,
            "video_path": files["video"]["path"],
            "video_sha256": files["video"]["sha256"],
            "wav_path": files["wav"]["path"],
            "wav_sha256": files["wav"]["sha256"],
            "transcript_lab_path": files["transcript_lab"]["path"],
            "transcript_lab_sha256": files["transcript_lab"]["sha256"],
            "textgrid_path": files["textgrid"]["path"] if forced else "",
            "textgrid_sha256": files["textgrid"]["sha256"] if forced else "",
            "openface_csv_path": files["openface_csv"]["path"],
            "openface_csv_sha256": files["openface_csv"]["sha256"],
            "feature_sha256": feature_digest,
            "original_word_count": len(words),
            "valid_length": int(sample["valid_length"]),
            "was_aggregated": bool(sample["was_aggregated"]),
            "alignment_method": sample["alignment_method"],
            "face_detection_failure_rate": float(sample["face_detection_failure_rate"]),
            "wav_duration_s": wav_duration,
            "wav_video_duration_gap_s": wav_video_gap,
            "openface_frame_count": len(csv_rows),
        })

    observed_wavs = {path.resolve() for path in (work_dir / "mfa_corpus").glob("*.wav")}
    observed_labs = {path.resolve() for path in (work_dir / "mfa_corpus").glob("*.lab")}
    observed_grids = {path.resolve() for path in (work_dir / "mfa_alignment").rglob("*.TextGrid")}
    observed_csvs = {path.resolve() for path in (work_dir / "openface").rglob("*.csv")}
    if (observed_wavs != wav_paths or observed_labs != lab_paths
            or observed_grids != grid_paths or observed_csvs != csv_paths):
        raise ValueError("intermediate modality files contain missing, duplicate, or unlabelled assets")

    if "source_word_span" in stacked and not np.array_equal(stacked["source_word_span"], np.stack(spans_all)):
        raise ValueError("stacked source word spans differ from samples")
    stacked["sample_index"] = np.arange(len(samples), dtype=np.int32)
    stacked["label_row"] = np.asarray(rows, dtype=np.int32)
    stacked["sequence_position"] = np.broadcast_to(
        np.arange(spans_all[0].shape[0], dtype=np.int32), (len(samples), spans_all[0].shape[0])
    ).copy()
    stacked["source_word_span"] = np.stack(spans_all)
    prior_audit = payload["metadata"].get("audit", {})
    if processing_log is None and prior_audit.get("processing_log"):
        processing_log = Path(prior_audit["processing_log"]["path"])
    processing_log_asset = _asset(processing_log) if processing_log is not None else None
    if prior_audit.get("processing_log") and prior_audit["processing_log"]["sha256"] != processing_log_asset["sha256"]:
        raise ValueError("processing log changed since prior audit")
    parameters = payload["metadata"]["parameters"]
    method_assets = _method_assets(parameters)
    for key, previous in prior_audit.get("audit_time_method_assets", {}).items():
        current = method_assets.get(key)
        if current is None or current["sha256"] != previous["sha256"]:
            raise ValueError(f"method asset changed since prior audit: {key}")
    tool_versions = payload["metadata"]["tool_versions"]
    if tool_versions.get("openface", "").startswith("Reading the landmark detector/tracker"):
        tool_versions["openface"] = (
            "version not exposed by FeatureExtraction; executable SHA-256 is recorded in audit"
        )
    recorded_libraries = {
        "torch": samples[0]["tool_metadata"]["bert"]["torch_version"],
        "transformers": samples[0]["tool_metadata"]["bert"]["transformers_version"],
        "librosa": samples[0]["tool_metadata"]["audio"]["librosa_version"],
    }
    audit_info = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "labels": label_asset,
        "raw_video_count": len(videos),
        "wav_count": len(observed_wavs),
        "transcript_lab_count": len(observed_labs),
        "textgrid_count": len(observed_grids),
        "openface_csv_count": len(observed_csvs),
        "label_count": len(records),
        "feature_count": len(samples),
        "one_to_one_mapping": True,
        "sequence_index_base": 0,
        "source_word_span_semantics": "half-open [start,end) indices into source_word_timeline; [-1,-1] means padding",
        "time_unit": "seconds from the start of each source MP4",
        "padding_rule": "right padding to target_length; zero features/intervals/quality and false availability",
        "aggregation_rule": "numpy.array_split into contiguous groups when original_word_count exceeds target_length",
        "alignment_fallback_count": fallback_count,
        "full_face_failure_count": zero_face_count,
        "max_wav_video_duration_gap_s": max_wav_video_gap,
        "wav_video_duration_tolerance_s": 0.25,
        "processing_log": processing_log_asset,
        "extraction_parameters": parameters,
        "recorded_tool_versions": tool_versions,
        "recorded_library_versions": recorded_libraries,
        "audit_time_method_assets": method_assets,
    }
    payload["metadata"]["audit"] = audit_info
    report = {"format": "problem1-audit-v1", **audit_info, "checks": {
        "all_label_rows_matched": True,
        "all_raw_videos_matched": True,
        "all_modality_files_present_unique_and_hashed": True,
        "all_timeline_positions_matched_source_words": True,
        "all_padding_and_stacked_rows_consistent": True,
        "all_feature_values_finite": True,
        "wav_format_and_duration_checked": True,
        "openface_csv_frame_count_positive": True,
    }}
    return payload, report, manifest


def write_audit_artifacts(
    output: Path, payload: dict[str, Any], report: dict[str, Any], manifest: list[dict[str, Any]],
    processing_log: Path | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_pickle_dump(payload, output)
    stem = output.with_suffix("")
    atomic_json_dump(report, Path(f"{stem}.audit.json"))
    atomic_json_dump(manifest, Path(f"{stem}.manifest.json"))
    csv_path = Path(f"{stem}.manifest.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    log_path = Path(f"{stem}.audit.jsonl")
    with log_path.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps({"event": "sample_verified", **row}, ensure_ascii=False) + "\n")
    if processing_log is not None:
        shutil.copyfile(processing_log, Path(f"{stem}.processing.log"))
    _write_reproduction(Path(f"{stem}.reproduce.md"), output, payload)


def _write_reproduction(path: Path, output: Path, payload: dict[str, Any]) -> None:
    params = payload["metadata"]["parameters"]
    options = [
        ("--labels", params["labels"]), ("--videos-root", params["videos_root"]),
        ("--work-dir", params["work_dir"]), ("--output", str(output)),
        ("--bert-model", params["bert_model"]),
        ("--mfa-dictionary", params["mfa_dictionary"]),
        ("--mfa-acoustic-model", params["mfa_acoustic_model"]),
        ("--ffmpeg", params["ffmpeg"]), ("--ffprobe", params["ffprobe"]),
        ("--mfa", params["mfa"]), ("--openface", params["openface"]),
        ("--openface-model", params.get("openface_model")),
        ("--openface-face-detector", params.get("openface_face_detector")),
        ("--device", params["device"]),
        ("--target-length", params["target_length"]),
        ("--visual-interpolation-gap", params["visual_interpolation_gap"]),
        ("--quality-threshold", params["quality_threshold"]),
    ]
    command = "problem1 run " + " ".join(
        f"{name} {shlex.quote(str(value))}" for name, value in options if value is not None
    )
    verify_command = "problem1 audit " + " ".join(
        f"{name} {shlex.quote(str(value))}" for name, value in (
            ("--features", output), ("--labels", params["labels"]),
            ("--videos-root", params["videos_root"]), ("--work-dir", params["work_dir"]),
        )
    ) + " --verify-only"
    project_dir = Path(__file__).resolve().parents[2]
    path.write_text(
        "# 问题1特征复现实验说明\n\n"
        "本次结果在 `MathematicalModeling` 环境生成。先进入同一项目目录；"
        "下列命令使用断点续跑的现有中间文件。若从原始视频重新提取，改用独立工作目录，"
        "并添加 `--no-resume --clean-mfa`。\n\n"
        f"```bash\nconda activate MathematicalModeling\ncd {shlex.quote(str(project_dir))}\n"
        "export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1\n"
        f"{command}\n```\n\n"
        "词区间以 MP4 开始时刻为零点；`sequence_position` 从 0 起，"
        "`source_word_span` 指向 `source_word_timeline` 的半开索引区间。"
        "超过 50 词时按连续词组聚合；右侧位置填零，`padding_mask=True` 表示忽略。"
        "文本使用冻结 BERT 与同词 WordPiece 均值；语音使用 74 维声学描述符，"
        "视觉使用 OpenFace 35 维帧特征，再按帧覆盖区间重叠时长乘质量分数池化。\n\n"
        "FFmpeg只提取16 kHz单声道PCM，不降噪或统一响度。"
        f"MFA使用 `{params['mfa_dictionary']}` 词典和 `{params['mfa_acoustic_model']}` 声学模型；"
        "仅数字静音且没有TextGrid的样本使用明确标记、置信度为零的均匀词区间；"
        "其他缺失TextGrid的样本会报错。"
        "语音帧长400点、帧移160点；"
        f"视觉短缺失插值阈值{params['visual_interpolation_gap']}秒，"
        f"质量掩码阈值{params['quality_threshold']}。\n\n"
        "逐样本原始视频、WAV、转写LAB、TextGrid、OpenFace CSV 和特征摘要见同名 `manifest.csv`。"
        "`audit.json` 给出覆盖与时序检查，`audit.jsonl` 是逐样本核验日志。"
        "若存在 `processing.log`，它是原始命令行处理输出。"
        "工具版本和关键参数保存在 pickle 元数据及 `audit.json`；模型与代码校验值为审计时采集。"
        "初次提取的运行期未记录逐文件哈希。\n\n"
        f"核验命令：\n\n```bash\n{verify_command}\n```\n",
        encoding="utf-8",
    )


def audit_file(
    features: Path, labels: Path, videos_root: Path, work_dir: Path,
    processing_log: Path | None = None, verify_only: bool = False,
    refresh_method_assets: bool = False,
) -> dict[str, Any]:
    with features.open("rb") as handle:
        payload = pickle.load(handle)
    if refresh_method_assets:
        if verify_only:
            raise ValueError("--refresh-method-assets cannot be combined with --verify-only")
        payload["metadata"].get("audit", {}).pop("audit_time_method_assets", None)
    payload, report, manifest = audit_payload(payload, labels, videos_root, work_dir, processing_log)
    if not verify_only:
        write_audit_artifacts(features, payload, report, manifest, processing_log)
        summary_path = features.with_suffix(".summary.json")
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["audit"] = report
            atomic_json_dump(summary, summary_path)
    return report
