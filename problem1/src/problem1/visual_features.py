"""OpenFace invocation and conversion to 35 frame-level visual features."""

from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .core import interpolate_short_gaps


AU_COLUMNS = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
    "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
]
POSE_COLUMNS = ["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"]
GAZE_COLUMNS = [
    "gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z",
    "gaze_angle_x", "gaze_angle_y",
]
LANDMARK_STATS = ["face_width", "face_height", "face_center_x", "face_center_y"]
VISUAL_FEATURE_NAMES = AU_COLUMNS + POSE_COLUMNS + GAZE_COLUMNS + LANDMARK_STATS
assert len(VISUAL_FEATURE_NAMES) == 35


def run_openface(
    video: Path,
    output_dir: Path,
    executable: str = "FeatureExtraction",
    model: Path | None = None,
    face_detector: Path | None = None,
) -> Path:
    """Run OpenFace FeatureExtraction and return its CSV path."""

    if shutil.which(executable) is None and not Path(executable).exists():
        raise FileNotFoundError(f"OpenFace executable not found: {executable!r}")
    for path in (model, face_detector):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"OpenFace model file not found: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [executable, "-f", str(video), "-out_dir", str(output_dir), "-aus", "-pose", "-gaze", "-2Dfp"]
    if model is not None:
        command.extend(["-mloc", str(model)])
    if face_detector is not None:
        command.extend(["-fdloc", str(face_detector)])
    subprocess.run(
        command,
        check=True,
    )
    csv_path = output_dir / f"{video.stem}.csv"
    if not csv_path.exists():
        candidates = sorted(output_dir.glob("*.csv"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"cannot identify OpenFace CSV in {output_dir}")
        csv_path = candidates[0]
    return csv_path


def _number(row: dict[str, str], name: str, default: float = np.nan) -> float:
    raw = row.get(name, "").strip()
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def load_openface35(
    csv_path: Path,
    max_interpolation_gap: float = 0.25,
    confidence_floor: float = 0.50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Read OpenFace output, retain long failures, and interpolate short failures."""

    times: list[float] = []
    rows: list[list[float]] = []
    quality: list[float] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        fields = {name.strip() for name in (reader.fieldnames or [])}
        required = set(
            AU_COLUMNS
            + POSE_COLUMNS
            + GAZE_COLUMNS
            + ["timestamp", "success", "confidence"]
            + [f"x_{i}" for i in range(68)]
            + [f"y_{i}" for i in range(68)]
        )
        missing = required - fields
        if missing:
            raise ValueError(f"OpenFace CSV lacks columns: {sorted(missing)}")
        for raw_row in reader:
            row = {key.strip(): value for key, value in raw_row.items()}
            xs = np.asarray([_number(row, f"x_{i}") for i in range(68)], dtype=np.float32)
            ys = np.asarray([_number(row, f"y_{i}") for i in range(68)], dtype=np.float32)
            landmark = [
                float(np.nanmax(xs) - np.nanmin(xs)),
                float(np.nanmax(ys) - np.nanmin(ys)),
                float(np.nanmean(xs)),
                float(np.nanmean(ys)),
            ]
            values = [_number(row, name) for name in AU_COLUMNS + POSE_COLUMNS + GAZE_COLUMNS] + landmark
            success = _number(row, "success", 0.0) >= 0.5
            confidence = np.clip(_number(row, "confidence", 0.0), 0.0, 1.0)
            times.append(_number(row, "timestamp"))
            rows.append(values)
            quality.append(float(confidence if success and confidence >= confidence_floor else 0.0))
    t = np.asarray(times, dtype=np.float64)
    x = np.asarray(rows, dtype=np.float32)
    q = np.asarray(quality, dtype=np.float32)
    if t.size == 0 or np.any(np.diff(t) <= 0):
        raise ValueError(f"OpenFace timestamps are empty or non-monotonic: {csv_path}")
    raw_failure_rate = float(np.mean(q <= 0))
    x, q = interpolate_short_gaps(t, x, q, max_interpolation_gap)
    metadata = {
        "extractor": "OpenFace FeatureExtraction",
        "source_csv": str(csv_path),
        "max_interpolation_gap_seconds": max_interpolation_gap,
        "confidence_floor": confidence_floor,
        "feature_names": VISUAL_FEATURE_NAMES,
        "face_detection_failure_rate": raw_failure_rate,
        "remaining_missing_rate_after_interpolation": float(np.mean(q <= 0)),
    }
    return t, x, q, metadata
