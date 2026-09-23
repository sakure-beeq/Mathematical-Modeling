"""Input discovery and atomic output helpers."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LabelRecord:
    video_id: str
    clip_id: str
    text: str
    label: float
    annotation: str
    video_path: Path

    @property
    def sample_id(self) -> str:
        return f"{self.video_id}$_${self.clip_id}"


def read_labels(xlsx: Path, videos_root: Path) -> list[LabelRecord]:
    try:
        import openpyxl
    except ImportError as exc:
        raise RuntimeError("reading xlsx labels requires openpyxl") from exc
    workbook = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    sheet = workbook.active
    rows = sheet.iter_rows(values_only=True)
    header = [str(value).strip() for value in next(rows)]
    required = ["video_id", "clip_id", "text", "label", "annotation"]
    missing = set(required) - set(header)
    if missing:
        raise ValueError(f"label sheet lacks columns: {sorted(missing)}")
    index = {name: header.index(name) for name in required}
    records: list[LabelRecord] = []
    for row_number, row in enumerate(rows, start=2):
        if not row or row[index["video_id"]] is None:
            continue
        video_id = str(row[index["video_id"]]).strip()
        clip_id = str(row[index["clip_id"]]).strip()
        if clip_id.endswith(".0"):
            clip_id = clip_id[:-2]
        video_path = videos_root / video_id / f"{clip_id}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"row {row_number}: video not found: {video_path}")
        records.append(
            LabelRecord(
                video_id=video_id,
                clip_id=clip_id,
                text=str(row[index["text"]]).strip(),
                label=float(row[index["label"]]),
                annotation=str(row[index["annotation"]]).strip(),
                video_path=video_path,
            )
        )
    return records


def atomic_pickle_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def atomic_json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(path)


def record_to_jsonable(record: LabelRecord) -> dict[str, Any]:
    result = asdict(record)
    result["video_path"] = str(result["video_path"])
    result["sample_id"] = record.sample_id
    return result

