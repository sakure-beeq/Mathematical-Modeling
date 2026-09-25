"""Map aligned BERT positions back to attachment 4 video time and transcript."""

from __future__ import annotations

import json
import os
from difflib import SequenceMatcher
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "problem1" / "src"))
from problem1.alignment import (  # noqa: E402
    parse_textgrid, prepare_mfa_item, run_mfa,
)
from problem1.core import WordInterval  # noqa: E402

WORDS = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")


def uniform_intervals(transcript: str, duration: float) -> list[WordInterval]:
    words = WORDS.findall(transcript)
    if not words or not np.isfinite(duration) or duration <= 0:
        raise ValueError("cannot construct fallback transcript timeline")
    boundaries = np.linspace(0.0, duration, len(words) + 1)
    return [WordInterval(word=word, start=float(boundaries[i]),
                         end=float(boundaries[i + 1])) for i, word in enumerate(words)]


def reconcile_words(raw_words: list[str], aligned: list[WordInterval]
                    ) -> list[WordInterval] | None:
    """Monotone mapping for MFA punctuation, normalized fillers, and split compounds."""
    a = [re.sub(r"[^a-z0-9]", "", word.lower()) for word in raw_words]
    b = [re.sub(r"[^a-z0-9]", "", word.word.lower()) for word in aligned]
    n, m = len(a), len(b)
    inf = float("inf")
    costs = [[inf] * (m + 1) for _ in range(n + 1)]
    previous: dict[tuple[int, int], tuple[int, int, tuple[int, int] | None]] = {}
    costs[0][0] = 0.0

    def offer(i: int, j: int, ni: int, nj: int, cost: float,
              span: tuple[int, int] | None) -> None:
        value = costs[i][j] + cost
        if value < costs[ni][nj]:
            costs[ni][nj] = value
            previous[(ni, nj)] = (i, j, span)

    for i in range(n + 1):
        for j in range(m + 1):
            if costs[i][j] == inf:
                continue
            if j < m:
                offer(i, j, i, j + 1, .1 if not b[j] else .8, None)
            if i >= n or j >= m:
                continue
            if a[i] == b[j]:
                cost = 0.0
            elif b[j] == "bracketed":
                cost = .5
            elif SequenceMatcher(None, a[i], b[j]).ratio() >= .75:
                cost = .5
            else:
                cost = 3.0
            offer(i, j, i + 1, j + 1, cost, (j, j + 1))
            if j + 1 < m and a[i] == b[j] + b[j + 1]:
                offer(i, j, i + 1, j + 2, .1, (j, j + 2))
    if costs[n][m] == inf or costs[n][m] > max(1.0, n * .25):
        return None
    mapping: list[tuple[int, int] | None] = [None] * n
    i, j = n, m
    while i or j:
        pi, pj, span = previous[(i, j)]
        if span is not None:
            mapping[pi] = span
        i, j = pi, pj
    if any(span is None for span in mapping):
        return None
    return [WordInterval(word=raw_words[k], start=float(aligned[span[0]].start),
                         end=float(aligned[span[1] - 1].end))
            for k, span in enumerate(mapping) if span is not None]


def video_duration(path: Path, ffprobe: str = "ffprobe") -> float:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    duration = float(result.stdout.strip())
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError(f"invalid video duration: {path}")
    return duration


def build_timelines(metadata: list[dict], video_dir: Path, bert_model: Path,
                    work_dir: Path, *, mfa: str | None = None,
                    ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe",
                    dictionary: str | None = None,
                    acoustic_model: str | None = None) -> list[dict]:
    """Force-align supplied transcripts; flag approximate fallbacks explicitly."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(bert_model), use_fast=True)
    corpus = work_dir / "mfa_corpus"
    aligned = work_dir / "mfa_alignment"
    corpus.mkdir(parents=True, exist_ok=True)
    aligned.mkdir(parents=True, exist_ok=True)
    durations = {}
    for item in metadata:
        sample_id = item["id"]
        video = video_dir / f"{sample_id}.mp4"
        if not video.is_file():
            raise FileNotFoundError(video)
        durations[sample_id] = video_duration(video, ffprobe)
        grid = aligned / f"{sample_id}.TextGrid"
        if not grid.exists():
            prepare_mfa_item(video, item["raw_text"], corpus, sample_id, ffmpeg)
    mfa_bin = mfa or shutil.which("mfa") or str(Path(sys.executable).with_name("mfa"))
    if any(not (aligned / f"{item['id']}.TextGrid").exists() for item in metadata):
        try:
            local_models = Path.home() / "Documents/MFA/pretrained_models"
            dictionary = dictionary or str(local_models / "dictionary/english_us_arpa.dict"
                                           if (local_models / "dictionary/english_us_arpa.dict").exists()
                                           else "english_us_arpa")
            acoustic_model = acoustic_model or str(local_models / "acoustic/english_us_arpa.zip"
                                                   if (local_models / "acoustic/english_us_arpa.zip").exists()
                                                   else "english_us_arpa")
            os.environ["MFA_ROOT_DIR"] = str(work_dir / "mfa_root")
            os.environ["PATH"] = str(Path(mfa_bin).parent) + os.pathsep + os.environ.get("PATH", "")
            (work_dir / "mfa_root").mkdir(parents=True, exist_ok=True)
            run_mfa(corpus, aligned, dictionary, acoustic_model, mfa_bin)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            print(f"MFA alignment failed: {exc}; affected samples will be flagged approximate",
                  file=sys.stderr, flush=True)

    timelines = []
    for item in metadata:
        sample_id, raw = item["id"], item["raw_text"]
        grid = aligned / f"{sample_id}.TextGrid"
        if grid.exists():
            try:
                intervals = parse_textgrid(grid)
                method = "mfa"
            except ValueError:
                intervals = uniform_intervals(raw, durations[sample_id])
                method = "uniform_fallback"
        else:
            intervals = uniform_intervals(raw, durations[sample_id])
            method = "uniform_fallback"

        word_matches = list(WORDS.finditer(raw))
        normalized_raw = [m.group().lower().replace("’", "'") for m in word_matches]
        normalized_aligned = [w.word.lower().replace("’", "'") for w in intervals]
        if len(word_matches) != len(intervals) or normalized_raw != normalized_aligned:
            repaired = reconcile_words([match.group() for match in word_matches], intervals)
            if repaired is None:
                intervals = uniform_intervals(raw, durations[sample_id])
                method = "uniform_word_mismatch"
            else:
                intervals = repaired
                if method == "mfa":
                    method = "mfa_reconciled"
        encoding = tokenizer(raw, return_offsets_mapping=True, truncation=True,
                             max_length=50)
        expected_ids = item["input_ids"][:item["valid_tokens"]]
        if list(map(int, encoding["input_ids"])) != list(map(int, expected_ids)):
            raise ValueError(f"BERT tokenizer does not reproduce text_bert for {sample_id}")
        token_times: list[list[float] | None] = []
        for start, end in encoding["offset_mapping"]:
            if end <= start:
                token_times.append(None)  # CLS or SEP
                continue
            overlapping = [k for k, match in enumerate(word_matches)
                           if match.start() < end and start < match.end()]
            if not overlapping:
                token_times.append(None)
            else:
                token_times.append([float(intervals[overlapping[0]].start),
                                    float(intervals[overlapping[-1]].end)])
        token_times.extend([None] * (50 - len(token_times)))
        timelines.append({
            "id": sample_id, "raw_text": raw,
            "duration": durations[sample_id], "alignment_method": method,
            "token_offsets": [[int(a), int(b)] for a, b in encoding["offset_mapping"]]
                             + [[0, 0]] * (50 - len(encoding["offset_mapping"])),
            "token_times": token_times,
            "words": [{"word": match.group(), "char_start": match.start(),
                       "char_end": match.end(), "start": float(interval.start),
                       "end": float(interval.end)}
                      for match, interval in zip(word_matches, intervals)],
        })
    return timelines


def evidence_location(timeline: dict, start: int, end: int) -> dict:
    """Return exact transcript substring and video span for [start,end) tokens."""
    offsets = timeline["token_offsets"][start:end]
    chars = [(a, b) for a, b in offsets if b > a]
    times = [pair for pair in timeline["token_times"][start:end] if pair is not None]
    if not chars or not times:
        return {"text": "", "char_start": None, "char_end": None,
                "start_sec": None, "end_sec": None, "mid_sec": None}
    char_start, char_end = min(a for a, _ in chars), max(b for _, b in chars)
    # BERT subwords can start or end inside a word; present whole source words.
    covering_words = [match for match in WORDS.finditer(timeline["raw_text"])
                      if match.start() < char_end and char_start < match.end()]
    if covering_words:
        char_start = covering_words[0].start()
        char_end = covering_words[-1].end()
    start_sec, end_sec = min(a for a, _ in times), max(b for _, b in times)
    return {"text": timeline["raw_text"][char_start:char_end],
            "char_start": char_start, "char_end": char_end,
            "start_sec": start_sec, "end_sec": end_sec,
            "mid_sec": (start_sec + end_sec) / 2}


def save_timelines(timelines: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(timelines, ensure_ascii=False, indent=2), encoding="utf-8")
