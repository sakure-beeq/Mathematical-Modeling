"""Transcript preparation, Montreal Forced Aligner invocation, and TextGrid I/O."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Iterable

import numpy as np

from .core import WordInterval


_INTERVAL = re.compile(
    r"intervals\s*\[\d+\]\s*:\s*"
    r"xmin\s*=\s*([0-9.eE+-]+)\s*"
    r"xmax\s*=\s*([0-9.eE+-]+)\s*"
    r'text\s*=\s*"((?:[^"\\]|\\.)*)"',
    re.DOTALL,
)
_TIER = re.compile(
    r"item\s*\[\d+\]\s*:\s*"
    r'class\s*=\s*"IntervalTier"\s*'
    r'name\s*=\s*"([^"]+)"(.*?)(?=\n\s*item\s*\[\d+\]\s*:|\Z)',
    re.DOTALL,
)
_TRANSCRIPT_WORD = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")


def parse_textgrid(path: Path, tier_names: Iterable[str] = ("words", "word")) -> list[WordInterval]:
    """Read non-empty words from a long-text Praat TextGrid."""

    body = path.read_text(encoding="utf-8-sig")
    wanted = {name.lower() for name in tier_names}
    for match in _TIER.finditer(body):
        if match.group(1).strip().lower() not in wanted:
            continue
        result: list[WordInterval] = []
        for interval in _INTERVAL.finditer(match.group(2)):
            start, end = float(interval.group(1)), float(interval.group(2))
            word = interval.group(3).replace(r'\"', '"').strip()
            if word and word.lower() not in {"<eps>", "sil", "sp", "spn"} and end > start:
                result.append(WordInterval(word=word, start=start, end=end))
        if result:
            return result
    raise ValueError(f"no non-empty word tier found in {path}")


def uniform_word_intervals(transcript: str, duration: float) -> list[WordInterval]:
    """Create an explicit zero-confidence fallback timeline for unalignable audio.

    This must only be used when forced alignment is unavailable (for example, a
    video whose audio stream is digital silence). It is deliberately kept
    separate from :func:`parse_textgrid` so callers can record the method.
    """

    tokens = _TRANSCRIPT_WORD.findall(transcript)
    if not tokens:
        raise ValueError("cannot create fallback intervals from an empty transcript")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError(f"invalid media duration for fallback alignment: {duration}")
    boundaries = np.linspace(0.0, float(duration), len(tokens) + 1)
    return [
        WordInterval(word=token, start=float(boundaries[i]), end=float(boundaries[i + 1]))
        for i, token in enumerate(tokens)
    ]


def safe_sample_name(video_id: str, clip_id: str) -> str:
    """Filesystem-safe reversible-enough sample name used for all intermediate files."""

    cleaned_video = re.sub(r"[^A-Za-z0-9_-]", "_", str(video_id))
    cleaned_clip = re.sub(r"[^A-Za-z0-9_-]", "_", str(clip_id))
    return f"{cleaned_video}__{cleaned_clip}"


def extract_wav(video: Path, wav: Path, ffmpeg: str = "ffmpeg", sample_rate: int = 16000) -> None:
    """Extract mono PCM without denoising or loudness normalization."""

    wav.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(wav),
    ]
    subprocess.run(command, check=True)


def prepare_mfa_item(video: Path, transcript: str, corpus_dir: Path, name: str, ffmpeg: str) -> None:
    extract_wav(video, corpus_dir / f"{name}.wav", ffmpeg=ffmpeg)
    (corpus_dir / f"{name}.lab").write_text(transcript.strip() + "\n", encoding="utf-8")


def is_digital_silence(wav: Path) -> bool:
    """Only an all-zero 16-bit PCM track qualifies for uniform timeline fallback."""

    with wave.open(str(wav), "rb") as handle:
        if handle.getsampwidth() != 2 or handle.getnchannels() != 1:
            raise ValueError(f"expected mono 16-bit PCM WAV: {wav}")
        while block := handle.readframes(65536):
            if any(block):
                return False
        return True


def run_mfa(
    corpus_dir: Path,
    output_dir: Path,
    dictionary: str,
    acoustic_model: str,
    mfa: str = "mfa",
    clean: bool = False,
) -> None:
    """Align the whole prepared corpus in one MFA call."""

    if shutil.which(mfa) is None:
        raise FileNotFoundError(f"MFA executable not found: {mfa!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [mfa, "align", str(corpus_dir), dictionary, acoustic_model, str(output_dir)]
    if clean:
        command.append("--clean")
    environment = os.environ.copy()
    environment.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "problem1-numba-cache"))
    subprocess.run(command, check=True, env=environment)
