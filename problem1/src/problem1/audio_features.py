"""Frame-level 74-dimensional acoustic features without aggressive enhancement."""

from __future__ import annotations

import os
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np


AUDIO_FEATURE_NAMES = (
    ["f0_hz", "voiced_probability", "log_rms", "rms", "zero_crossing_rate", "onset_strength"]
    + [f"mfcc_{i:02d}" for i in range(20)]
    + [f"mfcc_delta_{i:02d}" for i in range(20)]
    + [f"mfcc_delta2_{i:02d}" for i in range(20)]
    + [
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff",
        "spectral_flatness",
        "spectral_contrast_mean",
        "spectral_contrast_std",
        "harmonic_to_noise_ratio",
        "local_f0_jitter",
    ]
)
assert len(AUDIO_FEATURE_NAMES) == 74


def _match_frames(values: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    if values.size >= count:
        return values[:count]
    return np.pad(values, (0, count - values.size), mode="edge")


def extract_audio74(
    wav_path: Path,
    sample_rate: int = 16000,
    hop_length: int = 160,
    frame_length: int = 400,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Return centre times, 74-D frames, frame quality, and reproducibility metadata.

    This descriptor follows the requested MOSEI-like categories but is not claimed
    to be bit-compatible with proprietary/legacy COVAREP builds.
    """

    # Numba otherwise tries to cache librosa's JIT functions beside the installed
    # package, which fails in read-only Conda environments.
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "problem1-numba-cache"))
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("audio extraction requires librosa; install the extract extra") from exc
    y, sr = librosa.load(wav_path, sr=sample_rate, mono=True)
    if y.size < frame_length:
        y = np.pad(y, (0, frame_length - y.size))
    stft = np.abs(librosa.stft(y, n_fft=512, hop_length=hop_length, win_length=frame_length))
    count = stft.shape[1]
    rms = _match_frames(
        librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0], count
    )
    zcr = _match_frames(librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0], count)
    onset = _match_frames(librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length), count)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=512, hop_length=hop_length, win_length=frame_length)
    mfcc = mfcc[:, :count]
    delta_width = min(9, count if count % 2 else count - 1)
    if delta_width < 3:
        delta = np.zeros_like(mfcc)
        delta2 = np.zeros_like(mfcc)
    else:
        delta = librosa.feature.delta(mfcc, width=delta_width, mode="nearest")
        delta2 = librosa.feature.delta(mfcc, width=delta_width, order=2, mode="nearest")
    centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(S=stft)[0]
    # Five octave bands are safe below the 8 kHz Nyquist limit at 16 kHz.
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr, n_bands=5)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        frame_length=1024,
        hop_length=hop_length,
    )
    f0 = np.nan_to_num(_match_frames(f0, count), nan=0.0)
    voiced_prob = np.nan_to_num(_match_frames(voiced_prob, count), nan=0.0)
    voiced_flag = _match_frames(voiced_flag.astype(np.float32), count) > 0
    harmonic = librosa.effects.harmonic(y)
    percussive = y - harmonic
    h_rms = _match_frames(librosa.feature.rms(y=harmonic, frame_length=frame_length, hop_length=hop_length)[0], count)
    n_rms = _match_frames(librosa.feature.rms(y=percussive, frame_length=frame_length, hop_length=hop_length)[0], count)
    hnr = 20.0 * np.log10((h_rms + 1e-8) / (n_rms + 1e-8))
    previous = np.r_[f0[0], f0[:-1]]
    jitter = np.where(voiced_flag & (f0 > 0) & (previous > 0), np.abs(f0 - previous) / (f0 + 1e-8), 0.0)

    features = np.column_stack(
        [
            f0,
            voiced_prob,
            np.log(rms + 1e-8),
            rms,
            zcr,
            onset,
            mfcc.T,
            delta.T,
            delta2.T,
            centroid,
            bandwidth,
            rolloff,
            flatness,
            contrast.mean(axis=0),
            contrast.std(axis=0),
            hnr,
            jitter,
        ]
    ).astype(np.float32)
    finite = np.isfinite(features).all(axis=1)
    # Digital silence cannot carry acoustic emotion information and must not be
    # treated as a high-quality all-zero observation.
    audible = rms > 1e-6
    quality = (finite & audible).astype(np.float32)
    features[~np.isfinite(features)] = 0.0
    times = librosa.frames_to_time(np.arange(count), sr=sr, hop_length=hop_length).astype(np.float64)
    metadata = {
        "extractor": "problem1.audio_features.extract_audio74",
        "librosa_version": version("librosa"),
        "sample_rate": sr,
        "hop_length": hop_length,
        "frame_length": frame_length,
        "feature_names": AUDIO_FEATURE_NAMES,
        "silent_frame_rate": float(np.mean(~audible)),
        "silence_rms_threshold": 1e-6,
    }
    return times, features, quality, metadata
