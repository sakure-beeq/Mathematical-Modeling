"""Typical-sample visualization on the shared word time axis."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def plot_sample(sample: dict[str, Any], output: Path, ffmpeg: str = "ffmpeg") -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "problem1-matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "problem1-xdg-cache"))
    try:
        import matplotlib.pyplot as plt
        from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    except ImportError as exc:
        raise RuntimeError("plotting requires matplotlib; install the extract extra") from exc
    length = int(sample["valid_length"])
    intervals = np.asarray(sample["intervals"][:length])
    centres = intervals.mean(axis=1)
    words = sample["words"][:length]
    audio = np.asarray(sample["audio"][:length])
    modality_norms = np.column_stack(
        [
            np.linalg.norm(sample["text"][:length], axis=1),
            np.linalg.norm(audio, axis=1),
            np.linalg.norm(sample["vision"][:length], axis=1),
        ]
    ).T

    figure, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True, constrained_layout=True)
    title = f"{sample['id']} — {sample['raw_text']}".replace("$", r"\$")
    axes[0].set_title(title)
    for word, (start, end) in zip(words, intervals):
        axes[0].axvspan(start, end, alpha=0.12, color="C0")
        axes[0].text((start + end) / 2, 0.5, word, rotation=50, ha="right", va="center", fontsize=8)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("word")

    # Column 2 is log RMS in the documented 74-D descriptor.
    axes[1].plot(centres, audio[:, 2], color="C1", marker=".")
    axes[1].set_ylabel("log RMS")
    axes[1].grid(alpha=0.2)

    chosen = np.unique(np.linspace(0, max(0, length - 1), min(6, length)).astype(int))
    with tempfile.TemporaryDirectory(prefix="problem1_frames_") as temporary:
        for index in chosen:
            image_path = Path(temporary) / f"frame_{index}.jpg"
            subprocess.run(
                [ffmpeg, "-nostdin", "-loglevel", "error", "-y", "-ss", str(centres[index]), "-i", sample["video_path"], "-frames:v", "1", str(image_path)],
                check=True,
            )
            image = plt.imread(image_path)
            axes[2].add_artist(AnnotationBbox(OffsetImage(image, zoom=0.13), (centres[index], 0.5), frameon=False))
        axes[2].set_ylim(0, 1)
        axes[2].set_ylabel("keyframes")

    image = axes[3].imshow(
        modality_norms,
        aspect="auto",
        interpolation="nearest",
        extent=(intervals[0, 0], intervals[-1, 1], 2.5, -0.5),
        cmap="viridis",
    )
    axes[3].set_yticks([0, 1, 2], ["text", "audio", "vision"])
    axes[3].set_ylabel("feature norm")
    axes[3].set_xlabel("time (seconds)")
    figure.colorbar(image, ax=axes[3], pad=0.01)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
