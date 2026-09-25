"""Draw a source-backed multimodal alignment figure for one audited sample.

The upper panels use the source MP4, WAV, and OpenFace CSV. The lower panels
use the exact arrays stored in problem1_features.pkl. Every heatmap channel is
standardized only for display; this never changes the saved feature file.
"""

from __future__ import annotations

import json
import hashlib
import os
import pickle
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from problem1.audio_features import extract_audio74
from problem1.core import WordInterval, quality_weighted_pool
from problem1.visual_features import load_openface35


PROJECT = Path(__file__).resolve().parents[1]
DELIVERY = PROJECT / "delivery"
FEATURES = PROJECT / "outputs" / "problem1_features.pkl"


def display_values(frames: np.ndarray, quality: np.ndarray | None = None) -> np.ma.MaskedArray:
    """Return channel-wise z-scores, clipped solely to give comparable colors."""

    values = np.asarray(frames, dtype=np.float64)
    valid = np.ones(len(values), dtype=bool) if quality is None else np.asarray(quality) > 0
    valid &= np.isfinite(values).all(axis=1)
    if not valid.any():
        return np.ma.masked_all(values.T.shape)
    mean = values[valid].mean(axis=0)
    scale = values[valid].std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = np.clip((values - mean) / scale, -2.5, 2.5).T
    return np.ma.array(normalized, mask=np.broadcast_to((~valid)[None, :], normalized.shape))


def verify_source(asset: dict[str, object]) -> Path:
    path = Path(str(asset["path"]))
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != asset["sha256"]:
        raise ValueError(f"source file changed since feature audit: {path}")
    return path


def waveform_envelope(wav: Path, bins: int = 1600) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    with wave.open(str(wav), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError("expected mono 16-bit PCM WAV")
        sample_rate = handle.getframerate()
        signal = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    step = max(1, int(np.ceil(len(signal) / bins)))
    times, minimum, maximum = [], [], []
    for start in range(0, len(signal), step):
        block = signal[start:start + step]
        times.append((start + (len(block) - 1) / 2) / sample_rate)
        minimum.append(float(block.min()))
        maximum.append(float(block.max()))
    return np.asarray(times), np.asarray(minimum), np.asarray(maximum), len(signal) / sample_rate


def draw_video_frame(ax, path: Path, timestamp: float, half_width: float) -> None:
    import matplotlib.pyplot as plt

    with tempfile.TemporaryDirectory(prefix="problem1_figure_frame_") as temporary:
        image_path = Path(temporary) / "frame.png"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(path),
             "-ss", f"{timestamp:.6f}", "-frames:v", "1", str(image_path)],
            check=True,
        )
        frame = plt.imread(image_path)
        ax.imshow(
            frame,
            extent=(timestamp - half_width, timestamp + half_width, 0.18, 0.90),
            interpolation="nearest", aspect="auto", zorder=2,
        )
    ax.text(timestamp, 0.10, f"{timestamp:.2f} 秒", ha="center", va="center", fontsize=8, color="#25364A")


def make_figure(sample_index: int = 0) -> dict[str, object]:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "problem1-matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "problem1-xdg-cache"))
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Polygon, Rectangle
    from matplotlib.cm import ScalarMappable

    with FEATURES.open("rb") as handle:
        payload = pickle.load(handle)
    sample = payload["samples"][sample_index]
    length = int(sample["valid_length"])
    if length > 25 or sample["was_aggregated"]:
        raise ValueError("this figure layout requires at most 25 unaggregated word positions")
    source = sample["source_files"]
    video = verify_source(source["video"])
    wav = verify_source(source["wav"])
    visual_csv = verify_source(source["openface_csv"])
    verify_source(source["textgrid"])
    words = [WordInterval(row["word"], row["start_s"], row["end_s"]) for row in sample["source_word_timeline"]]
    assert len(words) == length
    audio_times, audio_frames, audio_quality, _ = extract_audio74(wav)
    visual_times, visual_frames, visual_quality, _ = load_openface35(visual_csv)
    audio_pooled, _, _ = quality_weighted_pool(audio_times, audio_frames, words, audio_quality)
    visual_pooled, _, _ = quality_weighted_pool(visual_times, visual_frames, words, visual_quality)
    audio_error = float(np.max(np.abs(audio_pooled - sample["audio"][:length])))
    visual_error = float(np.max(np.abs(visual_pooled - sample["vision"][:length])))
    if audio_error > 5e-4 or visual_error > 5e-4:
        raise ValueError("source frames do not reproduce saved aligned features")
    if not np.array_equal(sample["text"][:length], payload["stacked"]["text"][sample_index, :length]):
        raise ValueError("saved text rows differ from stacked output")

    wave_times, wave_low, wave_high, wav_duration = waveform_envelope(wav)
    duration = float(sample["video_duration"])
    if abs(wav_duration - duration) > 0.25:
        raise ValueError("WAV and video durations disagree")
    highlight = 5  # "solutions" in this sample; genuine interval from the TextGrid.
    accent = "#E69F00"
    blue = "#0072B2"
    green = "#009E73"
    cmap = plt.colormaps["RdBu_r"].copy()
    cmap.set_bad("#ECEFF2")
    plt.rcParams.update({
        "font.family": "PingFang SC", "font.sans-serif": ["PingFang SC", "Arial Unicode MS"],
        "axes.unicode_minus": False, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 300,
    })

    fig = plt.figure(figsize=(16, 13.7), facecolor="white")
    grid = fig.add_gridspec(
        9, 1, height_ratios=[1.20, 0.95, 1.20, 1.05, 0.80, 0.47, 1.28, 0.92, 0.85],
        left=0.125, right=0.91, bottom=0.095, top=0.925, hspace=0.24,
    )
    axes = [fig.add_subplot(grid[i]) for i in range(9)]
    video_ax, wave_ax, raw_audio_ax, raw_visual_ax, map_ax, word_ax, text_ax, audio_ax, visual_ax = axes
    escaped_id = sample["id"].replace("$", r"\$")
    fig.suptitle("典型样本：三模态特征的词级时序对齐", y=0.982, fontsize=17, fontweight="bold", color="#1E293B")
    fig.text(
        0.125, 0.95,
        f"样本 {escaped_id}  ｜  原视频 {duration:.3f} 秒  ｜  {length} 个有效词位  ｜  高亮词："
        f"{words[highlight].word} [{words[highlight].start:.2f}, {words[highlight].end:.2f}] 秒",
        fontsize=10, color="#475569",
    )

    raw_axes = [video_ax, wave_ax, raw_audio_ax, raw_visual_ax]
    for ax in raw_axes:
        ax.set_xlim(0, duration)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#CBD5E1")
        ax.tick_params(axis="x", labelbottom=False, bottom=False)
        ax.axvspan(words[highlight].start, words[highlight].end, color=accent, alpha=0.13, zorder=1)
    video_ax.set_ylim(0, 1)
    targets = np.linspace(words[0].start + 0.12, words[-1].end - 0.12, 5)
    frame_times = [float(visual_times[np.argmin(np.abs(visual_times - target))]) for target in targets]
    for timestamp in frame_times:
        draw_video_frame(video_ax, video, timestamp, half_width=0.28)
    video_ax.set_ylabel("原视频\nMP4抽帧", rotation=0, ha="right", va="center", labelpad=15, color=blue, fontweight="bold")

    wave_ax.fill_between(wave_times, wave_low, wave_high, color=blue, linewidth=0)
    wave_ax.axhline(0, color="#475569", lw=0.5)
    wave_peak = max(float(np.max(np.abs(wave_low))), float(np.max(np.abs(wave_high))), 1e-3)
    wave_ax.set_ylim(-1.12 * wave_peak, 1.12 * wave_peak)
    wave_ax.set_ylabel("原始语音\nPCM波形", rotation=0, ha="right", va="center", labelpad=15, color=blue, fontweight="bold")

    raw_audio_ax.imshow(
        display_values(audio_frames, audio_quality), origin="lower", interpolation="nearest", aspect="auto",
        extent=(audio_times[0], audio_times[-1], -0.5, 73.5), cmap=cmap, vmin=-2.5, vmax=2.5,
        rasterized=True,
    )
    raw_audio_ax.set_ylabel(f"语音帧特征\n74维·{len(audio_times)}帧", rotation=0, ha="right", va="center", labelpad=15, color=blue, fontweight="bold")
    raw_visual_ax.imshow(
        display_values(visual_frames, visual_quality), origin="lower", interpolation="nearest", aspect="auto",
        extent=(visual_times[0], visual_times[-1], -0.5, 34.5), cmap=cmap, vmin=-2.5, vmax=2.5,
        rasterized=True,
    )
    raw_visual_ax.set_ylabel(f"视觉帧特征\n35维·{len(visual_times)}帧", rotation=0, ha="right", va="center", labelpad=15, color=green, fontweight="bold")
    for ax in (wave_ax, raw_audio_ax, raw_visual_ax):
        for word in words:
            ax.axvline(word.start, color="#94A3B8", lw=0.35, alpha=0.55)
    raw_visual_ax.tick_params(axis="x", bottom=True, labelbottom=True, colors="#475569")
    raw_visual_ax.set_xlabel("原视频时间（秒）", labelpad=2)

    map_ax.set_xlim(0, 1)
    map_ax.set_ylim(-0.05, 1.25)
    map_ax.axis("off")
    for i, word in enumerate(words):
        left = word.start / duration
        right = word.end / duration
        bottom_left = i / length
        bottom_right = (i + 1) / length
        color = accent if i == highlight else ("#56B4E9" if i % 2 == 0 else "#CC79A7")
        map_ax.add_patch(Polygon(
            [[left, 1], [right, 1], [bottom_right, 0], [bottom_left, 0]],
            closed=True, facecolor=color, edgecolor="white", linewidth=0.3,
            alpha=0.35 if i == highlight else 0.24,
        ))
    map_ax.text(
        -0.02, 0.5, "时间→词位\n对应关系", transform=map_ax.transAxes,
        ha="right", va="center", color="#475569", fontweight="bold", clip_on=False,
    )

    word_ax.set_xlim(-0.5, length - 0.5)
    word_ax.set_ylim(0, 1)
    word_ax.axis("off")
    for i, word in enumerate(words):
        word_ax.add_patch(Rectangle(
            (i - 0.5, 0.03), 1, 0.94,
            facecolor="#FFF3D1" if i == highlight else ("#F2F7FA" if i % 2 == 0 else "#FAF7F8"),
            edgecolor="#CBD5E1", lw=0.55,
        ))
        word_ax.text(i, 0.68, word.word, ha="center", va="center", fontsize=8.1, rotation=35, color="#1E293B")
        word_ax.text(i, 0.20, str(i), ha="center", va="center", fontsize=7, color="#64748B")
    word_ax.text(
        -0.02, 0.5, "对齐词序列\n位置从0开始", transform=word_ax.transAxes,
        ha="right", va="center", color="#475569", fontweight="bold", clip_on=False,
    )

    pooled = (
        (text_ax, "BERT文本\n768维", sample["text"][:length], None, "#7C3AED"),
        (audio_ax, "词级语音\n74维", sample["audio"][:length], sample["modality_availability_mask"][:length, 1], blue),
        (visual_ax, "词级视觉\n35维", sample["vision"][:length], sample["modality_availability_mask"][:length, 2], green),
    )
    for ax, label, values, availability, label_color in pooled:
        ax.imshow(
            display_values(values, availability), origin="lower", interpolation="nearest", aspect="auto",
            extent=(-0.5, length - 0.5, -0.5, values.shape[1] - 0.5),
            cmap=cmap, vmin=-2.5, vmax=2.5, rasterized=True,
        )
        ax.set_xlim(-0.5, length - 0.5)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#CBD5E1")
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=15, color=label_color, fontweight="bold")
        ax.add_patch(Rectangle((highlight - 0.5, -0.5), 1, values.shape[1], fill=False, edgecolor=accent, lw=1.8))
    visual_ax.set_xticks(range(length))
    visual_ax.tick_params(axis="x", bottom=True, labelbottom=True, colors="#475569", labelsize=8)
    visual_ax.set_xlabel("对齐词位置（从0开始）")

    colorbar_ax = fig.add_axes([0.925, 0.115, 0.012, 0.20])
    colorbar = fig.colorbar(ScalarMappable(norm=Normalize(-2.5, 2.5), cmap=cmap), cax=colorbar_ax)
    colorbar.set_label("通道标准分（仅用于显示）", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    fig.text(
        0.125, 0.042,
        "上半部分来自原视频、WAV和OpenFace CSV；下半部分为特征文件中的实际向量。金色区域标出高亮词的原始时段与输出词位。\n"
        "颜色仅用于显示：各通道按有效帧或词位计算标准分，并截断到±2.5；灰色表示无效。重新从原始帧汇聚的语音、视觉向量已与保存结果核对。",
        fontsize=8.1, color="#475569", wrap=True,
    )
    DELIVERY.mkdir(parents=True, exist_ok=True)
    png = DELIVERY / "typical_alignment.png"
    pdf = DELIVERY / "typical_alignment.pdf"
    svg = DELIVERY / "typical_alignment.svg"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    fig.savefig(svg, facecolor="white")
    plt.close(fig)
    provenance = {
        "sample_index": sample_index,
        "sample_id": sample["id"],
        "video_duration_s": duration,
        "wav_duration_s": wav_duration,
        "valid_word_positions": length,
        "raw_audio_feature_shape": list(audio_frames.shape),
        "raw_visual_feature_shape": list(visual_frames.shape),
        "selected_video_frame_times_s": frame_times,
        "highlight_word": words[highlight].word,
        "highlight_interval_s": [words[highlight].start, words[highlight].end],
        "audio_recomputed_max_abs_diff": audio_error,
        "visual_recomputed_max_abs_diff": visual_error,
        "display_transform": "per-feature channel z-score across displayed valid frames or positions, clipped to [-2.5,2.5]; saved feature arrays unchanged",
        "source_files": source,
        "feature_sha256": sample["feature_sha256"],
    }
    (DELIVERY / "typical_alignment.provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {png}, {pdf}, and {svg}; raw frames {audio_frames.shape} / {visual_frames.shape}")
    return provenance


if __name__ == "__main__":
    make_figure()
