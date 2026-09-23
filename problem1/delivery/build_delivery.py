"""Build the Problem 1 report and submission archive from audited features."""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import shutil
import wave
import zipfile
from pathlib import Path

import numpy as np

from problem1.core import _frame_bounds


PROJECT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT / "outputs"
DELIVERY = PROJECT / "delivery"
PICKLE = OUTPUTS / "problem1_features.pkl"
MANIFEST = OUTPUTS / "problem1_features.manifest.csv"
AUDIT = OUTPUTS / "problem1_features.audit.json"


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_range(times: np.ndarray, start: float, end: float) -> str:
    left, right = _frame_bounds(times)
    matching = times[(right > start) & (left < end)]
    if matching.size == 0:
        return "无"
    return f"{matching[0]:.3f}–{matching[-1]:.3f} ({matching.size}帧)"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build() -> None:
    DELIVERY.mkdir(parents=True, exist_ok=True)
    with PICKLE.open("rb") as handle:
        payload = pickle.load(handle)
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    samples = payload["samples"]
    assert len(samples) == len(manifest) == 100
    assert audit["one_to_one_mapping"] and audit["feature_count"] == 100
    assert all(sample["id"] == row["sample_id"] for sample, row in zip(samples, manifest))

    modality_specs = (
        ("文本", "text", 768, "transcript_lab_path"),
        ("语音", "audio", 74, "wav_path"),
        ("视觉", "vision", 35, "openface_csv_path"),
    )
    full_rows: list[dict[str, object]] = []
    summary_rows: list[str] = []
    for i, (sample, source) in enumerate(zip(samples, manifest)):
        length = int(sample["valid_length"])
        start, end = float(sample["intervals"][0, 0]), float(sample["intervals"][length - 1, 1])
        available = np.asarray(sample["modality_availability_mask"][:length]).sum(axis=0)
        grain = "连续词组" if sample["was_aggregated"] else "词"
        method = "MFA" if sample["forced_alignment_available"] else "静音回退"
        summary_rows.append(
            f"| {i + 1} | {sample['id']} | {float(sample['video_duration']):.3f} | "
            f"{start:.3f}–{end:.3f} | {length} | "
            f"{int(available[0])}/{int(available[1])}/{int(available[2])} | "
            f"768/74/35 | {grain} | {method} |"
        )
        for modality_index, (modality, key, dimension, source_column) in enumerate(modality_specs):
            assert sample[key].shape == (50, dimension)
            full_rows.append({
                "sequence_number_1based": i + 1,
                "sample_index_0based": i,
                "label_excel_row": sample["label_row"],
                "sample_id": sample["id"],
                "modality": modality,
                "raw_video_duration_s": f"{float(sample['video_duration']):.6f}",
                "aligned_start_s": f"{start:.6f}",
                "aligned_end_s": f"{end:.6f}",
                "feature_dimension": dimension,
                "fixed_sequence_length": 50,
                "valid_sequence_length": length,
                "available_positions": int(available[modality_index]),
                "alignment_granularity": grain,
                "alignment_method": method,
                "raw_video_path": source["video_path"],
                "modality_source_path": source[source_column],
                "feature_sha256": source["feature_sha256"],
            })
    write_csv(DELIVERY / "full_results.csv", full_rows)

    exemplar = samples[0]
    exemplar_source = manifest[0]
    length = int(exemplar["valid_length"])
    with wave.open(exemplar_source["wav_path"], "rb") as handle:
        audio_frame_count = handle.getnframes() // 160 + 1
    audio_times = np.arange(audio_frame_count, dtype=np.float64) * 0.01
    with Path(exemplar_source["openface_csv_path"]).open(newline="", encoding="utf-8-sig") as handle:
        video_times = np.asarray(
            [float(row["timestamp"]) for row in csv.DictReader(handle, skipinitialspace=True)],
            dtype=np.float64,
        )
    example_rows: list[dict[str, object]] = []
    for position in range(length):
        start, end = map(float, exemplar["intervals"][position])
        example_rows.append({
            "position_0based": position,
            "word": exemplar["words"][position],
            "start_s": f"{start:.3f}",
            "end_s": f"{end:.3f}",
            "source_word_span_half_open": str(exemplar["source_word_span"][position].tolist()),
            "audio_candidate_frame_centres": frame_range(audio_times, start, end),
            "video_candidate_frame_centres": frame_range(video_times, start, end),
            "text_norm": f"{np.linalg.norm(exemplar['text'][position]):.3f}",
            "audio_log_rms": f"{float(exemplar['audio'][position, 2]):.3f}",
            "vision_AU12_r": f"{float(exemplar['vision'][position, 8]):.3f}",
            "modality_available_TAV": "/".join(
                "1" if flag else "0" for flag in exemplar["modality_availability_mask"][position]
            ),
        })
    write_csv(DELIVERY / "typical_alignment.csv", example_rows)
    shutil.copyfile(OUTPUTS / "sample_000_timeline.png", DELIVERY / "typical_alignment.png")

    selected = (0, 1, 4, 5, 10, 14)
    example_table = "\n".join(
        f"| {example_rows[j]['position_0based']} | {example_rows[j]['word']} | "
        f"{example_rows[j]['start_s']}–{example_rows[j]['end_s']} | "
        f"{example_rows[j]['audio_candidate_frame_centres']} | "
        f"{example_rows[j]['video_candidate_frame_centres']} | "
        f"{example_rows[j]['text_norm']} / {example_rows[j]['audio_log_rms']} / "
        f"{example_rows[j]['vision_AU12_r']} |"
        for j in selected
    )
    durations = np.asarray([float(sample["video_duration"]) for sample in samples])
    valid_lengths = np.asarray([int(sample["valid_length"]) for sample in samples])
    versions = audit["recorded_tool_versions"]
    libraries = audit["recorded_library_versions"]
    report = f"""# 问题1：特征提取、时序对齐与全量结果

## 1. 数据与处理流程

附件1的100条视频以Excel标签表中的 `video_id`、`clip_id` 确定样本ID。每条记录按以下流程处理：

`标签行 + MP4 → 16 kHz单声道WAV及转写LAB → MFA词级TextGrid → BERT文本特征 / librosa语音帧特征 / OpenFace视频帧特征 → 词区间质量加权池化 → 长序列连续聚合或右侧零填充 → 50位三模态特征及掩码 → 全量审计`。

原视频时长为 {durations.min():.3f}–{durations.max():.3f} 秒（中位数 {np.median(durations):.3f} 秒）；对齐后有效序列长度为 {valid_lengths.min()}–{valid_lengths.max()} 位（中位数 {np.median(valid_lengths):.0f} 位）。全部100条均保留，1条超过50词并按连续词组聚合。99条使用MFA对齐；1条数字静音、没有TextGrid的样本使用全时长均匀词区间，`alignment_confidence=0`，其他缺失TextGrid的样本会报错。4条样本未检测到有效人脸，视觉可用性由掩码标记。

### 1.1 特征定义与工具

| 模态 | 原始来源 | 工具与提取方法 | 每个对齐位置的维度 |
| --- | --- | --- | ---: |
| 文本 | Excel转写及MFA词序列 | 本地冻结的 `bert-base-uncased`；快速分词器将WordPiece映射回词，对同词子词的末层向量取算术均值 | 768 |
| 语音 | MP4音轨提取的16 kHz单声道PCM WAV | FFmpeg提取音频；`librosa` 以10 ms帧移生成74维描述符，多数短时谱特征使用25 ms窗，pYIN基频使用1024点分析帧：F0、浊音概率、log-RMS、RMS、过零率、起音强度共6维；20维MFCC及一、二阶差分共60维；频谱与谐噪/局部jitter共8维 | 74 |
| 视觉 | 原MP4视频帧 | OpenFace `FeatureExtraction`：17维AU强度、6维头姿、8维眼睛注视、4维68点面部几何统计；人脸检测 `success` 无效或置信度低于0.50的帧无效，短于等于0.25秒且两端有效的缺口做线性插值 | 35 |

74维语音描述符覆盖题目要求的声学类别，**不宣称与特定历史COVAREP二进制逐位相同**。视觉缺失没有被伪造为有效人脸特征。

### 1.2 时序对齐规则

MFA的非空词区间 `[s_k,e_k]` 是统一时间轴，单位为相对原视频起点的秒。语音帧中心由10 ms帧移产生，视觉帧中心取OpenFace CSV的 `timestamp`；相邻中心的中点界定帧覆盖区间。对词 `k` 与模态帧 `j`，权重为 `w(k,j)=|[s_k,e_k]∩[l_j,r_j]| × q_j`，其中 `q_j` 为帧质量。词级语音或视觉向量是所有有效帧按该权重的平均；没有有效帧时向量为零，并将模态可用掩码置为假。文本向量直接映射到相同词位。音频静音帧质量为零；视觉帧质量采用OpenFace置信度，并只对有界短缺口插值。

原词数不超过50时按词顺序排布，后续位置右侧零填充；超过50词时用连续分组覆盖全部词，并按词时长、质量及可用性再次加权汇聚。`source_word_span` 保存每个输出位置在原词序列中的半开索引范围，填充位置为 `[-1,-1]`；`valid_length`、`valid_mask`、`padding_mask` 和三模态质量/可用掩码同时保存。对齐粒度因此为词，唯一长样本为连续词组。

## 2. 特征文件规范与全量结果

原始特征文件为 `problem1_features.pkl`，使用Python pickle存储字典。`metadata` 包含格式版本、工具版本、参数、特征名和审计信息；`samples` 是按Excel标签顺序排列的100条逐样本记录；`stacked` 是直接建模的批量数组。核心数组形状为 `text=(100,50,768)`、`audio=(100,50,74)`、`vision=(100,50,35)`、`intervals=(100,50,2)`，三模态顺序固定为文本/语音/视觉。`stacked['id'][i]`、`samples[i]['id']` 与清单第 `i` 行一致。读取示例：

```python
import pickle
with open("problem1_features.pkl", "rb") as file:
    data = pickle.load(file)
i = 0
sample_id = data["stacked"]["id"][i]
length = int(data["stacked"]["valid_length"][i])
text = data["stacked"]["text"][i, :length]       # (length, 768)
audio = data["stacked"]["audio"][i, :length]     # (length, 74)
vision = data["stacked"]["vision"][i, :length]   # (length, 35)
intervals = data["stacked"]["intervals"][i, :length]
```

下表的“原视频有效时长”指原MP4片段长度；“词轴范围”指首个至末个有效对齐词的时段，两者因首尾静音可不同。“可用位”依次为文本/语音/视觉，维度亦依此顺序。序号为1–100，pickle行索引为序号减1。每个模态分别展开的300行机器可读表见 `full_results.csv`，其中还记录Excel行号、源路径、特征SHA-256。

| 序号 | 样本ID | 原视频有效时长(s) | 词轴范围(s) | 有效位 | T/A/V可用位 | T/A/V维度 | 粒度 | 来源 |
| ---: | --- | ---: | --- | ---: | --- | --- | --- | --- |
{chr(10).join(summary_rows)}

100条记录均通过独立审计：标签100、MP4 100、WAV 100、转写LAB 100、OpenFace CSV 100、输出特征100；TextGrid 99，另1条数字静音回退。`problem1_features.manifest.csv` 对每条样本保存源文件路径与SHA-256，`problem1_features.audit.json` 保存校验统计、参数、工具版本及模型/源码指纹；`problem1_features.audit.jsonl` 保存100条逐样本核验日志。

## 3. 典型样本时序验证

选取第1条样本 `{exemplar['id']}`（原视频 {float(exemplar['video_duration']):.3f} 秒、15个有效词位、文本/语音/视觉每个词位均可用）。原文：

> {exemplar['raw_text']}

下表列出部分位置。音频与视频列为**帧覆盖区间与词区间相交的候选帧中心时间范围**，实际池化仍按重叠时长与帧质量加权。最后一列依次为该位置的文本向量L2范数、语音 `log_rms`、视觉 `AU12_r`，用于展示三个数组确实在同一位置有数据；全部15位见 `typical_alignment.csv`。

| 位置(0起) | 文本词 | 对齐时段(s) | 对应语音候选帧 | 对应视频候选帧 | 文本范数 / log-RMS / AU12_r |
| ---: | --- | --- | --- | --- | --- |
{example_table}

![典型样本的词、语音、视频帧和三模态特征对应图](typical_alignment.png)

图中上方词区间、语音 `log_rms`、原视频抽取帧以及三模态向量范数共用时间轴。底部范数仅为可视化在各模态内部做0–1范围缩放，**不是模型输入或额外归一化步骤**。

## 4. 复现与附件

环境与流程见 `README.md`、`requirement.md` 和 `problem1_features.reproduce.md`。本次审计记录的主要工具版本：Python {versions['python']}、FFmpeg {versions['ffmpeg'].split()[2]}、MFA {versions['mfa']}、PyTorch {libraries['torch']}、Transformers {libraries['transformers']}、librosa {libraries['librosa']}。OpenFace命令不提供可靠版本号，审计记录了可执行文件和模型SHA-256。`problem1_features.processing.log` 是当前完整重跑日志；`problem1_features.initial_extraction.log` 记录首轮提取中间文件的过程。本次重跑复用了已生成的中间文件，从原视频重新提取请依照复现说明使用独立工作目录。来源与方法文件的逐文件哈希在审计时采集，原始首次提取时未记录逐文件哈希。

提交包包含完整原始特征pickle、100条来源清单、300行模态结果表、典型样本图与逐位表、审计与处理日志、复现说明和提取源码。`SHA256SUMS.txt` 可核对包内文件完整性。
"""
    (DELIVERY / "report.md").write_text(report, encoding="utf-8")

    files: list[tuple[Path, str]] = [
        (DELIVERY / "report.md", "report.md"),
        (DELIVERY / "full_results.csv", "full_results.csv"),
        (DELIVERY / "typical_alignment.csv", "typical_alignment.csv"),
        (DELIVERY / "typical_alignment.png", "typical_alignment.png"),
        (PICKLE, "problem1_features.pkl"),
    ]
    for extension in ("manifest.csv", "manifest.json", "audit.json", "audit.jsonl", "processing.log", "reproduce.md", "summary.json"):
        path = OUTPUTS / f"problem1_features.{extension}"
        files.append((path, path.name))
    files.append((PROJECT / "work" / "full-run.log", "problem1_features.initial_extraction.log"))
    for name in ("README.md", "requirement.md", "pyproject.toml"):
        files.append((PROJECT / name, name))
    for path in sorted((PROJECT / "src" / "problem1").glob("*.py")):
        files.append((path, f"src/problem1/{path.name}"))
    for path in sorted((PROJECT / "tests").glob("test_*.py")):
        files.append((path, f"tests/{path.name}"))
    files.append((Path(__file__), "build_delivery.py"))
    sums = "\n".join(f"{checksum(path)}  {name}" for path, name in files) + "\n"
    (DELIVERY / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    files.append((DELIVERY / "SHA256SUMS.txt", "SHA256SUMS.txt"))
    archive = DELIVERY / "problem1_submission.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for path, name in files:
            output.write(path, name)
    print(f"wrote {len(summary_rows)} sample rows, {len(full_rows)} modality rows, {len(example_rows)} example positions")
    print(f"archive: {archive} ({archive.stat().st_size} bytes)")


if __name__ == "__main__":
    build()
