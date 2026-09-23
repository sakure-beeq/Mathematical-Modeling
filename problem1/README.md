# 问题1：多模态特征提取与词级时序对齐

该目录给出一套从附件1原始视频生成问题1特征文件的完整、可复现代码。主时间轴是 MFA 强制对齐得到的英文词区间 `[s_k, e_k]`。语音和视觉帧按“帧覆盖区间与词区间的重叠时长 × 帧质量”加权池化；超过50词时将相邻词聚合成50个连续区间，因此不会丢弃视频后半段。

## 输出规格

最终 pickle 的 `stacked` 字段可直接用于建模：

- `text`: `(N, 50, 768)`，冻结 BERT，属于同一词的 WordPiece 取均值；
- `audio`: `(N, 50, 74)`，包括 F0、浊音概率、能量、MFCC 及差分、频谱、HNR 和 jitter；
- `vision`: `(N, 50, 35)`，17个 AU、6个头姿、8个注视特征和4个面部尺度/中心统计量；
- `intervals`: `(N, 50, 2)`；
- `valid_mask` 与 `padding_mask`：后者 `True` 明确表示应忽略的 padding；
- `modality_availability_mask`, `quality_scores`, `quality_mask`: `(N, 50, 3)`，模态顺序为文本、语音、视觉；
- `valid_length`、`alignment_confidence`、`face_detection_failure_rate`、`was_aggregated`、`was_truncated`。

`samples` 字段还逐样本保存原始文本、词、视频时长、标签、工具参数和质量诊断。`alignment_confidence` 是“标签文本与 TextGrid 词序列的一致率代理”，不是 MFA 的声学后验，代码会在元数据中明确记录，避免把代理量误报为声学置信度。所有输出使用 `float32`；无效和 padding 位置保持为零。

逐样本可追溯字段包括 `sample_index`（输出行号）、`label_row`（Excel行号）、`source_files`（MP4、WAV、转写LAB、TextGrid、OpenFace CSV路径与SHA-256）、`source_word_timeline`（原词及秒级区间）、`sequence_position`（0到49）、`source_word_span`（每个输出位置对应原词的半开索引区间）和 `feature_sha256`。填充位置的词索引为 `[-1,-1]`。因此即使超过50词而发生连续聚合，也能追溯每个位置覆盖的原词及时间范围。

若原始视频音轨为数字静音且 MFA 未生成 TextGrid，样本不会被删除：程序使用覆盖完整视频时长的均匀词区间作为显式兜底，并记录 `forced_alignment_available=False`、`alignment_method="uniform_text_timeline_fallback"`、`alignment_confidence=0`；静音语音帧的质量权重为零。其他缺失 TextGrid 的样本会报错，避免静默使用伪时间轴。

## 依赖

macOS完整依赖清单、Conda环境安装、OpenFace编译和验证命令见 [`requirement.md`](requirement.md)。

Python 依赖：

```bash
cd problem1
/opt/anaconda3/envs/MathematicalModeling/bin/python -m pip install -e '.[extract]'
```

还需要以下命令行工具：

1. `ffmpeg` 与 `ffprobe`：只转为 16 kHz 单声道 PCM，不降噪、不统一响度；
2. [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/) (`mfa`)；
3. [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) 的 `FeatureExtraction`；
4. 一个固定版本或固定本地目录的 `bert-base-uncased`。

首次使用 MFA 时安装英文模型（模型名称也可由命令行替换）：

```bash
mfa model download dictionary english_us_arpa
mfa model download acoustic english_us_arpa
```

当前机器若没有这些外部工具，程序会明确报错，不会用随机数、零向量或伪时间戳代替真实特征。

## 对附件1运行

在 `problem1` 目录执行。当前机器已验证下列本地模型和 OpenFace 路径：

```bash
problem1 run \
  --labels '../E题数据/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条/label-100.xlsx' \
  --videos-root '../E题数据/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条' \
  --work-dir work \
  --output outputs/problem1_features.pkl \
  --bert-model models/bert-base-uncased \
  --openface models/openface/FeatureExtraction \
  --openface-model models/openface/model/main_clnf_general.txt
```

流水线默认断点续跑，保留已生成的 WAV、TextGrid 和 OpenFace CSV。若工具参数发生变化，请使用 `--no-resume`；若还要清空 MFA 自身缓存，增加 `--clean-mfa`。

每次运行会检查标签行、原始MP4、WAV、转写LAB、TextGrid、视觉CSV和输出特征的一一对应关系，并在输出目录写入同名的 `manifest.csv`、`manifest.json`、`audit.json`、`audit.jsonl` 与 `reproduce.md`。`audit.json` 还记录提取参数、工具版本，以及审计时采集的模型、代码和来源文件校验值。已有特征文件可独立重验：

```bash
problem1 audit \
  --features outputs/problem1_features.pkl \
  --labels '../E题数据/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条/label-100.xlsx' \
  --videos-root '../E题数据/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条' \
  --work-dir work --verify-only
```

去掉 `--verify-only` 可为旧版pickle补写上述字段和旁证文件。加上 `--processing-log work/full-run.log` 会把原始命令行日志复制为同名 `processing.log`；审计生成的 `audit.jsonl` 始终记录每条样本的核验结果。
若审计后有意修改了本地提取代码或模型文件，旧哈希核验会失败；确认新版本后用 `--refresh-method-assets` 重新记录方法文件指纹。

## 典型样本图

同一时间轴上画词区间、语音 log-RMS、视频关键帧和三模态特征范数：

```bash
problem1 plot \
  --features outputs/problem1_features.pkl \
  --sample-index 0 \
  --output outputs/sample_000_timeline.png
```

## 测试

核心对齐、加权池化、短缺失插值、连续聚合、不可对齐兜底和 padding 不依赖外部模型，可直接测试：

```bash
PYTHONPATH=src /opt/anaconda3/envs/MathematicalModeling/bin/python -m unittest discover -s tests -v
```

## 代码结构

- `alignment.py`：音频准备、MFA 调用、TextGrid 解析；
- `text_features.py`：冻结 BERT 与 WordPiece 均值；
- `audio_features.py`：74维声学帧；
- `visual_features.py`：OpenFace 35维帧与短缺失插值；
- `core.py`：质量加权池化、连续区间聚合、右侧 padding；
- `pipeline.py`：附件1端到端处理、质量统计和 pickle 打包；
- `audit.py`：样本覆盖、时序与来源校验，生成清单和复现记录；
- `plotting.py`：典型样本同轴图。

说明：74维描述符覆盖题目要求的声学类别，但不声称与某个历史 COVAREP 二进制逐位一致。训练、验证、测试必须使用同一份代码和参数；如论文要求严格复现 COVAREP，应将 `extract_audio74` 替换为固定版本的 COVAREP 输出读取器，并保持后续对齐接口不变。
