# 问题1特征复现实验说明

本次结果在 `MathematicalModeling` 环境生成。先进入同一项目目录；下列命令使用断点续跑的现有中间文件。若从原始视频重新提取，改用独立工作目录，并添加 `--no-resume --clean-mfa`。

```bash
conda activate MathematicalModeling
cd '/Users/wangyiming/Desktop/E题/problem1'
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
problem1 run --labels '../E题数据/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条/label-100.xlsx' --videos-root '../E题数据/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条' --work-dir work --output outputs/problem1_features.pkl --bert-model models/bert-base-uncased --mfa-dictionary english_us_arpa --mfa-acoustic-model english_us_arpa --ffmpeg ffmpeg --ffprobe ffprobe --mfa mfa --openface models/openface/FeatureExtraction --openface-model models/openface/model/main_clnf_general.txt --device cpu --target-length 50 --visual-interpolation-gap 0.25 --quality-threshold 0.25
```

词区间以 MP4 开始时刻为零点；`sequence_position` 从 0 起，`source_word_span` 指向 `source_word_timeline` 的半开索引区间。超过 50 词时按连续词组聚合；右侧位置填零，`padding_mask=True` 表示忽略。文本使用冻结 BERT 与同词 WordPiece 均值；语音使用 74 维声学描述符，视觉使用 OpenFace 35 维帧特征，再按帧覆盖区间重叠时长乘质量分数池化。

FFmpeg只提取16 kHz单声道PCM，不降噪或统一响度。MFA使用 `english_us_arpa` 词典和 `english_us_arpa` 声学模型；仅数字静音且没有TextGrid的样本使用明确标记、置信度为零的均匀词区间；其他缺失TextGrid的样本会报错。语音帧长400点、帧移160点；视觉短缺失插值阈值0.25秒，质量掩码阈值0.25。

逐样本原始视频、WAV、转写LAB、TextGrid、OpenFace CSV 和特征摘要见同名 `manifest.csv`。`audit.json` 给出覆盖与时序检查，`audit.jsonl` 是逐样本核验日志。若存在 `processing.log`，它是原始命令行处理输出。工具版本和关键参数保存在 pickle 元数据及 `audit.json`；模型与代码校验值为审计时采集。初次提取的运行期未记录逐文件哈希。

核验命令：

```bash
problem1 audit --features outputs/problem1_features.pkl --labels '../E题数据/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条/label-100.xlsx' --videos-root '../E题数据/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条' --work-dir work --verify-only
```
