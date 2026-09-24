# 问题 2：局部模态缺失下的多模态情感预测

本目录是问题 2 的最终版本。选定模型为 **`fixed_gate`、随机种子 2028、dropout 0.3、第 9 轮权重**。模型同时预测 Negative / Neutral / Positive 三类情感和 [-3, 3] 连续情感强度，并通过连续区间遮挡训练处理局部模态缺失。

## 直接查看最终交付

- [最终权重与全部图表](release/seed2028/README.md)：包括 `best.pt`、15 张图的 PNG/PDF、指标表和附件 3 预测。
- [15 页图表合订 PDF](release/seed2028/问题2_种子2028_全部图表.pdf) · [逐图解读](release/seed2028/逐图解读.md) · [结果汇总](release/seed2028/结果汇总.md)
- [整合训练曲线](release/seed2028/charts/02_training_curves.png)：同图比较训练集完整输入、验证集完整输入与验证集 TAV 30% 缺失输入的逐轮 Accuracy、Macro F1、MAE。
- [附件 3 最终预测 CSV](release/seed2028/attachment3_predictions.csv)：30 个无标签样本的类别、强度、三类概率与原有缺失比例。

图表已去掉顶部总标题。最后一张架构消融图使用种子 2026、dropout 0.2 的历史实验，图中及说明文档均标明其来源；其余 14 张图使用最终种子 2028 权重。

## 数据来源与预处理

训练、验证、测试数据来自附件 2 的 **`aligned_50.pkl`**，分别有 3395、728、727 条样本。附件 3 的对齐版有 30 条无标签样本，仅用于最终推理。主模型没有读取问题 1 生成的特征文件；预处理仅复用同一份本地冻结 `bert-base-uncased` 权重，把附件 2、附件 3 的 `text_bert` 统一编码为文本表示。

每条样本最多 50 个对齐位置，文本、语音、视觉的输入维度依次是 768、74、35。`data.py` 分别维护真实位置掩码 P、模态可用掩码 A、质量掩码 Q，实际观测位置为 P×A×Q。语音和视觉标准化参数只从训练集有效位置估计，避免验证/测试信息泄漏。预处理缓存由 `prepare` 生成于 `problem2/cache/`，原始附件、BERT 权重和缓存不包含在本仓库中。

## 最终模型与训练设置

| 项目 | 设置 |
| --- | --- |
| 单模态投影 | 各自线性层 → LayerNorm → GELU，隐藏维度 128 |
| 时序编码 | 每模态两层 Transformer Encoder，4 个注意力头，前馈维度 256 |
| 跨模态编码 | 一层 Transformer Encoder，按 50×3 个模态位置处理 |
| 缺失处理 | 训练时随机遮挡连续区间；跨模态重建缺失位置表示 |
| 融合与输出 | 非填充模态等权融合 → 时间注意力池化 → 三分类头与 `3*tanh` 强度头 |
| Dropout | 0.3 |
| 优化器 | AdamW，batch size 32，初始学习率 2e-4，第 6 轮起 5e-5 |
| 训练与选模 | 种子 2028，最多 12 轮，patience 4；最佳权重在第 9 轮 |

联合目标包括缺失输入的加权交叉熵与 Smooth L1、权重 0.3 的完整输入任务损失、权重 0.1 的潜在重建损失和权重 0.1 的完整/缺失预测一致性损失。类别权重仅由训练集类别频数计算。每轮按验证集的 `0.5×完整输入 Macro F1 + 0.5×TAV 30% 缺失 Macro F1 − 0.05×缺失 MAE` 选择检查点；测试集不参与选模。

## 最终结果

| 划分 | 样本数 | Accuracy | Macro F1 | MAE | Pearson r |
| --- | ---: | ---: | ---: | ---: | ---: |
| 训练集 | 3395 | 0.7493 | 0.7265 | 0.4802 | 0.8235 |
| 验证集 | 728 | 0.6484 | 0.6303 | 0.6205 | 0.6460 |
| 测试集 | 727 | 0.6795 | 0.6398 | 0.6435 | 0.6776 |

第 9 轮在固定随机位置的验证集 TAV 30% 缺失输入上，Accuracy 为 0.6497、Macro F1 为 0.6340、MAE 为 0.6199。该单次遮挡不能代替稳健性评估；[缺失实验 CSV](release/seed2028/valid_robustness.csv) 包含 7 种模态组合、5 个比例、不同位置共 280 条条件记录。附件 3 无真实标签，不能计算其 Accuracy、F1 或 MAE。

逐标签指标见 [label_metrics.csv](release/seed2028/label_metrics.csv)。其中单标签 Accuracy 与 Macro F1 是“该标签对其余两类”的二分类指标；单标签 MAE 只统计真实标签属于该类的样本。

## 安装与运行

从仓库根目录执行。需要 Python、`problem2/requirements.txt` 中的依赖、附件 2/3 和本地冻结 BERT 权重。

```bash
python -m pip install -r problem2/requirements.txt
python problem2/main.py prepare \
  --attachment2 'E题数据/附件2-数据集特征文件/aligned_50.pkl' \
  --attachment3 'E题数据/附件3-模态缺失特征样本/对齐版本' \
  --bert-model problem1/models/bert-base-uncased
```

复现原始训练与逐轮训练集指标回放需要运行两次相同的随机种子。第二次只额外记录训练集指标；`replay_training_curve.py` 会逐轮核对指标并逐参数核对最佳权重。

```bash
python problem2/main.py train --out problem2/outputs/dropout03_lr_drop/seed2028 \
  --epochs 12 --batch-size 32 --hidden 128 --heads 4 --dropout 0.3 \
  --lr 0.0002 --patience 4 --seed 2028 --ablation fixed_gate \
  --lr-drop-epoch 6 --lr-drop-factor 0.25

python problem2/main.py train --out problem2/outputs/dropout03_lr_drop/seed2028_metrics_replay \
  --epochs 12 --batch-size 32 --hidden 128 --heads 4 --dropout 0.3 \
  --lr 0.0002 --patience 4 --seed 2028 --ablation fixed_gate \
  --lr-drop-epoch 6 --lr-drop-factor 0.25 --record-train-metrics

python problem2/replay_training_curve.py
```

以下命令生成验证/测试指标、逐标签指标、附件 3 预测、缺失实验及图表。`robustness` 运行 280 个条件，需要一定时间。

```bash
python problem2/main.py evaluate --split valid \
  --checkpoint problem2/outputs/dropout03_lr_drop/seed2028/best.pt \
  --out problem2/outputs/deliverable_seed2028
python problem2/main.py evaluate --split test \
  --checkpoint problem2/outputs/dropout03_lr_drop/seed2028/best.pt \
  --out problem2/outputs/deliverable_seed2028
python problem2/class_metrics.py \
  --checkpoint problem2/outputs/dropout03_lr_drop/seed2028/best.pt \
  --out problem2/outputs/deliverable_seed2028
python problem2/main.py predict3 \
  --checkpoint problem2/outputs/dropout03_lr_drop/seed2028/best.pt \
  --output problem2/outputs/deliverable_seed2028/attachment3_predictions.csv
python problem2/main.py robustness --split valid --seeds 5 --bootstrap 200 \
  --checkpoint problem2/outputs/dropout03_lr_drop/seed2028/best.pt \
  --out problem2/outputs/deliverable_seed2028
python problem2/main.py errors --data problem2/cache/valid.npz \
  --predictions problem2/outputs/deliverable_seed2028/valid_predictions.csv \
  --output problem2/outputs/deliverable_seed2028/valid_error_analysis.json
python problem2/main.py errors --data problem2/cache/test.npz \
  --predictions problem2/outputs/deliverable_seed2028/test_predictions.csv \
  --output problem2/outputs/deliverable_seed2028/test_error_analysis.json
python problem2/figures/build_seed2028_deliverables.py
python problem2/package_release.py
```

`outputs/` 为运行目录；`release/seed2028/` 是随代码提交的最终交付快照。历史架构消融的原始 CSV 已保存在交付目录中，图表脚本可直接读取。自动检查：

```bash
python -m unittest discover -s problem2/tests -v
```

## 目录说明

| 路径 | 内容 |
| --- | --- |
| `data.py`、`vendor/` | 附件 2/3 预处理、对齐掩码和冻结 BERT 编码 |
| `model.py` | 时序编码、跨模态重建、融合及两个预测头 |
| `experiment.py`、`main.py` | 训练、评价、缺失实验和附件 3 推理入口 |
| `class_metrics.py`、`error_analysis.py` | 三个数据划分及误差切片 |
| `figures/build_seed2028_deliverables.py` | 生成最终 15 张图及数值汇总 |
| `package_release.py` | 核验权重来源并整理 GitHub 交付文件 |
| `release/seed2028/` | 最终权重、图表、指标、预测和说明 |

`ema_report.py`、`dropout_report.py`、`hidden96_report.py` 等保留了调参实验的复现脚本；最终提交以本 README 上述种子 2028 的固定门控模型为准。
