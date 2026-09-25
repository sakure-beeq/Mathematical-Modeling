# 问题 3 主模型：面向附件 4 的时序交互网络与可核查解释

本目录是问题 3 的主模型，针对附件 4 的完整三模态样本设计。它不读取问题 2 或 `problem3/independent` 的预测权重，也不读取附件 3 的数据。模型参数只在附件 2 的训练集上学习，结构和参数只在附件 2 的验证集上选择；附件 4 没有标签，只用于最终推理和原视频证据回看。`problem3_2/prepare_training.py` 只读取附件 2 的 `aligned_50.pkl`，将训练、验证、测试特征写入本目录的 `cache/`。`prepare.py` 再读取附件 4，不依赖附件 3。

## 模型

- 文本、语音、视觉各有独立的投影、深度时序卷积和注意力池化编码器。
- 融合层同时输入三个模态表示、两两逐元素交互、三模态交互和有效位置比例。分类头输出三类 logit，回归头输出 `3*tanh(·)` 强度。
- 主训练目标使用完整三模态输入的加权交叉熵与 Smooth L1；另以权重 0.3 加入训练集上的随机模态子集任务损失，使解释时的模态移除属于模型见过的输入类型。该辅助任务不使用附件 3，也不改变附件 4 以完整输入预测的目标。
- 在验证集按 `Macro F1 + 0.2×Pearson r - 0.1×MAE` 选择权重，最佳为种子 2030 的第 3 轮。测试集和附件 4 未用于选模。

## 解释方法

对单条样本调用 `InteractionModel.predict_with_explanation(sample, timeline)`，同一次返回中包含 `pred_class`、`pred_score`、三类概率、`main_modality`、三模态带符号 Shapley 值、作用占比，以及 `windows` 中对应原文和视频秒数的局部证据。`forward` 保留为训练所需的可微预测接口。批量命令 `problem3_2/explain.py predict4` 对附件 4 全部样本调用上述统一接口，并生成提交 CSV 和解释卡。

以完整输入预测类别和次高类别的 logit 差为目标，对三模态全部 8 个子集精确计算 Shapley 值。三个带符号贡献之和等于完整输入与空子集的类别差值变化；绝对贡献归一化后用于比较作用程度。主要参考模态取对该判断正向支持最大的模态。

附件 4 的局部证据在每个模态的连续 2–5 个有效位置上逐窗遮挡，选择使原判断类别差值下降最多的窗口。输出英文片段、原视频起止秒数和视觉关键帧，并保存完整位置重要性曲线。这个窗口选择规则与其遮挡下降值使用同一实验，下降值是模型内的描述性解释，不能当作独立验证。

独立验证在附件 2 验证集每类随机取 20 条，用梯度乘输入**先选候选窗口**，再通过遮挡复算与同长度随机窗口比较，另记录只保留该窗口时是否维持原类别。60 条样本中候选窗口遮挡后的类别差值平均下降 0.128，随机窗口平均变化约为 -0.002；68.3% 的样本前者下降更多；只保留候选窗口时 78.3% 维持原预测类别。这是模型忠实性检查，不是真实情绪原因的证明。

## 预测指标

| 划分 | 样本数 | Accuracy | Macro F1 | MAE | Pearson r |
| --- | ---: | ---: | ---: | ---: | ---: |
| 新模型 valid | 728 | 0.6264 | 0.6164 | 0.6095 | 0.6339 |
| 新模型 test | 727 | 0.6451 | 0.6043 | 0.6364 | 0.6758 |
| 问题 2 最终模型 valid（对照） | 728 | 0.6484 | 0.6303 | 0.6205 | 0.6460 |
| 问题 2 最终模型 test（对照） | 727 | 0.6795 | 0.6398 | 0.6435 | 0.6776 |

本模型的强度 MAE 稍低，分类 Accuracy 和 Macro F1 较问题 2 最终模型低。附件 4 没有标签，因此无法计算附件 4 的上述四项指标。

## 结果文件

- `outputs/best.pt`、`training_config.json`、`training_history.json`、`result_manifest.json`：新模型权重、参数、逐轮记录与输入数据校验值。
- `outputs/valid_metrics.json`、`test_metrics.json` 与对应预测 CSV：附件 2 验证和测试结果，含 95% bootstrap 区间。
- `outputs/attachment4_predictions_explanations.csv`：附件 4 全部 20 条样本的预测、三模态 Shapley 值、主要模态、关键证据位置和视觉关键帧路径。
- `outputs/attachment4_explanation_cards.json`：每条样本的详细解释；`attachment4_position_importance.csv`、`importance_plots/`：位置重要性；`modality_comparison.png`：三模态作用占比。
- `outputs/valid_explanation_report.json`、`valid_explanation_faithfulness.csv`：独立候选与遮挡检验；`valid_error_analysis.json`：错误分析。
- `outputs/typical_explanation_cards.md`、`key_frames/`：典型样本说明与可回看的原视频关键帧。

附件 4 的 13 号样本虽然有原始视频，但对齐版视觉特征完全失效，因此视觉贡献为零且不输出视觉关键帧。时间定位使用 14 条原样 MFA 对齐与 6 条经词序修复的 MFA 对齐；修复样本应结合原视频复核。

## 复现

从仓库根目录运行。`prepare_training.py` 只处理附件 2；`prepare.py` 和 `timeline.py` 处理附件 4 与视频时间轴。三者复用仓库的特征预处理代码，不读取问题 2 的预测权重或附件 3。原始数据和下载的 BERT 权重不上传到 GitHub，运行者须按赛题材料和所记录的模型版本在本地准备。

```bash
PY=python
$PY problem3_2/prepare_training.py
$PY problem3_2/prepare.py --out problem3_2/cache --align
$PY problem3_2/main.py train
$PY problem3_2/main.py evaluate --split valid
$PY problem3_2/main.py evaluate --split test
$PY problem3_2/explain.py validate
$PY problem3_2/explain.py predict4
$PY problem3_2/report.py
$PY -m unittest discover -s problem3_2/tests -v
```

所需基础依赖见 `problem2/requirements.txt`；时间轴重建还需本地 `ffmpeg`、`ffprobe` 和 Montreal Forced Aligner。`cache/` 是可再生成的数据，不属于正式竞赛附件。竞赛提交时还须按题面核对总附件 50 MB 上限及匿名要求。
