# 问题 2：局部模态缺失下的情感预测

本目录包含数据准备、模型训练、评价、稳健性分析和附件 3 推理的源码。`tests/` 是测试源码，`vendor/mmer_preprocess/` 是所需的预处理模块副本。

## 数据与环境

主训练输入为本地 `E题数据/附件2-数据集特征文件/aligned_50.pkl`，附件 3 的对齐特征仅用于推理。默认复用本地 `problem1/models/bert-base-uncased` 的冻结 BERT 权重；可通过 `prepare --bert-model` 指定其他路径。训练与推理须使用相同的 BERT 检查点。

安装依赖：

```bash
python -m pip install -r problem2/requirements.txt
```

从仓库根目录运行：

```bash
python problem2/main.py prepare
python problem2/main.py train
python problem2/main.py evaluate --split valid
python problem2/main.py robustness --split valid --seeds 5
python problem2/main.py errors
python -m unittest discover -s problem2/tests -v
```

`prepare` 在本地生成 `problem2/cache/`，训练、评价及推理生成 `problem2/outputs/`，消融实验可写入 `problem2/ablations/`。这些目录包含数据缓存、模型权重或实验结果，不纳入本仓库的新提交。

## 方法与入口

`data.py` 处理文本、语音和视觉特征及可用性掩码，`model.py` 定义时序编码、跨模态重建和融合预测模型，`experiment.py` 实现训练与评价，`main.py` 提供命令行入口。`class_metrics.py`、`error_analysis.py`、`robustness_report.py` 等脚本用于生成本地分析结果，`figures/` 中的脚本生成图表。

使用 `python problem2/main.py --help` 查看全部子命令和参数。复现实验时先运行 `prepare`，再训练、选择模型并评价。原始数据、缓存和结果由使用者在本地准备或生成。
