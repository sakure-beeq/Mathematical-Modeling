# E题：多模态情感建模

本仓库包含数学建模文档、数据预处理代码、问题1的词级时序对齐特征提取程序，以及问题2的局部模态缺失预测程序。

## 目录

- [`Project/`](Project/)：通用多模态特征预处理程序和测试。
- [`problem1/`](problem1/)：从原始视频生成问题1特征的程序、环境说明和测试源码。
- [`problem2/`](problem2/)：使用附件2对齐特征训练鲁棒预测模型，并对附件3推理的程序和测试源码。
- `build_modeling_solution.py`：生成建模方案文档的脚本。

问题1的正式输出是 [`problem1/outputs/problem1_features.pkl`](problem1/outputs/problem1_features.pkl)；
同目录的 `problem1_features.manifest.csv` 记录100条样本与原始文件的对应关系。
复现步骤、依赖和参数见 [`problem1/README.md`](problem1/README.md)、
[`problem1/requirement.md`](problem1/requirement.md) 和
[`problem1/outputs/problem1_features.reproduce.md`](problem1/outputs/problem1_features.reproduce.md)。

原始数据位于本地 `E题数据/`，BERT、MFA 和 OpenFace 模型为外部资源；它们体积较大，
未纳入 Git 历史。问题2的缓存、模型权重、训练及测试结果和问题1的新交付产物也仅保留在本地。
从原始视频重新提取时，需按上述说明自行准备数据和模型。
