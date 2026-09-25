# E题：多模态情感识别（问题 1–3）

本仓库收录三个问题的正式代码、训练权重、结果、图表与复现说明。

| 问题 | 代码与说明 | 正式输出 | 模型 |
| --- | --- | --- | --- |
| 1 多模态特征提取 | [problem1/README.md](problem1/README.md) | [problem1/outputs/](problem1/outputs/)、[problem1/delivery/](problem1/delivery/) | 冻结 BERT、MFA、OpenFace 与 74 维声学特征 |
| 2 缺失模态情感预测 | [problem2/README.md](problem2/README.md) | [problem2/release/seed2028/](problem2/release/seed2028/)、[problem2/outputs/](problem2/outputs/) | RobustFusion fixed_gate，种子 2028 |
| 3 可解释情感预测 | [problem3_2/README.md](problem3_2/README.md) | [problem3_2/outputs/](problem3_2/outputs/) | 独立训练的 InteractionModel，种子 2030 |

原始附件 1–4、下载的 BERT/OpenFace 文件及预处理缓存体积约 12 GB，未放入本仓库。请按各问题的 README 和复现说明将附件放在项目根目录 `E题数据/`，将 BERT 放在 `problem1/models/bert-base-uncased/`；其他本地生成的缓存由相应准备命令重建。问题 2 的主方案使用附件 2 与附件 3 对齐版；问题 3 主方案使用附件 2 与附件 4 对齐版及其视频，不读取问题 2 的预测权重。

文件完整性可通过 [UPLOAD_MANIFEST.json](UPLOAD_MANIFEST.json) 核验。
