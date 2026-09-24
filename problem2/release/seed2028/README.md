# 问题 2 最终交付：随机种子 2028

本目录包含选中的第 9 轮模型权重、逐轮训练日志、15 张图的 PNG/PDF、图表合订 PDF、指标与预测 CSV。原始附件数据和预处理缓存不在仓库中。

- [图表目录](图表目录.md)
- [逐图解读](逐图解读.md)
- [结果汇总](结果汇总.md)
- [全部图表 PDF](问题2_种子2028_全部图表.pdf)
- [附件 3 预测](attachment3_predictions.csv)

使用 `torch.load('best.pt', map_location='cpu', weights_only=False)` 读取权重，由 `problem2.experiment.load_model` 构建网络。模型训练配置见 `training_config.json`。
`training_history_with_train_metrics.json` 是逐轮指标回放，已与原始最佳权重逐参数核对一致。
最后一张架构消融图来自种子 2026、dropout 0.2，是明确标注的历史对照。
