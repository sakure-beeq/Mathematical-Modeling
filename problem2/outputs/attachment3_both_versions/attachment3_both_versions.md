# 附件 3 对齐版与未对齐版预测类别

两个版本均有 30 条无标签样本。对齐版和未对齐版使用各自训练、验证选权重的模型。
未对齐音频与视觉按各自时间轴汇总为最多 50 个有序位置；文本从未对齐版 raw_text 使用相同冻结 BERT 编码。

| ID | 对齐版类别 | 强度 | 未对齐版类别 | 强度 | 类别一致 |
| --- | --- | ---: | --- | ---: | --- |
| 附件3_01 | Negative | -0.7162 | Negative | -0.9837 | 是 |
| 附件3_02 | Neutral | -0.1460 | Neutral | -0.1463 | 是 |
| 附件3_03 | Neutral | +0.0305 | Neutral | +0.2164 | 是 |
| 附件3_04 | Neutral | +0.0437 | Neutral | +0.1886 | 是 |
| 附件3_05 | Negative | -1.6359 | Negative | -0.4607 | 是 |
| 附件3_06 | Positive | +0.9822 | Positive | +1.1551 | 是 |
| 附件3_07 | Positive | +0.2568 | Neutral | +0.1060 | 否 |
| 附件3_08 | Positive | +0.3923 | Neutral | +0.3880 | 否 |
| 附件3_09 | Positive | +1.2456 | Positive | +0.8579 | 是 |
| 附件3_10 | Neutral | +0.3186 | Neutral | +0.1081 | 是 |
| 附件3_11 | Positive | +0.0708 | Positive | +0.2381 | 是 |
| 附件3_12 | Neutral | +0.2063 | Positive | +1.0668 | 否 |
| 附件3_13 | Positive | +0.5183 | Positive | +0.6990 | 是 |
| 附件3_14 | Negative | -0.6993 | Negative | -0.2533 | 是 |
| 附件3_15 | Positive | +0.4620 | Positive | +0.8332 | 是 |
| 附件3_16 | Positive | +1.3991 | Positive | +1.6811 | 是 |
| 附件3_17 | Negative | -0.3522 | Negative | -0.1614 | 是 |
| 附件3_18 | Positive | +0.5954 | Positive | +0.8256 | 是 |
| 附件3_19 | Negative | -0.8305 | Negative | -0.8982 | 是 |
| 附件3_20 | Neutral | -0.4422 | Neutral | +0.3874 | 是 |
| 附件3_21 | Positive | +0.5222 | Positive | +0.6540 | 是 |
| 附件3_22 | Positive | +0.1996 | Negative | -0.2311 | 否 |
| 附件3_23 | Neutral | +0.5685 | Positive | +1.4826 | 否 |
| 附件3_24 | Negative | +0.5494 | Positive | +1.2528 | 否 |
| 附件3_25 | Neutral | -0.2274 | Positive | +0.4039 | 否 |
| 附件3_26 | Positive | +0.6796 | Neutral | +0.5877 | 否 |
| 附件3_27 | Positive | +0.6832 | Positive | +0.7509 | 是 |
| 附件3_28 | Positive | +0.3827 | Positive | +0.9005 | 是 |
| 附件3_29 | Positive | +0.3256 | Positive | +1.5831 | 是 |
| 附件3_30 | Neutral | +0.3667 | Neutral | +0.8870 | 是 |
