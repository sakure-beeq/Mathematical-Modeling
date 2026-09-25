# 问题 3 典型样本解释卡

## 样本 16

原文：It's a terrible, this is a terrible movie

预测：Negative，强度 -1.733；主要模态：text。

| 模态 | 带符号 Shapley 贡献 | 绝对贡献占比 | 关键片段 | 遮挡后类别差值下降 |
| --- | ---: | ---: | --- | ---: |
| text | 3.708 | 91.8% | a terrible；1.66–2.15 s | 0.110 |
| audio | -0.112 | 2.8% | 1.51–2.44 s | 0.333 |
| vision | 0.221 | 5.5% | 1.51–2.44 s | 0.220 |

解释反映模型的决策依据；时间位置可在附件 4 原视频中回看。

## 样本 13

原文：For example, I could take a set of data and from that data, I can find a relationship between any two of the given factors or more.

预测：Neutral，强度 0.054；主要模态：audio。

| 模态 | 带符号 Shapley 贡献 | 绝对贡献占比 | 关键片段 | 遮挡后类别差值下降 |
| --- | ---: | ---: | --- | ---: |
| text | 0.291 | 40.9% | could take a set of；0.95–1.72 s | 0.162 |
| audio | 0.421 | 59.1% | 1.72–4.45 s | 0.053 |
| vision | 0.000 | 0.0% | 无有效特征 | — |

解释反映模型的决策依据；时间位置可在附件 4 原视频中回看。

## 样本 04

原文：If I blow it at the team exercise, should I kiss my chances of cheering "GO BLUE" goodbye?] Absolutely not.

预测：Positive，强度 0.751；主要模态：vision。

| 模态 | 带符号 Shapley 贡献 | 绝对贡献占比 | 关键片段 | 遮挡后类别差值下降 |
| --- | ---: | ---: | --- | ---: |
| text | -0.662 | 33.0% | blow it at the team；0.66–1.47 s | 0.379 |
| audio | 0.426 | 21.3% | 2.33–3.38 s | 0.110 |
| vision | 0.916 | 45.7% | 2.33–3.38 s | 0.081 |

解释反映模型的决策依据；时间位置可在附件 4 原视频中回看。
