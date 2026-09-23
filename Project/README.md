# 多模态情感识别数据预处理

本项目实现附件2、附件3、附件4对齐版数据的统一预处理。默认不修改原始数据，输出新的预处理文件。

## 处理原则

每个模态保存四种布尔掩码：

- `padding`：该位置是否属于真实序列；
- `availability`：该模态在此位置是否可用；
- `quality`：该位置是否通过基本质量检查；
- `effective`：前三者的逻辑与，供编码器、归一化和损失函数使用。

普通训练、验证和测试数据中的内部全零音视频行按质量失效处理。附件3是专门的缺失测试集，其中内部全零行按模态不可用处理。

## 安装

仅运行掩码、类型转换和标准化：

```bash
python -m pip install -e .
```

需要重新生成冻结BERT文本表示时：

```bash
python -m pip install -e '.[bert]'
```

## 附件2

先使用训练集拟合语音和视觉的均值、标准差，然后转换全部三个划分：

```bash
mmer-preprocess attachment2 \
  --input ../E题数据/附件2-数据集特征文件/aligned_50.pkl \
  --output-dir outputs/attachment2
```

输出包括：

- `train.pkl`、`valid.pkl`、`test.pkl`；
- `normalization_stats.json`；
- `preprocessing_summary.json`。

如需生成统一的冻结BERT表示，可增加：

```bash
--bert-model /path/to/local/bert-base-uncased
```

建议比赛实验固定本地模型目录或模型版本哈希，避免不同时间下载到不同权重。

## 附件3

```bash
mmer-preprocess attachment3 \
  --input-dir ../E题数据/附件3-模态缺失特征样本/对齐版本 \
  --stats outputs/attachment2/normalization_stats.json \
  --output outputs/attachment3.pkl
```

附件3没有样本ID字段，脚本使用文件名作为ID。

## 附件4

```bash
mmer-preprocess attachment4 \
  --input-dir ../E题数据/附件4-可解释专项视频样本与特征文件/附件4-可解释专项视频样本与特征文件/对齐版本 \
  --stats outputs/attachment2/normalization_stats.json \
  --output outputs/attachment4.pkl
```

## 只读验证

不生成预处理文件，只检查真实附件2结构、类型和掩码计数：

```bash
mmer-preprocess validate \
  --input ../E题数据/附件2-数据集特征文件/aligned_50.pkl
```

## 测试

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## 输出数据结构

每个输出文件包含：

- `text_bert`：统一为 `int64`；
- `audio`、`vision`：统一为 `float32`，只标准化有效位置；
- `masks`：文本、语音、视觉的四类掩码；
- `id`、`raw_text`：源文件存在时保留；
- `classification_labels`：统一为 `int64`；
- `regression_labels`：统一为 `float32`；
- `metadata`：样本数、时间长度、特征维数和处理角色。

无效位置在标准化后仍严格保持为零，避免零填充经过减均值后变成非零输入。
