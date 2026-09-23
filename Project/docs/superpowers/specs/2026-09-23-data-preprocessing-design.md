# 多模态情感数据预处理系统设计

## 目标

在 `Project` 目录中实现一套可复现、可测试的 Python 数据预处理流水线，统一处理附件2、附件3和附件4的对齐版特征。系统输出可直接供后续鲁棒多模态情感预测模型使用的数据文件、掩码、训练集标准化参数和审计报告。

交付物必须是可安装、可通过模块或命令行直接运行的 Python 工程，不交付一次性脚本、伪代码或只能复制粘贴的代码片段。

第一阶段只实现 `aligned_50` 主线，不把未对齐版混入同一输入接口。未对齐版需要独立解决 `vision_lengths` 异常和独立时间轴问题，应作为后续扩展。

## 成功标准

1. 附件2的 `train`、`valid`、`test` 保持原划分和原顺序，不发生样本泄漏。
2. 分类标签输出为 `int64`，回归标签输出为 `float32`，类别映射保持 0 负向、1 中性、2 正向。
3. 附件2、附件3和附件4的 `text_bert` 均通过同一个冻结 BERT 检查点生成 768 维文本表示。
4. 文本、语音、视觉分别输出填充掩码、可用性掩码、质量掩码和三者乘积形成的有效掩码。
5. 音频和视觉的均值、标准差只从附件2训练集有效位置估计；验证集、测试集和专项集只能复用这些统计量。
6. 标准化后所有无效位置重新置零；输出中不包含 NaN 或 Inf。
7. 源文件保持只读，所有结果写入用户指定的输出目录。
8. 每次运行生成配置、统计量、异常计数和输入文件指纹，能够复现实验。

## 范围

### 包含

- 附件2 `aligned_50.pkl` 的训练、验证、测试划分；
- 附件3对齐版本目录中的30个无标签缺失样本；
- 附件4对齐版本目录中的20个无标签解释样本；
- 统一BERT文本编码；
- 标签类型转换；
- 三类掩码构造；
- 训练集拟合的音频、视觉标准化；
- 训练集拟合的逐维极端值截尾边界；
- 数据审计报告和命令行入口；
- 单元测试和小型端到端测试。

### 不包含

- 原始视频的音频、视觉特征重新提取；
- 未对齐版特征处理；
- 人工连续缺失数据增强；
- 模型训练、预测和解释算法；
- 从附件3或附件4反向推断标签。

## 输入约定

### 附件2

顶层键为 `train`、`valid`、`test`。每个划分必须包含：

- `id`
- `raw_text`
- `text_bert`
- `audio`
- `vision`
- `classification_labels`
- `regression_labels`

允许存在原 `text` 字段，但主流水线不依赖该字段。

### 附件3

每个文件顶层包含 `test`，其下必须包含：

- `text_bert`
- `audio`
- `vision`

样本ID取文件名，不假设文件内部存在 `id`。附件3的 `text_bert` 即使以浮点类型保存，也必须先验证值为整数，再转换为整数类型。

### 附件4

每个文件必须包含：

- `id`
- `raw_text`
- `text_bert`
- `audio`
- `vision`

允许存在原 `text` 字段，但重新使用统一BERT检查点生成文本特征。

## 输出约定

主输出使用 Pickle 字典，以保留变长元数据和与原题相近的读取方式。每个划分或专项集合包含：

- `id`: 字符串数组；
- `raw_text`: 原始文本数组，不可用时保存空字符串；
- `text`: `float32`，形状为 `(N, 50, 768)`；
- `audio`: `float32`，形状为 `(N, 50, 74)`；
- `vision`: `float32`，形状为 `(N, 50, 35)`；
- `text_padding_mask`、`audio_padding_mask`、`vision_padding_mask`: 布尔数组；
- `text_availability_mask`、`audio_availability_mask`、`vision_availability_mask`: 布尔数组；
- `text_quality_mask`、`audio_quality_mask`、`vision_quality_mask`: 布尔数组；
- `text_effective_mask`、`audio_effective_mask`、`vision_effective_mask`: 布尔数组；
- 有标签数据额外包含 `classification_labels` 和 `regression_labels`；
- `metadata`: 数据角色、BERT模型标识、源文件指纹和处理版本。

同时输出：

- `normalization_stats.npz`: 音频、视觉的均值、标准差和截尾边界；
- `preprocessing_report.json`: 样本数、字段形状、无效行、全零模态、截断、NaN/Inf修复和标签分布；
- `run_config.json`: 本次运行的全部参数和输入路径。

## 掩码定义

设模态为 `m`，时间位置为 `t`：

\[
M_{m,t}=P_{m,t}A_{m,t}Q_{m,t}.
\]

### 填充掩码 P

- 文本：直接来自 `text_bert` 的 attention mask，但其最后一个有效位置用于确定真实序列边界；如果有效区间内部出现0，则该位置仍属于真实序列，只是不可用。
- 音频和视觉：与对齐文本共用时间轴。`[CLS]` 和 `[SEP]` 对应位置不包含真实音视频信号，因此音视频真实区间为文本有效边界内部的 `1` 到 `L-2`。
- 尾部固定长度填充位置为0。

### 可用性掩码 A

- 附件2和附件4是名义完整数据：真实区间内默认可用，局部全零特征不归入缺失，而归入质量问题。
- 附件3是缺失专项数据：真实区间内的全零音频或视觉行视为不可用，即 `A=0`。
- 文本位置在真实边界内但 attention mask 为0时视为文本不可用。

### 质量掩码 Q

- 任一特征行包含 NaN 或 Inf 时，质量为0，并把该行替换为零。
- 附件2和附件4中，真实区间内的全零音频或视觉行质量为0。
- 附件3中，内部全零行优先解释为题目给出的可用性缺失；只有非有限值等独立质量异常使 `Q=0`。
- BERT编码输出包含非有限值时文本质量为0并中止该批次输出，避免静默生成损坏结果。

## 文本编码

文本编码器通过接口注入：

```python
class TextEncoder(Protocol):
    def encode(self, text_bert: np.ndarray, batch_size: int) -> np.ndarray: ...
```

正式实现使用 Hugging Face `AutoModel` 加载冻结检查点，默认模型标识为 `bert-base-uncased`。输入为 token ID、attention mask、segment ID，输出最后隐藏层 `(N, 50, 768)`。模型处于 `eval` 模式，参数不求梯度。模型名称、解析后的本地路径或revision、transformers版本和torch版本写入运行元数据。

测试使用确定性的假编码器，不访问网络，也不依赖本地模型缓存。

## 标准化与截尾

仅在附件2训练集的有效位置上，为音频和视觉逐特征维度计算：

- 均值 `mean`；
- 标准差 `std`；
- 下界 `mean - 5 * std`；
- 上界 `mean + 5 * std`。

标准差小于 `1e-8` 的维度使用1作为除数，避免除零。变换顺序为：

1. 依据质量掩码处理非有限值；
2. 对有效位置截尾；
3. 使用训练集均值和标准差标准化；
4. 将所有无效位置重新置零；
5. 转为 `float32`。

验证集、测试集和专项集不得重新估计统计量。

## 模块划分

```text
Project/
  pyproject.toml
  README.md
  src/emotion_preprocessing/
    __init__.py
    schemas.py          输入输出数据结构与形状验证
    io.py               可信Pickle读取、原子写入、文件指纹
    masks.py            三类掩码构造
    normalization.py    训练集统计量拟合与变换
    text_encoder.py     冻结BERT编码接口与实现
    pipeline.py         附件2、3、4流程编排
    reporting.py        JSON审计报告
    cli.py              命令行入口
  tests/
    test_schemas.py
    test_masks.py
    test_normalization.py
    test_text_encoder.py
    test_pipeline.py
    test_cli.py
```

## 命令行接口

提供三个子命令：

```text
emotion-preprocess fit-transform --input aligned_50.pkl --output-dir outputs/attachment2
emotion-preprocess transform-missing --input-dir 附件3/对齐版本 --stats outputs/attachment2/normalization_stats.npz --output-dir outputs/attachment3
emotion-preprocess transform-explain --input-dir 附件4/对齐版本 --stats outputs/attachment2/normalization_stats.npz --output-dir outputs/attachment4
```

共同参数包括BERT模型标识、设备、批大小、随机种子和是否只允许本地模型文件。专项集命令必须读取已有训练集统计量，不允许自行拟合。

## 错误处理

- 缺少必需字段、形状不符、样本数不一致时直接失败并说明字段和实际形状；
- token ID为浮点型但存在非整数值时直接失败；
- 标签不在 `{0,1,2}` 或回归标签超出 `[-3,3]` 时直接失败；
- 输出目录已存在同名最终文件时，除非显式指定覆盖，否则拒绝覆盖；
- 所有输出先写临时文件，再原子替换，避免中途失败留下半成品；
- Pickle只能用于题目提供的可信文件，README中明确提示不要读取未知来源Pickle。

## 测试策略

### 单元测试

- attention mask包含内部空洞时，填充边界和可用性必须区分；
- 音视频的 `[CLS]`、`[SEP]` 和尾部填充位置必须被排除；
- 同一全零视觉行在附件2角色下形成质量失败，在附件3角色下形成可用性缺失；
- NaN和Inf触发质量失败并在输出中被置零；
- 统计量只使用训练集有效行；
- 标准化后无效位置仍为严格零；
- 零方差特征不会产生除零；
- 浮点token ID只有在全部为整数值时才能安全转换；
- 标签类型、范围和类别映射得到验证。

### 端到端测试

构造小型合成附件2、附件3、附件4数据，使用假文本编码器运行三个子命令，验证：

- 输出形状和dtype；
- 统计量被专项集复用；
- 输入文件没有被修改；
- 报告中的计数与合成异常一致；
- 固定输入得到确定性输出；
- 中途验证失败时不产生最终输出文件。

### 真实数据冒烟测试

在不执行完整BERT编码的情况下读取真实数据并运行结构审计；随后对少量样本使用真实BERT编码器验证接口。完整运行作为最终验收，不把大型输出提交到源码目录。

## 可复现性和存储约束

- 固定 NumPy、PyTorch 随机种子；
- BERT在推理模式运行；
- 输出记录Python、NumPy、PyTorch和Transformers版本；
- 记录输入文件SHA-256；
- 不在源码仓库保存BERT权重和大体积预处理输出；
- BERT权重通过确定的模型标识和revision获取，以满足竞赛附件大小限制；
- README给出从附件2拟合统计量，再处理附件3和附件4的固定运行顺序。

## 验收命令

实现完成后至少执行：

```text
pytest -q
python -m emotion_preprocessing.cli --help
python -m emotion_preprocessing.cli audit --input <aligned_50.pkl>
```

真实BERT和完整数据处理只有在检查点可用且用户允许所需依赖或模型下载时执行；缺少模型缓存时，程序必须给出明确错误，不得自动换用其他检查点。
