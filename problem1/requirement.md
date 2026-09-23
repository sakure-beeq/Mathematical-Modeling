# problem1 运行依赖与 macOS 安装文档

本文档适用于 `problem1` 多模态特征提取代码，目标 Conda 环境名为 `MathematicalModeling`。以下命令使用当前机器已有的 `MathematicalModeling` 环境。

## 1. 依赖清单

### 1.1 Python 直接依赖

| 名称 | 最低版本 | 用途 | 是否必需 |
|---|---:|---|---|
| Python | 3.10；推荐3.11 | 运行环境 | 是 |
| NumPy | 1.24 | 张量、掩码和加权池化 | 是 |
| SciPy | 1.10 | librosa及数值处理依赖 | 是 |
| OpenPyXL | 3.1 | 读取 `label-100.xlsx` | 是 |
| librosa | 0.10.1 | 提取74维声学特征 | 是 |
| SoundFile | 0.12 | 音频文件读取后端 | 是 |
| PyTorch | 2.1 | 运行冻结BERT | 是 |
| Transformers | 4.38 | 加载BERT模型和Tokenizer | 是 |
| Hugging Face Hub | 最新稳定版 | 下载固定的BERT模型目录 | 建议 |
| Matplotlib | 3.7 | 绘制典型样本时间轴图 | 绘图时必需 |
| pytest | 7.4 | 开发测试；项目测试也可用unittest | 建议 |
| setuptools | 68 | 安装本地Python包 | 是 |
| wheel | 最新稳定版 | 安装二进制Python包 | 建议 |
| seaborn | 0.13 | MFA 3.4.2 的 Python 依赖 | 是 |

`pip` 或 Conda 会自动安装上述库的传递依赖，例如 `numba`、`llvmlite`、`scikit-learn`、`audioread`、`soxr`、`cffi`、`tokenizers`、`safetensors`、`requests`、`filelock`、`fsspec`、`Jinja2`、`Pillow`、`fonttools`、`python-dateutil` 等，不需要逐个手工安装。

### 1.2 外部命令行工具

| 名称 | 用途 | 安装来源 |
|---|---|---|
| FFmpeg | 从MP4提取16 kHz单声道WAV | Conda Forge或Homebrew |
| FFprobe | 读取视频总时长；随FFmpeg提供 | Conda Forge或Homebrew |
| Montreal Forced Aligner（MFA） | 标签英文转写到词级时间区间 | Conda Forge |
| `english_us_arpa` dictionary | MFA英文发音词典 | MFA模型仓库 |
| `english_us_arpa` acoustic | MFA英文声学模型 | MFA模型仓库 |
| OpenFace `FeatureExtraction` | AU、头姿、眼球和注视特征 | macOS源码编译 |

### 1.3 OpenFace在macOS上的构建依赖

- Xcode Command Line Tools；
- Homebrew；
- Git、CMake、wget；
- C++17编译器（Apple Clang；Homebrew GCC作为备用）；
- Boost、TBB、OpenBLAS、dlib、OpenCV。

## 2. 核验并激活Conda环境

先确认当前终端调用的是预期的Conda：

```bash
which conda
conda info --envs
```

环境列表中应存在 `MathematicalModeling`。然后激活：

```bash
conda activate "MathematicalModeling"
python --version
which python
echo "$CONDA_PREFIX"
```

建议使用Python 3.11。如果环境确实存在但不是3.11，可执行：

```bash
conda install -n "MathematicalModeling" -c conda-forge python=3.11 -y
conda activate "MathematicalModeling"
```

如果 `conda info --envs` 中没有该环境，说明它可能创建在另一套Conda中，或尚未真正创建。确认后可用下面的命令创建：

```bash
conda create -n "MathematicalModeling" -c conda-forge python=3.11 -y
conda activate "MathematicalModeling"
```

## 3. 安装FFmpeg与MFA

优先在同一个Conda环境中安装，避免系统路径混乱：

```bash
conda activate "MathematicalModeling"
conda install -c conda-forge ffmpeg montreal-forced-aligner kalpy -y
```

验证：

```bash
ffmpeg -version
ffprobe -version
mfa version
```

下载问题1使用的英文词典和声学模型：

```bash
mfa model download dictionary english_us_arpa
mfa model download acoustic english_us_arpa
mfa model list dictionary
mfa model list acoustic
```

MFA官方推荐通过Conda Forge安装；如需更新，可执行：

```bash
conda update -c conda-forge montreal-forced-aligner kalpy --update-deps
```

如果不希望通过Conda安装FFmpeg，也可使用Homebrew：

```bash
brew install ffmpeg
```

## 4. 安装Python依赖和本项目

进入本目录后，通过 `pyproject.toml` 一次安装所有直接Python依赖：

```bash
conda activate "MathematicalModeling"
cd "/Users/wangyiming/Desktop/E题/problem1"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[extract,test]'
python -m pip install --upgrade huggingface_hub
```

其中 `.[extract,test]` 会安装：

```text
numpy
scipy
openpyxl
librosa
soundfile
torch
transformers
matplotlib
pytest
```

PyTorch官方在macOS上使用pip安装预编译包，不需要安装CUDA。Apple Silicon可使用MPS，但本项目默认CPU即可运行。

## 5. 下载固定的BERT模型

建议把模型下载到项目本地目录，运行时传入固定路径，避免每次运行取到不同修订版本：

```bash
conda activate "MathematicalModeling"
cd "/Users/wangyiming/Desktop/E题/problem1"
mkdir -p models/bert-base-uncased
hf download google-bert/bert-base-uncased --local-dir models/bert-base-uncased
```

验证模型文件：

```bash
ls models/bert-base-uncased/config.json
ls models/bert-base-uncased/tokenizer.json
```

如需完全固定模型版本，可在 `hf download` 后增加 `--revision COMMIT_HASH`，并在实验记录中保存该哈希。

## 6. 在macOS上安装OpenFace

OpenFace没有适合本项目的官方macOS Python轮子，必须获得可执行文件 `FeatureExtraction`。以下步骤参考OpenFace官方macOS构建说明。

### 6.1 Xcode命令行工具

```bash
xcode-select --install
sudo xcodebuild -license accept
```

如果第一条提示已安装，可以继续。当前电脑曾出现“尚未同意Xcode许可协议”的提示，因此第二条不能省略。

### 6.2 安装Homebrew

已有Homebrew时跳过本步。验证：

```bash
brew --version
```

未安装时，使用Homebrew官网安装命令：

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

安装程序结束时会打印一至两条用于配置 `PATH` 的命令，请按它给出的内容执行，然后重新打开终端。

### 6.3 安装OpenFace构建库

```bash
brew update
brew install git cmake wget gcc boost tbb openblas dlib opencv@4
```

### 6.4 下载、编译和下载模型

当前机器已编译的可执行文件位于 `models/openface/FeatureExtraction`（指向本目录中的构建产物）。本次使用 OpenFace 官方提交 `3d4b5cf8d96138be42bed229447f36cbb09a5a29` 的 CLNF 和 MTCNN 模型；运行时用 `--openface-model` 指定 `models/openface/model/main_clnf_general.txt`。若另行从源码构建，应使用 OpenCV 4，并让 CMake 找到 Homebrew 的 OpenBLAS：

```bash
conda activate "MathematicalModeling"
mkdir -p "$CONDA_PREFIX/opt"
cd "$CONDA_PREFIX/opt"
git clone https://github.com/TadasBaltrusaitis/OpenFace.git
cd OpenFace
git checkout 3d4b5cf8d96138be42bed229447f36cbb09a5a29
OpenBLAS_HOME="$(brew --prefix openblas)" cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(brew --prefix)" -DOpenCV_DIR="$(brew --prefix opencv@4)/lib/cmake/opencv4"
cmake --build build --target FeatureExtraction --config Release --parallel
cd "/Users/wangyiming/Desktop/E题/problem1"
ln -s "$CONDA_PREFIX/opt/OpenFace/build/bin" models/openface
```

上面的 `--openface-model` 使用仓库自带的 CLNF 模型。`download_models.sh` 下载的是默认 CEN 模型，本配置无需运行。

验证可执行文件：

```bash
test -x models/openface/FeatureExtraction
models/openface/FeatureExtraction -help
```

不同OpenFace版本的产物也可能位于 `build/bin/FeatureExtraction` 之外。若上面的 `test` 失败，可定位它：

```bash
find "$CONDA_PREFIX/opt/OpenFace/build" -type f -name FeatureExtraction
```

Apple Silicon上若OpenBLAS与TBB线程冲突，运行特征提取前可设置：

```bash
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

## 7. 完整安装验证

### 7.1 Python库

```bash
conda activate "MathematicalModeling"
python -m pip check
python -c 'import numpy, scipy, openpyxl, librosa, soundfile, torch, transformers, matplotlib, huggingface_hub; print("Python dependencies: OK"); print("torch", torch.__version__); print("MPS available", torch.backends.mps.is_available())'
```

### 7.2 外部工具

```bash
command -v ffmpeg
command -v ffprobe
command -v mfa
test -x models/openface/FeatureExtraction
test -f models/openface/model/main_clnf_general.txt
mfa version
```

### 7.3 项目测试

```bash
cd "/Users/wangyiming/Desktop/E题/problem1"
python -m unittest discover -s tests -v
problem1 --help
```

应看到8项测试全部通过。

2026-09-23 本机核验：8项测试通过，`pip check` 无损坏依赖；100条样本完整运行成功，输出三模态张量尺寸分别为 `(100,50,768)`、`(100,50,74)`、`(100,50,35)`。另用单条 WAV 和转写独立运行 MFA，成功生成 15 词的 TextGrid。

## 8. 运行问题1代码

```bash
conda activate "MathematicalModeling"
cd "/Users/wangyiming/Desktop/E题/problem1"
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

problem1 run \
  --labels '../E题数据/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条/label-100.xlsx' \
  --videos-root '../E题数据/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条' \
  --work-dir work \
  --output outputs/problem1_features.pkl \
  --bert-model models/bert-base-uncased \
  --openface models/openface/FeatureExtraction \
  --openface-model models/openface/model/main_clnf_general.txt
```

本机 `MathematicalModeling` 环境的 `librosa` 首次运行需要可写的 Numba 缓存。代码会自动使用系统临时目录；如想固定位置，可在运行前设置 `NUMBA_CACHE_DIR`。单样本环境验证已生成 `outputs/smoke.pkl`，完整运行结果写入 `outputs/problem1_features.pkl`。

### 8.1 交付完整性与独立核验

`problem1 run` 会同时生成 `problem1_features.manifest.csv`、`problem1_features.audit.json`、`problem1_features.audit.jsonl` 和 `problem1_features.reproduce.md`。清单每行给出Excel行号、MP4/WAV/TextGrid/OpenFace CSV路径及SHA-256、特征摘要、有效长度与对齐方法。pickle内的 `source_word_timeline` 和 `source_word_span` 可逐位置核对原始词时间区间；右侧填充使用 `[-1,-1]` 词索引、零特征与 `padding_mask=True`。

```bash
problem1 audit \
  --features outputs/problem1_features.pkl \
  --labels '../E题数据/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条/label-100.xlsx' \
  --videos-root '../E题数据/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条' \
  --work-dir work --verify-only
```

此命令只读核验；要给旧版pickle补齐审计字段，去掉 `--verify-only`。本次审计的原始运行日志另存为 `outputs/problem1_features.processing.log`。初次提取时未逐文件记录哈希，清单与模型哈希是在审计时采集，这一点在复现说明中明确标注。

生成典型样本图：

```bash
problem1 plot \
  --features outputs/problem1_features.pkl \
  --sample-index 0 \
  --output outputs/sample_000_timeline.png
```

## 9. 常见macOS问题

### `You have not agreed to the Xcode license agreements`

```bash
sudo xcodebuild -license accept
```

### `mfa: command not found`

确认已激活环境，并检查：

```bash
conda activate "MathematicalModeling"
echo "$CONDA_PREFIX"
conda list montreal-forced-aligner
```

### `FeatureExtraction` 找不到模型

确认 `models/openface/model/main_clnf_general.txt`、`models/openface/model/mtcnn_detector/` 和 `models/openface/AU_predictors/` 均存在，并向 `--openface` 传入可执行文件路径、向 `--openface-model` 传入 CLNF 主模型路径。

### Apple Silicon上MPS报不支持的算子

本项目可以直接使用默认的 `--device cpu`。只有明确希望使用MPS时才传 `--device mps`；若部分算子不支持，可先设置：

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## 10. 官方资料

- [MFA安装文档](https://montreal-forced-aligner.readthedocs.io/en/stable/installation.html)
- [PyTorch macOS安装](https://docs.pytorch.org/get-started/locally/)
- [Hugging Face模型下载](https://huggingface.co/docs/huggingface_hub/guides/download)
- [librosa安装](https://librosa.org/doc/main/install.html)
- [Matplotlib安装](https://matplotlib.org/stable/install/index.html)
- [FFmpeg Homebrew公式](https://formulae.brew.sh/formula/ffmpeg)
- [OpenFace macOS构建说明](https://github.com/TadasBaltrusaitis/OpenFace/wiki/mac-installation)
- [OpenFace模型下载说明](https://github.com/TadasBaltrusaitis/OpenFace/wiki/model-download)
