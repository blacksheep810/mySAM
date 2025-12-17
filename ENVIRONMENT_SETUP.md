# 环境配置指南

本文档说明如何在新的服务器上配置 mySAM 项目的运行环境。

## 目录

- [系统要求](#系统要求)
- [Conda 环境配置](#conda-环境配置)
- [项目依赖安装](#项目依赖安装)
- [外部依赖配置](#外部依赖配置)
- [数据准备](#数据准备)
- [模型检查点准备](#模型检查点准备)
- [验证安装](#验证安装)
- [常见问题](#常见问题)

---

## 系统要求

### 硬件要求

- **GPU**: NVIDIA GPU（推荐 A30 或更高，至少 24GB 显存）
- **内存**: 建议 64GB 或更多
- **存储**: 至少 100GB 可用空间（用于数据集和模型检查点）

### 软件要求

- **操作系统**: Linux (Ubuntu 20.04+ 推荐)
- **CUDA**: 11.8 或更高版本
- **Python**: 3.10.x
- **Conda**: Miniconda 或 Anaconda

### 当前环境信息

- **Conda 环境名**: `py310env`
- **Python 版本**: 3.10.16
- **PyTorch 版本**: 2.7.0+cu118
- **CUDA 版本**: 11.8
- **GPU**: NVIDIA A30 (24GB)

---

## Conda 环境配置

### 1. 安装 Miniconda/Anaconda

如果服务器上还没有安装 Conda，请先安装：

```bash
# 下载 Miniconda（推荐）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 重新加载 shell 配置
source ~/.bashrc
```

### 2. 创建 Conda 环境

```bash
# 创建 Python 3.10 环境
conda create -n py310env python=3.10 -y

# 激活环境
conda activate py310env
```

### 3. 安装 CUDA Toolkit

```bash
# 在激活的环境中安装 CUDA 11.8
conda install -c conda-forge cudatoolkit=11.8 -y
```

---

## 项目依赖安装

### 1. 克隆项目仓库

```bash
# 克隆项目（如果还没有）
cd /root/workspace
git clone <repository-url> mySAM
cd mySAM

# 或者如果已经克隆，拉取最新代码
git pull origin main
```

### 2. 安装 PyTorch（CUDA 11.8 版本）

```bash
# 激活环境
conda activate py310env

# 安装 PyTorch 2.7.0 with CUDA 11.8
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
```

### 3. 安装 Segment Anything Model (SAM)

项目依赖 Facebook Research 的 segment-anything 库，需要从源码安装：

```bash
# 克隆 segment-anything 仓库（如果还没有）
cd /root/workspace
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything

# 安装依赖
pip install -e .

# 返回项目目录
cd /root/workspace/mySAM
```

**注意**: segment-anything 需要安装在 `/root/workspace/segment-anything` 目录，项目通过 `-e git+https://github.com/facebookresearch/segment-anything@dca509fe793f601edb92606367a655c15ac00fdf#egg=segment_anything` 方式安装。

### 4. 安装其他核心依赖

```bash
# 激活环境
conda activate py310env

# 核心深度学习库
pip install pytorch-lightning==2.5.1.post0
pip install torchmetrics==1.7.2
pip install torchsummary==1.5.1

# 图像处理
pip install opencv-python==4.7.0.72
pip install opencv-python-headless==4.11.0.86
pip install albumentations==1.3.1
pip install Pillow

# 分割模型相关
pip install segmentation-models-pytorch==0.3.2
pip install efficientnet-pytorch==0.7.1
pip install fft-conv-pytorch==1.2.0

# 数据处理
pip install numpy==1.23.5
pip install pandas
pip install scikit-learn
pip install scikit-image

# 工具库
pip install tqdm
pip install yacs==0.1.8
pip install tensorboard
pip install wandb  # 可选：用于实验跟踪

# 其他依赖
pip install batchgenerators==0.25.1
pip install batchgeneratorsv2==0.2.3
pip install connected-components-3d==3.23.0
pip install acvl-utils==0.2.5
```

### 5. 完整依赖列表（可选）

如果需要完全复现环境，可以使用以下命令导出并安装：

```bash
# 导出当前环境的完整依赖（在旧服务器上）
conda activate py310env
conda list --export > environment_conda.txt
pip freeze > requirements.txt

# 在新服务器上安装（注意：可能需要根据实际情况调整）
conda activate py310env
# 安装 conda 包（如果有）
conda install --file environment_conda.txt -y
# 安装 pip 包
pip install -r requirements.txt
```

---

## 外部依赖配置

### 1. Segment Anything 模型检查点

项目需要 SAM 预训练模型检查点，请下载并放置在正确位置：

```bash
# 创建检查点目录
mkdir -p /root/workspace/mySAM/checkpoints

# 下载 SAM 模型检查点（根据需要选择）
# SAM ViT-H (Huge) - 推荐用于训练
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
     -O /root/workspace/mySAM/checkpoints/sam_vit_h_4b8939.pth

# SAM ViT-L (Large) - 可选
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth \
     -O /root/workspace/mySAM/checkpoints/sam_vit_l_0b3195.pth

# SAM ViT-B (Base) - 可选
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
     -O /root/workspace/mySAM/checkpoints/sam_vit_b_01ec64.pth
```

### 2. Git LFS 配置（用于大文件管理）

项目使用 Git LFS 管理大型模型文件：

```bash
# 安装 Git LFS（如果还没有）
git lfs install

# 验证 Git LFS 配置
git lfs track
```

`.gitattributes` 文件已配置跟踪以下文件类型：
- `*.pth`, `*.pt` - PyTorch 模型文件
- `*.bin`, `*.safetensors` - 其他模型格式
- `*.npy`, `*.npz` - NumPy 数组文件

---

## 数据准备

### 1. ISIC 数据集结构

项目使用 ISIC 数据集，需要按以下结构组织：

```
data/
└── ISIC/
    ├── train_boxes.csv      # 训练集边界框标注
    ├── test_boxes.csv       # 测试集边界框标注（可选）
    └── images/              # 图像文件目录
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

### 2. 数据集路径配置

在训练脚本中指定数据路径：

```bash
--data_root ./data/ISIC
--train_box_csv ./data/ISIC/train_boxes.csv
--test_box_csv ./data/ISIC/test_boxes.csv
```

---

## 模型检查点准备

### 1. 预训练模型

确保 SAM 检查点已下载（见[外部依赖配置](#外部依赖配置)）

### 2. 训练输出目录

项目会在以下目录保存训练输出：

- `output_base/` - Base 模型训练输出
- `outputs_huge/` - Huge 模型训练输出
- `checkpoints/` - 模型检查点目录

这些目录会在首次训练时自动创建。

---

## 验证安装

### 1. 验证 Python 和 Conda 环境

```bash
conda activate py310env
python --version  # 应该显示 Python 3.10.x
which python       # 应该指向 conda 环境中的 python
```

### 2. 验证 PyTorch 和 CUDA

```bash
conda activate py310env
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

预期输出：
```
PyTorch: 2.7.0+cu118
CUDA available: True
CUDA version: 11.8
```

### 3. 验证 GPU

```bash
nvidia-smi
```

应该显示 GPU 信息，包括：
- GPU 型号（如 NVIDIA A30）
- CUDA 版本
- 显存大小

### 4. 验证项目依赖

```bash
conda activate py310env
cd /root/workspace/mySAM

# 测试导入核心模块
python -c "from segment_anything import sam_model_registry; print('SAM imported successfully')"
python -c "import torch; import torchvision; import cv2; import numpy as np; print('Core dependencies OK')"
```

### 5. 验证项目结构

```bash
cd /root/workspace/mySAM

# 检查关键目录和文件
ls -la config/          # 应该包含 args.py
ls -la models/          # 应该包含模型定义
ls -la training/        # 应该包含训练器
ls -la utils/           # 应该包含工具函数
ls -la checkpoints/     # 应该包含 SAM 检查点
```

### 6. 运行简单测试（可选）

```bash
conda activate py310env
cd /root/workspace/mySAM

# 检查训练脚本是否可以正常解析参数
python train.py --help
```

---

## 快速安装脚本

以下是一个快速安装脚本，可以自动化大部分配置过程：

```bash
#!/bin/bash
# 快速环境配置脚本

set -e

echo "=== 1. 创建 Conda 环境 ==="
conda create -n py310env python=3.10 -y
conda activate py310env

echo "=== 2. 安装 CUDA Toolkit ==="
conda install -c conda-forge cudatoolkit=11.8 -y

echo "=== 3. 安装 PyTorch ==="
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118

echo "=== 4. 安装 Segment Anything ==="
cd /root/workspace
if [ ! -d "segment-anything" ]; then
    git clone https://github.com/facebookresearch/segment-anything.git
fi
cd segment-anything
pip install -e .
cd /root/workspace/mySAM

echo "=== 5. 安装项目依赖 ==="
pip install pytorch-lightning==2.5.1.post0 torchmetrics==1.7.2 torchsummary==1.5.1
pip install opencv-python==4.7.0.72 opencv-python-headless==4.11.0.86 albumentations==1.3.1
pip install segmentation-models-pytorch==0.3.2 efficientnet-pytorch==0.7.1 fft-conv-pytorch==1.2.0
pip install numpy==1.23.5 pandas scikit-learn scikit-image
pip install tqdm yacs==0.1.8 tensorboard
pip install batchgenerators==0.25.1 batchgeneratorsv2==0.2.3 connected-components-3d==3.23.0 acvl-utils==0.2.5

echo "=== 6. 下载 SAM 检查点 ==="
mkdir -p checkpoints
if [ ! -f "checkpoints/sam_vit_h_4b8939.pth" ]; then
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
         -O checkpoints/sam_vit_h_4b8939.pth
fi

echo "=== 7. 配置 Git LFS ==="
git lfs install

echo "=== 环境配置完成 ==="
echo "请运行以下命令验证安装："
echo "  conda activate py310env"
echo "  python -c \"import torch; print(torch.cuda.is_available())\""
```

---

## 常见问题

### 问题 1: CUDA 不可用

**症状**: `torch.cuda.is_available()` 返回 `False`

**解决方案**:
1. 检查 NVIDIA 驱动: `nvidia-smi`
2. 确认 CUDA 版本匹配: PyTorch 需要 CUDA 11.8
3. 重新安装匹配的 PyTorch 版本

### 问题 2: 导入 segment_anything 失败

**症状**: `ImportError: No module named 'segment_anything'`

**解决方案**:
1. 确认 segment-anything 已安装在 `/root/workspace/segment-anything`
2. 使用 `pip install -e /root/workspace/segment-anything` 重新安装
3. 检查 Python 路径: `python -c "import sys; print(sys.path)"`

### 问题 3: 显存不足 (OOM)

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
1. 减小 batch size: `--batch_size 1`
2. 使用梯度累积: `--gradient_accumulation_steps 4`
3. 启用梯度检查点: `--use_gradient_checkpointing`
4. 使用混合精度训练: `--use_amp`

### 问题 4: 依赖版本冲突

**症状**: 包版本不兼容错误

**解决方案**:
1. 创建新的干净环境
2. 按照文档顺序安装依赖
3. 如果仍有问题，检查具体包的兼容性要求

### 问题 5: Git LFS 文件未下载

**症状**: 大文件显示为指针文件而非实际文件

**解决方案**:
```bash
git lfs install
git lfs pull
```

### 问题 6: 数据路径错误

**症状**: `FileNotFoundError` 或数据集加载失败

**解决方案**:
1. 检查数据目录结构是否正确
2. 确认 CSV 文件路径正确
3. 检查文件权限

---

## 环境变量配置（可选）

如果需要设置特定的环境变量：

```bash
# 添加到 ~/.bashrc 或环境激活脚本中
export CUDA_VISIBLE_DEVICES=0  # 指定使用的 GPU
export OMP_NUM_THREADS=4        # OpenMP 线程数
export PYTHONPATH=/root/workspace/mySAM:$PYTHONPATH
```

---

## 维护和更新

### 更新依赖

```bash
conda activate py310env
pip install --upgrade <package-name>
```

### 备份环境

```bash
conda activate py310env
conda env export > environment_backup.yml
pip freeze > requirements_backup.txt
```

### 恢复环境

```bash
conda env create -f environment_backup.yml
conda activate py310env
pip install -r requirements_backup.txt
```

---

## 联系和支持

如果遇到配置问题，请检查：
1. 本文档的[常见问题](#常见问题)部分
2. 项目的其他文档（`CODE_READING_GUIDE.md`, `MODULE_DESIGN.md`）
3. Git 仓库的 Issues

---

**最后更新**: 2025-12-17  
**环境版本**: Python 3.10.16, PyTorch 2.7.0+cu118, CUDA 11.8

