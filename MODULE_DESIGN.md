# 模块化设计文档

本文档详细说明了 mySAM 项目的模块化架构设计，包括每个模块的职责、功能和接口。

## ? 目录结构

```
mySAM/
├── model.py                    # 主入口文件（简化后）
├── config/
│   ├── __init__.py
│   └── args.py                # 参数解析和配置管理
├── models/
│   ├── __init__.py
│   ├── losses.py              # 损失函数模块
│   ├── projection.py          # 投影头模块
│   └── sam_wrapper.py         # SAM 模型封装
├── utils/
│   ├── __init__.py
│   ├── metrics.py             # 评估指标模块
│   ├── prompts.py             # Prompt 处理模块
│   └── training_utils.py      # 训练工具函数
└── training/
    ├── __init__.py
    ├── trainer.py             # 训练主逻辑
    └── data_utils.py          # 数据工具函数
```

---

## ? 模块详细说明

### 1. `model.py` - 主入口文件

**职责**：
- 作为程序的入口点
- 解析命令行参数
- 初始化训练器并启动训练

**主要功能**：
- 导入所有必要的模块
- 调用参数解析器
- 创建 Trainer 实例并启动训练

**代码结构**：
```python
from config.args import build_argparser
from training.trainer import Trainer

if __name__ == '__main__':
    parser = build_argparser()
    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.train()
```

**依赖关系**：
- `config.args` - 参数解析
- `training.trainer` - 训练逻辑

---

### 2. `config/args.py` - 参数配置模块

**职责**：
- 定义所有命令行参数
- 提供参数解析功能
- 参数验证和默认值设置

**主要功能**：
- `build_argparser()` - 构建参数解析器
  - 数据相关参数（data_root, train_box_csv等）
  - 模型相关参数（sam_checkpoint, model_type等）
  - 训练相关参数（batch_size, epochs, lr等）
  - 优化相关参数（use_amp, gradient_checkpointing等）
  - GPU相关参数（device, multi_gpu, gpu_ids等）

**参数分类**：

1. **数据参数**：
   - `--data_root`: 数据集根目录
   - `--train_box_csv`: 训练集 box CSV 文件
   - `--test_box_csv`: 测试集 box CSV 文件（可选）
   - `--img_size`: 图像尺寸

2. **模型参数**：
   - `--sam_checkpoint`: SAM checkpoint 路径
   - `--model_type`: 模型类型（vit_b/vit_l/vit_h）
   - `--unfreeze_last_k`: 解冻最后K层

3. **训练参数**：
   - `--batch_size`: 批次大小
   - `--epochs`: 训练轮数
   - `--lr_encoder`: Encoder 学习率
   - `--lr_decoder`: Decoder 学习率
   - `--weight_decay`: 权重衰减

4. **优化参数**：
   - `--use_amp`: 混合精度训练
   - `--use_gradient_checkpointing`: 梯度检查点
   - `--gradient_accumulation_steps`: 梯度累积步数

5. **对比学习参数**：
   - `--proj_dim`: 投影维度
   - `--pos_samples`: 正样本数量
   - `--neg_samples`: 负样本数量
   - `--temperature`: 温度参数
   - `--entropy_thresh`: 熵阈值

6. **GPU参数**：
   - `--device`: 设备（cuda/cpu）
   - `--multi_gpu`: 多GPU训练
   - `--gpu_ids`: 指定GPU ID

7. **输出参数**：
   - `--output_dir`: 输出目录
   - `--save_every`: 保存频率

**依赖关系**：
- 无（独立模块）

---

### 3. `models/losses.py` - 损失函数模块

**职责**：
- 定义所有损失函数
- 实现对比学习损失
- 实现分割损失

**主要功能**：

1. **`DiceLoss` 类**：
   - Dice 损失实现
   - 用于分割任务的监督损失
   - 公式：`1 - (2*intersect + eps) / (union + eps)`

2. **`pixel_info_nce()` 函数**：
   - 像素级 InfoNCE 对比损失
   - 输入：anchors, positives, negatives, temperature
   - 输出：对比损失值
   - 用于对比学习训练

**接口定义**：
```python
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6)
    def forward(self, logits, target)

def pixel_info_nce(anchors, positives, negatives, temperature=0.1)
```

**依赖关系**：
- `torch`, `torch.nn`, `torch.nn.functional`

---

### 4. `models/projection.py` - 投影头模块

**职责**：
- 定义特征投影头
- 将 encoder 特征投影到对比学习空间

**主要功能**：

**`PixelProjHead` 类**：
- 将 image_encoder 输出投影到低维空间
- 输入：`(B, C, H, W)` - encoder 特征
- 输出：`(B, D, H, W)` - 归一化的投影特征
- 结构：Conv2d -> BatchNorm -> GELU -> Conv2d -> L2 Normalize

**接口定义**：
```python
class PixelProjHead(nn.Module):
    def __init__(self, in_dim, proj_dim=64)
    def forward(self, x)  # (B, C, H, W) -> (B, D, H, W)
```

**依赖关系**：
- `torch.nn`, `torch.nn.functional`

---

### 5. `models/sam_wrapper.py` - SAM 模型封装

**职责**：
- 封装 SAM 模型的加载和初始化
- 处理多GPU设置
- 管理 EMA teacher 模型
- 处理模型冻结策略

**主要功能**：

1. **`SAMWrapper` 类**（可选）：
   - 封装 SAM 模型的加载
   - 处理 checkpoint 加载
   - 管理模型设备

2. **`setup_model()` 函数**：
   - 加载 SAM 模型
   - 设置冻结策略
   - 创建 EMA teacher
   - 配置多GPU

3. **`setup_optimizer()` 函数**：
   - 创建优化器
   - 设置学习率
   - 配置参数组

**接口定义**：
```python
def load_sam_model(checkpoint_path, model_type, device, unfreeze_last_k=0)
def create_teacher_model(sam_model, device)
def setup_optimizer(model, lr_encoder, lr_decoder, weight_decay)
def setup_multi_gpu(model, device_ids)
```

**依赖关系**：
- `segment_anything.sam_model_registry`
- `torch`, `torch.nn`

---

### 6. `utils/metrics.py` - 评估指标模块

**职责**：
- 实现各种评估指标
- 计算模型性能指标

**主要功能**：

1. **`compute_miou()` 函数**：
   - 计算 mean Intersection over Union
   - 输入：pred_logits, gt_masks, threshold
   - 输出：平均 IoU 值
   - 用于评估分割性能

2. **`mask_entropy_logits()` 函数**：
   - 计算 mask logits 的熵
   - 用于伪标签质量评估
   - 输入：mask_logits (B, 1, H, W)
   - 输出：平均熵值 (B,)

**接口定义**：
```python
def compute_miou(pred_logits, gt_masks, threshold=0.5) -> float
def mask_entropy_logits(mask_logits) -> torch.Tensor
```

**依赖关系**：
- `torch`, `torch.nn.functional`

---

### 7. `utils/prompts.py` - Prompt 处理模块

**职责**：
- 处理各种 prompt 类型
- 采样点生成
- 框处理工具

**主要功能**：

1. **`sample_points_in_ring()` 函数**：
   - 在大小框之间的环形区域采样点
   - 用于生成点 prompt
   - 输入：small_box, big_box, num_points, img_size
   - 输出：points (N, 2), labels (N,)

2. **`prepare_box_prompts()` 函数**（新增）：
   - 准备框 prompt（大框+小框）
   - 将多个框合并为 tensor
   - 输入：big_box, small_box
   - 输出：boxes_tensor (N, 4)

3. **`prepare_point_prompts()` 函数**（新增）：
   - 准备点 prompt
   - 处理点坐标和标签
   - 输入：points, labels
   - 输出：points_tensor, labels_tensor

**接口定义**：
```python
def sample_points_in_ring(small_box, big_box, num_points=10, img_size=1024)
def prepare_box_prompts(big_box, small_box, device)
def prepare_point_prompts(points, labels, device)
```

**依赖关系**：
- `torch`, `numpy`

---

### 8. `utils/training_utils.py` - 训练工具模块

**职责**：
- 提供训练相关的通用工具函数
- 设置随机种子
- 文件系统操作

**主要功能**：

1. **`set_seed()` 函数**：
   - 设置随机种子
   - 确保实验可复现
   - 设置 Python, NumPy, PyTorch 的随机种子

2. **`mkdir()` 函数**：
   - 创建目录（如果不存在）
   - 支持递归创建

3. **`setup_cuda_memory()` 函数**（新增）：
   - 配置 CUDA 内存分配
   - 清理显存缓存
   - 显示显存状态

**接口定义**：
```python
def set_seed(seed=42)
def mkdir(path)
def setup_cuda_memory(device_id=0)
```

**依赖关系**：
- `torch`, `numpy`, `random`, `pathlib`

---

### 9. `training/data_utils.py` - 数据工具模块

**职责**：
- 数据加载相关工具
- Collate 函数
- 数据预处理

**主要功能**：

1. **`collate_fn_isic()` 函数**：
   - ISIC 数据集的 batch 整理函数
   - 处理图像、框、mask 的批处理
   - 返回格式化的 batch 字典

**接口定义**：
```python
def collate_fn_isic(batch) -> dict:
    """
    返回:
    {
        'image': torch.Tensor,      # (B, 3, H, W)
        'boxes': list,               # list of big_boxes
        'big_boxes': list,           # list of big_boxes
        'small_boxes': list,         # list of small_boxes
        'mask': torch.Tensor,        # (B, 1, H, W)
        'img_names': list            # list of image names
    }
    """
```

**依赖关系**：
- `torch`, `torch.utils.data`

---

### 10. `training/trainer.py` - 训练主逻辑模块

**职责**：
- 实现完整的训练流程
- 管理训练循环
- 处理损失计算
- 管理 checkpoint 保存

**主要功能**：

**`Trainer` 类**：

1. **初始化 (`__init__`)**：
   - 接收配置参数
   - 初始化模型、优化器、数据加载器
   - 设置设备、多GPU

2. **`train()` 方法**：
   - 主训练循环
   - 遍历所有 epoch
   - 调用 `train_epoch()` 进行每个 epoch 的训练

3. **`train_epoch()` 方法**：
   - 单个 epoch 的训练逻辑
   - 遍历数据加载器
   - 前向传播、损失计算、反向传播
   - 更新 EMA teacher

4. **`forward_pass()` 方法**：
   - 前向传播逻辑
   - Teacher 生成伪标签
   - Student 前向传播
   - 计算各种损失

5. **`compute_losses()` 方法**：
   - 计算所有损失
   - Mask loss
   - Contrastive loss
   - IoU loss
   - Distillation loss

6. **`save_checkpoint()` 方法**：
   - 保存模型 checkpoint
   - 保存优化器状态
   - 保存训练进度

**接口定义**：
```python
class Trainer:
    def __init__(self, args)
    def train(self)
    def train_epoch(self, epoch)
    def forward_pass(self, batch)
    def compute_losses(self, preds, targets, features)
    def save_checkpoint(self, epoch)
```

**依赖关系**：
- `models.losses` - 损失函数
- `models.projection` - 投影头
- `models.sam_wrapper` - SAM 模型
- `utils.metrics` - 评估指标
- `utils.prompts` - Prompt 处理
- `utils.training_utils` - 训练工具
- `training.data_utils` - 数据工具

---

## ? 模块依赖关系图

```
model.py
  ├── config/args.py
  └── training/trainer.py
        ├── models/sam_wrapper.py
        │     └── segment_anything (外部)
        ├── models/projection.py
        ├── models/losses.py
        ├── utils/metrics.py
        ├── utils/prompts.py
        ├── utils/training_utils.py
        └── training/data_utils.py
              └── dataset.ISIC (外部)
```

---

## ? 使用示例

### 基本使用

```python
# model.py
from config.args import build_argparser
from training.trainer import Trainer

if __name__ == '__main__':
    parser = build_argparser()
    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.train()
```

### 单独使用某个模块

```python
# 使用损失函数
from models.losses import DiceLoss, pixel_info_nce

dice_loss = DiceLoss()
loss = dice_loss(pred_logits, gt_masks)

# 使用评估指标
from utils.metrics import compute_miou

miou = compute_miou(pred_logits, gt_masks)

# 使用 Prompt 工具
from utils.prompts import prepare_box_prompts

boxes_tensor = prepare_box_prompts(big_box, small_box, device)
```

---

## ? 模块设计原则

1. **单一职责原则**：每个模块只负责一个明确的功能
2. **低耦合高内聚**：模块之间依赖关系清晰，内部功能相关
3. **接口清晰**：每个模块提供明确的接口定义
4. **易于测试**：模块可以独立测试
5. **易于扩展**：新增功能时不影响现有模块

---

## ? 迁移计划

### 阶段1：创建目录结构
- 创建所有必要的目录和 `__init__.py` 文件

### 阶段2：拆分工具模块
- `utils/metrics.py`
- `utils/prompts.py`
- `utils/training_utils.py`

### 阶段3：拆分模型模块
- `models/losses.py`
- `models/projection.py`
- `models/sam_wrapper.py`

### 阶段4：拆分配置和数据
- `config/args.py`
- `training/data_utils.py`

### 阶段5：重构训练逻辑
- `training/trainer.py`
- 重构 `model.py` 为主入口

### 阶段6：测试和验证
- 确保所有功能正常
- 验证训练流程完整

---

## ? 注意事项

1. **向后兼容**：保持命令行接口不变
2. **导入路径**：使用相对导入或绝对导入保持一致
3. **错误处理**：每个模块应该有适当的错误处理
4. **文档字符串**：每个函数和类都应该有文档字符串
5. **类型提示**：建议添加类型提示以提高代码可读性

---

## ? 模块化后的优势

1. **代码组织清晰**：每个文件职责明确，易于理解
2. **维护成本低**：修改某个功能时只需关注对应模块
3. **可复用性强**：模块可以在其他项目中复用
4. **测试友好**：可以针对每个模块编写单元测试
5. **协作开发**：多人可以并行开发不同模块
6. **扩展性好**：新增功能时只需添加新模块或扩展现有模块

