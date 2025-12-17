# mySAM 项目代码阅读指南

## 📚 项目概述

这是一个基于 SAM (Segment Anything Model) 的图像编码器微调项目，采用**像素级对比学习**方法，只微调 `image_encoder`，冻结 `decoder`。

## 🗺️ 推荐阅读路径

### 第一阶段：理解项目整体架构（30分钟）

#### 1. 阅读文档（必读）
- ✅ **`TRAINING_PIPELINE.md`** - 训练流程详解
  - 理解训练的整体流程和策略
  - 了解 Teacher-Student 架构
  - 理解对比学习的原理
  
- ✅ **`MODULE_DESIGN.md`** - 模块设计文档
  - 了解项目的模块化架构
  - 理解各模块的职责和接口

#### 2. 入口文件（5分钟）
- 📄 **`train.py`** (21行)
  - 项目入口，非常简单
  - 解析参数 → 创建 Trainer → 开始训练

```python
# 阅读重点：
# 1. 如何解析参数
# 2. 如何初始化 Trainer
# 3. 训练如何启动
```

---

### 第二阶段：理解配置和数据（20分钟）

#### 3. 配置模块
- 📄 **`config/args.py`** (71行)
  - 所有命令行参数的定义
  - 参数分类：数据、模型、训练、对比学习、GPU等
  - **阅读重点**：了解有哪些可配置的参数

#### 4. 数据集模块
- 📄 **`dataset/ISIC.py`**
  - ISIC 数据集加载
  - 数据格式：图像、大小框、mask
  - **阅读重点**：数据如何组织，batch 格式是什么

- 📄 **`training/data_utils.py`**
  - `collate_fn_isic()` - batch 整理函数
  - **阅读重点**：数据如何被组织成 batch

---

### 第三阶段：理解核心模型组件（40分钟）

#### 5. SAM 模型封装
- 📄 **`models/sam_wrapper.py`**
  - `load_sam_model()` - 加载 SAM 模型
  - `create_teacher_model()` - 创建 EMA Teacher
  - `setup_multi_gpu()` - 多GPU设置
  - **阅读重点**：
    - 如何加载预训练 SAM
    - 如何设置冻结策略（只训练 encoder）
    - Teacher 如何创建和更新

#### 6. 投影头
- 📄 **`models/projection.py`**
  - `PixelProjHead` 类
  - 将 encoder 特征投影到对比学习空间
  - **阅读重点**：
    - 为什么需要投影头？
    - 投影头的结构（Conv → BN → GELU → Conv → L2 Norm）

#### 7. 损失函数
- 📄 **`models/losses.py`**
  - `DiceLoss` - Dice 损失（用于监督）
  - `pixel_info_nce()` - InfoNCE 对比损失（核心）
  - **阅读重点**：
    - InfoNCE 损失的计算公式
    - 如何拉近 anchor 和 positive，推远 negative

---

### 第四阶段：理解训练流程（60分钟）

#### 8. 训练器核心逻辑（最重要！）
- 📄 **`training/trainer.py`** (~750行)
  
  **阅读顺序**：
  
  a. **`__init__()`** (初始化)
     - 设置设备、GPU
     - 加载模型（SAM + Teacher + Proj）
     - 设置优化器、数据加载器
  
  b. **`train()`** (主训练循环)
     - 遍历所有 epoch
     - 调用 `train_epoch()`
  
  c. **`train_epoch()`** (单个 epoch)
     - 遍历数据加载器
     - 前向传播 → 计算损失 → 反向传播
     - 更新 EMA Teacher
  
  d. **`forward_pass()`** (前向传播，核心！)
     - Teacher 生成伪标签
     - Student 前向传播
     - 采样正负样本
     - 计算各种损失
     - **这是整个训练的核心逻辑！**
  
  e. **`_compute_contrastive_loss()`** (对比损失计算)
     - 如何选择正样本（小框 + 伪标签重叠区域）
     - 如何选择困难负样本（漏检、误检）
     - InfoNCE 损失计算
  
  f. **`save_checkpoint()`** (保存检查点)
     - 保存哪些参数（只保存 encoder + proj）

**阅读重点**：
- 理解 Teacher-Student 的交互过程
- 理解伪标签如何生成和使用
- 理解对比学习的样本采样策略
- 理解损失函数的组合方式

---

### 第五阶段：理解工具函数（20分钟）

#### 9. Prompt 处理
- 📄 **`utils/prompts.py`**
  - `sample_points_in_ring()` - 在环形区域采样点
  - **阅读重点**：为什么要在大小框之间采样点？

#### 10. 评估指标
- 📄 **`utils/metrics.py`**
  - `compute_miou()` - 计算 mIoU
  - `mask_entropy_logits()` - 计算熵（用于筛选高质量伪标签）
  - **阅读重点**：熵阈值如何筛选高质量样本

#### 11. 训练工具
- 📄 **`utils/training_utils.py`**
  - `set_seed()` - 设置随机种子
  - `setup_cuda_memory()` - CUDA 内存管理
  - **阅读重点**：工具函数，了解即可

---

## 🎯 核心概念理解检查清单

阅读完代码后，确保理解以下概念：

### ✅ 架构层面
- [ ] 为什么只微调 encoder，冻结 decoder？
- [ ] Teacher-Student 架构的作用是什么？
- [ ] EMA Teacher 如何更新？

### ✅ 训练流程
- [ ] Teacher 如何生成伪标签？
- [ ] 熵阈值如何筛选高质量样本？
- [ ] 正样本如何选择？（双重验证：小框 + 伪标签）
- [ ] 困难负样本如何选择？（漏检、误检）
- [ ] InfoNCE 损失如何计算？

### ✅ 损失函数
- [ ] 总损失由哪些部分组成？
- [ ] 各损失的权重是什么？
- [ ] 哪些损失参与梯度更新，哪些只是监控？

### ✅ 数据流
- [ ] 数据如何从数据集 → DataLoader → Trainer？
- [ ] batch 的格式是什么？
- [ ] 图像如何预处理？

---

## 📖 详细阅读建议

### 对于初学者

1. **先读文档，再读代码**
   - `TRAINING_PIPELINE.md` 提供了完整的训练流程
   - 理解文档后再看代码会事半功倍

2. **从入口开始，逐步深入**
   - `train.py` → `trainer.py` → 各个模块
   - 不要一开始就深入某个细节

3. **重点关注 `trainer.py`**
   - 这是整个项目的核心
   - `forward_pass()` 方法包含了所有关键逻辑

4. **画流程图**
   - 画出数据流：输入 → 模型 → 损失 → 更新
   - 画出 Teacher-Student 的交互过程

### 对于有经验的开发者

1. **快速浏览文档**
   - 了解整体架构即可

2. **重点阅读核心模块**
   - `trainer.py` 的 `forward_pass()`
   - `models/losses.py` 的 `pixel_info_nce()`
   - `models/sam_wrapper.py` 的模型加载逻辑

3. **关注设计模式**
   - 模块化设计
   - 配置管理
   - 多GPU支持

---

## 🔍 关键代码片段位置

| 功能 | 文件位置 | 关键函数/类 |
|------|---------|------------|
| 训练入口 | `train.py` | `main()` |
| 训练循环 | `training/trainer.py` | `Trainer.train()` |
| 前向传播 | `training/trainer.py` | `Trainer.forward_pass()` |
| 对比损失 | `models/losses.py` | `pixel_info_nce()` |
| 伪标签生成 | `training/trainer.py` | `forward_pass()` 中 Teacher 部分 |
| 样本采样 | `training/trainer.py` | `_compute_contrastive_loss()` |
| 模型加载 | `models/sam_wrapper.py` | `load_sam_model()` |
| 投影头 | `models/projection.py` | `PixelProjHead` |
| 点采样 | `utils/prompts.py` | `sample_points_in_ring()` |

---

## 💡 调试建议

如果想调试代码，建议：

1. **设置断点位置**：
   - `trainer.py` 的 `forward_pass()` 开始处
   - `_compute_contrastive_loss()` 中样本采样后
   - `pixel_info_nce()` 中损失计算处

2. **打印关键变量**：
   - `img_emb` 的形状
   - `mask_logits_stack` 的形状和值
   - `loss_contrast` 的值
   - 正负样本的数量

3. **可视化**：
   - 可视化伪标签 mask
   - 可视化采样的正负样本位置
   - 可视化特征图

---

## 📝 总结

**最短路径理解核心逻辑**：
1. 读 `TRAINING_PIPELINE.md` 理解流程
2. 读 `train.py` 看入口
3. 读 `trainer.py` 的 `forward_pass()` 理解核心逻辑
4. 读 `models/losses.py` 理解损失函数
5. 其他模块按需阅读

**预计总时间**：2-3小时（初学者），30分钟（有经验者）

---

## ❓ 常见问题

**Q: 为什么代码中有中文注释？**
A: 项目使用 UTF-8 编码，支持中文注释。如果显示乱码，请确保编辑器使用 UTF-8 编码。

**Q: `model_original.py` 是什么？**
A: 这是重构前的原始代码文件，现在已经被模块化拆分。可以忽略，或者作为参考。

**Q: 如何运行训练？**
A: 参考 `TRAINING_PIPELINE.md` 中的使用示例，或查看 `train_huge.sh` 脚本。

---

祝阅读愉快！🎉

