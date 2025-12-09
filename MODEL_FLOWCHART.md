# SAM-based Pixel-wise Contrastive Learning 模型流程图

## 一、整体架构概览

```
输入数据 (Images + Boxes)
    ↓
┌─────────────────────────────────────────────────────────┐
│                   训练流程                                │
├─────────────────────────────────────────────────────────┤
│  1. Teacher Model (EMA, Frozen)                         │
│     └─> 生成伪标签 (Pseudo Labels)                      │
│                                                          │
│  2. Student Model (Trainable)                            │
│     ├─> Image Encoder (ViT, Trainable)                   │
│     ├─> Prompt Encoder (Frozen)                         │
│     ├─> Mask Decoder (Frozen)                           │
│     └─> Pixel Projection Head (Trainable)               │
│                                                          │
│  3. 对比学习 (Contrastive Learning)                      │
│     ├─> Anchor: 高置信伪标签内像素                       │
│     ├─> Positive: Teacher特征对应位置                    │
│     └─> Negative: 困难负样本 + in-batch negatives       │
│                                                          │
│  4. 损失计算                                             │
│     ├─> Contrastive Loss (InfoNCE, α=1.0)               │
│     ├─> Mask Loss (BCE + Dice, 监控)                    │
│     ├─> IoU Loss (MSE, β=0.5, 监控)                     │
│     └─> Distillation Loss (L2, γ=0.1)                   │
└─────────────────────────────────────────────────────────┘
    ↓
输出: 训练好的 Image Encoder + Projection Head
```

## 二、详细数据流程图

### 2.1 数据输入阶段

```
输入数据格式:
├─ Images: (B, 3, H, W)          # B=batch_size, H=W=1024
├─ Big Boxes: List[List[4]]       # 大框坐标 [x1, y1, x2, y2]
├─ Small Boxes: List[List[4]]     # 小框坐标 [x1, y1, x2, y2]
└─ GT Masks: (B, 1, H, W)        # Ground truth masks (用于监控)

数据预处理:
1. Images resize 到 1024×1024 (SAM输入尺寸)
2. Boxes 坐标按比例缩放
3. 在大小框中间区域采样离散点作为 prompt
```

### 2.2 Teacher 模型前向传播（生成伪标签）

```
┌─────────────────────────────────────────────────────────┐
│  Teacher Model (EMA, 完全冻结)                           │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input: Images (B, 3, 1024, 1024)                       │
│    ↓                                                      │
│  Teacher Image Encoder (ViT, Frozen)                     │
│    ↓                                                      │
│  Image Embeddings: (B, 256, H', W')                     │
│    ↓                                                      │
│  Prompt Encoder (Frozen)                                 │
│    ├─> 输入: 环形区域采样点 (points)                     │
│    └─> 输出: sparse_embeddings, dense_embeddings         │
│    ↓                                                      │
│  Mask Decoder (Frozen)                                   │
│    ├─> 输入: image_embeddings + prompt_embeddings       │
│    └─> 输出: Low-res masks (B, 1, 256, 256)             │
│    ↓                                                      │
│  上采样到原始尺寸: (B, 1, 1024, 1024)                    │
│    ↓                                                      │
│  计算熵 (Entropy)                                        │
│    ↓                                                      │
│  筛选: entropy <= threshold (默认0.2)                    │
│    ↓                                                      │
│  输出: Pseudo Labels (高置信度)                          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**关键步骤：**
- 使用环形区域（大小框之间）采样点作为 prompt
- 计算每个伪标签的熵值
- 只保留低熵（高置信度）的伪标签用于训练

### 2.3 Student 模型前向传播

```
┌─────────────────────────────────────────────────────────┐
│  Student Model (可训练)                                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input: Images (B, 3, 1024, 1024)                       │
│    ↓                                                      │
│  Student Image Encoder (ViT, Trainable) ?               │
│    ↓                                                      │
│  Image Embeddings: (B, 256, H', W')                     │
│    ├─> 用于对比学习: img_emb                             │
│    └─> 用于Mask预测: img_emb                             │
│    ↓                                                      │
│  ┌──────────────────────────────────────┐               │
│  │ 分支1: Mask预测路径                   │               │
│  ├──────────────────────────────────────┤               │
│  │ Prompt Encoder (Frozen)              │               │
│  │   └─> 使用相同的环形区域采样点        │               │
│  │     ↓                                 │               │
│  │ Mask Decoder (Frozen)                │               │
│  │   └─> 输出: Predicted Masks          │               │
│  └──────────────────────────────────────┘               │
│                                                          │
│  ┌──────────────────────────────────────┐               │
│  │ 分支2: 对比学习路径                   │               │
│  ├──────────────────────────────────────┤               │
│  │ Pixel Projection Head (Trainable) ?  │               │
│  │   └─> 256 → 64 维特征投影            │               │
│  │     ↓                                 │               │
│  │ 投影特征: z (B, 64, H', W')          │               │
│  │   └─> 用于对比学习                    │               │
│  └──────────────────────────────────────┘               │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 2.4 对比学习采样策略

```
┌─────────────────────────────────────────────────────────┐
│  像素级对比学习采样                                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  对于每个高置信度样本 (trusted_idx):                     │
│                                                          │
│  1. Anchor 采样 (正样本)                                │
│     └─> 位置: 伪标签 AND 小框 (高度重叠区域)             │
│     └─> 数量: pos_samples (默认256)                     │
│                                                          │
│  2. Positive 采样                                        │
│     └─> 来源: Teacher Image Encoder 对应位置特征        │
│     └─> 数量: 与 Anchor 相同                            │
│                                                          │
│  3. Negative 采样 (困难负样本)                          │
│     ├─> 类型1: 小框内但伪标签未覆盖 (漏检)              │
│     ├─> 类型2: 大框外但伪标签覆盖 (误检)                │
│     └─> 数量: neg_samples (默认1024)                    │
│                                                          │
│  4. InfoNCE 损失计算                                    │
│     └─> L_contrast = -log(exp(sim_pos) / (exp(sim_pos) + Σexp(sim_neg))) │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 2.5 损失计算与反向传播

```
┌─────────────────────────────────────────────────────────┐
│  损失计算                                                │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Contrastive Loss (主要训练信号) ?                   │
│     L_contrast = InfoNCE(anchors, positives, negatives) │
│     权重: α = 1.0                                        │
│                                                          │
│  2. Mask Loss (监控用，不参与训练)                       │
│     L_mask = BCE(pred, gt) + Dice(pred, gt)            │
│     权重: 1.0 (但decoder冻结，无梯度)                    │
│                                                          │
│  3. IoU Loss (监控用)                                   │
│     L_iou = MSE(pred_iou, gt_iou)                       │
│     权重: β = 0.5                                        │
│                                                          │
│  4. Distillation Loss (防止遗忘)                         │
│     L_distill = MSE(student_feat, ref_feat)            │
│     权重: γ = 0.1                                        │
│                                                          │
│  总损失:                                                 │
│  L_total = L_mask + α·L_contrast + β·L_iou + γ·L_distill│
│                                                          │
│  ┌──────────────────────────────────────┐               │
│  │ 反向传播                              │               │
│  ├──────────────────────────────────────┤               │
│  │ 只更新:                               │               │
│  │   - Image Encoder (部分或全部)        │               │
│  │   - Pixel Projection Head             │               │
│  │ 冻结:                                 │               │
│  │   - Prompt Encoder                   │               │
│  │   - Mask Decoder                     │               │
│  └──────────────────────────────────────┘               │
│                                                          │
│  EMA Teacher 更新:                                       │
│  teacher_param = 0.999 * teacher_param + 0.001 * student_param │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## 三、模块详细说明

### 3.1 Image Encoder (ViT)

**功能：** 提取图像特征表示

**输入：** Images (B, 3, 1024, 1024)

**输出：** Image Embeddings (B, 256, H', W')
- H', W' 通常为 64×64 (ViT-B)

**状态：**
- Student: Trainable (可全部或部分解冻)
- Teacher: Frozen

**创新点：**
- 只微调 encoder，保持 decoder 能力
- 支持部分解冻策略（只训练最后 K 个 transformer blocks）

### 3.2 Prompt Encoder

**功能：** 编码 prompt（点或框）为 embedding

**输入：**
- Points: (N, 2) 坐标 + (N,) 标签
- Boxes: (1, 4) 坐标

**输出：**
- Sparse embeddings: 点/框的 embedding
- Dense embeddings: 密集 prompt embedding

**状态：** Frozen（使用 SAM 预训练权重）

### 3.3 Mask Decoder

**功能：** 根据 image embeddings 和 prompt embeddings 生成分割 mask

**输入：**
- Image embeddings
- Sparse prompt embeddings
- Dense prompt embeddings

**输出：**
- Low-res masks: (B, 1, 256, 256)
- IoU predictions: (B, 1)

**状态：** Frozen（使用 SAM 预训练权重）

### 3.4 Pixel Projection Head

**功能：** 将 image encoder 特征投影到对比学习空间

**结构：**
```
Conv2d(256 → 256) → BatchNorm → GELU → Conv2d(256 → 64) → L2 Normalize
```

**输入：** Image Embeddings (B, 256, H', W')

**输出：** Projected Features (B, 64, H', W')

**状态：** Trainable

**创新点：**
- 专门为像素级对比学习设计
- L2 归一化确保特征在单位球面上

### 3.5 EMA Teacher

**功能：** 生成稳定的伪标签用于自训练

**更新策略：**
```python
teacher_param = 0.999 * teacher_param + 0.001 * student_param
```

**状态：** 完全冻结（不参与反向传播）

**创新点：**
- EMA 更新确保伪标签稳定
- 避免直接使用 student 的预测（可能不稳定）

## 四、核心创新点

### 4.1 像素级对比学习

**创新：** 在像素级别进行对比学习，而非图像级别

**优势：**
- 学习细粒度的特征表示
- 更好地处理边界区域
- 适合分割任务

**实现：**
- Anchor: 高置信伪标签内像素
- Positive: Teacher 特征对应位置
- Negative: 困难负样本（漏检+误检）

### 4.2 环形区域点采样

**创新：** 在大小框之间的环形区域采样点作为 prompt

**优势：**
- 提供边界信息
- 比单纯使用大框更精确
- 比使用小框更稳定

**实现：**
```python
# 在 big_box 内但不在 small_box 内的区域采样点
points = sample_points_in_ring(small_box, big_box, num_points=10)
```

### 4.3 困难负样本挖掘

**创新：** 针对性地采样困难负样本

**类型：**
1. **漏检样本：** 小框内但伪标签未覆盖的区域
2. **误检样本：** 大框外但伪标签覆盖的区域

**优势：**
- 提高模型对边界区域的敏感性
- 减少假阳性和假阴性

### 4.4 熵筛选伪标签

**创新：** 使用熵值筛选高置信度伪标签

**实现：**
```python
entropy = mask_entropy_logits(pseudo_labels)
trusted_idx = entropy <= threshold  # 默认 0.2
```

**优势：**
- 只使用高质量伪标签
- 避免噪声标签干扰训练

### 4.5 部分解冻策略

**创新：** 支持只解冻 encoder 的最后 K 个 transformer blocks

**优势：**
- 减少显存占用
- 保持预训练知识
- 灵活的训练策略

**实现：**
```python
if unfreeze_last_k > 0:
    # 只解冻最后 K 个 blocks
    for i in range(total - unfreeze_last_k, total):
        unfreeze_block(i)
```

### 4.6 特征蒸馏

**创新：** 使用 L2 损失约束 encoder 不要过度偏离预训练状态

**优势：**
- 防止灾难性遗忘
- 保持 SAM 的通用能力
- 权重较小（γ=0.1），不阻碍适应新任务

### 4.7 弱监督学习

**创新：** 只需要大小框标注，无需精确 mask

**优势：**
- 降低标注成本
- 利用 SAM 的强泛化能力
- 通过自训练逐步提升性能

## 五、完整训练流程时序图

```
Epoch Start
    ↓
┌─────────────────────────────────────────────────────────┐
│  For each batch:                                        │
│                                                          │
│  1. 数据加载                                            │
│     └─> Images, Boxes, GT Masks                         │
│                                                          │
│  2. Teacher 前向 (no_grad)                              │
│     └─> 生成伪标签 + 熵筛选                              │
│                                                          │
│  3. Student 前向                                        │
│     ├─> Image Encoder → Image Embeddings                │
│     ├─> Prompt Encoder → Prompt Embeddings              │
│     ├─> Mask Decoder → Predicted Masks                 │
│     └─> Projection Head → Projected Features            │
│                                                          │
│  4. 对比学习采样                                        │
│     ├─> Anchor: 伪标签 ∩ 小框                           │
│     ├─> Positive: Teacher 特征                           │
│     └─> Negative: 困难负样本                             │
│                                                          │
│  5. 损失计算                                            │
│     ├─> Contrastive Loss                                │
│     ├─> Mask Loss (监控)                                │
│     ├─> IoU Loss (监控)                                 │
│     └─> Distillation Loss                               │
│                                                          │
│  6. 反向传播                                            │
│     └─> 只更新 Image Encoder + Projection Head         │
│                                                          │
│  7. EMA Teacher 更新                                    │
│     └─> teacher_param = 0.999*teacher + 0.001*student  │
│                                                          │
└─────────────────────────────────────────────────────────┘
    ↓
Epoch End → Save Checkpoint → Next Epoch
```

## 六、数据维度变化

```
输入:
  Images:      (B, 3, 1024, 1024)
  Boxes:       List[List[4]]
  GT Masks:    (B, 1, 1024, 1024)

Teacher 路径:
  Image Encoder:    (B, 3, 1024, 1024) → (B, 256, 64, 64)
  Mask Decoder:     (B, 256, 64, 64) → (B, 1, 256, 256)
  上采样:           (B, 1, 256, 256) → (B, 1, 1024, 1024)
  Pseudo Labels:   (B, 1, 1024, 1024)

Student 路径:
  Image Encoder:    (B, 3, 1024, 1024) → (B, 256, 64, 64)
  Projection Head: (B, 256, 64, 64) → (B, 64, 64, 64)
  Mask Decoder:     (B, 256, 64, 64) → (B, 1, 256, 256)
  上采样:           (B, 1, 256, 256) → (B, 1, 1024, 1024)

对比学习:
  Anchors:     (N, 64)  # N = pos_samples * num_trusted
  Positives:   (N, 64)
  Negatives:   (M, 64)  # M = neg_samples * num_trusted
```

## 七、关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lr_encoder` | 2e-6 | Image Encoder 学习率 |
| `pos_samples` | 256 | 正样本采样数量 |
| `neg_samples` | 1024 | 负样本采样数量 |
| `temperature` | 0.1 | InfoNCE 温度参数 |
| `entropy_thresh` | 0.2 | 熵筛选阈值 |
| `unfreeze_last_k` | 0 | 解冻最后 K 个 blocks (0=全部) |
| `ema_decay` | 0.999 | EMA Teacher 衰减率 |
| `alpha` | 1.0 | Contrastive Loss 权重 |
| `beta` | 0.5 | IoU Loss 权重 |
| `gamma` | 0.1 | Distillation Loss 权重 |

## 八、输出结果

**训练输出：**
- Checkpoint: 包含 `sam_image_encoder` 和 `proj` 的权重
- 训练日志: 损失值和 mIOU

**评估指标：**
- mIOU: 平均 IoU 值
- Mask Loss: 监控用
- Contrast Loss: 主要训练信号
- IoU Loss: 监控用

## 九、与标准 SAM 的区别

| 特性 | 标准 SAM | 本模型 |
|------|---------|--------|
| **训练方式** | 冻结所有组件 | 微调 Image Encoder |
| **训练信号** | 监督学习 | 对比学习 + 弱监督 |
| **Prompt** | 用户提供 | 环形区域自动采样 |
| **标注需求** | 精确 mask | 大小框即可 |
| **特征学习** | 固定特征 | 任务特定特征 |
| **应用场景** | 通用分割 | 特定领域微调 |

## 十、总结

本模型的核心思想是：
1. **保持 SAM 的强泛化能力**（冻结 decoder）
2. **通过对比学习优化特征表示**（微调 encoder）
3. **利用弱监督和自训练**（大小框 + 伪标签）
4. **像素级细粒度学习**（而非图像级）

这种设计在保持 SAM 通用能力的同时，通过对比学习在特定任务上获得更好的性能。

