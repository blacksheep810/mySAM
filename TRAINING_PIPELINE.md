# SAM 图像编码器微调训练流程详解

## 📁 代码文件总览

| 文件路径 | 主要功能 | 关键类/函数 |
|---------|---------|------------|
| `train.py` | 训练入口 | `main()` |
| `training/trainer.py` | 训练器主类 | `Trainer` 类（~750行） |
| `config/args.py` | 参数配置 | `build_argparser()` |
| `models/sam_wrapper.py` | SAM 模型封装 | `load_sam_model()`, `create_teacher_model()` |
| `models/projection.py` | 投影头 | `PixelProjHead` 类 |
| `models/losses.py` | 损失函数 | `DiceLoss`, `pixel_info_nce()` |
| `utils/prompts.py` | Prompt 处理 | `prepare_box_prompts()` |
| `utils/metrics.py` | 评估指标 | `compute_miou()`, `mask_entropy_logits()` |
| `training/data_utils.py` | 数据工具 | `collate_fn_isic()` |
| `dataset/ISIC.py` | ISIC 数据集 | `ISIC2016Dataset` 类 |

**核心文件**：`training/trainer.py` 包含所有训练逻辑，建议重点阅读。

---

## 一、模型架构概述

**相关代码文件**：
- `train.py` - 训练入口
- `training/trainer.py` - 训练器主类
- `models/sam_wrapper.py` - SAM 模型封装
- `models/projection.py` - 投影头实现

### 1.1 核心组件

本训练框架基于 SAM (Segment Anything Model)，采用**只微调 image_encoder，冻结 decoder** 的策略。

```
SAM 模型结构：
├── image_encoder (ViT)      # ✅ 可训练 - 提取图像特征
├── prompt_encoder           # ❌ 冻结 - 处理点/框 prompt
└── mask_decoder            # ❌ 冻结 - 生成分割 mask

辅助组件：
├── PixelProjHead           # ✅ 可训练 - 特征投影到对比学习空间
└── Teacher (EMA)           # ❌ 冻结 - 生成伪标签
```

### 1.2 训练策略

- **可训练参数**：`image_encoder` + `PixelProjHead`
- **冻结参数**：`mask_decoder` + `prompt_encoder` + `Teacher`
- **主要训练信号**：像素级对比学习（InfoNCE）
- **辅助训练信号**：特征蒸馏损失

---

## 二、完整训练流程

### 阶段 0：初始化与准备

**相关代码文件**：
- `training/trainer.py::Trainer.__init__()` - 初始化方法
- `training/trainer.py::Trainer._setup_models()` - 模型设置
- `models/sam_wrapper.py::load_sam_model()` - SAM 模型加载
- `models/sam_wrapper.py::create_teacher_model()` - Teacher 模型创建
- `models/projection.py::PixelProjHead` - 投影头类

#### 2.1 模型加载与参数设置

**作用**：加载预训练 SAM 模型，设置参数冻结策略，初始化投影头。

**代码位置**：`training/trainer.py::Trainer._setup_models()` (第108-143行)

```python
# 加载 Student 模型（可训练）
sam = sam_model_registry[args.model_type](checkpoint=checkpoint_path)
sam.to(device)

# 加载 Teacher 模型（EMA，完全冻结）
teacher = sam_model_registry[args.model_type](checkpoint=checkpoint_path)
teacher.to(device)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

# 冻结策略：只训练 image_encoder
for p in sam.mask_decoder.parameters():
    p.requires_grad = False  # 冻结 decoder
for p in sam.prompt_encoder.parameters():
    p.requires_grad = False  # 冻结 prompt_encoder

# image_encoder 可训练
for p in sam.image_encoder.parameters():
    p.requires_grad = True  # 或通过 unfreeze_last_k 控制部分解冻

# 初始化投影头
proj = PixelProjHead(in_dim=256, proj_dim=64).to(device)

# 可训练参数
trainable_params = list(proj.parameters())
for p in sam.image_encoder.parameters():
    if p.requires_grad:
        trainable_params.append(p)
```

**关键点**：
- Teacher 完全冻结，用于生成稳定的伪标签
- Decoder 冻结，保持 SAM 预训练的分割能力
- 只训练 encoder，通过对比学习优化特征表示

---

### 阶段 1：数据准备

**相关代码文件**：
- `training/trainer.py::Trainer._setup_dataloader()` - 数据加载器设置
- `training/data_utils.py::collate_fn_isic()` - Batch 整理函数
- `dataset/ISIC.py::ISIC2016Dataset` - ISIC 数据集类
- `training/trainer.py::Trainer._prepare_images_and_boxes()` - 图像预处理

#### 1.1 数据加载

**作用**：从 ISIC 数据集加载图像、大小框、mask 等信息。

**代码位置**：
- 数据集加载：`training/trainer.py::Trainer._setup_dataloader()` (第177-201行)
- Batch 整理：`training/data_utils.py::collate_fn_isic()` (第12-54行)

```python
# 数据格式
batch = {
    'image': (B, 3, H, W),           # 图像
    'boxes': [big_box1, ...],        # 大框列表 [x1, y1, x2, y2]
    'small_boxes': [small_box1, ...], # 小框列表 [x1, y1, x2, y2]
    'mask': (B, 1, H, W),            # Ground truth mask
    'img_names': [...]
}

# 提取数据
images = batch['image'].to(device)
big_boxes = batch['boxes']
small_boxes = batch['small_boxes']
gt_masks = batch['mask'].to(device)
B = images.size(0)
```

#### 1.2 图像和框的预处理

**作用**：将图像 resize 到 SAM 输入尺寸（1024x1024），并相应缩放 box 坐标。

**代码位置**：`training/trainer.py::Trainer._prepare_images_and_boxes()` (第203-232行)

```python
def _prepare_images_and_boxes(images, boxes_list, small_boxes_list):
    """
    准备图像和框（resize 到 SAM 输入尺寸）
    """
    B = images.size(0)
    
    if images.shape[-1] != 1024 or images.shape[-2] != 1024:
        # Resize 图像到 SAM 期望的尺寸
        images_sam = F.interpolate(
            images, 
            size=(1024, 1024), 
            mode='bilinear', 
            align_corners=False
        )
        # 缩放 box 坐标
        scale_factor = 1024 / args.img_size
        boxes_list_sam = [coord * scale_factor for coord in boxes_list]
        small_boxes_list_sam = [coord * scale_factor for coord in small_boxes_list]
    else:
        images_sam = images
        boxes_list_sam = boxes_list
        small_boxes_list_sam = small_boxes_list
    
    return images_sam, boxes_list_sam, small_boxes_list_sam
```

**关键点**：
- SAM 预训练时使用 1024x1024 输入，需要将图像 resize 到此尺寸
- box 坐标需要按比例缩放
- 大小框提供弱监督信息（目标的大致范围和核心区域）
- GT mask 用于监督损失（但 decoder 冻结，仅用于监控）

---

### 阶段 2：Teacher 生成伪标签

**相关代码文件**：
- `training/trainer.py::Trainer._teacher_forward()` - Teacher 前向传播
- `utils/prompts.py::prepare_box_prompts()` - 框 prompt 准备
- `utils/metrics.py::mask_entropy_logits()` - 熵计算

#### 2.1 大小框信息输入作为 prompt

**作用**：使用大小框作为 prompt 输入给 SAM，提供弱监督信息。

**代码位置**：`utils/prompts.py::prepare_box_prompts()` (第94-114行)

```python
def prepare_box_prompts(big_box, small_box, device):
    """
    准备框 prompt（大框+小框）
    将多个框合并为 tensor
    
    原理：
    - 大框：目标大致范围
    - 小框：目标核心区域
    - SAM 可以同时使用多个框作为 prompt
    """
    if small_box is not None:
        # 将大框和小框合并，形状为 (2, 4)
        boxes_tensor = torch.tensor([big_box, small_box], device=device, dtype=torch.float32)
    else:
        # 如果没有 small_box，只使用 big_box
        boxes_tensor = torch.tensor([big_box], device=device, dtype=torch.float32)
    
    return boxes_tensor  # (N, 4)，N=2 如果有 small_box，否则 N=1
```
```

```
┌─────────────────────────┐
│   Big Box (大框)        │
│  ┌───────────────────┐  │
│  │  Small Box (小框) │  │
│  │  ┌─────────────┐  │  │
│  │  │  核心区域   │  │  │
│  │  └─────────────┘  │  │
│  │                   │  │ 
│  └───────────────────┘  │
└─────────────────────────┘
```

#### 2.2 Teacher 前向传播

**作用**：使用冻结的 Teacher 模型生成伪标签，作为对比学习的监督信号。

**代码位置**：`training/trainer.py::Trainer._teacher_forward()` (第234-309行)

```python
with torch.no_grad():  # Teacher 不参与梯度计算
    # 1. Teacher 提取图像特征
    t_img_emb = teacher.image_encoder(images)  # (B, 256, H', W')
    
    all_mask_logits = []
    all_mask_entropy = []
    
            for b in range(B):
                # 2. 准备框 prompt（大框+小框）
                big_box = boxes_list_sam[b]
                small_box = small_boxes_list_sam[b]
                boxes_tensor = prepare_box_prompts(big_box, small_box, device)  # (N, 4)
                
                # 3. 编码 prompt
                sparse_p, dense_p = teacher.prompt_encoder(
                    points=None,
                    boxes=boxes_tensor,
                    masks=None
                )
        
        # 4. 生成伪标签 mask
        out = teacher.mask_decoder(
            image_embeddings=t_img_emb[b:b+1],
            image_pe=teacher.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_p,
            dense_prompt_embeddings=dense_p,
            multimask_output=False
        )
        
        logits = out[0]  # (1, 1, H_low, W_low)
        logits = F.interpolate(logits, size=(H, W), mode='bilinear')  # 上采样
        
        all_mask_logits.append(logits)
        all_mask_entropy.append(mask_entropy_logits(logits))  # 计算熵
    
    # 5. 堆叠所有伪标签
    mask_logits_stack = torch.cat(all_mask_logits, dim=0)  # (B, 1, H, W)
    mask_entropy_vals = torch.stack(all_mask_entropy)  # (B,)
```

**关键点**：
- Teacher 完全冻结，提供稳定的伪标签
- 使用框 prompt（大框+小框），提供弱监督信息
- 计算熵用于筛选高置信样本
- 图像会被 resize 到 SAM 输入尺寸（1024x1024），box 坐标也会相应缩放

#### 2.3 熵筛选

**作用**：基于熵值筛选高置信度的伪标签，只对高质量样本进行对比学习。

**代码位置**：
- 熵计算：`utils/metrics.py::mask_entropy_logits()` (第44-59行)
- 熵筛选：`training/trainer.py::Trainer._compute_contrastive_loss()` (第460行)

```python
def mask_entropy_logits(mask_logits):
    """
    计算 mask 的熵（不确定性）
    
    原理：
    - 低熵：模型很确定（概率接近 0 或 1）→ 高置信
    - 高熵：模型不确定（概率接近 0.5）→ 低置信
    """
    p = torch.sigmoid(mask_logits)  # 转换为概率
    # 二分类熵: H = -[p*log(p) + (1-p)*log(1-p)]
    ent = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
    return ent.view(ent.size(0), -1).mean(dim=1)  # 平均熵

# 筛选高置信样本
trusted_idx = (mask_entropy_vals <= args.entropy_thresh).nonzero()  # 默认 0.2
```

**关键点**：
- 熵阈值：`entropy_thresh=0.2`，只保留低熵（高置信）样本
- 减少噪声伪标签的影响，提高训练稳定性

---

### 阶段 3：Student 前向传播

**相关代码文件**：
- `training/trainer.py::Trainer._student_forward()` - Student 前向传播
- `models/projection.py::PixelProjHead.forward()` - 特征投影

#### 3.1 保存参考特征

**作用**：保存冻结状态的 encoder 特征，用于特征蒸馏损失。

**代码位置**：`training/trainer.py::Trainer._student_forward()` (第314-316行)

```python
# 在 encoder 更新前保存参考特征
with torch.no_grad():
    ref_feats = sam.image_encoder(images).detach()  # 冻结版本的特征
```

**关键点**：
- 用于蒸馏损失，防止 encoder 过度偏离预训练状态
- 只在训练开始时保存一次（或每个 step 保存）

#### 3.2 Student Encoder 前向

**作用**：Student 的 image_encoder 提取特征，这是**唯一可训练**的组件。

**代码位置**：`training/trainer.py::Trainer._student_forward()` (第318-324行)

```python
# Student image_encoder 前向（可训练，需要梯度）
img_emb = sam.image_encoder(images)  # (B, 256, H', W') - 有梯度
```

**关键点**：
- `img_emb` 有梯度，用于反向传播
- 这是对比学习的主要特征来源

#### 3.3 Student Decoder 前向（冻结）

**作用**：使用冻结的 decoder 生成预测 mask，用于监控和损失计算（但不参与训练）。

**代码位置**：`training/trainer.py::Trainer._student_forward()` (第326-395行)

```python
# student decoder 前向传播（冻结，不需要梯度）
preds = []
ious = []
with torch.no_grad():  # decoder 冻结，不需要梯度
    for b in range(B):
        # 使用相同的 prompt（与 teacher 一致）
        big_box = boxes_list_sam[b]
        small_box = small_boxes_list_sam[b]
        boxes_tensor = prepare_box_prompts(big_box, small_box, device)
        
        # 编码 prompt
        sp, dp = sam.prompt_encoder(
            points=None,
            boxes=boxes_tensor,
            masks=None
        )
        
        # 通过 decoder（使用 detach 的 embeddings）
        outb = sam.mask_decoder(
            image_embeddings=img_emb[b:b+1].detach(),  # decoder 不需要梯度
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sp,
            dense_prompt_embeddings=dp,
            multimask_output=False
        )
        
        masks_upsampled = F.interpolate(outb[0], size=(H, W), mode='bilinear')
        preds.append(masks_upsampled)
        ious.append(outb[1])  # IoU 预测

pred_logits = torch.cat(preds, dim=0)  # (B, 1, H, W)
pred_iou = torch.cat(ious, dim=0)  # (B, 1)
```

**关键点**：
- Decoder 冻结，`pred_logits` 无梯度
- 主要用于监控和计算损失值（但不参与训练）
- 使用 `img_emb.detach()` 确保 decoder 不参与梯度计算

---

### 阶段 4：损失计算

**相关代码文件**：
- `training/trainer.py::Trainer._compute_losses()` - 总损失计算
- `training/trainer.py::Trainer._compute_contrastive_loss()` - 对比损失计算
- `models/losses.py::DiceLoss` - Dice 损失
- `models/losses.py::pixel_info_nce()` - InfoNCE 对比损失
- `utils/metrics.py::compute_miou()` - mIoU 计算

#### 4.1 监督 Mask 损失（仅监控）

**作用**：计算预测 mask 与 GT mask 的差异，**仅用于监控**，不参与训练。

**代码位置**：`training/trainer.py::Trainer._compute_losses()` (第402-411行)

```python
# Mask 损失（不参与训练，decoder 冻结）
loss_mask = torch.tensor(0.0, device=device)
if gt_masks is not None:
    with torch.no_grad():  # 不参与梯度计算
        gt_resized = F.interpolate(gt_masks, size=pred_logits.shape[-2:], mode='nearest')
        
        # BCE 损失
        loss_bce = nn.BCEWithLogitsLoss()(pred_logits, gt_resized)
        
        # Dice 损失
        loss_dice = DiceLoss()(pred_logits, gt_resized)
        
        loss_mask = loss_bce + loss_dice
```

**Dice Loss 公式**：
```
Dice = (2 * |pred ∩ gt| + ε) / (|pred| + |gt| + ε)
Loss = 1 - Dice
```

**关键点**：
- 仅用于监控训练进度
- Decoder 冻结，无法通过此损失训练 encoder

#### 4.2 像素级对比损失（核心）

**作用**：通过对比学习优化 encoder 特征表示，这是**主要的训练信号**。

**代码位置**：`training/trainer.py::Trainer._compute_contrastive_loss()` (第454-594行)

##### 4.2.1 特征投影

**代码位置**：`models/projection.py::PixelProjHead.forward()` (第36-51行)

```python
# 投影到对比学习空间
z = proj(img_emb)  # (B, 64, Hf, Wf) - L2 归一化的特征
Bz, D, Hf, Wf = z.shape
```

##### 4.2.2 创建空间掩码

**代码位置**：`training/trainer.py::Trainer._compute_contrastive_loss()` (第495-513行)

```python
for b in range(B):
    if b not in trusted_idx:  # 只处理高置信样本
        continue
    
    # 将 box 坐标缩放到特征图尺寸
    scale_h = Hf / args.img_size
    scale_w = Wf / args.img_size
    
    # 创建 small_box 和 big_box 的 mask
    small_box_mask = torch.zeros(Hf, Wf, device=device)
    small_box_mask[sy1_f:sy2_f+1, sx1_f:sx2_f+1] = 1.0
    
    big_box_mask = torch.zeros(Hf, Wf, device=device)
    big_box_mask[by1_f:by2_f+1, bx1_f:bx2_f+1] = 1.0
    
    # Teacher 伪标签 mask（二值化）
    tmask_resized = (torch.sigmoid(mask_logits_stack[b]) > 0.5).float()
```

##### 4.2.3 选择显著性正样本

**代码位置**：`training/trainer.py::Trainer._compute_contrastive_loss()` (第515-541行)

```python
# 正样本 = 伪标签 AND 小框（高度重叠区域）
salient_pos_mask = (tmask_resized > 0.5) & (small_box_mask > 0.5)
pos_idx = salient_pos_mask.view(-1).nonzero()

# 采样固定数量
npos = min(pos_idx.numel(), args.pos_samples)  # 默认 256
chosen_pos = pos_idx[torch.randperm(pos_idx.numel())[:npos]]

# 提取 anchor 和 positive
z_b = z[b].reshape(-1, D)  # (Hf*Wf, 64)
anchors = z_b[chosen_pos]  # (npos, 64) - Student 特征

# Teacher 特征作为 positive
t_img_emb_b = teacher.image_encoder(images[b:b+1])
t_z = proj(t_img_emb_b).reshape(-1, D)
positives = t_z[chosen_pos]  # (npos, 64) - Teacher 特征
```

**正样本选择原理**：
```
┌─────────────────────────┐
│  Small Box              │
│  ┌───────────────────┐  │
│  │ 伪标签覆盖区域    │  │ ← 正样本（双重验证）
│  │ ┌───────────────┐ │  │
│  │ │ 高度重叠区域  │ │  │
│  │ └───────────────┘ │  │
│  └───────────────────┘  │
└─────────────────────────┘
```

##### 4.2.4 选择困难负样本

**代码位置**：`training/trainer.py::Trainer._compute_contrastive_loss()` (第543-568行)

```python
hard_neg_list = []

# 类型1：小框内但伪标签未覆盖（漏掉的部分）
missed_in_small = (small_box_mask > 0.5) & (tmask_resized < 0.5)
missed_idx = missed_in_small.view(-1).nonzero()
if missed_idx.numel() > 0:
    n_missed = min(missed_idx.numel(), args.neg_samples // 2)
    hard_neg_type1 = z_b[missed_idx[torch.randperm(missed_idx.numel())[:n_missed]]]
    hard_neg_list.append(hard_neg_type1)

# 类型2：大框外但伪标签覆盖（溢出的部分）
overflow_out_big = (big_box_mask < 0.5) & (tmask_resized > 0.5)
overflow_idx = overflow_out_big.view(-1).nonzero()
if overflow_idx.numel() > 0:
    n_overflow = min(overflow_idx.numel(), args.neg_samples // 2)
    hard_neg_type2 = z_b[overflow_idx[torch.randperm(overflow_idx.numel())[:n_overflow]]]
    hard_neg_list.append(hard_neg_type2)

# 合并困难负样本
if len(hard_neg_list) > 0:
    hard_negs = torch.cat(hard_neg_list, dim=0)
else:
    hard_negs = random_sample(all_pixels)  # 回退到随机负样本
```

**困难负样本原理**：
```
类型1（漏检）：
┌─────────────────────────┐
│  Small Box              │
│  ┌───────────────────┐  │
│  │ [未覆盖区域]      │  │ ← 困难负样本类型1
│  │ 伪标签覆盖区域    │  │
│  └───────────────────┘  │
└─────────────────────────┘

类型2（误检）：
┌─────────────────────────┐
│ [溢出区域] ← 类型2      │
│  Big Box                │
│  ┌───────────────────┐  │
│  │ 伪标签覆盖区域    │  │
│  └───────────────────┘  │
└─────────────────────────┘
```

##### 4.2.5 计算 InfoNCE 损失

**代码位置**：
- InfoNCE 函数：`models/losses.py::pixel_info_nce()` (第49-70行)
- 调用位置：`training/trainer.py::Trainer._compute_contrastive_loss()` (第593行)

```python
def pixel_info_nce(anchors, positives, negatives, temperature=0.1):
    """
    Pixel-wise InfoNCE 损失
    
    原理：
    - 拉近 anchor 和 positive（同目标不同视图）
    - 推远 anchor 和 negatives（不同目标/背景）
    """
    # 归一化
    anchors = F.normalize(anchors, dim=1)
    positives = F.normalize(positives, dim=1)
    negatives = F.normalize(negatives, dim=1)
    
    # 计算相似度
    sim_pos = torch.exp(torch.sum(anchors * positives, dim=1) / temperature)  # (N,)
    sim_neg = torch.exp(torch.matmul(anchors, negatives.T) / temperature)  # (N, M)
    
    # InfoNCE
    denom = sim_pos + sim_neg.sum(dim=1)  # (N,)
    loss = -torch.log(sim_pos / (denom + 1e-12) + 1e-12)
    return loss.mean()

# 计算对比损失
loss_contrast = pixel_info_nce(anchors, positives, hard_negs, temperature=0.1)
```

**InfoNCE 公式**：
```
L = -log(exp(sim(anchor, positive) / τ) / 
         (exp(sim(anchor, positive) / τ) + Σexp(sim(anchor, negative) / τ)))
```

**关键点**：
- 这是**主要的训练信号**，用于优化 encoder
- 通过对比学习学习区分前景/背景特征
- 困难负样本提高模型鲁棒性

#### 4.3 IoU 损失（仅监控）

**作用**：计算预测 IoU 与实际 IoU 的差异，**仅用于监控**。

**代码位置**：`training/trainer.py::Trainer._compute_losses()` (第419-427行)

```python
loss_iou = torch.tensor(0.0, device=device)
if pred_iou is not None:
    with torch.no_grad():  # 不参与训练
        # 计算实际 IoU（Student 预测 vs Teacher 伪标签）
        pred_bin = (torch.sigmoid(pred_logits) > 0.5).float()
        teacher_bin = (torch.sigmoid(mask_logits_stack) > 0.5).float()
        
        inter = (pred_bin * teacher_bin).sum(dim=[1,2,3])  # 交集
        union = ((pred_bin + teacher_bin) > 0).float().sum(dim=[1,2,3])  # 并集
        gt_iou = (inter / (union + 1e-6)).unsqueeze(-1)  # (B, 1)
        
        # MSE 损失
        loss_iou = F.mse_loss(pred_iou, gt_iou)
```

**关键点**：
- 仅用于监控，不参与训练
- 评估模型对预测质量的估计能力
- 计算的是 Student 预测与 Teacher 伪标签之间的 IoU

#### 4.4 特征蒸馏损失

**作用**：约束 encoder 不要过度偏离预训练状态，防止灾难性遗忘。

**代码位置**：`training/trainer.py::Trainer._compute_losses()` (第429-446行)

```python
loss_distill = torch.tensor(0.0, device=device)
try:
    s_feat = img_emb  # 当前 encoder 输出（可训练）
    r_feat = ref_feats  # 冻结的参考特征（初始状态）
    
    if s_feat.shape == r_feat.shape:
        loss_distill = F.mse_loss(s_feat, r_feat)
    else:
        # 如果尺寸不同，先上采样
        r_pool = F.interpolate(r_feat, size=s_feat.shape[-2:], mode='bilinear')
        loss_distill = F.mse_loss(s_feat, r_pool)
except:
    loss_distill = torch.tensor(0.0, device=device)
```

**关键点**：
- 权重较小（gamma=0.1），避免过度约束
- 帮助保持预训练知识，同时允许适应新任务

#### 4.5 总损失

**代码位置**：`training/trainer.py::Trainer._compute_losses()` (第448-452行)

```python
# 损失权重
alpha = 1.0   # 对比损失（主要）
beta = 0.5    # IoU 损失（监控）
gamma = 0.1   # 蒸馏损失（辅助）

# 总损失
loss = loss_mask + alpha * loss_contrast + beta * loss_iou + gamma * loss_distill
```

**损失组成**：
- `loss_mask`：监控用（decoder 冻结，无梯度）
- `loss_contrast`：**主要训练信号**（有梯度）
- `loss_iou`：监控用（无梯度）
- `loss_distill`：辅助训练信号（有梯度，权重小）

---

### 阶段 5：反向传播与更新

**相关代码文件**：
- `training/trainer.py::Trainer.train()` - 主训练循环
- `training/trainer.py::Trainer._update_ema_teacher()` - EMA Teacher 更新

#### 5.1 反向传播

**作用**：计算梯度并更新可训练参数（image_encoder + proj）。

**代码位置**：`training/trainer.py::Trainer.train()` (第666-694行)

```python
# 反向传播
# 支持梯度累积
loss = loss / gradient_accumulation_steps
loss.backward()

# 梯度更新（每 gradient_accumulation_steps 步更新一次）
if (step + 1) % gradient_accumulation_steps == 0:
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
    
    # 优化器更新（支持混合精度）
    if use_amp:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    
    optimizer.zero_grad()
    
    # 更新学习率调度器
    scheduler.step()  # CosineAnnealingLR
```

**关键点**：
- 使用 **AdamW** 优化器
- 使用 **CosineAnnealingLR** 学习率调度器
- 支持梯度累积以模拟更大的 batch size
- 支持混合精度训练（AMP）
- 只有 `image_encoder` 和 `proj` 的参数会被更新
- Decoder 和 prompt_encoder 保持冻结

#### 5.2 EMA Teacher 更新

**作用**：使用指数移动平均更新 Teacher 模型，提供更稳定的伪标签。

**代码位置**：`training/trainer.py::Trainer._update_ema_teacher()` (第596-606行)

```python
# EMA 更新 Teacher
with torch.no_grad():
    ema_decay = 0.999
    for t_param, s_param in zip(teacher.parameters(), sam.parameters()):
        # 指数移动平均
        t_param.data = ema_decay * t_param.data + (1 - ema_decay) * s_param.data
```

**EMA 公式**：
```
teacher_param = 0.999 * teacher_param + 0.001 * student_param
```

**关键点**：
- Teacher 缓慢更新，提供稳定的伪标签
- Student 快速学习，逐步改善
- 形成自训练循环

---

## 三、训练流程图

```
输入图像 (B, 3, H, W)
    ↓
    resize 到 1024x1024，缩放 box 坐标
    ↓
┌─────────────────────────────────────┐
│  Teacher (冻结)                     │
│  ├─ image_encoder → t_img_emb       │
│  ├─ prompt_encoder (框 prompt)      │
│  └─ mask_decoder → 伪标签           │
│     └─ 熵筛选 → 高置信样本           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Student (可训练)                   │
│  ├─ image_encoder → img_emb ✅      │
│  │   └─ proj → z (对比特征) ✅      │
│  ├─ prompt_encoder (框 prompt) ❌   │
│  └─ mask_decoder → pred_logits ❌   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  损失计算                            │
│  ├─ loss_mask (监控) ❌             │
│  ├─ loss_contrast (主要) ✅         │
│  │   ├─ anchors: z[正样本位置]      │
│  │   ├─ positives: t_z[正样本位置]  │
│  │   └─ negatives: z[困难负样本]    │
│  ├─ loss_iou (监控) ❌              │
│  └─ loss_distill (辅助) ✅          │
└─────────────────────────────────────┘
    ↓
反向传播 → 更新 image_encoder + proj → EMA 更新 Teacher
```

**图例**：
- ✅ 可训练，有梯度
- ❌ 冻结，无梯度

---

## 四、关键超参数

**相关代码文件**：
- `config/args.py` - 所有超参数定义
- `training/trainer.py::Trainer._setup_optimizer()` - 优化器和调度器设置

### 4.1 学习率

**代码位置**：`config/args.py` (第40-41行)

```python
lr_encoder = 2e-6      # encoder 学习率（较小，因为是大模型）
lr_decoder = 1e-4      # decoder 学习率（不使用，decoder 冻结）
```

### 4.2 对比学习

**代码位置**：`config/args.py` (第45-48行)

```python
pos_samples = 256      # 正样本数量
neg_samples = 1024     # 负样本数量
temperature = 0.1      # InfoNCE 温度参数
```

### 4.3 筛选阈值

**代码位置**：`config/args.py` (第49行)

```python
entropy_thresh = 0.2   # 熵阈值（低熵 = 高置信）
```

### 4.4 损失权重

**代码位置**：`training/trainer.py::Trainer._compute_losses()` (第449行)

```python
alpha = 1.0   # 对比损失权重（主要）
beta = 0.5    # IoU 损失权重（监控）
gamma = 0.1   # 蒸馏损失权重（辅助）
```

### 4.5 EMA

**代码位置**：`training/trainer.py::Trainer.__init__()` (第77行)

```python
ema_decay = 0.999      # Teacher 更新系数
```

---

## 五、训练特点总结

### 5.1 优势

1. **参数高效**：只训练 encoder（~100M 参数），decoder 冻结
2. **特征学习**：通过对比学习优化特征表示
3. **保持能力**：利用 SAM 预训练的 decoder 能力
4. **弱监督**：只需要大小框，无需精确 mask 标注
5. **自训练**：Teacher 生成伪标签，Student 学习

### 5.2 训练信号

- **主要信号**：对比损失（InfoNCE）
- **辅助信号**：蒸馏损失
- **监控指标**：Mask 损失、IoU 损失

### 5.3 关键技术

1. **框 prompt**：使用大小框作为 prompt，提供弱监督信息
2. **图像预处理**：resize 到 SAM 输入尺寸（1024x1024），缩放 box 坐标
3. **显著性正样本**：双重验证（小框 + 伪标签）
4. **困难负样本**：针对两类错误（漏检、误检）
5. **熵筛选**：只使用高置信伪标签
6. **EMA Teacher**：稳定的伪标签生成
7. **多GPU支持**：使用 DataParallel 支持多GPU训练
8. **混合精度训练**：支持 AMP 以节省显存

---

## 六、使用示例

**相关代码文件**：
- `train.py` - 训练入口脚本
- `config/args.py::build_argparser()` - 参数解析器

**代码位置**：`train.py` (第13-20行)

```bash
python train.py \
  --data_root ./data/ISIC \
  --train_box_csv ./data/ISIC/train_boxes.csv \
  --sam_checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
  --output_dir ./outputs \
  --batch_size 4 \
  --epochs 30 \
  --img_size 1024 \
  --lr_encoder 2e-6 \
  --unfreeze_last_k 0 \  # 0=训练整个encoder，>0=只训练最后K个blocks
  --pos_samples 256 \
  --neg_samples 1024 \
  --temperature 0.1 \
  --entropy_thresh 0.2 \
  --use_amp \  # 可选：使用混合精度训练
  --multi_gpu \  # 可选：使用多GPU训练
  --use_gradient_checkpointing  # 可选：使用梯度检查点节省显存
```

---

## 七、训练监控

训练过程中会输出以下指标：

```
E1 L=2.3226 mask=0.2250 cont=2.0826 iou=0.0299 mIOU=0.8234 GPU:2.45/3.21GB
```

- `L`：总损失
- `mask`：Mask 损失（监控）
- `cont`：对比损失（主要训练信号）
- `iou`：IoU 损失（监控）
- `mIOU`：平均 IoU（监控指标）
- `GPU`：显存使用情况（已分配/已保留）

**期望趋势**：
- `cont` 应持续下降（主要优化目标）
- `mask` 和 `iou` 用于监控，可能波动
- `mIOU` 应逐步提升
- 总损失 `L` 应整体下降

---

## 八、Checkpoint 保存

**代码位置**：`training/trainer.py::Trainer._save_checkpoint()` (第608-624行)

```python
ckpt = {
    'sam_image_encoder': sam.image_encoder.state_dict(),  # 只保存 encoder
    'proj': proj.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch
}
torch.save(ckpt, 'checkpoint_epoch_{epoch+1}.pth')
```

**注意**：只保存 `image_encoder` 和 `proj`，因为 decoder 和 prompt_encoder 使用预训练权重。

---

## 九、总结

本训练框架通过**只微调 image_encoder** 的策略，结合**像素级对比学习**和**困难负样本挖掘**，在保持 SAM decoder 预训练能力的同时，优化 encoder 的特征表示能力。主要训练信号来自对比损失，通过拉近正样本、推远负样本，学习区分前景/背景的特征表示。

### 9.2 代码文件索引

**核心文件**：
- `train.py` - 训练入口
- `training/trainer.py` - 训练器主类（~750行），包含所有训练逻辑
- `config/args.py` - 参数配置（71行）

**模型相关**：
- `models/sam_wrapper.py` - SAM 模型封装（183行）
- `models/projection.py` - 投影头（52行）
- `models/losses.py` - 损失函数（71行）

**工具函数**：
- `utils/prompts.py` - Prompt 处理（134行）
- `utils/metrics.py` - 评估指标（61行）
- `utils/training_utils.py` - 训练工具函数
- `training/data_utils.py` - 数据工具（55行）

**数据集**：
- `dataset/ISIC.py` - ISIC 数据集类

### 9.1 关键修正说明

1. **Prompt 类型**：实际代码使用**框 prompt**（大框+小框），而非点 prompt
2. **图像预处理**：图像会被 resize 到 SAM 输入尺寸（1024x1024），box 坐标相应缩放
3. **IoU 损失**：计算的是 Student 预测与 Teacher 伪标签之间的 IoU
4. **多GPU支持**：代码支持 DataParallel 多GPU训练
5. **混合精度**：支持 AMP 混合精度训练以节省显存
6. **梯度累积**：支持梯度累积以模拟更大的 batch size

