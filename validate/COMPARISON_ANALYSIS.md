# mIOU计算方法对比分析

## 问题：为什么mySAM和wesam的mIOU表现差异很大？

## 两种计算方法对比

### 1. wesam的方法（使用smp库）

**代码位置**: `wesam/utils/eval_utils.py` 第66-72行

```python
batch_stats = smp.metrics.get_stats(
    pred_mask,
    gt_mask.int(),
    mode='binary',
    threshold=0.5,
)
batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
```

**特点**:
- 使用 `segmentation_models_pytorch` (smp) 库
- `reduction="micro-imagewise"`：**先对所有图像的所有像素计算TP/FP/FN，然后计算一个全局IoU**
- 这是**micro级别的聚合**（像素级别）

**计算公式**:
```
TP_total = 所有图像的TP之和
FP_total = 所有图像的FP之和  
FN_total = 所有图像的FN之和
IoU_micro = TP_total / (TP_total + FP_total + FN_total)
```

### 2. mySAM的方法（自定义实现）

**代码位置**: `mySAM/model.py` 第151-179行

```python
def compute_miou(pred_logits, gt_masks, threshold=0.5):
    pred_bin = (torch.sigmoid(pred_logits) > 0.5).float()
    gt_bin = (gt_masks > threshold).float()
    
    # 计算每个样本的IoU
    intersection = (pred_bin * gt_bin).sum(dim=[1, 2, 3])  # (B,)
    union = ((pred_bin + gt_bin) > 0).float().sum(dim=[1, 2, 3])  # (B,)
    iou_per_sample = intersection / (union + 1e-6)  # (B,)
    
    # 返回平均IoU（macro）
    return iou_per_sample.mean().item()
```

**特点**:
- **先计算每个样本的IoU，然后取平均**
- 这是**macro级别的聚合**（样本级别）

**计算公式**:
```
IoU_i = TP_i / (TP_i + FP_i + FN_i)  # 每个样本的IoU
mIOU_macro = (1/N) * Σ IoU_i  # 样本平均
```

## 关键差异

### Micro vs Macro的区别

| 方法 | 聚合级别 | 计算公式 | 特点 |
|------|---------|---------|------|
| **Micro (wesam)** | 像素级别 | `TP_total / (TP_total + FP_total + FN_total)` | 偏向大目标，大目标像素多，权重高 |
| **Macro (mySAM)** | 样本级别 | `mean(IoU_1, IoU_2, ..., IoU_N)` | 所有样本平等，小目标和大目标权重相同 |

### 为什么会有差异？

**示例**：
- 样本1：大目标，IoU=0.8，像素数=1000
- 样本2：小目标，IoU=0.2，像素数=10

**Macro方法**（mySAM）:
```
mIOU = (0.8 + 0.2) / 2 = 0.5
```
两个样本权重相同

**Micro方法**（wesam）:
```
mIOU ≈ 0.8  (因为大目标的1000个像素占主导)
```
大目标权重高

## 为什么表现差这么多？

### 可能的原因

1. **评估方法不同**
   - wesam使用micro-imagewise（偏向大目标）
   - mySAM使用macro（所有样本平等）
   - **如果数据集中大目标多，micro会给出更高的mIOU**

2. **数据预处理差异**
   - 图像尺寸、归一化方式可能不同
   - Box坐标的处理方式可能不同

3. **Prompt方式差异**
   - wesam可能使用不同的prompt策略
   - Point采样方式可能不同

4. **模型训练差异**
   - wesam可能训练了整个模型（包括decoder）
   - mySAM只训练了encoder

## 如何统一评估方法？

### 方案1：修改mySAM使用smp库（推荐）

```python
import segmentation_models_pytorch as smp

def compute_miou_smp(pred_logits, gt_masks, threshold=0.5):
    """使用smp库计算mIOU（与wesam一致）"""
    pred_bin = (torch.sigmoid(pred_logits) > threshold).float()
    gt_bin = (gt_masks > threshold).float()
    
    # 对每个样本计算
    batch_ious = []
    for pred, gt in zip(pred_bin, gt_bin):
        batch_stats = smp.metrics.get_stats(
            pred.unsqueeze(0),
            gt.int().unsqueeze(0),
            mode='binary',
            threshold=0.5,
        )
        iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
        batch_ious.append(iou)
    
    return torch.tensor(batch_ious).mean().item()
```

### 方案2：修改wesam使用macro方法

需要修改wesam的评估代码，使用macro reduction。

## 建议

1. **统一评估方法**：建议使用smp库的micro-imagewise方法，与wesam保持一致
2. **同时报告两种方法**：可以同时计算macro和micro，便于对比
3. **检查数据预处理**：确保图像尺寸、归一化等与wesam一致
4. **检查prompt方式**：确保使用的prompt类型和采样方式一致

## 参考

- [segmentation_models_pytorch文档](https://github.com/qubvel/segmentation_models.pytorch)
- wesam论文中的评估方法说明

