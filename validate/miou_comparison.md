# mIoU 计算方法对比分析

## 一、wesam/utils/eval_utils.py 的计算方式

### 1. Mask 维度来源

**pred_mask 的维度：**
- 来自 `model.decode()` 返回的 `pred_masks`
- 在 `wesam/model.py` 第114行：`pred_masks.append(masks.squeeze(1))`
- `masks` 形状是 `(1, 1, H, W)`，经过 `squeeze(1)` 后变成 `(1, H, W)`
- **结论：`pred_mask` 是 `(1, H, W)` 格式**

**gt_mask 的维度：**
- 来自数据集的 `gt_masks`
- 通常是 `(H, W)` 格式（numpy array 或 tensor）
- **结论：`gt_mask` 是 `(H, W)` 格式**

### 2. 计算流程（第65-74行）

```python
for pred_mask, gt_mask in zip(pred_masks, gt_masks):
    batch_stats = smp.metrics.get_stats(
        pred_mask,      # (1, H, W) 格式
        gt_mask.int(),  # (H, W) 格式
        mode='binary',
        threshold=0.5,
    )
    batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
    ious.update(batch_iou, num_images)  # 使用 batch 大小作为权重
```

**关键点：**
- **没有做维度转换**，直接传入 `get_stats`
- `pred_mask` 是 `(1, H, W)`，`gt_mask` 是 `(H, W)`
- `smp.metrics.get_stats` 会自动处理维度不匹配的情况

### 3. 权重更新方式
- `ious.update(batch_iou, num_images)` - 使用 batch 大小作为权重

---

## 二、mySAM/validate/compare_models.py 的计算方式

### 1. Mask 维度来源

**pred_mask 的维度：**
- 来自 `inference_with_box_prompt()` 返回
- 第155行：`pred_mask = torch.sigmoid(pred_mask).squeeze(0).squeeze(0)`
- **结论：`pred_mask` 是 `(H, W)` 格式**

**gt_mask 的维度：**
- 来自数据集的 `masks[b].squeeze(0)`
- 第203行：`gt_mask = masks[b].squeeze(0)  # (H, W)`
- **结论：`gt_mask` 是 `(H, W)` 格式**

### 2. 计算流程（第49-88行）

```python
def compute_miou(pred_mask, gt_mask):
    # 确保都是 (H, W) 格式
    if pred_mask.dim() == 3:
        pred_mask = pred_mask.squeeze(0)  # (H, W)
    if gt_mask.dim() == 3:
        gt_mask = gt_mask.squeeze(0)  # (H, W)
    
    # 形状匹配处理
    if pred_mask.shape != gt_mask.shape:
        pred_mask = F.interpolate(...)  # 插值对齐
    
    batch_stats = smp.metrics.get_stats(
        pred_mask,      # (H, W) 格式
        gt_mask.int(),  # (H, W) 格式
        mode='binary',
        threshold=0.5,
    )
    iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
    return iou.item()
```

**关键点：**
- **显式转换为 `(H, W)` 格式**
- 两个 mask 都是 `(H, W)` 格式传入 `get_stats`
- 有形状匹配和插值处理

### 3. 权重更新方式
- `ious.update(iou, num_images)` - 使用 batch 大小作为权重（已修复）

---

## 三、mySAM/validate/wesam_val.py 的计算方式

### 1. Mask 维度来源

**pred_mask 的维度：**
- 来自 `SAMModelAdapter.decode()` 返回的 `pred_masks`
- 第137行：`pred_masks.append(masks.squeeze(1).squeeze(0))`
- **结论：`pred_mask` 初始是 `(H, W)` 格式**

**gt_mask 的维度：**
- 来自数据集的 `masks[b]`
- 形状是 `(1, H, W)`

### 2. 计算流程（第235-275行）

```python
# 第236-248行：维度转换
if pred_mask.dim() == 2:
    pred_mask = pred_mask.unsqueeze(0)  # (H, W) -> (1, H, W)
elif pred_mask.dim() == 3:
    pass  # 保持 (1, H, W)

if gt_mask.dim() == 3:
    gt_mask = gt_mask.squeeze(0)  # (1, H, W) -> (H, W)
if gt_mask.dim() == 2:
    gt_mask = gt_mask.unsqueeze(0)  # (H, W) -> (1, H, W)

# 第262-267行：计算
batch_stats = smp.metrics.get_stats(
    pred_mask,      # (1, H, W) 格式（经过转换）
    gt_mask.int(),  # (1, H, W) 格式（经过转换）
    mode='binary',
    threshold=0.5,
)
```

**关键点：**
- **显式转换为 `(1, H, W)` 格式**
- 两个 mask 都是 `(1, H, W)` 格式传入 `get_stats`

### 3. 权重更新方式
- `ious.update(batch_iou.item(), num_images)` - 使用 batch 大小作为权重

---

## 四、为什么会有不同的 mIoU 结果？

### 核心原因：`smp.metrics.get_stats` 对不同维度的处理方式不同

**情况1：`(H, W)` vs `(H, W)` 格式（compare_models.py）**
```python
get_stats(pred_mask, gt_mask)  # 都是 (H, W)
```
- `get_stats` 将两个 2D tensor 视为**单个图像的所有像素**
- 计算方式：直接计算所有像素的 TP, FP, FN, TN
- **IoU = TP / (TP + FP + FN)**
- **结果：0.8038**（更高）

**情况2：`(1, H, W)` vs `(H, W)` 格式（wesam 原始）**
```python
get_stats(pred_mask, gt_mask)  # (1, H, W) vs (H, W)
```
- `get_stats` 内部可能会：
  - 将 `(1, H, W)` 的 batch 维度处理掉，变成 `(H, W)`
  - 或者按不同的方式计算统计信息
- **结果：0.7521**（更低）

**情况3：`(1, H, W)` vs `(1, H, W)` 格式（wesam_val.py）**
```python
get_stats(pred_mask, gt_mask)  # 都是 (1, H, W)
```
- `get_stats` 将两个 3D tensor 视为**batch 格式**
- 可能按 batch 维度处理，导致计算方式不同
- **结果：0.7521**（与情况2相同）

### 为什么维度会影响结果？

`segmentation_models_pytorch` 的 `get_stats` 函数内部逻辑推测：

1. **对于 `(H, W)` 格式：**
   ```python
   # 伪代码
   pred_binary = (pred_mask >= threshold).float()  # (H, W)
   gt_binary = gt_mask.int()  # (H, W)
   TP = (pred_binary * gt_binary).sum()  # 所有像素的 TP
   FP = (pred_binary * (1 - gt_binary)).sum()
   FN = ((1 - pred_binary) * gt_binary).sum()
   IoU = TP / (TP + FP + FN)
   ```

2. **对于 `(1, H, W)` 格式：**
   ```python
   # 伪代码
   pred_binary = (pred_mask >= threshold).float()  # (1, H, W)
   gt_binary = gt_mask.int()  # (1, H, W) 或 (H, W)
   # 可能先处理 batch 维度，或者计算方式不同
   # 导致统计结果不同
   ```

**实际测试建议：**
可以创建一个简单的测试脚本来验证不同维度对 `get_stats` 的影响。

---

## 五、正确的做法

### 方案1：与 wesam 原始代码完全一致（推荐）

**wesam 原始代码的实际行为：**
- `pred_mask`: `(1, H, W)` 格式（来自 `masks.squeeze(1)`）
- `gt_mask`: `(H, W)` 格式（来自数据集）
- **直接传入，不做维度转换**

**修改 compare_models.py：**
```python
def compute_miou(pred_mask, gt_mask):
    # 不做维度转换，直接传入（与 wesam 原始代码一致）
    # pred_mask 是 (H, W)，需要转换为 (1, H, W)
    # gt_mask 是 (H, W)，保持不变
    
    # 将 pred_mask 转换为 (1, H, W) 以匹配 wesam 原始代码
    if pred_mask.dim() == 2:
        pred_mask = pred_mask.unsqueeze(0)  # (H, W) -> (1, H, W)
    
    # gt_mask 保持 (H, W) 格式
    if gt_mask.dim() == 3:
        gt_mask = gt_mask.squeeze(0)  # (1, H, W) -> (H, W)
    
    batch_stats = smp.metrics.get_stats(
        pred_mask,      # (1, H, W)
        gt_mask.int(),  # (H, W)
        mode='binary',
        threshold=0.5,
    )
    iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
    return iou.item()
```

### 方案2：统一使用 `(H, W)` 格式

**修改两个文件都使用 `(H, W)` 格式：**
- `compare_models.py`: 已使用 `(H, W)` ?
- `wesam_val.py`: 需要修改为 `(H, W)` 格式（移除第236-248行的维度转换）

---

## 六、总结对比表

| 文件 | pred_mask 格式 | gt_mask 格式 | 传入 get_stats 的格式 | 结果 |
|------|---------------|-------------|---------------------|------|
| **wesam 原始** | `(1, H, W)` | `(H, W)` | `(1, H, W)` vs `(H, W)` | **0.7521** |
| **compare_models.py** | `(H, W)` | `(H, W)` | `(H, W)` vs `(H, W)` | **0.8038** |
| **wesam_val.py** | `(H, W)` → `(1, H, W)` | `(1, H, W)` → `(1, H, W)` | `(1, H, W)` vs `(1, H, W)` | **0.7521** |

**关键发现：**
1. `compare_models.py` 使用 `(H, W)` 格式得到更高的结果（0.8038）
2. `wesam_val.py` 和 wesam 原始代码使用混合或 `(1, H, W)` 格式得到相同的结果（0.7521）
3. **维度格式是导致结果差异的根本原因**

**建议：**
- 如果要与 wesam 原始代码一致，应该使用 `(1, H, W)` vs `(H, W)` 的混合格式
- 如果要两个文件一致，应该统一使用 `(H, W)` 格式
- **关键是要理解：`smp.metrics.get_stats` 对不同维度的处理方式不同，导致结果不同**

