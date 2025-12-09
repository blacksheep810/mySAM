# 模型评估方法文档

## 概述

本文档描述了基于 SAM 的像素级对比学习模型的评估方法，参考 WeSAM 论文中的评估标准，使用 mIOU (mean Intersection over Union) 作为主要评价指标。

## 评估指标

### 1. mIOU (mean Intersection over Union)

mIOU 是语义分割任务中最常用的评价指标之一，用于衡量预测 mask 与 ground truth mask 之间的重叠程度。

#### 计算公式

对于每个样本：
```
IoU = Intersection / Union
    = (pred_mask ∩ gt_mask) / (pred_mask ∪ gt_mask)
```

对于整个数据集：
```
mIOU = (1/N) * Σ IoU_i
```
其中 N 是样本总数。

#### 计算步骤

1. **二值化预测结果**：使用阈值 0.5 将预测的 logits 转换为二值 mask
   ```python
   pred_bin = (torch.sigmoid(pred_logits) > 0.5).float()
   ```

2. **计算交集和并集**：
   ```python
   intersection = (pred_bin * gt_mask).sum()
   union = ((pred_bin + gt_mask) > 0).float().sum()
   ```

3. **计算 IoU**：
   ```python
   iou = intersection / (union + epsilon)  # epsilon 防止除零
   ```

4. **计算平均值**：对所有样本的 IoU 求平均得到 mIOU

### 2. F1 Score (可选)

F1 Score 是精确率 (Precision) 和召回率 (Recall) 的调和平均数，也可以作为辅助评价指标。

## 评估流程

### 1. 模型准备

- 加载训练好的 checkpoint（包含 `sam_image_encoder` 和 `proj` 的权重）
- 将模型设置为评估模式 (`model.eval()`)
- 禁用梯度计算 (`torch.no_grad()`)

### 2. 数据准备

- 加载测试集/验证集数据
- 使用与训练时相同的数据预处理流程
- 准备 ground truth masks

### 3. 预测过程

对于每个样本：
1. 将图像输入到模型的 `image_encoder` 获取特征
2. 使用 prompt（box 或 point）通过 `prompt_encoder` 编码
3. 通过 `mask_decoder` 生成预测 mask
4. 将低分辨率 mask 上采样到原始图像尺寸

### 4. 指标计算

- 对每个 batch 计算 IoU
- 使用 `AverageMeter` 累计所有样本的 IoU
- 最终得到整个数据集的平均 mIOU

## 评估代码结构

### 主要函数

1. **`compute_miou(pred_logits, gt_masks, threshold=0.5)`**
   - 计算预测 mask 和 ground truth mask 之间的 mIOU
   - 自动处理尺寸不匹配的情况
   - 支持自定义二值化阈值

2. **`validate(model, dataloader, device, ...)`**
   - 完整的评估流程
   - 遍历数据加载器
   - 计算并返回 mIOU 和 F1 Score

3. **`load_checkpoint(checkpoint_path, model, device)`**
   - 加载训练好的 checkpoint
   - 恢复模型权重

## 评估参数

### 关键参数

- **threshold**: 二值化阈值，默认 0.5
- **img_size**: 图像尺寸，需要与训练时一致
- **sam_input_size**: SAM 模型期望的输入尺寸（1024x1024）
- **prompt_type**: prompt 类型（"box" 或 "point"）

### 评估设置

- **batch_size**: 评估时的 batch size，可以比训练时大（因为不需要梯度）
- **num_workers**: 数据加载的线程数
- **device**: 使用的设备（cuda 或 cpu）

## 结果保存

评估结果会保存到 CSV 文件中，包含以下信息：
- 模型名称
- Prompt 类型
- Mean IoU
- Mean F1 Score
- 评估时的迭代次数/epoch

## 使用示例

### 从项目根目录运行

```bash
cd /root/workspace/mySAM

python validate/evaluate.py \
    --checkpoint_path ./outputs/checkpoint_epoch_30.pth \
    --data_root ./data/ISIC \
    --test_box_csv ./data/ISIC/test_boxes.csv \
    --sam_checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
    --model_type vit_b \
    --img_size 1024 \
    --batch_size 8 \
    --output_dir ./validate/results
```

### 从 validate 目录运行（推荐）

脚本已自动处理路径，可以从 `validate` 目录直接运行：

```bash
cd /root/workspace/mySAM/validate

python evaluate.py \
    --checkpoint_path ./outputs/checkpoint_epoch_30.pth \
    --data_root ./data/ISIC \
    --test_box_csv ./data/ISIC/test_boxes.csv \
    --sam_checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
    --model_type vit_b \
    --img_size 1024 \
    --batch_size 8 \
    --output_dir ./results
```

**注意**: 脚本会自动将相对路径转换为相对于项目根目录的绝对路径，因此可以从任何目录运行。

## 参考

- WeSAM 论文评估方法
- Segmentation Models PyTorch (smp) 库的评估函数
- SAM (Segment Anything Model) 官方实现

## 注意事项

1. **路径处理**：脚本会自动处理相对路径，将其转换为相对于项目根目录的绝对路径，可以从任何目录运行
2. **尺寸一致性**：确保评估时使用的图像尺寸与训练时一致（默认 1024）
3. **Prompt 类型**：评估时使用的 prompt 类型（box/point）应与训练时一致
4. **数据预处理**：评估时的数据预处理应与训练时完全一致
5. **模型状态**：评估前务必将模型设置为 `eval()` 模式（脚本已自动处理）
6. **内存管理**：评估完成后及时清理 GPU 缓存
7. **数据路径**：
   - 数据集根目录应为 `./data/ISIC`
   - 测试集 CSV 文件应为 `./data/ISIC/test_boxes.csv`

## 常见问题

### Q: mIOU 值很低怎么办？
A: 检查以下几点：
- 模型是否加载正确
- 数据预处理是否与训练时一致
- Prompt 是否正确
- 阈值设置是否合理

### Q: 如何评估多个 checkpoint？
A: 可以编写脚本循环加载不同的 checkpoint 进行评估，或使用 `--checkpoint_dir` 参数批量评估。

### Q: 评估速度慢怎么办？
A: 
- 增大 batch_size（评估时不需要梯度，可以使用更大的 batch）
- 使用多 GPU（如果可用）
- 减少数据加载的 num_workers（如果 CPU 资源有限）

