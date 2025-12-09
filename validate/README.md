# 模型评估工具

本目录包含模型评估相关的文档和代码。

## 文件说明

- `EVALUATION_METHOD.md`: 详细的评估方法文档，说明 mIOU 的计算方法和评估流程
- `evaluate.py`: 模型评估脚本，用于评估训练好的模型
- `README.md`: 本文件，使用说明

## 快速开始

### 1. 基本使用（从项目根目录运行）

```bash
# 从项目根目录运行
cd /root/workspace/mySAM

python validate/evaluate.py \
    --checkpoint_path ./outputs/checkpoint_epoch_30.pth \
    --data_root ./data/ISIC \
    --test_box_csv ./data/ISIC/test_boxes.csv \
    --sam_checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
    --model_type vit_b \
    --img_size 1024 \
    --batch_size 8
```

### 2. 从 validate 目录运行（推荐）

脚本已自动处理路径，可以从 `validate` 目录直接运行：

```bash
# 从 validate 目录运行（路径会自动转换为相对于项目根目录）
cd /root/workspace/mySAM/validate

python evaluate.py \
    --checkpoint_path ./outputs/checkpoint_epoch_30.pth \
    --data_root ./data/ISIC \
    --test_box_csv ./data/ISIC/test_boxes.csv \
    --sam_checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
    --model_type vit_b \
    --img_size 1024 \
    --batch_size 8
```

### 3. 使用 Point Prompt

```bash
cd /root/workspace/mySAM/validate

python evaluate.py \
    --checkpoint_path ./outputs/checkpoint_epoch_30.pth \
    --data_root ./data/ISIC \
    --test_box_csv ./data/ISIC/test_boxes.csv \
    --sam_checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
    --model_type vit_b \
    --prompt_type point \
    --num_points 10 \
    --img_size 1024 \
    --batch_size 8
```

## 参数说明

### 必需参数

- `--checkpoint_path`: 训练好的模型 checkpoint 路径
- `--data_root`: 数据集根目录
- `--test_box_csv`: 测试集 box CSV 文件路径
- `--sam_checkpoint`: SAM 预训练模型 checkpoint 路径

### 可选参数

- `--model_type`: SAM 模型类型，可选 `vit_b`, `vit_l`, `vit_h`，默认 `vit_b`
- `--img_size`: 图像尺寸，默认 1024
- `--batch_size`: 评估时的 batch size，默认 8（评估时可以使用更大的 batch）
- `--prompt_type`: Prompt 类型，`box` 或 `point`，默认 `box`
- `--num_points`: 使用 point prompt 时的采样点数，默认 10
- `--num_workers`: 数据加载线程数，默认 4
- `--output_dir`: 结果保存目录，默认 `./validate/results`
- `--device`: 使用的设备，`cuda` 或 `cpu`，默认 `cuda`

## 输出结果

评估结果会：
1. 打印到控制台，包括：
   - Checkpoint 路径
   - Epoch 编号
   - Prompt 类型
   - 数据集信息
   - 样本数量
   - mIOU 值

2. 保存到 CSV 文件 (`validate/results/evaluation_results.csv`)，包含：
   - 时间戳
   - Checkpoint 路径
   - Prompt 类型
   - 数据集路径
   - mIOU 值
   - 样本数量
   - 图像尺寸
   - Batch size

## 评估指标

主要评估指标是 **mIOU (mean Intersection over Union)**，范围在 0-1 之间：
- mIOU > 0.7: 较好
- mIOU > 0.8: 很好
- mIOU > 0.9: 优秀

详细的计算方法请参考 `EVALUATION_METHOD.md`。

## 注意事项

1. **路径处理**: 脚本会自动将相对路径转换为相对于项目根目录的绝对路径，可以从任何目录运行
2. **数据路径**: 确保数据路径正确，测试集 CSV 文件格式与训练时一致
   - 数据集根目录: `./data/ISIC`
   - 测试集 CSV: `./data/ISIC/test_boxes.csv`
3. **模型类型**: `--model_type` 必须与训练时使用的 SAM 模型类型一致
4. **图像尺寸**: `--img_size` 应与训练时使用的图像尺寸一致（默认 1024）
5. **Prompt 类型**: 评估时使用的 prompt 类型应与训练时一致（如果训练时使用了特定 prompt）
6. **GPU 内存**: 如果遇到 GPU 内存不足，可以减小 `--batch_size` 或使用 `--device cpu`

## 批量评估多个 Checkpoint

可以编写简单的脚本来批量评估多个 checkpoint：

```bash
#!/bin/bash
# 从项目根目录运行
cd /root/workspace/mySAM

for epoch in {1..30}; do
    python validate/evaluate.py \
        --checkpoint_path ./outputs/checkpoint_epoch_${epoch}.pth \
        --data_root ./data/ISIC \
        --test_box_csv ./data/ISIC/test_boxes.csv \
        --sam_checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
        --model_type vit_b \
        --img_size 1024 \
        --batch_size 8
done
```

或者从 validate 目录运行：

```bash
#!/bin/bash
# 从 validate 目录运行
cd /root/workspace/mySAM/validate

for epoch in {1..30}; do
    python evaluate.py \
        --checkpoint_path ./outputs/checkpoint_epoch_${epoch}.pth \
        --data_root ./data/ISIC \
        --test_box_csv ./data/ISIC/test_boxes.csv \
        --sam_checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
        --model_type vit_b \
        --img_size 1024 \
        --batch_size 8
done
```

## 常见问题

### Q: 评估时出现 CUDA out of memory 错误
A: 减小 `--batch_size`，或者使用 `--device cpu`（速度会较慢）

### Q: 找不到 segment_anything 模块
A: 确保 segment-anything 目录在正确的位置，或者修改脚本中的路径设置

### Q: mIOU 值很低
A: 检查：
- Checkpoint 是否正确加载
- 数据预处理是否与训练时一致
- Prompt 是否正确
- 图像尺寸是否匹配

### Q: 如何评估其他数据集？
A: 需要创建对应的数据集类（参考 `dataset/ISIC.py`），并修改评估脚本中的数据加载部分

## 参考

- `EVALUATION_METHOD.md`: 详细的评估方法文档
- `../model.py`: 训练脚本，包含 `compute_miou` 等函数
- WeSAM 论文: 评估方法的参考来源

