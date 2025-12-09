# mIOU计算方法对比：mySAM vs wesam

## 核心差异

### wesam的方法
- **使用库**: `segmentation_models_pytorch` (smp)
- **Reduction**: `"micro-imagewise"`
- **计算方式**: 先对所有图像的**所有像素**计算TP/FP/FN，然后计算一个全局IoU
- **特点**: **偏向大目标**（像素多的目标权重高）

### mySAM的方法
- **使用库**: 自定义实现
- **Reduction**: `"macro"`（样本级别）
- **计算方式**: 先计算**每个样本**的IoU，然后取平均
- **特点**: **所有样本平等**（小目标和大目标权重相同）

## 为什么表现差这么多？

### 1. 评估方法不同（主要原因）

**Micro方法**（wesam）：
```
将所有像素合并：
TP_total = Σ TP_i (所有图像的TP之和)
FP_total = Σ FP_i
FN_total = Σ FN_i
IoU_micro = TP_total / (TP_total + FP_total + FN_total)
```

**Macro方法**（mySAM）：
```
先计算每个样本：
IoU_i = TP_i / (TP_i + FP_i + FN_i)
然后取平均：
mIOU_macro = mean(IoU_1, IoU_2, ..., IoU_N)
```

### 2. 实际影响

**示例场景**：
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

### 3. 数据集特征影响

- **如果数据集中大目标多**：Micro方法会给出更高的mIOU
- **如果数据集中小目标多**：Macro方法可能更高
- **如果目标大小均匀**：两种方法结果接近

## 如何使用新功能

### 1. 使用wesam的方法（micro-imagewise）

```bash
cd /root/workspace/mySAM/validate

python evaluate.py \
    --checkpoint_path ./outputs/checkpoint_epoch_30.pth \
    --data_root ./data/ISIC \
    --test_box_csv ./data/ISIC/test_boxes.csv \
    --sam_checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
    --model_type vit_b \
    --use_smp \
    --img_size 1024 \
    --batch_size 8
```

### 2. 同时对比两种方法（推荐）

```bash
python evaluate.py \
    --checkpoint_path ./outputs/checkpoint_epoch_30.pth \
    --data_root ./data/ISIC \
    --test_box_csv ./data/ISIC/test_boxes.csv \
    --sam_checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
    --model_type vit_b \
    --compare_methods \
    --img_size 1024 \
    --batch_size 8
```

这会同时计算两种方法并显示对比结果。

### 3. 安装smp库（如果还没有）

```bash
pip install segmentation-models-pytorch
```

## 输出示例

使用 `--compare_methods` 时会看到：

```
======================================================================
METHOD COMPARISON (Macro vs Micro)
======================================================================
Model                          Macro mIOU       Micro mIOU       Difference      
----------------------------------------------------------------------
SAM Pretrained                 0.2231           0.3500           +0.1269
Your Trained                   0.3536           0.4800           +0.1264
======================================================================

Note:
  - Macro: Sample-level average (mySAM method) - treats all samples equally
  - Micro: Pixel-level aggregation (wesam method) - favors large objects
  - If micro > macro: dataset has many large objects
  - If macro > micro: dataset has many small objects
```

## 建议

1. **统一评估方法**：如果要与wesam对比，使用 `--use_smp` 参数
2. **同时报告两种方法**：使用 `--compare_methods` 了解数据集特征
3. **理解差异**：Micro方法偏向大目标，Macro方法对所有样本平等
4. **选择合适的方法**：
   - 如果关注大目标性能：使用Micro
   - 如果关注所有样本的平衡性能：使用Macro

## 参考

- wesam评估代码：`wesam/utils/eval_utils.py`
- smp库文档：https://github.com/qubvel/segmentation_models.pytorch

