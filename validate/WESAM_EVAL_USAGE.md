# 使用 wesam/utils/eval_utils.py 测试 SAM Huge 性能

## 概述

`wesam/utils/eval_utils.py` 提供了 `validate` 函数来评估 SAM 模型的性能。这个函数会计算 mIoU 和 F1 分数。

## 使用方法

### 1. 基本使用步骤

使用 `wesam/utils/eval_utils.py` 需要以下步骤：

1. **创建配置对象**（Box 格式）
2. **创建 Lightning Fabric**（用于分布式训练/评估）
3. **创建 Model 实例**并加载 SAM checkpoint
4. **加载数据集**
5. **调用 validate 函数**

### 2. 关键依赖

```python
from model import Model  # wesam 的 Model 类
from utils.eval_utils import validate
from datasets import call_load_dataset  # 数据集加载函数
import lightning as L  # Lightning Fabric
from box import Box  # 配置管理
```

### 3. 配置示例

```python
cfg = Box({
    "gpu_ids": "0",  # GPU ID
    "val_batchsize": 4,
    "num_workers": 4,
    "dataset": "ISIC",
    "prompt": "box",  # 或 "point"
    "out_dir": "./outputs/eval",
    "name": "SAM_Huge",
    "model": {
        "type": "vit_h",  # vit_h, vit_l, vit_b
        "checkpoint": "./checkpoints",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": True,
        }
    },
    "datasets": {
        "ISIC": {
            "root_dir": "./data/ISIC/",
            "test_list": "./data/ISIC/test.csv"
        }
    },
    "csv_keys": ["Name", "Prompt", "Mean IoU", "Mean F1", "iters"],
})
```

### 4. 完整示例代码

参考 `test_sam_huge_wesam.py` 文件，其中包含完整的使用示例。

### 5. 运行示例

```bash
# 确保在 wesam 目录下，或者正确设置了 Python 路径
cd /root/workspace/wesam

# 运行评估脚本
python ../mySAM/validate/test_sam_huge_wesam.py
```

### 6. validate 函数参数说明

```python
validate(
    fabric: L.Fabric,      # Lightning Fabric 实例
    cfg: Box,              # 配置对象
    model: Model,          # SAM Model 实例
    val_dataloader: DataLoader,  # 验证数据加载器
    name: str,             # 模型名称（用于保存结果）
    iters: int = 0         # 迭代次数（用于日志）
)
```

**返回值：**
- `miou`: 平均 IoU 分数
- `f1_score`: 平均 F1 分数

### 7. 输出结果

- **控制台输出**：实时显示评估进度和结果
- **CSV 文件**：保存在 `cfg.out_dir` 目录下，文件名格式为 `{dataset}-{prompt}.csv`
- **TensorBoard 日志**：保存在 `cfg.out_dir` 目录下

### 8. 注意事项

1. **数据格式要求**：
   - 数据集必须返回 `(images, bboxes, gt_masks)` 格式
   - `images`: `(B, 3, H, W)` tensor
   - `bboxes`: list of bbox tensors
   - `gt_masks`: list of mask tensors

2. **模型接口要求**：
   - Model 必须实现 `forward(images, prompts)` 方法
   - 返回格式：`(image_embeddings, pred_masks, ious, res_masks)`

3. **Fabric 使用**：
   - 如果不需要分布式训练，可以只使用单个 GPU
   - 设置 `gpu_ids = "0"` 和 `num_devices = 1`

### 9. 与 mySAM 的对比

如果你已经在使用 `mySAM/validate/wesam_val.py` 或 `compare_models.py`，这些脚本已经适配了 wesam 的评估逻辑，可以直接使用，无需使用原始的 `wesam/utils/eval_utils.py`。

### 10. 常见问题

**Q: 如何测试不同的 prompt 类型？**
A: 修改 `cfg.prompt` 为 `"box"` 或 `"point"`，然后重新运行。

**Q: 如何测试不同的模型类型？**
A: 修改 `cfg.model.type` 为 `"vit_h"`、`"vit_l"` 或 `"vit_b"`。

**Q: 如何加载训练好的模型？**
A: 在创建 Model 后，使用 `fabric.load(ckpt_path)` 加载 checkpoint。

