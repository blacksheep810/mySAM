## 项目对比：`mySAM` vs `mySAM_amend`

### 1. 总体定位与目标

- **`wesam_amend`（上游原始项目）**
  - 面向多数据集、多场景的弱监督 SAM 适配框架，支持 COCO、VOC、Kvasir-SEG、ISIC 等多种自然图像与医学图像数据集。
  - 代码结构围绕单一入口 `adaptation.py` 与统一的 `configs/config.py` 展开，偏重**论文复现与多数据集实验**。
- **`mySAM` / `mySAM_amend`（本地定制版本）**
  - 聚焦于 **ISIC2016 皮肤病变分割** 场景，围绕 “只微调 SAM image encoder + 像素级对比学习 + 框级弱标注” 做了工程化重构。
  - 将原始 Wesam 的训练逻辑拆分为清晰的模块（`config/`, `models/`, `utils/`, `training/`, `dataset/`），并额外加入了数据准备、Web 弱标注工具与验证脚本，偏向**针对性实验与工程落地**。

**小结**：`wesam_amend` 更像“论文代码仓库”，`mySAM` / `mySAM_amend` 是在其思想基础上的 **ISIC 专项、模块化、本地化重构版本**。

---

### 2. 目录结构与模块化改动

- **`wesam_amend` 的典型结构**
  - 以 `adaptation.py` 为单一训练入口，内部同时负责：
    - 解析配置、加载数据与模型
    - 定义训练 / 验证循环
    - 管理损失与日志
  - 数据集、模型、损失等模块虽然拆分在 `datasets/`, `models/` 下，但整体耦合在一个大脚本中。

- **`mySAM` / `mySAM_amend` 的结构重构**
  - 核心入口：
    - `train.py`：通用训练入口（通过 `config/args.py` 解析命令行参数）。
    - `training/trainer.py`：`Trainer` 类封装完整训练逻辑。
    - `model_original.py`：保留原始 Wesam 风格入口以便对照。
  - 模块拆分：
    - `config/args.py`：集中管理训练/评估超参数与路径（替代原始 `configs/config.py` 的部分功能）。
    - `models/sam_wrapper.py`：封装 SAM 加载、Teacher 创建、多 GPU 设置等逻辑。
    - `models/projection.py`：像素级投影头 `PixelProjHead`。
    - `models/losses.py`：DiceLoss 与 `pixel_info_nce` 对比损失等。
    - `utils/metrics.py`、`utils/prompts.py`、`utils/training_utils.py`：指标计算、Prompt 处理、训练工具函数。
    - `training/data_utils.py`：`collate_fn_isic` 等数据打包函数。
    - `dataset/ISIC.py`：单独的 ISIC2016 数据集类。
  - `mySAM_amend` 在此基础上进一步：
    - 引入 `docs/` 目录记录二次实验与脚本更新说明。
    - 增加多个训练脚本（如 `train_huge_update.sh`, `train_with_poor_detection.sh`），用于快速切换不同实验配置。

**改动价值**：由单文件训练脚本演进为**高度模块化 + 多入口脚本**的结构，便于后续维护、迁移与扩展到新数据集或新损失。

---

### 3. 训练策略与超参数调整

#### 3.1 共同点

- 都基于 **Teacher–Student 自训练框架**：
  - Teacher：冻结或 EMA 更新的 SAM，用于生成伪标签。
  - Student：在框 prompt 约束下学习更强的 encoder 特征。
- 都采用 **像素级 InfoNCE 对比损失** 作为核心训练信号，辅以分割损失 / 蒸馏损失做约束。
- 都主要微调 **image encoder + 轻量投影头**，保持原始 SAM decoder 能力。

#### 3.2 `mySAM` 的训练细化

- 在 `TRAINING_PIPELINE.md` 中给出了完整的训练阶段拆解（模型初始化 → 数据准备 → Teacher 伪标签 → Student 前向 → 多种损失 → EMA 更新）。
- 显式区分：
  - **主要训练信号**：像素对比损失 `loss_contrast`（InfoNCE）。
  - **辅助信号**：特征蒸馏损失 `loss_distill`。
  - **监控项**：Mask 损失、IoU 损失与 mIoU 指标。
- 采用参数配置：
  - `lr_encoder`、`unfreeze_last_k`、`pos_samples`、`neg_samples`、`entropy_thresh` 等通过命令行传入，便于 grid search。

#### 3.3 `mySAM_amend` 的实验型调整

- 在 `train_huge_update.sh` 中进行了针对 Huge 模型的 **超参优化**：
  - `lr_encoder`: 从 `2e-6` 提升到 `5e-6`，加快 encoder 收敛。
  - `unfreeze_last_k`: 从 `2` 提升为 `3`，解冻更多 encoder block，提高表示能力。
  - `batch_size`: 从 2 提升到 4（在显存允许的前提下提升稳定性）。
  - 使用新的 `train_boxes_update.csv`，对应改进后的弱标注质量。
- 这些脚本改变的是 **训练强度与可学习自由度**，不改变整体框架设计。

**改动价值**：在保持原始 Wesam 思想的前提下，`mySAM` 明确了训练 pipeline，`mySAM_amend` 则围绕 ISIC 场景做了多轮 **实践驱动的超参调优与解冻策略探索**。

---

### 4. 数据与标注流程变化

- **`wesam_amend`**
  - 面向多数据集的通用配置，标注形式多样（COCO bbox、mask 等）。
  - ISIC 仅是其中一个数据集，没有单独针对的弱标注工具。

- **`mySAM` / `mySAM_amend`**
  - 专门为 ISIC2016 定义了：
    - `data/ISIC/ISBI2016_ISIC_Part1_Training_Data` / `..._GroundTruth` 目录结构。
    - `ISIC2016Dataset`（`dataset/ISIC.py`）与 `collate_fn_isic`。
    - `train_boxes.csv` / `test_boxes.csv`：包含 **大框（max_boxes\*）+ 小框（min_boxes\*）** 的弱标注文件。
  - 标注生成有两条路线：
    - 利用 GT `mask` 计算 **最小外接矩形 + 最大内接矩形**（见 `dataset/ISIC_deal.ipynb` 与 `utils/smart_box_generator.py` 的 `find_max_inner_rectangle`）。
    - 基于 **Web 弱标注工具** `utils/web_annotation.py`，在浏览器中点 4 个点生成大框，再用智能方法（轮廓 / 梯度 / 最大内接矩形）生成小框。

**改动价值**：从“假定已有边界框标注”的设置，升级为支持 **从 mask 自动生成、从 Web 工具交互生成** 两种 box 数据构建方式，极大方便了 ISIC 下游任务的快速实验。

---

### 5. 验证与可视化工具

- **`wesam_amend`**
  - 使用 `validate.py` 在多数据集上评估适配后的模型。

- **`mySAM` / `mySAM_amend`**
  - 新增 `validate/compare_models.py`：
    - 自动检测 checkpoint 中 encoder 的维度，推断模型类型（vit\_b / vit\_l / vit\_h）。
    - 加载对应类型的原始 SAM 作为基线，对比 mIoU。
    - 在 ISIC2016 上统一 box prompt 评估流程。
  - `validate/wesam_val.py`：适配原 Wesam 的评估工具，将 `mySAM` 训练出的 SAM 模型封装为 Wesam 风格接口，便于横向对比。
  - 提供可选可视化保存功能，输出 “原图 + 大小框 + 预测 mask + GT mask” 叠加图，便于误检/漏检分析。

**改动价值**：评估逻辑从“单 checkpoint + 单模型”扩展为“原始 SAM vs 适配后模型”的自动对比，并针对 ISIC 提供可视化分析能力。

---

### 6. 工程与环境层面的增强

- **配置与环境**
  - `mySAM`：新增 `ENVIRONMENT_SETUP.md`，详细记录 Conda 环境、依赖安装步骤，以及与 `segment-anything` 子模块的关系。
  - `mySAM_amend`：训练脚本中直接包含 `conda activate py310env`、`cd /mnt/mySAM` 等生产环境路径，方便在固定服务器上一键启动。

- **日志与断点**
  - `mySAM` 中的 `training/trainer.py` 支持：
    - 梯度累积、AMP 混合精度、多 GPU（`DataParallel`）训练。
    - 定期保存 checkpoint（仅 encoder + proj + optimizer），简化模型迁移。
  - `mySAM_amend` 在此基础上主要增加了不同输出目录（如 `outputs_huge_update_optimized`）以区分多组实验。

**改动价值**：将原项目中分散的环境说明、脚本参数固化到了文档与 shell 脚本中，使得在服务器上的批量实验更加稳定、可复现。

---

### 7. 总结：`mySAM` / `mySAM_amend` 相对 `wesam_amend` 的主要改进点

1. **场景聚焦**：从多数据集通用框架，聚焦到 ISIC2016 皮肤病变分割，训练与数据管线围绕该任务深度定制。
2. **结构模块化**：将原本高度耦合的 `adaptation.py` 拆分为 `config/`, `models/`, `utils/`, `training/`, `dataset/` 等清晰模块，降低维护难度。
3. **弱标注体系升级**：引入基于 mask 的自动外接/内接矩形计算 + Web 交互式弱标注工具，统一输出 `train/test_boxes.csv`，支持不同标注强度的实验。
4. **训练策略细化**：显式拆解并记录 Teacher–Student + 像素对比 + 困难负样本 + 特征蒸馏的完整 pipeline，并通过脚本在 `mySAM_amend` 中探索更激进的学习率与解冻策略。
5. **评估与可视化增强**：新增对比评估脚本与可视化工具，支持“原始 SAM vs 适配模型”在 ISIC2016 上的一键对比与结果分析。

整体来看，`mySAM` / `mySAM_amend` 可以被视为在 `wesam_amend` 思想之上的 **工程化、任务化、易用化重构版本**，更适合在 ISIC 这类单数据集场景下做系统实验与进一步研究。

