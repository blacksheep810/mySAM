#!/bin/bash
# 解决 OOM 问题的训练脚本
# 包含所有显存优化选项

# 设置 PyTorch CUDA 内存分配配置（减少碎片化）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ========== 推荐配置：平衡显存和性能 ==========
# batch_size=2 + gradient_accumulation_steps=2 = 有效 batch_size=4
# 比 batch_size=1 更稳定，训练效果更好
python model.py \
  --data_root ./data/ISIC \
  --train_box_csv ./data/ISIC/train_boxes.csv \
  --sam_checkpoint ./checkpoints/sam_vit_h_4b8939.pth \
  --unfreeze_last_k 2 \
  --batch_size 2 \
  --use_amp \
  --gradient_accumulation_steps 2 \
  --epochs 30 \
  --output_dir ./outputs

# ========== 如果还是 OOM，使用最小显存配置 ==========
# batch_size=1 + gradient_accumulation_steps=4 = 有效 batch_size=4
# 显存占用最小，但训练速度较慢，梯度估计不稳定
# 取消下面的注释以使用此配置：
# python model.py \
#   --data_root ./data/ISIC \
#   --train_box_csv ./data/ISIC/train_boxes.csv \
#   --sam_checkpoint ./checkpoints/sam_vit_h_4b8939.pth \
#   --unfreeze_last_k 2 \
#   --batch_size 1 \
#   --use_amp \
#   --use_gradient_checkpointing \
#   --gradient_accumulation_steps 4 \
#   --epochs 30 \
#   --output_dir ./outputs

# ========== 其他优化选项 ==========
# 1. 使用更小的模型：--model_type vit_b --sam_checkpoint ./checkpoints/sam_vit_b_01ec64.pth
# 2. 减小图像尺寸：--img_size 512（可能影响精度）
# 3. 只解冻更少的层：--unfreeze_last_k 1
