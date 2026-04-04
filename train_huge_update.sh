#!/bin/bash
# 优化后的训练脚本 - 提高学习率和解冻层数
# 使用 train_boxes_update.csv 微调 SAM Huge (vit_h) 模型
# 优化点：
# 1. lr_encoder: 2e-6 -> 5e-6 (提高2.5倍，加快训练)
# 2. unfreeze_last_k: 2 -> 3 (增加解冻层数，提高学习能力，避免显存不足)

cd /mnt/mySAM

conda activate py310env

python train.py \
  --data_root ./data/ISIC \
  --train_box_csv ./data/ISIC/train_boxes_update.csv \
  --test_box_csv ./data/ISIC/test_boxes.csv \
  --sam_checkpoint ./checkpoints/sam_vit_h_4b8939.pth \
  --model_type vit_h \
  --output_dir ./outputs_huge_update_optimized \
  --batch_size 4 \
  --epochs 30 \
  --img_size 1024 \
  --lr_encoder 5e-6 \
  --lr_decoder 1e-4 \
  --weight_decay 1e-4 \
  --proj_dim 64 \
  --pos_samples 256 \
  --neg_samples 1024 \
  --temperature 0.1 \
  --entropy_thresh 0.2 \
  --unfreeze_last_k 3 \
  --use_amp \
  --use_gradient_checkpointing \
  --gradient_accumulation_steps 2 \
  --device cuda \
  --save_every 1

