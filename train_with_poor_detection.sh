#!/bin/bash
# 训练脚本 - 平台期 3.0 配置
# 功能：
# 1. 在训练过程中实时检测 mIoU 表现差的样本并保存可视化结果
# 2. 自动统计和均衡正负样本数量，确保训练稳定性
# 3. 使用自适应并集交集策略进行正负样本选择
# 4. 启用过渡区域采样密度控制，强化边界判别能力
# 5. 启用 pseudo Dice 强化（高置信区域 + 边界加权）
# 6. 使用慢 EMA teacher 与 TS 分歧驱动困难样本采样
# 调优参考：docs/训练日志分析与调优建议.md

# 进入脚本所在目录（项目根目录），兼容不同部署路径
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 激活 py310env（PyTorch+CUDA 11.8）
if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate py310env
fi

python -u train.py \
  --data_root ./data/ISIC \
  --train_box_csv ./data/ISIC/train_boxes_update.csv \
  --test_box_csv ./data/ISIC/test_boxes.csv \
  --sam_checkpoint ./checkpoints/sam_vit_h_4b8939.pth \
  --model_type vit_h \
  --output_dir ./outputs_huge_with_poor_detection_3_0 \
  --batch_size 4 \
  --epochs 30 \
  --img_size 1024 \
  --lr_encoder 5e-6 \
  --lr_decoder 5e-5 \
  --weight_decay 1e-4 \
  --proj_dim 64 \
  --pos_samples 256 \
  --neg_samples 1024 \
  --temperature 0.1 \
  --entropy_thresh 0.15 \
  --pos_neg_ratio 0.25 \
  --unfreeze_last_k 3 \
  --use_amp \
  --use_gradient_checkpointing \
  --gradient_accumulation_steps 2 \
  --device "${GPU_DEVICE:-cuda}" \
  --save_every 1 \
  --save_best_ckpt \
  --early_stop_patience 10 \
  --hard_sample_max_weight 0.12 \
  --hard_sample_start_progress 0.75 \
  --detect_poor_samples \
  --poor_sample_iou_threshold_severe 0.75 \
  --poor_sample_iou_threshold_moderate 0.8 \
  --poor_sample_save_interval 1 \
  --teacher_warmup_epochs 3 \
  --hard_sample_min_confidence 0.75 \
  --transition_region_enabled \
  --transition_region_sampling_ratio 1.2 \
  --transition_region_min_confidence 0.65 \
  --transition_region_max_ratio 0.15 \
  --student_strong_aug \
  --pseudo_dice_weight 0.2 \
  --pseudo_dice_use_confidence_mask \
  --pseudo_dice_boundary_weighted \
  --pseudo_dice_boundary_alpha 1.0 \
  --ema_fixed_decay 0.999 \
  --ema_update_interval 1 \
  --hard_sample_use_disagreement \
  --hardness_alpha 0.5 \
  --hardness_beta 0.5
