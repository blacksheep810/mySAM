#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Parameter configuration module
Define all command line arguments and configuration management
"""

import argparse


def build_argparser():
    """
    Build argument parser
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    p = argparse.ArgumentParser(
        description='SAM-based pixel-level contrastive learning fine-tuning image encoder'
    )
    
    # ========== Data parameters ==========
    p.add_argument('--data_root', type=str, required=True, help='ISIC dataset root directory')
    p.add_argument('--train_box_csv', type=str, required=True, help='Training set box CSV file path')
    p.add_argument('--test_box_csv', type=str, default=None, help='Test set box CSV file path (optional)')
    p.add_argument('--img_size', type=int, default=1024, help='Image size')
    
    # ========== Model parameters ==========
    p.add_argument('--sam_checkpoint', type=str, required=True, help='SAM checkpoint path')
    p.add_argument('--model_type', type=str, default='vit_h', 
                   choices=['vit_b', 'vit_l', 'vit_h'],
                   help='sam model type: vit_b, vit_l, vit_h (default: vit_h)')
    p.add_argument('--unfreeze_last_k', type=int, default=0, 
                   help='Unfreeze last K transformer blocks; 0 means train entire encoder')
    
    # ========== Training parameters ==========
    p.add_argument('--batch_size', type=int, default=4, help='Batch size')
    p.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    p.add_argument('--lr_decoder', type=float, default=1e-4, help='Decoder learning rate')
    p.add_argument('--lr_encoder', type=float, default=2e-6, help='Encoder learning rate')
    p.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    
    # ========== Contrastive learning parameters ==========
    p.add_argument('--proj_dim', type=int, default=64, help='Projection dimension')
    p.add_argument('--pos_samples', type=int, default=256, help='Positive samples count')
    p.add_argument('--neg_samples', type=int, default=1024, help='Negative samples count')
    p.add_argument('--temperature', type=float, default=0.1, help='Temperature parameter')
    p.add_argument('--entropy_thresh', type=float, default=0.2, help='Entropy threshold (for pseudo-label filtering)')
    p.add_argument('--pos_neg_ratio', type=float, default=0.25, help='Target positive/negative ratio (default 0.25 means 1:4 ratio)')
    
    # ========== Optimization parameters ==========
    p.add_argument('--use_gradient_checkpointing', action='store_true', 
                   help='Use gradient checkpointing to save memory (time for space)')
    p.add_argument('--use_amp', action='store_true', 
                   help='Use mixed precision training (AMP) to save memory')
    p.add_argument('--gradient_accumulation_steps', type=int, default=1, 
                   help='Gradient accumulation steps (equivalent to larger batch size)')
    
    # ========== GPU parameters ==========
    p.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    p.add_argument('--multi_gpu', action='store_true', 
                   help='Use multi-GPU training (auto-detect all available GPUs)')
    p.add_argument('--gpu_ids', type=str, default=None, 
                   help='Specify GPU IDs, e.g. "0,1" or "0,1,2,3", default uses all GPUs')
    
    # ========== Output parameters ==========
    p.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    p.add_argument('--save_every', type=int, default=1, help='Save checkpoint every N epochs')
    
    # ========== Poor sample detection parameters ==========
    p.add_argument('--detect_poor_samples', action='store_true',
                   help='Enable poor sample detection during training')
    p.add_argument('--poor_sample_iou_threshold_severe', type=float, default=0.75,
                   help='IoU threshold for severe poor samples (default: 0.75, samples with IoU < 0.75 are severe)')
    p.add_argument('--poor_sample_iou_threshold_moderate', type=float, default=0.8,
                   help='IoU threshold for moderate poor samples (default: 0.8, samples with 0.75 <= IoU < 0.8 are moderate)')
    p.add_argument('--poor_sample_save_interval', type=int, default=1,
                   help='Save poor samples every N epochs (default: 1)')
    
    # ========== Teacher-Student training parameters ==========
    p.add_argument('--teacher_warmup_epochs', type=int, default=3,
                   help='Number of epochs to wait before updating teacher (warm-up, default: 3)')
    p.add_argument('--hard_sample_min_confidence', type=float, default=0.7,
                   help='Minimum confidence threshold for hard samples (default: 0.7, only use high-confidence hard samples)')
    p.add_argument('--hard_sample_max_weight', type=float, default=0.2,
                   help='Maximum weight cap for hard samples (default: 0.2, doc suggests 0.2 to reduce miou degradation)')
    p.add_argument('--hard_sample_start_progress', type=float, default=0.6,
                   help='Training progress to start introducing hard samples (default: 0.6=60%%, 0.7=70%% to delay)')
    
    # ========== Checkpoint and early stopping (调优建议) ==========
    p.add_argument('--save_best_ckpt', action='store_true',
                   help='Also save checkpoint when mIOU reaches new best (for eval/deploy)')
    p.add_argument('--early_stop_patience', type=int, default=0,
                   help='Stop training if no mIOU improvement for N epochs (0=disabled, 5 suggested in doc)')
    
    # ========== Transition region sampling parameters ==========
    p.add_argument('--transition_region_enabled', action='store_true',
                   help='Enable transition region (between small box and big box) sampling for boundary learning')
    p.add_argument('--transition_region_sampling_ratio', type=float, default=1.5,
                   help='Sampling ratio for transition region compared to small box region (default: 1.5, means 1.5x more samples)')
    p.add_argument('--transition_region_min_confidence', type=float, default=0.6,
                   help='Minimum confidence threshold for transition region samples (default: 0.6, lower than small box due to higher uncertainty)')
    p.add_argument('--transition_region_max_ratio', type=float, default=0.3,
                   help='Maximum ratio of transition region samples in total positive samples (default: 0.3, means max 30% of positives from transition)')
    
    # ========== 强数据增强与伪标签 Dice（提升 mIOU 见效） ==========
    p.add_argument('--student_strong_aug', action='store_true',
                   help='Apply strong color augmentation to Student input (Teacher uses original; improves robustness)')
    p.add_argument('--pseudo_dice_weight', type=float, default=0.0,
                   help='Weight for pseudo-label Dice loss (Student pred vs Teacher pred, 0=disabled, 0.1-0.3 suggested)')
    p.add_argument('--pseudo_dice_use_confidence_mask', action='store_true',
                   help='Only compute pseudo Dice on teacher high-confidence pixels')
    p.add_argument('--pseudo_dice_confidence_thresh', type=float, default=0.8,
                   help='Teacher confidence threshold used by pseudo Dice confidence mask')
    p.add_argument('--pseudo_dice_boundary_weighted', action='store_true',
                   help='Upweight high-entropy / disagreement pixels in pseudo Dice')
    p.add_argument('--pseudo_dice_boundary_alpha', type=float, default=1.0,
                   help='Extra weight scale for boundary-aware pseudo Dice')

    # ========== Teacher 更新策略（平台期阶段重点） ==========
    p.add_argument('--ema_fixed_decay', type=float, default=0.0,
                   help='Fixed EMA decay; 0 means use built-in dynamic schedule, recommended ablation: 0.999 / 0.9995')
    p.add_argument('--ema_update_interval', type=int, default=1,
                   help='Update teacher every N optimizer steps (default 1 means every step)')

    # ========== TS 分歧驱动困难样本挖掘 ==========
    p.add_argument('--hard_sample_use_disagreement', action='store_true',
                   help='Rank hard samples by teacher-student disagreement + teacher entropy')
    p.add_argument('--hardness_alpha', type=float, default=0.5,
                   help='Weight of |p_t - p_s| in hardness score')
    p.add_argument('--hardness_beta', type=float, default=0.5,
                   help='Weight of teacher entropy in hardness score')
    
    return p
