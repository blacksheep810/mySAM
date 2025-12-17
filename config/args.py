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
    
    return p
