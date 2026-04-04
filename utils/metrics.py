#!/usr/bin/env python
# coding: utf-8

"""
评估指标模块
实现各种评估指标，如 mIOU、熵等
"""

import torch
import torch.nn.functional as F


def compute_miou(pred_logits, gt_masks, threshold=0.5):
    """
    计算预测mask和ground truth mask之间的mIOU (mean Intersection over Union)
    
    Args:
        pred_logits: (B, 1, H, W) 预测的logits
        gt_masks: (B, 1, H, W) ground truth masks
        threshold: 二值化阈值
    
    Returns:
        miou: 标量，平均IoU值
    """
    # 确保尺寸一致
    if pred_logits.shape[-2:] != gt_masks.shape[-2:]:
        pred_logits = F.interpolate(pred_logits, size=gt_masks.shape[-2:], mode='bilinear', align_corners=False)
    
    # 二值化预测
    pred_bin = (torch.sigmoid(pred_logits) > threshold).float()  # (B, 1, H, W)
    gt_bin = (gt_masks > threshold).float()  # (B, 1, H, W)
    
    # 计算每个样本的IoU
    intersection = (pred_bin * gt_bin).sum(dim=[1, 2, 3])  # (B,)
    union = ((pred_bin + gt_bin) > 0).float().sum(dim=[1, 2, 3])  # (B,)
    
    # 避免除零
    iou_per_sample = intersection / (union + 1e-6)  # (B,)
    
    # 返回平均IoU
    return iou_per_sample.mean().item()


def compute_iou_per_sample(pred_logits, gt_masks, threshold=0.5):
    """
    计算每个样本的IoU值（返回每个样本的IoU，而不是平均值）
    
    Args:
        pred_logits: (B, 1, H, W) 预测的logits
        gt_masks: (B, 1, H, W) ground truth masks
        threshold: 二值化阈值
    
    Returns:
        iou_per_sample: (B,) tensor，每个样本的IoU值
    """
    # 确保尺寸一致
    if pred_logits.shape[-2:] != gt_masks.shape[-2:]:
        pred_logits = F.interpolate(pred_logits, size=gt_masks.shape[-2:], mode='bilinear', align_corners=False)
    
    # 二值化预测
    pred_bin = (torch.sigmoid(pred_logits) > threshold).float()  # (B, 1, H, W)
    gt_bin = (gt_masks > threshold).float()  # (B, 1, H, W)
    
    # 计算每个样本的IoU
    intersection = (pred_bin * gt_bin).sum(dim=[1, 2, 3])  # (B,)
    union = ((pred_bin + gt_bin) > 0).float().sum(dim=[1, 2, 3])  # (B,)
    
    # 避免除零
    iou_per_sample = intersection / (union + 1e-6)  # (B,)
    
    return iou_per_sample


def mask_entropy_logits(mask_logits):
    """
    计算每个 mask 的 logits 熵 (二分类) 以及平均熵
    
    Args:
        mask_logits: (B, 1, H, W) mask logits
    
    Returns:
        entropy: (B,) 每个样本的平均熵
    """
    p = torch.sigmoid(mask_logits)
    # binary entropy per pixel
    eps = 1e-8
    ent = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
    # avg per image
    return ent.view(ent.size(0), -1).mean(dim=1)


def mask_entropy_map_logits(mask_logits):
    """
    计算逐像素二元熵图
    
    Args:
        mask_logits: (B, 1, H, W) mask logits
    
    Returns:
        entropy_map: (B, 1, H, W)
    """
    p = torch.sigmoid(mask_logits)
    eps = 1e-8
    return -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
