#!/usr/bin/env python
# coding: utf-8

"""
损失函数模块
定义所有损失函数，包括分割损失和对比学习损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice 损失函数
    用于分割任务的监督损失
    公式：1 - (2*intersect + eps) / (union + eps)
    """
    
    def __init__(self, eps=1e-6):
        """
        Args:
            eps: 平滑项，避免除零
        """
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        """
        计算 Dice 损失
        
        Args:
            logits: 预测的 logits (B, 1, H, W)
            target: 目标 mask (B, 1, H, W)
        
        Returns:
            loss: Dice 损失值
        """
        pred = torch.sigmoid(logits)
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        intersect = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        loss = 1 - (2. * intersect + self.eps) / (union + self.eps)
        return loss.mean()


def weighted_dice_loss(logits, target, weight=None, eps=1e-6):
    """
    加权 Dice，用于边界/高熵区域强化。
    
    Args:
        logits: (B, 1, H, W)
        target: (B, 1, H, W)
        weight: (B, 1, H, W) or None
    """
    pred = torch.sigmoid(logits)
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    if weight is None:
        intersect = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        loss = 1 - (2. * intersect + eps) / (union + eps)
        return loss.mean()

    weight = weight.view(weight.size(0), -1)
    if torch.all(weight <= 0):
        intersect = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        loss = 1 - (2. * intersect + eps) / (union + eps)
        return loss.mean()

    intersect = (pred * target * weight).sum(dim=1)
    union = (pred * weight).sum(dim=1) + (target * weight).sum(dim=1)
    loss = 1 - (2. * intersect + eps) / (union + eps)
    return loss.mean()


def pixel_info_nce(anchors, positives, negatives, temperature=0.1):
    """
    像素级 InfoNCE 对比损失（数值稳定版：logits 上做 log-sum-exp）
    
    Args:
        anchors: anchor 特征 (N, D)，通常为 Student 特征
        positives: positive 特征 (N, D)，通常为 Teacher 同位置特征
        negatives: negative 特征 (M, D)
        temperature: 温度参数
    
    Returns:
        loss: 对比损失值
    """
    anchors = F.normalize(anchors, dim=1)
    positives = F.normalize(positives, dim=1)
    negatives = F.normalize(negatives, dim=1)

    # logits（未除以 temperature 前），避免 exp 溢出
    logit_pos = torch.sum(anchors * positives, dim=1, keepdim=True) / temperature  # (N, 1)
    logit_neg = torch.matmul(anchors, negatives.T) / temperature  # (N, M)
    # 数值稳定：log(exp(a)/(exp(a)+sum(exp(b)))) = a - logsumexp([a, b...])
    logits_all = torch.cat([logit_pos, logit_neg], dim=1)  # (N, 1+M)
    logits_max = logits_all.max(dim=1, keepdim=True)[0].detach()
    logits_stable = logits_all - logits_max
    loss = -(logits_stable[:, 0] - torch.log(torch.exp(logits_stable).sum(dim=1) + 1e-12))
    return loss.mean()

