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


def pixel_info_nce(anchors, positives, negatives, temperature=0.1):
    """
    像素级 InfoNCE 对比损失
    
    Args:
        anchors: anchor 特征 (N, D)
        positives: positive 特征 (N, D)
        negatives: negative 特征 (M, D)
        temperature: 温度参数
    
    Returns:
        loss: 对比损失值
    """
    anchors = F.normalize(anchors, dim=1)
    positives = F.normalize(positives, dim=1)
    negatives = F.normalize(negatives, dim=1)

    sim_pos = torch.exp(torch.sum(anchors * positives, dim=1) / temperature)  
    sim_neg = torch.exp(torch.matmul(anchors, negatives.T) / temperature) 
    denom = sim_pos + sim_neg.sum(dim=1)
    loss = -torch.log(sim_pos / (denom + 1e-12) + 1e-12)
    return loss.mean()

