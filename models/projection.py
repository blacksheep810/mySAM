#!/usr/bin/env python
# coding: utf-8

"""
投影头模块
将 encoder 特征投影到对比学习空间
"""

import torch.nn as nn
import torch.nn.functional as F


class PixelProjHead(nn.Module):
    """
    像素级投影头
    将 image_encoder 输出投影到低维空间并归一化
    
    输入: (B, C, H, W) - encoder 特征
    输出: (B, D, H, W) - 归一化的投影特征
    """
    
    def __init__(self, in_dim, proj_dim=64):
        """
        Args:
            in_dim: 输入特征维度（encoder 输出通道数）
            proj_dim: 投影维度（默认64）
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1),
            nn.BatchNorm2d(in_dim),
            nn.GELU(),
            nn.Conv2d(in_dim, proj_dim, kernel_size=1)
        )

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 (B, C, H, W)
        
        Returns:
            z: 投影后的特征 (B, D, H, W)，已 L2 归一化
        """
        z = self.net(x)  # (B, D, H, W)
        # L2 normalize per pixel
        z = z.permute(0, 2, 3, 1)  # (B, H, W, D)
        z = F.normalize(z, p=2, dim=-1)
        z = z.permute(0, 3, 1, 2)  # (B, D, H, W)
        return z

