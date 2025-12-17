#!/usr/bin/env python
# coding: utf-8

"""
Prompt 处理模块
处理各种 prompt 类型，包括点采样和框处理
"""

import numpy as np
import torch


def sample_points_in_ring(small_box, big_box, num_points=10, img_size=1024):
    """
    在 small_box 和 big_box 之间的环形区域随机采样点
    
    Args:
        small_box: [x1, y1, x2, y2] 小框
        big_box: [x1, y1, x2, y2] 大框
        num_points: 采样点数量
        img_size: 图像尺寸
    
    Returns:
        points: (num_points, 2) tensor, 每行是 [x, y]
        labels: (num_points,) tensor, 1 表示前景点（在环形区域内）
    """
    if isinstance(small_box, torch.Tensor):
        small_box = small_box.tolist()
    if isinstance(big_box, torch.Tensor):
        big_box = big_box.tolist()
    
    sx1, sy1, sx2, sy2 = small_box
    bx1, by1, bx2, by2 = big_box
    
    points = []
    labels = []
    
    # 在环形区域采样点：在 big_box 内但不在 small_box 内
    attempts = 0
    max_attempts = num_points * 10
    
    while len(points) < num_points and attempts < max_attempts:
        attempts += 1
        # 在 big_box 内随机采样
        x = np.random.uniform(bx1, bx2)
        y = np.random.uniform(by1, by2)
        
        # 检查是否在环形区域内（在 big_box 内但不在 small_box 内）
        in_big = (bx1 <= x <= bx2) and (by1 <= y <= by2)
        in_small = (sx1 <= x <= sx2) and (sy1 <= y <= sy2)
        
        if in_big and not in_small:
            points.append([x, y])
            labels.append(1)  # 前景点（在边界区域）
    
    # 如果采样点不够，用边界上的点补充
    if len(points) < num_points:
        # 在 big_box 边界上采样
        remaining = num_points - len(points)
        for _ in range(remaining):
            # 随机选择一条边
            edge = np.random.randint(4)
            if edge == 0:  # 上边
                x = np.random.uniform(bx1, bx2)
                y = by1
            elif edge == 1:  # 下边
                x = np.random.uniform(bx1, bx2)
                y = by2
            elif edge == 2:  # 左边
                x = bx1
                y = np.random.uniform(by1, by2)
            else:  # 右边
                x = bx2
                y = np.random.uniform(by1, by2)
            
            # 确保不在 small_box 内
            if not ((sx1 <= x <= sx2) and (sy1 <= y <= sy2)):
                points.append([x, y])
                labels.append(1)
    
    if len(points) == 0:
        # 如果还是没采样到，使用 big_box 的中心点
        cx = (bx1 + bx2) / 2
        cy = (by1 + by2) / 2
        points = [[cx, cy]]
        labels = [1]
    
    points = torch.tensor(points, dtype=torch.float32)  # (N, 2)
    labels = torch.tensor(labels, dtype=torch.long)  # (N,)
    
    return points, labels


def prepare_box_prompts(big_box, small_box, device):
    """
    准备框 prompt（大框+小框）
    将多个框合并为 tensor
    
    Args:
        big_box: [x1, y1, x2, y2] 大框
        small_box: [x1, y1, x2, y2] 小框，可以为 None
        device: 设备
    
    Returns:
        boxes_tensor: (N, 4) tensor，N=2 如果有 small_box，否则 N=1
    """
    if small_box is not None:
        # 将大框和小框合并，形状为 (2, 4)
        boxes_tensor = torch.tensor([big_box, small_box], device=device, dtype=torch.float32)
    else:
        # 如果没有 small_box，只使用 big_box
        boxes_tensor = torch.tensor([big_box], device=device, dtype=torch.float32)
    
    return boxes_tensor


def prepare_point_prompts(points, labels, device):
    """
    准备点 prompt
    处理点坐标和标签
    
    Args:
        points: (N, 2) 点坐标
        labels: (N,) 点标签
        device: 设备
    
    Returns:
        points_tensor: (1, N, 2) tensor
        labels_tensor: (1, N) tensor
    """
    points_tensor = points.unsqueeze(0).to(device)  # (1, N, 2)
    labels_tensor = labels.unsqueeze(0).to(device)  # (1, N)
    return points_tensor, labels_tensor

