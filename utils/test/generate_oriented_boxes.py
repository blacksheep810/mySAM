#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成带方向的小框数据集
功能：
1. 读取ISIC数据集的大框
2. 内缩50%生成小框
3. 小框方向与病灶方向一致
4. 生成新的数据集图片（带标注框）
5. 生成新的CSV文件
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import argparse


def calculate_lesion_orientation(image: np.ndarray, big_box: List[int]) -> float:
    """
    根据大框和图像内容粗略估计病灶的主方向（角度，单位：度）
    
    不使用mask，仅基于图像梯度信息估计方向
    
    Args:
        image: 图像数组 (RGB格式)
        big_box: 大框坐标 [x1, y1, x2, y2]
    
    Returns:
        angle: 角度（度），范围[-90, 90]
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = big_box
    
    # 确保大框在图像范围内
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(x1, min(x2, w))
    y2 = max(y1, min(y2, h))
    
    # 提取大框区域
    roi = image[y1:y2, x1:x2].copy()
    
    if roi.size == 0:
        return 0.0
    
    # 转换为灰度图
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    else:
        gray = roi
    
    # 方法1: 使用图像梯度计算主方向
    # 计算Sobel梯度
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算梯度幅值和方向
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # 只考虑梯度较大的像素（边缘）
    threshold = np.percentile(magnitude, 70)  # 使用70%分位数作为阈值
    edge_mask = magnitude > threshold
    
    if np.sum(edge_mask) < 10:
        # 如果边缘点太少，返回0度
        return 0.0
    
    # 获取边缘点的梯度方向
    edge_grad_x = grad_x[edge_mask]
    edge_grad_y = grad_y[edge_mask]
    
    # 计算梯度方向的加权平均（权重为梯度幅值）
    weights = magnitude[edge_mask]
    
    # 计算主方向（使用PCA）
    # 构建数据矩阵：每个边缘点的梯度向量
    data = np.column_stack((edge_grad_x.flatten(), edge_grad_y.flatten()))
    weights_flat = weights.flatten()
    
    # 加权PCA
    # 中心化
    mean = np.average(data, axis=0, weights=weights_flat)
    data_centered = data - mean
    
    # 计算加权协方差矩阵
    cov_matrix = np.cov(data_centered.T, aweights=weights_flat)
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # 主方向对应最大特征值的特征向量
    main_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]
    
    # 计算角度（梯度方向垂直于边缘方向，所以需要加90度）
    angle = np.arctan2(main_eigenvector[1], main_eigenvector[0]) * 180 / np.pi
    
    # 规范化角度到[-90, 90]
    while angle > 90:
        angle -= 180
    while angle < -90:
        angle += 180
    
    # 限制角度范围
    angle = max(-90, min(90, angle))
    
    return angle


def shrink_box(big_box: List[int], shrink_ratio: float = 0.3) -> List[int]:
    """
    将大框内缩指定比例生成小框
    
    Args:
        big_box: 大框坐标 [x1, y1, x2, y2]
        shrink_ratio: 内缩比例，0.5表示内缩50%
    
    Returns:
        [x1, y1, x2, y2]: 小框坐标
    """
    x1, y1, x2, y2 = big_box
    
    # 计算中心点和尺寸
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1
    
    # 内缩
    new_width = width * shrink_ratio
    new_height = height * shrink_ratio
    
    # 计算新框的坐标
    new_x1 = int(center_x - new_width / 2.0)
    new_y1 = int(center_y - new_height / 2.0)
    new_x2 = int(center_x + new_width / 2.0)
    new_y2 = int(center_y + new_height / 2.0)
    
    return [new_x1, new_y1, new_x2, new_y2]


def get_rotated_box_corners(box: List[int], center: Tuple[float, float], angle: float) -> np.ndarray:
    """
    获取旋转后框的四个角点
    
    Args:
        box: 框坐标 [x1, y1, x2, y2]
        center: 旋转中心 (cx, cy)
        angle: 旋转角度（度）
    
    Returns:
        rotated_corners: 旋转后的四个角点坐标 (4, 2)
    """
    x1, y1, x2, y2 = box
    cx, cy = center
    
    # 框的四个角点
    corners = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ], dtype=np.float32)
    
    # 转换为相对于中心的坐标
    corners_centered = corners - np.array([cx, cy])
    
    # 旋转矩阵
    angle_rad = np.deg2rad(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])
    
    # 旋转角点
    rotated_corners = corners_centered @ rotation_matrix.T
    
    # 转换回绝对坐标
    rotated_corners += np.array([cx, cy])
    
    return rotated_corners


def rotate_box(box: List[int], center: Tuple[float, float], angle: float) -> List[int]:
    """
    旋转框使其与指定角度对齐，返回轴对齐边界框
    
    Args:
        box: 框坐标 [x1, y1, x2, y2]
        center: 旋转中心 (cx, cy)
        angle: 旋转角度（度）
    
    Returns:
        [x1, y1, x2, y2]: 旋转后的框坐标（轴对齐边界框）
    """
    rotated_corners = get_rotated_box_corners(box, center, angle)
    
    # 计算轴对齐边界框
    x_coords = rotated_corners[:, 0]
    y_coords = rotated_corners[:, 1]
    
    new_x1 = int(np.min(x_coords))
    new_y1 = int(np.min(y_coords))
    new_x2 = int(np.max(x_coords))
    new_y2 = int(np.max(y_coords))
    
    return [new_x1, new_y1, new_x2, new_y2]


def estimate_foreground_mask_from_image(image: np.ndarray, big_box: List[int]) -> np.ndarray:
    """
    从图像估计前景mask（不使用真实mask）
    
    使用多种方法结合提高精度：
    1. Otsu阈值分割
    2. 自适应阈值
    3. 梯度边缘检测
    4. 轮廓分析
    
    Args:
        image: 图像数组 (RGB格式)
        big_box: 大框坐标 [x1, y1, x2, y2]
    
    Returns:
        estimated_mask: 估计的前景mask（前景=255，背景=0）
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = big_box
    
    # 确保大框在图像范围内
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(x1, min(x2, w))
    y2 = max(y1, min(y2, h))
    
    # 提取ROI
    roi = image[y1:y2, x1:x2].copy()
    
    if roi.size == 0:
        return np.zeros((h, w), dtype=np.uint8)
    
    # 转换为灰度
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    else:
        gray = roi
    
    # 方法1: Otsu阈值分割
    _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 方法2: 自适应阈值（处理光照不均）
    binary_adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # 方法3: 基于梯度的边缘检测
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude_norm = (magnitude / (magnitude.max() + 1e-6) * 255).astype(np.uint8)
    _, binary_gradient = cv2.threshold(magnitude_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 结合多种方法：取交集（最保守）
    combined_binary = cv2.bitwise_and(binary_otsu, binary_adaptive)
    combined_binary = cv2.bitwise_and(combined_binary, binary_gradient)
    
    # 形态学操作去噪
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_medium = np.ones((5, 5), np.uint8)
    
    # 闭运算：填充小洞
    combined_binary = cv2.morphologyEx(combined_binary, cv2.MORPH_CLOSE, kernel_small)
    # 开运算：去除小点
    combined_binary = cv2.morphologyEx(combined_binary, cv2.MORPH_OPEN, kernel_small)
    
    # 找到最大连通区域作为前景
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        combined_binary, connectivity=8
    )
    
    if num_labels > 1:
        # 找到面积最大的区域（排除背景，label=0）
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        estimated_mask_roi = (labels == largest_label).astype(np.uint8) * 255
        
        # 如果最大区域太小，尝试使用Otsu结果
        if stats[largest_label, cv2.CC_STAT_AREA] < roi.size * 0.1:
            # 使用Otsu结果，但进行形态学操作
            estimated_mask_roi = cv2.morphologyEx(binary_otsu, cv2.MORPH_CLOSE, kernel_medium)
            estimated_mask_roi = cv2.morphologyEx(estimated_mask_roi, cv2.MORPH_OPEN, kernel_small)
    else:
        # 如果没有找到连通区域，使用Otsu结果
        estimated_mask_roi = cv2.morphologyEx(binary_otsu, cv2.MORPH_CLOSE, kernel_medium)
        estimated_mask_roi = cv2.morphologyEx(estimated_mask_roi, cv2.MORPH_OPEN, kernel_small)
    
    # 扩展到全图尺寸
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = estimated_mask_roi
    
    return full_mask


def check_rotated_box_foreground_only(mask: np.ndarray, box: List[int], 
                                      center: Tuple[float, float], angle: float,
                                      threshold: float = 0.99, strict: bool = True) -> Tuple[bool, float]:
    """
    检查旋转框内是否只包含前景
    
    Args:
        mask: mask图像（灰度图，前景>0，背景=0）
        box: 原始框坐标 [x1, y1, x2, y2]（旋转前的）
        center: 旋转中心 (cx, cy)
        angle: 旋转角度（度）
        threshold: 前景像素比例阈值
        strict: 如果True，要求小框完全在mask内（100%），否则使用threshold
    
    Returns:
        (is_valid, foreground_ratio): 是否满足要求，前景像素比例
    """
    h, w = mask.shape[:2]
    
    # 获取旋转框的四个角点
    rotated_corners = get_rotated_box_corners(box, center, angle)
    rotated_corners_int = rotated_corners.astype(np.int32)
    
    # 如果strict=True，首先检查所有角点是否都在mask内
    if strict:
        all_corners_in_mask = True
        for corner in rotated_corners_int:
            x, y = corner[0], corner[1]
            # 确保坐标在图像范围内
            if x < 0 or x >= w or y < 0 or y >= h:
                all_corners_in_mask = False
                break
            # 检查角点是否在mask内
            if mask[y, x] == 0:
                all_corners_in_mask = False
                break
        
        # 如果角点不在mask内，直接返回False
        if not all_corners_in_mask:
            return False, 0.0
    
    # 创建mask来标记旋转框内的区域
    box_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(box_mask, [rotated_corners_int], 255)
    
    # 计算旋转框内的前景像素
    box_region = (box_mask > 0)
    if np.sum(box_region) == 0:
        return False, 0.0
    
    foreground_in_box = np.sum((mask > 0) & box_region)
    total_in_box = np.sum(box_region)
    foreground_ratio = foreground_in_box / total_in_box if total_in_box > 0 else 0.0
    
    # 如果strict=True，要求100%在mask内
    if strict:
        is_valid = foreground_ratio >= 1.0
    else:
        is_valid = foreground_ratio >= threshold
    
    return is_valid, foreground_ratio


def check_rotated_box_foreground_only_no_mask(image: np.ndarray, big_box: List[int],
                                              box: List[int], center: Tuple[float, float], 
                                              angle: float, threshold: float = 0.99,
                                              strict: bool = True) -> Tuple[bool, float]:
    """
    不使用真实mask检查旋转框内是否只包含前景（基于图像估计）
    
    Args:
        image: 图像数组 (RGB格式)
        big_box: 大框坐标（用于估计前景区域）
        box: 原始框坐标 [x1, y1, x2, y2]（旋转前的）
        center: 旋转中心 (cx, cy)
        angle: 旋转角度（度）
        threshold: 前景像素比例阈值
        strict: 如果True，要求小框完全在mask内（100%），否则使用threshold
    
    Returns:
        (is_valid, foreground_ratio): 是否满足要求，前景像素比例
    """
    # 估计前景mask
    estimated_mask = estimate_foreground_mask_from_image(image, big_box)
    
    # 使用估计的mask检查旋转框
    return check_rotated_box_foreground_only(estimated_mask, box, center, angle, threshold, strict=strict)


def find_optimal_shrink_ratio(image: np.ndarray, mask: np.ndarray, big_box: List[int],
                              angle: float, threshold: float = 0.99,
                              min_ratio: float = 0.1, max_ratio: float = 0.8,
                              step: float = 0.01, use_mask: bool = True,
                              strict: bool = True) -> Tuple[float, List[int], float]:
    """
    为每张图像自动找到满足前景要求的最大缩放比例
    
    Args:
        image: 图像数组 (RGB格式)
        mask: mask图像（灰度图），如果use_mask=False则可以为None
        big_box: 大框坐标 [x1, y1, x2, y2]
        angle: 旋转角度（度）
        threshold: 前景像素比例阈值
        min_ratio: 最小缩放比例（降低到0.1以确保能找到满足条件的比例）
        max_ratio: 最大缩放比例
        step: 搜索步长（减小到0.01以提高精度）
        use_mask: 是否使用真实mask，False则从图像估计
        strict: 如果True，要求小框完全在mask内（100%）
    
    Returns:
        (best_ratio, best_small_box_rotated, foreground_ratio): 
        最佳缩放比例，旋转后的小框坐标，前景比例
    """
    # 计算大框中心
    big_center = (
        (big_box[0] + big_box[2]) / 2.0,
        (big_box[1] + big_box[3]) / 2.0
    )
    
    best_ratio = min_ratio
    best_small_box_rotated = None
    best_fg_ratio = 0.0
    
    # 从最大比例开始向下搜索，找到第一个满足要求的比例
    ratios = np.arange(max_ratio, min_ratio - step, -step)
    
    for ratio in ratios:
        # 内缩生成小框
        small_box = shrink_box(big_box, ratio)
        
        # 检查旋转框是否满足要求
        if use_mask and mask is not None:
            # 使用真实mask，strict=True确保小框完全在mask内
            is_valid, fg_ratio = check_rotated_box_foreground_only(
                mask, small_box, big_center, angle, threshold, strict=strict
            )
        else:
            # 不使用mask，从图像估计，也需要传递strict参数
            is_valid, fg_ratio = check_rotated_box_foreground_only_no_mask(
                image, big_box, small_box, big_center, angle, threshold, strict=strict
            )
        
        if is_valid:
            # 找到满足要求的最大比例
            best_ratio = ratio
            best_small_box_rotated = rotate_box(small_box, big_center, angle)
            best_fg_ratio = fg_ratio
            break
    
    # 如果没有找到满足要求的，继续缩小直到找到或达到最小比例
    if best_small_box_rotated is None:
        # 继续缩小，步长更小
        fine_step = 0.005
        current_ratio = min_ratio
        while current_ratio >= 0.05:  # 最小比例降低到0.05
            small_box = shrink_box(big_box, current_ratio)
            if use_mask and mask is not None:
                is_valid, fg_ratio = check_rotated_box_foreground_only(
                    mask, small_box, big_center, angle, threshold, strict=strict
                )
            else:
                is_valid, fg_ratio = check_rotated_box_foreground_only_no_mask(
                    image, big_box, small_box, big_center, angle, threshold, strict=strict
                )
            
            if is_valid:
                best_ratio = current_ratio
                best_small_box_rotated = rotate_box(small_box, big_center, angle)
                best_fg_ratio = fg_ratio
                break
            
            current_ratio -= fine_step
        
        # 如果仍然没有找到，使用最小比例（即使不满足要求）
        if best_small_box_rotated is None:
            small_box = shrink_box(big_box, 0.05)
            best_ratio = 0.05
            best_small_box_rotated = rotate_box(small_box, big_center, angle)
            if use_mask and mask is not None:
                _, best_fg_ratio = check_rotated_box_foreground_only(
                    mask, small_box, big_center, angle, threshold, strict=strict
                )
            else:
                _, best_fg_ratio = check_rotated_box_foreground_only_no_mask(
                    image, big_box, small_box, big_center, angle, threshold, strict=strict
                )
    
    return best_ratio, best_small_box_rotated, best_fg_ratio


def draw_oriented_box(image: np.ndarray, box: List[int], center: Tuple[float, float], 
                      angle: float, color: Tuple[int, int, int], thickness: int = 2):
    """
    在图像上绘制旋转的框
    
    Args:
        image: 图像数组 (RGB格式)
        box: 框坐标 [x1, y1, x2, y2]
        center: 旋转中心 (cx, cy)
        angle: 旋转角度（度）
        color: 颜色 (R, G, B) - RGB格式
        thickness: 线宽
    """
    rotated_corners = get_rotated_box_corners(box, center, angle)
    rotated_corners = rotated_corners.astype(np.int32)
    
    # 绘制旋转的矩形
    cv2.polylines(image, [rotated_corners], isClosed=True, color=color, thickness=thickness)


def process_isic_dataset(data_root: str, output_root: str, shrink_ratio: float = None,
                         auto_adapt: bool = True, threshold: float = 0.99,
                         use_mask: bool = True, strict: bool = True):
    """
    处理ISIC数据集，生成带方向的小框（仅处理训练集）
    
    Args:
        data_root: ISIC数据集根目录
        output_root: 输出数据集根目录（将创建与ISIC相同的目录结构）
        shrink_ratio: 内缩比例（如果auto_adapt=True则忽略此参数）
        auto_adapt: 是否自动适应缩放比例，确保小框内都是前景
        threshold: 前景像素比例阈值（auto_adapt=True时使用）
        use_mask: 是否使用真实mask验证，False则从图像估计前景（精度可能略低）
        strict: 若True，要求旋转小框四个角点都在前景内（易导致小框很小）；若False，只要求框内前景比例>=threshold，小框会更大
    """
    # 路径配置
    train_csv = os.path.join(data_root, 'train_boxes.csv')
    
    train_img_dir = os.path.join(data_root, 'ISBI2016_ISIC_Part1_Training_Data')
    train_mask_dir = os.path.join(data_root, 'ISBI2016_ISIC_Part1_Training_GroundTruth')
    
    # 创建输出目录结构（仿照ISIC结构）
    output_train_img_dir = os.path.join(output_root, 'ISBI2016_ISIC_Part1_Training_Data')
    output_train_mask_dir = os.path.join(output_root, 'ISBI2016_ISIC_Part1_Training_GroundTruth')
    os.makedirs(output_train_img_dir, exist_ok=True)
    os.makedirs(output_train_mask_dir, exist_ok=True)
    
    # 读取CSV文件（仅训练集）
    train_df = pd.read_csv(train_csv)
    
    # 处理训练集
    print("处理训练集...")
    train_results = []
    for idx, row in train_df.iterrows():
        img_name = row['image_file']
        mask_name = row['mask_file']
        big_box = [
            int(row['max_boxes_x1']),
            int(row['max_boxes_y1']),
            int(row['max_boxes_x2']),
            int(row['max_boxes_y2'])
        ]
        
        # 读取图像和mask
        img_path = os.path.join(train_img_dir, img_name)
        mask_path = os.path.join(train_mask_dir, mask_name)
        
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            print(f"警告: 文件不存在 - {img_name}")
            continue
        
        image = cv2.imread(img_path)
        if image is None:
            print(f"警告: 无法读取图像 - {img_name}")
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"警告: 无法读取mask - {mask_name}")
            continue
        
        # 计算病灶方向（基于图像和大框，不使用mask）
        angle = calculate_lesion_orientation(image, big_box)
        
        # 计算大框中心
        big_center = (
            (big_box[0] + big_box[2]) / 2.0,
            (big_box[1] + big_box[3]) / 2.0
        )
        
        # 自动适应缩放比例或使用固定比例
        if auto_adapt:
            # strict=True 时要求小框四角都在前景内，易偏小；strict=False 时只要求前景比例>=threshold
            best_ratio, small_box_rotated, fg_ratio = find_optimal_shrink_ratio(
                image, mask, big_box, angle, threshold, use_mask=use_mask, strict=strict
            )
            # 重新计算原始小框用于绘制
            small_box = shrink_box(big_box, best_ratio)
        else:
            # 使用固定缩放比例
            if shrink_ratio is None:
                shrink_ratio = 0.5
            best_ratio = shrink_ratio
            small_box = shrink_box(big_box, shrink_ratio)
            small_box_rotated = rotate_box(small_box, big_center, angle)
            if use_mask and mask is not None:
                _, fg_ratio = check_rotated_box_foreground_only(
                    mask, small_box, big_center, angle, threshold, strict=strict
                )
            else:
                _, fg_ratio = check_rotated_box_foreground_only_no_mask(
                    image, big_box, small_box, big_center, angle, threshold, strict=strict
                )
        
        # 获取旋转框的四个角点坐标
        rotated_corners = get_rotated_box_corners(small_box, big_center, angle)
        # 确保角点在图像范围内
        h, w = image.shape[:2]
        rotated_corners[:, 0] = np.clip(rotated_corners[:, 0], 0, w)
        rotated_corners[:, 1] = np.clip(rotated_corners[:, 1], 0, h)
        
        # 绘制框
        image_with_boxes = image.copy()
        # 绘制大框（红色）
        cv2.rectangle(image_with_boxes, 
                     (big_box[0], big_box[1]), 
                     (big_box[2], big_box[3]), 
                     (255, 0, 0), 2)
        
        # 绘制旋转的小框（绿色）- 使用四个角点
        rotated_corners_int = rotated_corners.astype(np.int32)
        cv2.polylines(image_with_boxes, [rotated_corners_int], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # 保存图像（转换回BGR格式）
        image_bgr = cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR)
        output_img_path = os.path.join(output_train_img_dir, img_name)
        cv2.imwrite(output_img_path, image_bgr)
        
        # 复制mask文件到输出目录
        output_mask_path = os.path.join(output_train_mask_dir, mask_name)
        cv2.imwrite(output_mask_path, mask)
        
        # 保存结果（保存四个角点坐标，而不是AABB+角度）
        train_results.append({
            'image_file': img_name,
            'mask_file': mask_name,
            'max_boxes_x1': big_box[0],
            'max_boxes_y1': big_box[1],
            'max_boxes_x2': big_box[2],
            'max_boxes_y2': big_box[3],
            # 保存旋转框的四个角点坐标（8个值）
            'small_box_corner1_x': float(rotated_corners[0, 0]),
            'small_box_corner1_y': float(rotated_corners[0, 1]),
            'small_box_corner2_x': float(rotated_corners[1, 0]),
            'small_box_corner2_y': float(rotated_corners[1, 1]),
            'small_box_corner3_x': float(rotated_corners[2, 0]),
            'small_box_corner3_y': float(rotated_corners[2, 1]),
            'small_box_corner4_x': float(rotated_corners[3, 0]),
            'small_box_corner4_y': float(rotated_corners[3, 1]),
            # 为了向后兼容，也保存AABB（用于训练时的快速访问）
            'min_boxes_x1': float(np.min(rotated_corners[:, 0])),
            'min_boxes_y1': float(np.min(rotated_corners[:, 1])),
            'min_boxes_x2': float(np.max(rotated_corners[:, 0])),
            'min_boxes_y2': float(np.max(rotated_corners[:, 1]))
        })
        
        # 打印调试信息（如果使用自动适应）
        if auto_adapt and (idx + 1) % 100 == 0:
            print(f"  已处理 {idx + 1}/{len(train_df)} 张图像，当前图像缩放比例: {best_ratio:.3f}, 前景比例: {fg_ratio:.4f}")
        
        if (idx + 1) % 10 == 0:
            print(f"  已处理 {idx + 1}/{len(train_df)} 张训练图像")
    
    # 保存CSV文件到原始ISIC目录
    train_output_csv = os.path.join(data_root, 'train_boxes_update.csv')
    
    train_df_new = pd.DataFrame(train_results)
    train_df_new.to_csv(train_output_csv, index=False)
    
    print(f"\n处理完成！")
    print(f"训练集: {len(train_results)} 张图像")
    print(f"输出训练图像目录: {output_train_img_dir}")
    print(f"输出训练mask目录: {output_train_mask_dir}")
    print(f"训练集CSV: {train_output_csv}")


def main():
    parser = argparse.ArgumentParser(description='Generate oriented small boxes dataset')
    parser.add_argument('--data_root', type=str, 
                       default='/mnt/mySAM/data/ISIC',
                       help='ISIC dataset root directory')
    parser.add_argument('--output_root', type=str,
                       default='/mnt/mySAM/data/update_ISIC',
                       help='Output dataset root directory')
    parser.add_argument('--shrink_ratio', type=float,
                       default=0.5,
                       help='Shrink ratio (default 0.5, ignored if --auto_adapt is True)')
    parser.add_argument('--auto_adapt', action='store_true',
                       help='Automatically adapt shrink ratio for each image to ensure foreground only')
    parser.add_argument('--threshold', type=float,
                       default=0.99,
                       help='Foreground ratio threshold (default 0.99, used when --auto_adapt is True)')
    parser.add_argument('--no_mask', action='store_true',
                       help='Do not use mask, estimate foreground from image (lower accuracy but works without mask)')
    parser.add_argument('--no_strict', action='store_true',
                       help='Do not require all 4 corners inside foreground; only require foreground ratio >= threshold (produces larger small boxes)')
    
    args = parser.parse_args()
    
    process_isic_dataset(args.data_root, args.output_root, args.shrink_ratio,
                        args.auto_adapt, args.threshold, use_mask=not args.no_mask,
                        strict=not args.no_strict)


if __name__ == '__main__':
    main()

