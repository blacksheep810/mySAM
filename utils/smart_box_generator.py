#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
智能小框生成模块
基于大框内的图像内容，生成物体的内接矩形（最大内接矩形）
而不是简单的比例缩放
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Union
import os


def find_inner_bounding_box(image_path: str, 
                            big_box: List[int],
                            method: str = 'contour',
                            threshold_ratio: float = 0.3) -> List[int]:
    """
    基于图像内容找到大框内物体的内接矩形
    
    Args:
        image_path: 图像文件路径
        big_box: 大框坐标 [x1, y1, x2, y2]
        method: 方法 ('contour', 'gradient', 'sam')
        threshold_ratio: 阈值比例（用于二值化）
    
    Returns:
        [x1, y1, x2, y2]: 内接矩形坐标
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    # 确保大框在图像范围内
    x1, y1, x2, y2 = big_box
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(x1, min(x2, w))
    y2 = max(y1, min(y2, h))
    
    # 提取大框区域
    roi = image[y1:y2, x1:x2].copy()
    
    if method == 'contour':
        inner_box = _find_inner_box_by_contour(roi, threshold_ratio)
    elif method == 'gradient':
        inner_box = _find_inner_box_by_gradient(roi)
    elif method == 'sam':
        inner_box = _find_inner_box_by_sam(image_path, big_box)
    else:
        raise ValueError(f"未知方法: {method}")
    
    # 将相对坐标转换为绝对坐标
    inner_box[0] += x1
    inner_box[1] += y1
    inner_box[2] += x1
    inner_box[3] += y1
    
    return inner_box


def _find_inner_box_by_contour(roi: np.ndarray, threshold_ratio: float = 0.3) -> List[int]:
    """
    基于轮廓检测找到内接矩形
    
    方法：
    1. 将ROI转换为灰度图
    2. 使用自适应阈值或Otsu阈值进行二值化
    3. 找到最大轮廓
    4. 计算轮廓的内接矩形
    """
    # 转换为灰度图
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    else:
        gray = roi
    
    # 方法1: 使用Otsu阈值
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 方法2: 如果Otsu效果不好，尝试自适应阈值
    # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #                                cv2.THRESH_BINARY, 11, 2)
    
    # 形态学操作：去除噪声
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 找到轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # 如果没有找到轮廓，返回ROI的中心区域（缩小20%）
        h, w = roi.shape[:2]
        return [int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)]
    
    # 找到最大轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 计算轮廓的边界框
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # 如果轮廓太小，尝试找到内接矩形
    # 使用旋转矩形的方法
    rect = cv2.minAreaRect(largest_contour)
    box_points = cv2.boxPoints(rect)
    box_points = np.int0(box_points)
    
    # 计算内接矩形（非旋转）
    x_coords = [p[0] for p in box_points]
    y_coords = [p[1] for p in box_points]
    
    inner_x1 = max(0, min(x_coords))
    inner_y1 = max(0, min(y_coords))
    inner_x2 = min(roi.shape[1], max(x_coords))
    inner_y2 = min(roi.shape[0], max(y_coords))
    
    # 确保内接矩形有效
    if inner_x2 <= inner_x1 or inner_y2 <= inner_y1:
        # 如果计算失败，使用边界框
        inner_x1, inner_y1 = x, y
        inner_x2, inner_y2 = x + w, y + h
    
    return [inner_x1, inner_y1, inner_x2, inner_y2]


def _find_inner_box_by_gradient(roi: np.ndarray) -> List[int]:
    """
    基于梯度信息找到内接矩形
    
    方法：
    1. 计算图像梯度
    2. 找到梯度较大的区域（物体边缘）
    3. 计算物体的内接矩形
    """
    # 转换为灰度图
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    else:
        gray = roi
    
    # 计算梯度
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # 归一化
    gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)
    
    # 阈值化：保留梯度较大的区域
    _, binary = cv2.threshold(gradient_magnitude, 50, 255, cv2.THRESH_BINARY)
    
    # 形态学操作
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 找到轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # 如果没有找到，返回中心区域
        h, w = roi.shape[:2]
        return [int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)]
    
    # 找到最大轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 计算边界框
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return [x, y, x + w, y + h]


def _find_inner_box_by_sam(image_path: str, big_box: List[int]) -> List[int]:
    """
    使用SAM模型找到内接矩形（需要SAM模型）
    
    注意：这个方法需要SAM模型，如果不可用会回退到轮廓方法
    """
    try:
        import sys
        import os
        
        # 尝试导入SAM
        sam_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                'segment-anything')
        if sam_path not in sys.path:
            sys.path.insert(0, sam_path)
        
        from segment_anything import sam_model_registry, SamPredictor
        
        # 这里需要SAM模型路径和类型，暂时返回None让调用者处理
        # 实际使用时需要提供checkpoint_path和model_type
        return None
        
    except ImportError:
        # 如果SAM不可用，回退到轮廓方法
        return _find_inner_box_by_contour(
            cv2.imread(image_path), 
            threshold_ratio=0.3
        )


def find_max_inner_rectangle(mask: np.ndarray) -> List[int]:
    """
    找到二值mask的最大内接矩形（非旋转）
    
    这是一个经典的算法问题，使用动态规划求解
    
    Args:
        mask: 二值mask，1表示物体，0表示背景
    
    Returns:
        [x1, y1, x2, y2]: 最大内接矩形坐标
    """
    h, w = mask.shape
    
    # 计算每个位置向上能延伸的最大高度
    heights = np.zeros((h, w), dtype=np.int32)
    for i in range(h):
        for j in range(w):
            if mask[i, j] > 0:
                heights[i, j] = heights[i-1, j] + 1 if i > 0 else 1
    
    max_area = 0
    best_box = [0, 0, 0, 0]
    
    # 对每一行，使用单调栈找到最大矩形
    for i in range(h):
        stack = []
        for j in range(w):
            while stack and heights[i, stack[-1]] > heights[i, j]:
                height = heights[i, stack.pop()]
                width = j - stack[-1] - 1 if stack else j
                area = height * width
                if area > max_area:
                    max_area = area
                    start_col = stack[-1] + 1 if stack else 0
                    best_box = [start_col, i - height + 1, start_col + width, i + 1]
            stack.append(j)
        
        # 处理栈中剩余元素
        while stack:
            height = heights[i, stack.pop()]
            width = w - stack[-1] - 1 if stack else w
            area = height * width
            if area > max_area:
                max_area = area
                start_col = stack[-1] + 1 if stack else 0
                best_box = [start_col, i - height + 1, start_col + width, i + 1]
    
    return best_box


def generate_smart_small_box(image_path: str,
                             big_box: List[int],
                             method: str = 'contour',
                             fallback_ratio: float = 0.6) -> List[int]:
    """
    智能生成小框：基于图像内容找到内接矩形
    
    Args:
        image_path: 图像文件路径
        big_box: 大框坐标 [x1, y1, x2, y2]
        method: 方法 ('contour', 'gradient', 'max_inner')
        fallback_ratio: 如果方法失败，使用的回退比例
    
    Returns:
        [x1, y1, x2, y2]: 小框坐标
    """
    try:
        if method == 'max_inner':
            # 使用最大内接矩形算法
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            x1, y1, x2, y2 = big_box
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(x1, min(x2, w))
            y2 = max(y1, min(y2, h))
            
            roi = image[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if len(roi.shape) == 3 else roi
            
            # 二值化
            _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 找到最大内接矩形
            inner_box = find_max_inner_rectangle(binary)
            
            # 转换为绝对坐标
            inner_box[0] += x1
            inner_box[1] += y1
            inner_box[2] += x1
            inner_box[3] += y1
            
            return inner_box
        else:
            # 使用轮廓或梯度方法
            return find_inner_bounding_box(image_path, big_box, method=method)
    except Exception as e:
        print(f"警告: 智能方法失败 ({e})，使用回退方法")
        # 回退到简单比例方法
        x1, y1, x2, y2 = big_box
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = (x2 - x1) * fallback_ratio
        height = (y2 - y1) * fallback_ratio
        
        return [
            int(center_x - width / 2),
            int(center_y - height / 2),
            int(center_x + width / 2),
            int(center_y + height / 2)
        ]


def process_csv_with_smart_boxes(input_csv: str,
                                 output_csv: str,
                                 image_dir: str,
                                 method: str = 'contour'):
    """
    处理CSV文件，使用智能方法为大框生成小框
    
    Args:
        input_csv: 输入CSV文件路径（包含大框信息）
        output_csv: 输出CSV文件路径（包含大小框信息）
        image_dir: 图像目录路径
        method: 方法 ('contour', 'gradient', 'max_inner')
    """
    import pandas as pd
    
    # 读取CSV
    df = pd.read_csv(input_csv)
    
    # 检查必要的列
    required_cols = ['image_file', 'max_boxes_x1', 'max_boxes_y1', 'max_boxes_x2', 'max_boxes_y2']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV文件缺少必要的列: {col}")
    
    # 生成小框
    small_boxes = []
    for idx, row in df.iterrows():
        image_file = row['image_file']
        image_path = os.path.join(image_dir, image_file)
        
        big_box = [
            int(row['max_boxes_x1']),
            int(row['max_boxes_y1']),
            int(row['max_boxes_x2']),
            int(row['max_boxes_y2'])
        ]
        
        print(f"[{idx+1}/{len(df)}] 处理: {image_file}")
        
        try:
            small_box = generate_smart_small_box(image_path, big_box, method=method)
            small_boxes.append(small_box)
            print(f"  ✓ 生成小框: {small_box}")
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            # 使用回退方法
            x1, y1, x2, y2 = big_box
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            width = (x2 - x1) * 0.8
            height = (y2 - y1) * 0.8
            small_box = [
                int(center_x - width / 2),
                int(center_y - height / 2),
                int(center_x + width / 2),
                int(center_y + height / 2)
            ]
            small_boxes.append(small_box)
    
    # 添加小框列到DataFrame
    df['min_boxes_x1'] = [box[0] for box in small_boxes]
    df['min_boxes_y1'] = [box[1] for box in small_boxes]
    df['min_boxes_x2'] = [box[2] for box in small_boxes]
    df['min_boxes_y2'] = [box[3] for box in small_boxes]
    
    # 保存
    df.to_csv(output_csv, index=False)
    print(f"\n✓ 已生成小框并保存到: {output_csv}")
    print(f"  共处理 {len(df)} 条记录")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="智能小框生成工具：基于图像内容生成内接矩形")
    parser.add_argument("--input_csv", type=str, required=True, help="输入CSV文件路径（包含大框）")
    parser.add_argument("--output_csv", type=str, required=True, help="输出CSV文件路径（包含大小框）")
    parser.add_argument("--image_dir", type=str, required=True, help="图像目录路径")
    parser.add_argument("--method", type=str, default="contour",
                       choices=['contour', 'gradient', 'max_inner'],
                       help="生成方法: contour(轮廓检测), gradient(梯度), max_inner(最大内接矩形)")
    
    args = parser.parse_args()
    
    process_csv_with_smart_boxes(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        image_dir=args.image_dir,
        method=args.method
    )

