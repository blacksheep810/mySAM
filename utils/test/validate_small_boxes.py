#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
验证小框是否只包含前景
功能：
1. 读取train_boxes_update.csv中的小框坐标
2. 检查小框在mask中是否只包含前景（白色像素）
3. 统计不同缩放比例下的满足情况
4. 保存不满足要求的小框图像
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
from tqdm import tqdm

# 从generate_oriented_boxes.py导入必要的函数
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_oriented_boxes import (
    calculate_lesion_orientation,
    shrink_box,
    get_rotated_box_corners,
    draw_oriented_box,
    rotate_box
)


def check_box_foreground_only(mask: np.ndarray, box: List[int], 
                              threshold: float = 0.99) -> Tuple[bool, float]:
    """
    检查框内是否只包含前景
    
    Args:
        mask: mask图像（灰度图，前景>0，背景=0）
        box: 框坐标 [x1, y1, x2, y2]
        threshold: 前景像素比例阈值，默认0.99表示99%以上是前景才算满足
    
    Returns:
        (is_valid, foreground_ratio): 是否满足要求，前景像素比例
    """
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = box
    
    # 确保框在图像范围内
    x1 = max(0, min(int(x1), w))
    y1 = max(0, min(int(y1), h))
    x2 = max(x1, min(int(x2), w))
    y2 = max(y1, min(int(y2), h))
    
    # 提取框内区域
    box_mask = mask[y1:y2, x1:x2]
    
    if box_mask.size == 0:
        return False, 0.0
    
    # 计算前景像素比例（前景像素值>0）
    foreground_pixels = np.sum(box_mask > 0)
    total_pixels = box_mask.size
    foreground_ratio = foreground_pixels / total_pixels if total_pixels > 0 else 0.0
    
    # 判断是否满足要求
    is_valid = foreground_ratio >= threshold
    
    return is_valid, foreground_ratio


def reverse_engineer_shrink_ratio(big_box: List[int], target_aabb: List[int], 
                                   angle: float, tolerance: int = 5) -> float:
    """
    从目标AABB反推使用的缩放比例
    
    Args:
        big_box: 大框坐标 [x1, y1, x2, y2]
        target_aabb: 目标轴对齐边界框坐标（CSV中保存的）
        angle: 旋转角度（度）
        tolerance: 允许的坐标误差（像素）
    
    Returns:
        shrink_ratio: 最匹配的缩放比例，如果找不到则返回None
    """
    big_center = (
        (big_box[0] + big_box[2]) / 2.0,
        (big_box[1] + big_box[3]) / 2.0
    )
    
    # 尝试不同的缩放比例
    ratios = np.arange(0.3, 0.85, 0.01)
    best_ratio = None
    min_diff = float('inf')
    
    for ratio in ratios:
        # 生成小框并旋转
        small_box = shrink_box(big_box, ratio)
        rotated_aabb = rotate_box(small_box, big_center, angle)
        
        # 计算与目标AABB的差异
        diff = sum(abs(rotated_aabb[i] - target_aabb[i]) for i in range(4))
        
        if diff < min_diff:
            min_diff = diff
            best_ratio = ratio
        
        # 如果差异在容差范围内，返回该比例
        if diff <= tolerance * 4:  # 4个坐标的容差
            return ratio
    
    # 如果找不到精确匹配，返回最接近的比例
    return best_ratio if min_diff < tolerance * 10 else None


def check_rotated_box_from_corners(mask: np.ndarray, corners: np.ndarray,
                                   threshold: float = 0.99, strict: bool = True) -> Tuple[bool, float]:
    """
    从四个角点检查旋转框内是否只包含前景
    
    Args:
        mask: mask图像（灰度图，前景>0，背景=0）
        corners: 四个角点坐标 (4, 2)
        threshold: 前景像素比例阈值
        strict: 如果True，要求小框完全在mask内（100%），否则使用threshold
    
    Returns:
        (is_valid, foreground_ratio): 是否满足要求，前景像素比例
    """
    h, w = mask.shape[:2]
    
    # 转换为整数坐标
    corners_int = corners.astype(np.int32)
    
    # 创建mask来标记旋转框内的区域
    box_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(box_mask, [corners_int], 255)
    
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


def check_rotated_box_foreground_only(mask: np.ndarray, box: List[int], 
                                      center: Tuple[float, float], angle: float,
                                      threshold: float = 0.99) -> Tuple[bool, float]:
    """
    检查旋转框内是否只包含前景
    
    Args:
        mask: mask图像（灰度图，前景>0，背景=0）
        box: 原始框坐标 [x1, y1, x2, y2]（旋转前的）
        center: 旋转中心 (cx, cy)
        angle: 旋转角度（度）
        threshold: 前景像素比例阈值
    
    Returns:
        (is_valid, foreground_ratio): 是否满足要求，前景像素比例
    """
    h, w = mask.shape[:2]
    
    # 获取旋转框的四个角点
    rotated_corners = get_rotated_box_corners(box, center, angle)
    rotated_corners_int = rotated_corners.astype(np.int32)
    
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
    
    is_valid = foreground_ratio >= threshold
    
    return is_valid, foreground_ratio


def test_shrink_ratios(mask: np.ndarray, big_box: List[int], 
                      shrink_ratios: List[float]) -> Dict[float, Tuple[bool, float]]:
    """
    测试不同缩放比例下的小框是否满足要求
    
    Args:
        mask: mask图像
        big_box: 大框坐标 [x1, y1, x2, y2]
        shrink_ratios: 缩放比例列表
    
    Returns:
        dict: {shrink_ratio: (is_valid, foreground_ratio)}
    """
    results = {}
    
    for ratio in shrink_ratios:
        # 计算中心点和尺寸
        center_x = (big_box[0] + big_box[2]) / 2.0
        center_y = (big_box[1] + big_box[3]) / 2.0
        width = big_box[2] - big_box[0]
        height = big_box[3] - big_box[1]
        
        # 内缩
        new_width = width * ratio
        new_height = height * ratio
        
        # 计算新框的坐标
        small_box = [
            int(center_x - new_width / 2.0),
            int(center_y - new_height / 2.0),
            int(center_x + new_width / 2.0),
            int(center_y + new_height / 2.0)
        ]
        
        is_valid, fg_ratio = check_box_foreground_only(mask, small_box)
        results[ratio] = (is_valid, fg_ratio)
    
    return results


def visualize_box_on_image(image: np.ndarray, mask: np.ndarray, 
                           big_box: List[int], small_box_rotated_aabb: List[int],
                           foreground_ratio: float, angle: float, 
                           shrink_ratio: float = 0.5,
                           small_box_corners: np.ndarray = None) -> np.ndarray:
    """
    在图像上可视化大框和旋转的小框
    
    Args:
        image: 原始图像 (RGB格式)
        mask: mask图像
        big_box: 大框坐标
        small_box_rotated_aabb: 旋转后小框的轴对齐边界框坐标（CSV中保存的）
        foreground_ratio: 前景像素比例
        angle: 旋转角度（度）
        shrink_ratio: 内缩比例
        small_box_corners: 旋转框的四个角点（如果存在）
    
    Returns:
        vis_image: 可视化图像
    """
    vis_image = image.copy()
    
    # 绘制mask轮廓（半透明）
    mask_binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_image, contours, -1, (255, 255, 0), 2)  # 黄色轮廓
    
    # 绘制大框（红色）
    cv2.rectangle(vis_image, 
                 (big_box[0], big_box[1]), 
                 (big_box[2], big_box[3]), 
                 (255, 0, 0), 2)
    
    # 计算大框中心
    big_center = (
        (big_box[0] + big_box[2]) / 2.0,
        (big_box[1] + big_box[3]) / 2.0
    )
    
    # 绘制旋转的小框（绿色表示满足要求，橙色表示不满足）
    color = (0, 255, 0) if foreground_ratio >= 0.99 else (0, 165, 255)  # 绿色或橙色
    
    if small_box_corners is not None:
        # 使用四个角点直接绘制
        corners_int = small_box_corners.astype(np.int32)
        cv2.polylines(vis_image, [corners_int], isClosed=True, color=color, thickness=2)
        rotated_corners = small_box_corners
    else:
        # 向后兼容：重新计算原始小框（内缩后的框，未旋转）
        original_small_box = shrink_box(big_box, shrink_ratio)
        draw_oriented_box(vis_image, original_small_box, big_center, angle, color, 2)
        rotated_corners = get_rotated_box_corners(original_small_box, big_center, angle)
    
    # 添加文本信息
    angle_text = f"{angle:.1f}°" if angle is not None else "N/A"
    text = f"FG Ratio: {foreground_ratio:.3f}, Angle: {angle_text}"
    # 找到旋转框的左上角位置用于放置文本
    text_x = int(np.min(rotated_corners[:, 0]))
    text_y = int(np.min(rotated_corners[:, 1])) - 10
    if text_y < 0:
        text_y = int(np.max(rotated_corners[:, 1])) + 20
    cv2.putText(vis_image, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return vis_image


def validate_small_boxes(data_root: str, csv_path: str, output_dir: str,
                        shrink_ratios: List[float] = None, threshold: float = 0.99,
                        auto_adapt_mode: bool = False):
    """
    验证小框是否只包含前景
    
    Args:
        data_root: ISIC数据集根目录
        csv_path: train_boxes_update.csv路径
        output_dir: 输出目录（保存不满足要求的图像和统计结果）
        shrink_ratios: 要测试的缩放比例列表
        threshold: 前景像素比例阈值
        auto_adapt_mode: 是否使用自动适应模式（从CSV反推每张图像的缩放比例）
    """
    if shrink_ratios is None:
        shrink_ratios = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    # 路径配置
    train_mask_dir = os.path.join(data_root, 'ISBI2016_ISIC_Part1_Training_GroundTruth')
    train_img_dir = os.path.join(data_root, 'ISBI2016_ISIC_Part1_Training_Data')
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    invalid_images_dir = os.path.join(output_dir, 'invalid_boxes')
    os.makedirs(invalid_images_dir, exist_ok=True)
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 统计结果
    current_results = []  # 当前CSV中小框的结果
    ratio_results = {ratio: {'valid': 0, 'invalid': 0, 'fg_ratios': []} 
                     for ratio in shrink_ratios}  # 不同缩放比例的结果
    
    print("开始验证小框...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理图像"):
        img_name = row['image_file']
        mask_name = row['mask_file']
        
        # 读取大框坐标
        big_box = [
            int(row['max_boxes_x1']),
            int(row['max_boxes_y1']),
            int(row['max_boxes_x2']),
            int(row['max_boxes_y2'])
        ]
        
        # 优先读取四个角点（如果存在），否则使用AABB
        has_corners = 'small_box_corner1_x' in row
        if has_corners:
            # 读取旋转框的四个角点
            small_box_corners = np.array([
                [row['small_box_corner1_x'], row['small_box_corner1_y']],
                [row['small_box_corner2_x'], row['small_box_corner2_y']],
                [row['small_box_corner3_x'], row['small_box_corner3_y']],
                [row['small_box_corner4_x'], row['small_box_corner4_y']]
            ], dtype=np.float32)
            # 计算AABB用于向后兼容
            small_box = [
                int(np.min(small_box_corners[:, 0])),
                int(np.min(small_box_corners[:, 1])),
                int(np.max(small_box_corners[:, 0])),
                int(np.max(small_box_corners[:, 1]))
            ]
        else:
            # 向后兼容：使用AABB
            small_box = [
                int(row['min_boxes_x1']),
                int(row['min_boxes_y1']),
                int(row['min_boxes_x2']),
                int(row['min_boxes_y2'])
            ]
            small_box_corners = None
        
        # 读取图像和mask
        img_path = os.path.join(train_img_dir, img_name)
        mask_path = os.path.join(train_mask_dir, mask_name)
        
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            print(f"警告: 文件不存在 - {img_name}")
            continue
        
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            print(f"警告: 无法读取文件 - {img_name}")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 如果存在四个角点，直接使用它们进行验证（无需旋转信息）
        if has_corners and small_box_corners is not None:
            # 使用四个角点直接检查旋转框，无需计算角度和缩放比例
            # 可视化函数可以直接使用角点绘制，不需要角度信息
            # strict=True确保小框完全在mask内
            is_valid, fg_ratio = check_rotated_box_from_corners(mask, small_box_corners, threshold, strict=True)
            angle = None  # 有四个角点时不需要角度
            shrink_ratio = None  # 有四个角点时不需要缩放比例
        else:
            # 向后兼容：没有四个角点时，需要重新计算旋转角度
            angle = calculate_lesion_orientation(image_rgb, big_box)
            
            # 计算大框中心
            big_center = (
                (big_box[0] + big_box[2]) / 2.0,
                (big_box[1] + big_box[3]) / 2.0
            )
            
            # 根据模式确定缩放比例
            if auto_adapt_mode:
                # 自动适应模式：从CSV中的AABB反推实际使用的缩放比例
                inferred_ratio = reverse_engineer_shrink_ratio(big_box, small_box, angle)
                if inferred_ratio is not None:
                    shrink_ratio = inferred_ratio
                else:
                    # 如果反推失败，使用默认值
                    shrink_ratio = 0.5
                    print(f"警告: 无法反推缩放比例 - {img_name}，使用默认值0.5")
            else:
                # 固定模式：假设使用固定缩放比例
                shrink_ratio = 0.5
            
            # 重新计算原始小框
            original_small_box = shrink_box(big_box, shrink_ratio)
            
            # 检查旋转后的小框是否满足要求
            is_valid, fg_ratio = check_rotated_box_foreground_only(
                mask, original_small_box, big_center, angle, threshold
            )
            small_box_corners = None  # 需要重新计算用于可视化
        
        current_results.append({
            'image_file': img_name,
            'mask_file': mask_name,
            'is_valid': is_valid,
            'foreground_ratio': fg_ratio,
            'angle': angle,
            'inferred_shrink_ratio': shrink_ratio if auto_adapt_mode else None,
            'small_box_aabb': small_box,  # CSV中保存的轴对齐边界框
            'has_corners': has_corners  # 是否使用四个角点
        })
        
        # 如果不满足要求，保存可视化图像（使用旋转框）
        if not is_valid:
            vis_image = visualize_box_on_image(
                image_rgb, mask, big_box, small_box, fg_ratio, angle, shrink_ratio, small_box_corners
            )
            vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            output_path = os.path.join(invalid_images_dir, f"{img_name}")
            cv2.imwrite(output_path, vis_image_bgr)
        
        # 测试不同缩放比例
        ratio_test_results = test_shrink_ratios(mask, big_box, shrink_ratios)
        for ratio, (valid, fg_r) in ratio_test_results.items():
            if valid:
                ratio_results[ratio]['valid'] += 1
            else:
                ratio_results[ratio]['invalid'] += 1
            ratio_results[ratio]['fg_ratios'].append(fg_r)
    
    # 保存当前CSV小框的验证结果
    current_df = pd.DataFrame(current_results)
    current_output_path = os.path.join(output_dir, 'current_box_validation.csv')
    current_df.to_csv(current_output_path, index=False)
    print(f"\n当前小框验证结果已保存到: {current_output_path}")
    
    # 统计当前结果
    total_current = len(current_results)
    valid_current = sum(1 for r in current_results if r['is_valid'])
    invalid_current = total_current - valid_current
    avg_fg_ratio_current = np.mean([r['foreground_ratio'] for r in current_results])
    
    print(f"\n当前CSV中小框的统计结果:")
    print(f"  总数: {total_current}")
    print(f"  满足要求: {valid_current} ({valid_current/total_current*100:.2f}%)")
    print(f"  不满足要求: {invalid_current} ({invalid_current/total_current*100:.2f}%)")
    print(f"  平均前景比例: {avg_fg_ratio_current:.4f}")
    
    # 统计不同缩放比例的结果
    ratio_stats = []
    for ratio in sorted(shrink_ratios):
        stats = ratio_results[ratio]
        total = stats['valid'] + stats['invalid']
        valid_count = stats['valid']
        invalid_count = stats['invalid']
        avg_fg_ratio = np.mean(stats['fg_ratios']) if stats['fg_ratios'] else 0.0
        min_fg_ratio = np.min(stats['fg_ratios']) if stats['fg_ratios'] else 0.0
        
        ratio_stats.append({
            'shrink_ratio': ratio,
            'total': total,
            'valid_count': valid_count,
            'invalid_count': invalid_count,
            'valid_percentage': valid_count / total * 100 if total > 0 else 0.0,
            'avg_foreground_ratio': avg_fg_ratio,
            'min_foreground_ratio': min_fg_ratio
        })
        
        print(f"\n缩放比例 {ratio:.2f}:")
        print(f"  满足要求: {valid_count}/{total} ({valid_count/total*100:.2f}%)")
        print(f"  平均前景比例: {avg_fg_ratio:.4f}")
        print(f"  最小前景比例: {min_fg_ratio:.4f}")
    
    # 保存不同缩放比例的统计结果
    ratio_df = pd.DataFrame(ratio_stats)
    ratio_output_path = os.path.join(output_dir, 'shrink_ratio_statistics.csv')
    ratio_df.to_csv(ratio_output_path, index=False)
    print(f"\n缩放比例统计结果已保存到: {ratio_output_path}")
    
    # 保存不满足要求的图像列表
    invalid_list = [r for r in current_results if not r['is_valid']]
    invalid_df = pd.DataFrame(invalid_list)
    invalid_output_path = os.path.join(output_dir, 'invalid_boxes_list.csv')
    invalid_df.to_csv(invalid_output_path, index=False)
    print(f"不满足要求的图像列表已保存到: {invalid_output_path}")
    print(f"不满足要求的可视化图像已保存到: {invalid_images_dir}")
    
    # 推荐缩放比例
    print("\n推荐缩放比例分析:")
    for ratio_stat in ratio_stats:
        if ratio_stat['valid_percentage'] >= 95.0:
            print(f"  缩放比例 {ratio_stat['shrink_ratio']:.2f}: "
                  f"{ratio_stat['valid_percentage']:.2f}% 满足要求 "
                  f"(平均前景比例: {ratio_stat['avg_foreground_ratio']:.4f})")
    
    print(f"\n处理完成！所有结果保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Validate small boxes foreground only')
    parser.add_argument('--data_root', type=str,
                       default='/mnt/mySAM/data/ISIC',
                       help='ISIC dataset root directory')
    parser.add_argument('--csv_path', type=str,
                       default='/mnt/mySAM/data/ISIC/train_boxes_update.csv',
                       help='Path to train_boxes_update.csv')
    parser.add_argument('--output_dir', type=str,
                       default='/mnt/mySAM/utils/test/box_validation',
                       help='Output directory for validation results')
    parser.add_argument('--shrink_ratios', type=float, nargs='+',
                       default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                       help='Shrink ratios to test')
    parser.add_argument('--threshold', type=float,
                       default=0.99,
                       help='Foreground ratio threshold (default 0.99)')
    parser.add_argument('--auto_adapt_mode', action='store_true',
                       help='Enable auto-adapt mode: reverse engineer shrink ratio from CSV AABB')
    
    args = parser.parse_args()
    
    validate_small_boxes(
        args.data_root,
        args.csv_path,
        args.output_dir,
        args.shrink_ratios,
        args.threshold,
        args.auto_adapt_mode
    )


if __name__ == '__main__':
    main()

