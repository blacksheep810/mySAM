#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在原始图像上可视化小框
将训练时保存的512x512坐标映射回原始图像尺寸
"""

import cv2
import numpy as np
import pandas as pd
import os
import sys

def visualize_on_original_image(img_name, csv_path, data_root, output_dir, 
                                corners_512=None, big_box_512=None, small_box_512=None):
    """
    在原始图像上可视化框
    
    Args:
        img_name: 图像文件名
        csv_path: CSV文件路径
        data_root: 数据根目录
        output_dir: 输出目录
        corners_512: 512x512坐标系下的四个角点 (4, 2)
        big_box_512: 512x512坐标系下的大框 [x1, y1, x2, y2]
        small_box_512: 512x512坐标系下的小框AABB [x1, y1, x2, y2]
    """
    # 读取CSV
    df = pd.read_csv(csv_path)
    row = df[df['image_file'] == img_name].iloc[0]
    
    # 读取原始图像
    img_path = os.path.join(data_root, 'ISBI2016_ISIC_Part1_Training_Data', img_name)
    img_orig = cv2.imread(img_path)
    if img_orig is None:
        print(f"无法读取图像: {img_path}")
        return
    
    orig_h, orig_w = img_orig.shape[:2]
    img_size = 512
    
    # 计算缩放比例
    scale_x = img_size / orig_w
    scale_y = img_size / orig_h
    
    # 将坐标从512x512映射回原始尺寸
    inv_scale_x = 1.0 / scale_x
    inv_scale_y = 1.0 / scale_y
    
    img_vis = img_orig.copy()
    
    # 绘制大框（红色）
    if big_box_512 is not None:
        bx1_orig = big_box_512[0] * inv_scale_x
        by1_orig = big_box_512[1] * inv_scale_y
        bx2_orig = big_box_512[2] * inv_scale_x
        by2_orig = big_box_512[3] * inv_scale_y
        cv2.rectangle(img_vis, (int(bx1_orig), int(by1_orig)), 
                     (int(bx2_orig), int(by2_orig)), (0, 0, 255), 3)
    
    # 绘制小框（绿色）- 优先使用四个角点
    if corners_512 is not None:
        # 将角点从512x512映射回原始尺寸
        corners_orig = corners_512.copy()
        corners_orig[:, 0] *= inv_scale_x
        corners_orig[:, 1] *= inv_scale_y
        corners_int = corners_orig.astype(np.int32)
        cv2.polylines(img_vis, [corners_int], isClosed=True, color=(0, 255, 0), thickness=3)
    elif small_box_512 is not None:
        # 使用AABB
        sx1_orig = small_box_512[0] * inv_scale_x
        sy1_orig = small_box_512[1] * inv_scale_y
        sx2_orig = small_box_512[2] * inv_scale_x
        sy2_orig = small_box_512[3] * inv_scale_y
        cv2.rectangle(img_vis, (int(sx1_orig), int(sy1_orig)), 
                     (int(sx2_orig), int(sy2_orig)), (0, 255, 0), 3)
    
    # 同时绘制CSV中的原始坐标（用于对比）
    if 'small_box_corner1_x' in row:
        corners_csv = np.array([
            [row['small_box_corner1_x'], row['small_box_corner1_y']],
            [row['small_box_corner2_x'], row['small_box_corner2_y']],
            [row['small_box_corner3_x'], row['small_box_corner3_y']],
            [row['small_box_corner4_x'], row['small_box_corner4_y']]
        ], dtype=np.float32)
        corners_csv_int = corners_csv.astype(np.int32)
        cv2.polylines(img_vis, [corners_csv_int], isClosed=True, color=(255, 255, 0), thickness=2)
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{os.path.splitext(img_name)[0]}_original.jpg')
    cv2.imwrite(output_path, img_vis)
    print(f"已保存到: {output_path}")
    print(f"原始图像尺寸: {orig_w} x {orig_h}")
    print(f"缩放比例: {scale_x:.4f} x {scale_y:.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_name', type=str, required=True, help='图像文件名')
    parser.add_argument('--csv_path', type=str, 
                       default='/root/Desktop/我的网盘/mySAM/data/ISIC/train_boxes_update.csv')
    parser.add_argument('--data_root', type=str,
                       default='/root/Desktop/我的网盘/mySAM/data/update_ISIC')
    parser.add_argument('--output_dir', type=str,
                       default='/root/Desktop/我的网盘/mySAM/utils/test/original_visualization')
    args = parser.parse_args()
    
    visualize_on_original_image(
        args.img_name, args.csv_path, args.data_root, args.output_dir
    )
