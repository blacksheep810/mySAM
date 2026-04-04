#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
调试可视化坐标问题
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import cv2
import numpy as np
import pandas as pd
import torch
from dataset.ISIC import ISIC2016Dataset

# 测试图像
img_name = 'ISIC_0000048.jpg'

# 读取CSV
df = pd.read_csv('/root/Desktop/我的网盘/mySAM/data/ISIC/train_boxes_update.csv')
row = df[df['image_file'] == img_name].iloc[0]

# 读取原始图像
img_path = f'/root/Desktop/我的网盘/mySAM/data/update_ISIC/ISBI2016_ISIC_Part1_Training_Data/{img_name}'
img_orig = cv2.imread(img_path)
orig_h, orig_w = img_orig.shape[:2]
print(f'原始图像尺寸: {orig_w} x {orig_h}')

# CSV中的坐标（原始图像尺寸）
corners_orig = np.array([
    [row['small_box_corner1_x'], row['small_box_corner1_y']],
    [row['small_box_corner2_x'], row['small_box_corner2_y']],
    [row['small_box_corner3_x'], row['small_box_corner3_y']],
    [row['small_box_corner4_x'], row['small_box_corner4_y']]
], dtype=np.float32)
print(f'\nCSV中的角点坐标（原始尺寸）:')
print(corners_orig)

# 使用数据集加载
dataset = ISIC2016Dataset(
    root='/root/Desktop/我的网盘/mySAM/data/ISIC',
    box_csv='/root/Desktop/我的网盘/mySAM/data/ISIC/train_boxes_update.csv',
    img_size=512,
    split='train'
)

# 找到对应的索引
idx = df[df['image_file'] == img_name].index[0]
image, big_box, small_box, mask, img_name_loaded, small_box_corners_scaled = dataset[idx]

print(f'\n数据集加载的图像尺寸: {image.shape}')
print(f'缩放后的大框坐标: {big_box.tolist()}')
print(f'缩放后的小框AABB: {small_box.tolist()}')
print(f'缩放后的角点坐标:')
if small_box_corners_scaled is not None:
    print(small_box_corners_scaled)
else:
    print('None')

# 计算缩放比例
scale_x = 512 / orig_w
scale_y = 512 / orig_h
print(f'\n缩放比例: scale_x={scale_x:.4f}, scale_y={scale_y:.4f}')

# 手动缩放角点
corners_manual_scaled = corners_orig.copy()
corners_manual_scaled[:, 0] *= scale_x
corners_manual_scaled[:, 1] *= scale_y
print(f'\n手动缩放后的角点坐标:')
print(corners_manual_scaled)

# 比较
if small_box_corners_scaled is not None:
    diff = np.abs(small_box_corners_scaled - corners_manual_scaled)
    print(f'\n差异（数据集加载 vs 手动缩放）:')
    print(diff)
    print(f'最大差异: {diff.max():.4f}')

# 可视化原始图像上的框
img_vis_orig = img_orig.copy()
corners_int_orig = corners_orig.astype(np.int32)
cv2.polylines(img_vis_orig, [corners_int_orig], isClosed=True, color=(0, 255, 0), thickness=3)
cv2.imwrite('/root/Desktop/我的网盘/mySAM/utils/test/debug_orig_image.jpg', img_vis_orig)
print(f'\n已保存原始图像可视化: debug_orig_image.jpg')

# 可视化缩放后的图像上的框
img_scaled = cv2.resize(img_orig, (512, 512))
img_vis_scaled = img_scaled.copy()
if small_box_corners_scaled is not None:
    corners_int_scaled = small_box_corners_scaled.astype(np.int32)
    cv2.polylines(img_vis_scaled, [corners_int_scaled], isClosed=True, color=(0, 255, 0), thickness=2)
    # 绘制AABB
    sx1, sy1, sx2, sy2 = small_box.int().tolist()
    cv2.rectangle(img_vis_scaled, (sx1, sy1), (sx2, sy2), (255, 0, 0), 1)
cv2.imwrite('/root/Desktop/我的网盘/mySAM/utils/test/debug_scaled_image.jpg', img_vis_scaled)
print(f'已保存缩放图像可视化: debug_scaled_image.jpg')
