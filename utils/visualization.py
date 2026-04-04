#!/usr/bin/env python
# coding: utf-8

"""
可视化工具模块
用于可视化预测结果、大小框和mask
"""

import cv2
import numpy as np
import torch


def get_rotated_box_corners(box, center, angle: float) -> np.ndarray:
    """
    获取旋转后框的四个角点
    
    Args:
        box: 框坐标 [x1, y1, x2, y2]
        center: 旋转中心 (cx, cy)
        angle: 旋转角度（度）
    
    Returns:
        rotated_corners: 旋转后的四个角点坐标 (4, 2)
    """
    if isinstance(box, (list, tuple)):
        x1, y1, x2, y2 = box
    else:
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    
    cx, cy = center
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


def visualize_prediction_with_boxes(image, pred_mask, gt_mask, big_box, small_box, 
                                     save_path, img_name=None, iou_value=None, small_box_corners=None):
    """
    可视化预测结果，包含大小框和mask
    
    Args:
        image: 原始图像，形状为 (3, H, W)，值在 [0, 1] 或 (H, W, 3) 值在 [0, 255]
        pred_mask: 预测 mask，形状为 (H, W) 或 (1, H, W)，值在 [0, 1] 或 logits
        gt_mask: 真实 mask，形状为 (H, W) 或 (1, H, W)，值为 0 或 1
        big_box: 大框坐标，形状为 (4,)，格式为 [x1, y1, x2, y2]
        small_box: 小框坐标，形状为 (4,)，格式为 [x1, y1, x2, y2]，可以为None
        save_path: 保存路径
        img_name: 图像名称（可选，用于显示）
        iou_value: IoU值（可选，用于显示）
    """
    # 转换为 numpy 格式用于可视化
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:
            # (C, H, W) -> (H, W, C)
            img_np = image.permute(1, 2, 0).cpu().numpy()
        else:
            img_np = image.cpu().numpy()
        
        # 如果值在 [0, 1]，转换为 [0, 255]
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
    else:
        img_np = np.array(image)
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
    
    # 确保是RGB格式
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    
    # 处理预测mask
    if isinstance(pred_mask, torch.Tensor):
        pred_mask_np = pred_mask.cpu().numpy()
        if len(pred_mask_np.shape) == 3:
            pred_mask_np = pred_mask_np.squeeze(0)
    else:
        pred_mask_np = np.array(pred_mask)
        if len(pred_mask_np.shape) == 3:
            pred_mask_np = pred_mask_np.squeeze(0)
    
    # 如果是logits，先应用sigmoid
    if pred_mask_np.min() < 0 or pred_mask_np.max() > 1:
        pred_mask_np = 1 / (1 + np.exp(-pred_mask_np))
    
    pred_mask_bin = (pred_mask_np >= 0.5).astype(np.uint8)
    
    # 处理真实mask
    if isinstance(gt_mask, torch.Tensor):
        gt_mask_np = gt_mask.cpu().numpy()
        if len(gt_mask_np.shape) == 3:
            gt_mask_np = gt_mask_np.squeeze(0)
    else:
        gt_mask_np = np.array(gt_mask)
        if len(gt_mask_np.shape) == 3:
            gt_mask_np = gt_mask_np.squeeze(0)
    
    gt_mask_bin = (gt_mask_np > 0.5).astype(np.uint8)
    
    # 创建可视化图像
    vis_img = img_np.copy()
    
    # 绘制大框（红色）
    if big_box is not None:
        if isinstance(big_box, torch.Tensor):
            x1, y1, x2, y2 = big_box.int().cpu().numpy()
        else:
            x1, y1, x2, y2 = big_box
        cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        # 添加标签
        cv2.putText(vis_img, 'Big Box', (int(x1), int(y1) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # 绘制小框（绿色）- 支持旋转框（使用四个角点）
    if small_box is not None:
        if isinstance(small_box, torch.Tensor):
            sx1, sy1, sx2, sy2 = small_box.int().cpu().numpy()
        else:
            sx1, sy1, sx2, sy2 = small_box
        
        # 如果提供了四个角点，绘制旋转框
        if small_box_corners is not None:
            try:
                # 转换为numpy数组
                if isinstance(small_box_corners, (list, tuple)):
                    corners = np.array(small_box_corners, dtype=np.float32)
                else:
                    corners = small_box_corners
                
                # 确保是 (4, 2) 形状
                if corners.shape == (4, 2):
                    corners_int = corners.astype(np.int32)
                    # 绘制旋转框（多边形）
                    cv2.polylines(vis_img, [corners_int], isClosed=True, color=(0, 255, 0), thickness=2)
                    
                    # 标签位置使用旋转框的最小y坐标
                    label_y = int(np.min(corners_int[:, 1])) - 10
                    if label_y < 0:
                        label_y = int(np.max(corners_int[:, 1])) + 20
                    label_x = int(np.min(corners_int[:, 0]))
                    cv2.putText(vis_img, 'Small Box (Rotated)', (label_x, label_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    # 角点格式不正确，回退到普通矩形
                    cv2.rectangle(vis_img, (int(sx1), int(sy1)), (int(sx2), int(sy2)), (0, 255, 0), 2)
                    cv2.putText(vis_img, 'Small Box', (int(sx1), int(sy1) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except Exception as e:
                # 如果旋转框绘制失败，回退到普通矩形
                cv2.rectangle(vis_img, (int(sx1), int(sy1)), (int(sx2), int(sy2)), (0, 255, 0), 2)
                cv2.putText(vis_img, 'Small Box', (int(sx1), int(sy1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # 没有角点，绘制普通矩形（AABB）
            cv2.rectangle(vis_img, (int(sx1), int(sy1)), (int(sx2), int(sy2)), (0, 255, 0), 2)
            cv2.putText(vis_img, 'Small Box', (int(sx1), int(sy1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 绘制预测 mask（绿色，半透明）
    pred_colored = np.zeros_like(vis_img)
    pred_colored[pred_mask_bin > 0] = [0, 255, 0]
    vis_img = cv2.addWeighted(vis_img, 0.7, pred_colored, 0.3, 0)
    
    # 绘制真实 mask（蓝色边框）
    gt_contours, _ = cv2.findContours(gt_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_img, gt_contours, -1, (0, 0, 255), 2)
    
    # 添加文本信息
    text_y = 30
    if img_name:
        cv2.putText(vis_img, f'Image: {img_name}', (10, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        text_y += 25
    
    if iou_value is not None:
        cv2.putText(vis_img, f'IoU: {iou_value:.4f}', (10, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        text_y += 25
    
    # 添加图例
    legend_y = vis_img.shape[0] - 80
    cv2.putText(vis_img, 'Red: Big Box, Green: Small Box', (10, legend_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(vis_img, 'Green overlay: Pred Mask, Blue contour: GT Mask', (10, legend_y + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 保存图像（BGR格式）
    cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    
    return vis_img


def visualize_prediction_on_original_image(original_img_path, pred_mask_512, gt_mask_512, 
                                           big_box_512, small_box_512, save_path, 
                                           img_name=None, iou_value=None, small_box_corners_512=None,
                                           img_size=512):
    """
    在原始图像上可视化预测结果（坐标从512x512映射回原始尺寸）
    
    Args:
        original_img_path: 原始图像路径
        pred_mask_512: 预测mask（512x512尺寸），形状为 (H, W) 或 (1, H, W)
        gt_mask_512: 真实mask（512x512尺寸），形状为 (H, W) 或 (1, H, W)
        big_box_512: 大框坐标（512x512坐标系），格式为 [x1, y1, x2, y2]
        small_box_512: 小框坐标（512x512坐标系），格式为 [x1, y1, x2, y2]，可以为None
        save_path: 保存路径
        img_name: 图像名称（可选，用于显示）
        iou_value: IoU值（可选，用于显示）
        small_box_corners_512: 小框四个角点（512x512坐标系），形状为 (4, 2) 或 list
        img_size: 训练时使用的图像尺寸（默认512）
    """
    # 读取原始图像
    img_orig = cv2.imread(original_img_path)
    if img_orig is None:
        raise ValueError(f"无法读取原始图像: {original_img_path}")
    
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_orig.shape[:2]
    
    # 计算缩放比例
    scale_x = img_size / orig_w
    scale_y = img_size / orig_h
    inv_scale_x = 1.0 / scale_x
    inv_scale_y = 1.0 / scale_y
    
    # 处理预测mask：resize回原始尺寸
    if isinstance(pred_mask_512, torch.Tensor):
        pred_mask_np = pred_mask_512.cpu().numpy()
        if len(pred_mask_np.shape) == 3:
            pred_mask_np = pred_mask_np.squeeze(0)
    else:
        pred_mask_np = np.array(pred_mask_512)
        if len(pred_mask_np.shape) == 3:
            pred_mask_np = pred_mask_np.squeeze(0)
    
    # 如果是logits，先应用sigmoid
    if pred_mask_np.min() < 0 or pred_mask_np.max() > 1:
        pred_mask_np = 1 / (1 + np.exp(-pred_mask_np))
    
    # Resize到原始尺寸
    pred_mask_orig = cv2.resize(pred_mask_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    pred_mask_bin = (pred_mask_orig >= 0.5).astype(np.uint8)
    
    # 处理真实mask：resize回原始尺寸
    if isinstance(gt_mask_512, torch.Tensor):
        gt_mask_np = gt_mask_512.cpu().numpy()
        if len(gt_mask_np.shape) == 3:
            gt_mask_np = gt_mask_np.squeeze(0)
    else:
        gt_mask_np = np.array(gt_mask_512)
        if len(gt_mask_np.shape) == 3:
            gt_mask_np = gt_mask_np.squeeze(0)
    
    gt_mask_orig = cv2.resize(gt_mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    gt_mask_bin = (gt_mask_orig > 0.5).astype(np.uint8)
    
    # 创建可视化图像
    vis_img = img_orig.copy()
    
    # 根据原始图像尺寸调整线宽和字体大小
    thickness = max(2, int(orig_w / 500))
    font_scale = max(0.6, orig_w / 1000)
    text_thickness = max(2, int(orig_w / 500))
    
    # 将坐标从512x512映射回原始尺寸
    # 绘制大框（红色）
    if big_box_512 is not None:
        if isinstance(big_box_512, torch.Tensor):
            x1_512, y1_512, x2_512, y2_512 = big_box_512.cpu().numpy()
        else:
            x1_512, y1_512, x2_512, y2_512 = big_box_512
        
        x1_orig = int(x1_512 * inv_scale_x)
        y1_orig = int(y1_512 * inv_scale_y)
        x2_orig = int(x2_512 * inv_scale_x)
        y2_orig = int(y2_512 * inv_scale_y)
        
        cv2.rectangle(vis_img, (x1_orig, y1_orig), (x2_orig, y2_orig), (255, 0, 0), thickness)
        # 添加标签
        cv2.putText(vis_img, 'Big Box', (x1_orig, y1_orig - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
    
    # 绘制小框（绿色）- 支持旋转框（使用四个角点）
    if small_box_512 is not None:
        if isinstance(small_box_512, torch.Tensor):
            sx1_512, sy1_512, sx2_512, sy2_512 = small_box_512.cpu().numpy()
        else:
            sx1_512, sy1_512, sx2_512, sy2_512 = small_box_512
        
        # 如果提供了四个角点，绘制旋转框
        if small_box_corners_512 is not None:
            try:
                # 转换为numpy数组
                if isinstance(small_box_corners_512, (list, tuple)):
                    corners_512 = np.array(small_box_corners_512, dtype=np.float32)
                else:
                    corners_512 = small_box_corners_512
                
                # 确保是 (4, 2) 形状
                if corners_512.shape == (4, 2):
                    # 将角点从512x512映射回原始尺寸
                    corners_orig = corners_512.copy()
                    corners_orig[:, 0] *= inv_scale_x
                    corners_orig[:, 1] *= inv_scale_y
                    corners_int = corners_orig.astype(np.int32)
                    
                    # 绘制旋转框（多边形）
                    cv2.polylines(vis_img, [corners_int], isClosed=True, color=(0, 255, 0), thickness=thickness)
                    
                    # 标签位置使用旋转框的最小y坐标
                    label_y = int(np.min(corners_int[:, 1])) - 10
                    if label_y < 0:
                        label_y = int(np.max(corners_int[:, 1])) + 20
                    label_x = int(np.min(corners_int[:, 0]))
                    cv2.putText(vis_img, 'Small Box (Rotated)', (label_x, label_y),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
                else:
                    # 角点格式不正确，回退到普通矩形
                    sx1_orig = int(sx1_512 * inv_scale_x)
                    sy1_orig = int(sy1_512 * inv_scale_y)
                    sx2_orig = int(sx2_512 * inv_scale_x)
                    sy2_orig = int(sy2_512 * inv_scale_y)
                    cv2.rectangle(vis_img, (sx1_orig, sy1_orig), (sx2_orig, sy2_orig), (0, 255, 0), thickness)
                    cv2.putText(vis_img, 'Small Box', (sx1_orig, sy1_orig - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
            except Exception as e:
                # 如果旋转框绘制失败，回退到普通矩形
                sx1_orig = int(sx1_512 * inv_scale_x)
                sy1_orig = int(sy1_512 * inv_scale_y)
                sx2_orig = int(sx2_512 * inv_scale_x)
                sy2_orig = int(sy2_512 * inv_scale_y)
                cv2.rectangle(vis_img, (sx1_orig, sy1_orig), (sx2_orig, sy2_orig), (0, 255, 0), thickness)
                cv2.putText(vis_img, 'Small Box', (sx1_orig, sy1_orig - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
        else:
            # 没有角点，绘制普通矩形（AABB）
            sx1_orig = int(sx1_512 * inv_scale_x)
            sy1_orig = int(sy1_512 * inv_scale_y)
            sx2_orig = int(sx2_512 * inv_scale_x)
            sy2_orig = int(sy2_512 * inv_scale_y)
            cv2.rectangle(vis_img, (sx1_orig, sy1_orig), (sx2_orig, sy2_orig), (0, 255, 0), thickness)
            cv2.putText(vis_img, 'Small Box', (sx1_orig, sy1_orig - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
    
    # 绘制预测 mask（绿色，半透明）
    pred_colored = np.zeros_like(vis_img)
    pred_colored[pred_mask_bin > 0] = [0, 255, 0]
    vis_img = cv2.addWeighted(vis_img, 0.7, pred_colored, 0.3, 0)
    
    # 绘制真实 mask（蓝色边框）
    gt_contours, _ = cv2.findContours(gt_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_img, gt_contours, -1, (0, 0, 255), thickness)
    
    # 添加文本信息
    text_y = max(30, int(orig_h / 30))
    
    if img_name:
        cv2.putText(vis_img, f'Image: {img_name}', (10, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness)
        text_y += int(orig_h / 25)
    
    if iou_value is not None:
        cv2.putText(vis_img, f'IoU: {iou_value:.4f}', (10, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness)
        text_y += int(orig_h / 25)
    
    # 添加图例
    legend_y = orig_h - int(orig_h / 15)
    cv2.putText(vis_img, 'Red: Big Box, Green: Small Box', (10, legend_y),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (255, 255, 255), text_thickness - 1)
    cv2.putText(vis_img, 'Green overlay: Pred Mask, Blue contour: GT Mask', (10, legend_y + int(orig_h / 30)),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (255, 255, 255), text_thickness - 1)
    
    # 保存图像（BGR格式）
    cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    
    return vis_img
