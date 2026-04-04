#!/usr/bin/env python
# coding: utf-8

"""
数据工具模块
数据加载相关工具，包括 collate 函数
"""

import torch


def collate_fn_isic(batch):
    """
    将 ISIC2016Dataset 的 batch 转换为训练所需的格式
    
    Args:
        batch: 从 ISIC2016Dataset 获取的 batch
    
    Returns:
        dict: 包含以下键的字典
            - 'image': (B, 3, H, W) 图像tensor
            - 'boxes': list of big_boxes
            - 'big_boxes': list of big_boxes
            - 'small_boxes': list of small_boxes
            - 'mask': (B, 1, H, W) mask tensor
            - 'img_names': list of image names
    """
    images = []
    big_boxes_list = []
    small_boxes_list = []
    masks = []
    img_names = []
    small_box_corners_list = []  # 存储四个角点
    
    for item in batch:
        if len(item) == 6:
            image, big_box, small_box, mask, img_name, small_box_corners = item
        else:
            # 兼容旧格式（没有角点）
            image, big_box, small_box, mask, img_name = item
            small_box_corners = None
        images.append(image)
        big_boxes_list.append(big_box.tolist())  # 转换为 list
        small_boxes_list.append(small_box.tolist())
        masks.append(mask)
        img_names.append(img_name)
        # 保存四个角点（如果存在）
        if small_box_corners is not None:
            small_box_corners_list.append(small_box_corners.tolist() if hasattr(small_box_corners, 'tolist') else small_box_corners)
        else:
            small_box_corners_list.append(None)
    
    # 堆叠图像和 masks
    images = torch.stack(images, dim=0)  # (B, 3, H, W)
    masks = torch.stack(masks, dim=0)  # (B, 1, H, W)
    
    # 返回格式：与原来的 SimpleMaskDataset 兼容
    return {
        'image': images,
        'boxes': big_boxes_list,  # 使用 big_box 作为主要 box
        'big_boxes': big_boxes_list,  # 保留 big_box
        'small_boxes': small_boxes_list,  # 保留 small_box (AABB，用于向后兼容)
        'mask': masks,
        'img_names': img_names,
        'small_box_corners': small_box_corners_list  # 添加四个角点列表（如果存在）
    }

