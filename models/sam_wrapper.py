#!/usr/bin/env python
# coding: utf-8

"""
SAM 模型封装模块
封装 SAM 模型的加载、初始化和配置
"""

import os
import sys

import torch
import torch.nn as nn


def setup_sam_path():
    """
    设置 segment_anything 模块路径
    """
    segment_anything_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        'segment-anything'
    )
    if os.path.exists(segment_anything_path) and segment_anything_path not in sys.path:
        sys.path.insert(0, segment_anything_path)
    
    try:
        from segment_anything import sam_model_registry
        return sam_model_registry
    except ImportError as e:
        # 如果还是找不到，尝试其他可能的路径
        alternative_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'segment-anything'),
            '/root/workspace/segment-anything',
            './segment-anything',
            '../segment-anything',
        ]
        for alt_path in alternative_paths:
            if os.path.exists(alt_path) and alt_path not in sys.path:
                sys.path.insert(0, alt_path)
                try:
                    from segment_anything import sam_model_registry
                    return sam_model_registry
                except ImportError:
                    continue
        raise ImportError(f"无法找到 segment_anything 模块。请确保 segment-anything 目录在正确的位置。错误: {e}")


def load_sam_model(checkpoint_path, model_type, device, unfreeze_last_k=0, use_gradient_checkpointing=False):
    """
    加载 SAM 模型并配置冻结策略
    
    Args:
        checkpoint_path: checkpoint 路径
        model_type: 模型类型（vit_b/vit_l/vit_h）
        device: 设备
        unfreeze_last_k: 解冻最后K层，0表示解冻全部
        use_gradient_checkpointing: 是否使用梯度检查点
    
    Returns:
        sam: 配置好的 SAM 模型
    """
    sam_model_registry = setup_sam_path()
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    sam.eval()
    
    # 冻结 mask_decoder 和 prompt_encoder
    for p in sam.mask_decoder.parameters():
        p.requires_grad = False
    for p in sam.prompt_encoder.parameters():
        p.requires_grad = False
    
    # 梯度检查点
    if use_gradient_checkpointing:
        if hasattr(sam.image_encoder, 'blocks'):
            for block in sam.image_encoder.blocks:
                if hasattr(block, 'gradient_checkpointing'):
                    block.gradient_checkpointing = True
    
    # 部分解冻策略
    if unfreeze_last_k > 0:
        # 先冻结所有 encoder 参数
        for p in sam.image_encoder.parameters():
            p.requires_grad = False
        
        # 查找 blocks 容器
        block_container = None
        for attr in ['blocks', 'transformer', 'resblocks', 'layers']:
            if hasattr(sam.image_encoder, attr):
                block_container = getattr(sam.image_encoder, attr)
                break
        
        if block_container is not None:
            total = len(block_container)
            start = max(0, total - unfreeze_last_k)
            print(f"[INFO] Unfreezing blocks {start} to {total-1} out of {total} total blocks")
            for i in range(start, total):
                for p in block_container[i].parameters():
                    p.requires_grad = True
        else:
            print("[WARN] Cannot find encoder blocks container; will train all encoder parameters.")
            for p in sam.image_encoder.parameters():
                p.requires_grad = True
    else:
        # 解冻整个 image_encoder
        for p in sam.image_encoder.parameters():
            p.requires_grad = True
        print("[INFO] Training entire image_encoder")
    
    return sam


def create_teacher_model(checkpoint_path, model_type, device):
    """
    创建 EMA teacher 模型（完全冻结）
    
    Args:
        checkpoint_path: checkpoint 路径
        model_type: 模型类型
        device: 设备
    
    Returns:
        teacher: 冻结的 teacher 模型
    """
    sam_model_registry = setup_sam_path()
    teacher = sam_model_registry[model_type](checkpoint=checkpoint_path)
    teacher.to(device)
    teacher.eval()
    
    # Teacher 完全冻结
    for p in teacher.parameters():
        p.requires_grad = False
    
    return teacher


def get_encoder_feature_dim(sam_model):
    """
    从模型结构推断 encoder 输出特征维度
    
    Args:
        sam_model: SAM 模型
    
    Returns:
        in_dim: 特征维度
    """
    in_dim = 256  # SAM ViT-B 的默认输出通道数
    try:
        # 从 neck 层推断输出通道数
        if hasattr(sam_model.image_encoder, 'neck'):
            # neck 是 Sequential，查找最后一个 Conv2d
            for layer in reversed(sam_model.image_encoder.neck):
                if isinstance(layer, nn.Conv2d):
                    in_dim = layer.out_channels
                    break
        print(f"[INFO] Using image_encoder feature channels: {in_dim} (inferred from model structure)")
    except Exception as e:
        print(f"[WARN] Could not infer feature dimension, using default: {in_dim}")
        print(f"[WARN] Error: {e}")
    
    return in_dim


def setup_multi_gpu(models_dict, device_ids):
    """
    设置多GPU（DataParallel）
    
    Args:
        models_dict: 模型字典，例如 {'sam': sam, 'teacher': teacher, 'proj': proj}
        device_ids: GPU ID 列表
    
    Returns:
        wrapped_models: 包装后的模型字典
    """
    wrapped_models = {}
    for name, model in models_dict.items():
        if model is not None:
            wrapped_models[name] = nn.DataParallel(model, device_ids=device_ids)
        else:
            wrapped_models[name] = None
    return wrapped_models

