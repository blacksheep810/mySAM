#!/usr/bin/env python
# coding: utf-8

"""
训练工具函数模块
提供训练相关的通用工具函数
"""

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed=42):
    """
    设置随机种子，确保实验可复现
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir(path):
    """
    创建目录（如果不存在）
    
    Args:
        path: 目录路径
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def setup_cuda_memory(device_id=0):
    """
    配置 CUDA 内存分配和清理缓存
    
    Args:
        device_id: GPU 设备ID
    
    Returns:
        dict: 包含显存状态信息的字典
    """
    import os
    
    # 设置可扩展内存段以减少碎片化
    if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    if torch.cuda.is_available():
        # 清理之前的缓存
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        # 重置内存统计
        torch.cuda.reset_peak_memory_stats()
        
        # 显示初始显存状态
        allocated = torch.cuda.memory_allocated(device_id) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device_id) / 1024**3    # GB
        total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3  # GB
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'total': total,
            'device_id': device_id
        }
    return None

