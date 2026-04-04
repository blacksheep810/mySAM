#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型对比脚本
对比 SAM 原始模型和训练后的第30轮模型的 mIoU 性能
使用 box prompt 进行推理
"""

import os
import sys
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.sam_wrapper import load_sam_model, setup_sam_path
from dataset.ISIC import ISIC2016Dataset
from training.data_utils import collate_fn_isic
from utils.prompts import prepare_box_prompts


class AverageMeter:
    """计算和存储平均值和当前值"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_miou(pred_mask, gt_mask):
    """
    计算 mIoU（使用 segmentation_models_pytorch 库的方法）
    与 wesam 原始代码保持一致：pred_mask 是 logits，gt_mask 是 (H, W) 格式
    
    Args:
        pred_mask: 预测的 mask（logits），形状为 (H, W) 或 (1, H, W)，值域可能是 (-∞, +∞)
        gt_mask: 真实 mask，形状为 (H, W) 或 (1, H, W)，值为 0 或 1
    
    Returns:
        iou: IoU 分数
    """
    # 与 wesam 原始代码保持一致：pred_mask 是 logits，直接传递
    # 确保 pred_mask 是 (H, W) 格式（smp.metrics.get_stats 要求维度匹配）
    if pred_mask.dim() == 3:
        pred_mask = pred_mask.squeeze(0)  # (1, H, W) -> (H, W)
    elif pred_mask.dim() != 2:
        raise ValueError(f"Unexpected pred_mask shape: {pred_mask.shape}")
    
    # gt_mask 保持 (H, W) 格式（与 wesam 原始代码一致）
    if gt_mask.dim() == 3:
        gt_mask = gt_mask.squeeze(0)  # (1, H, W) -> (H, W)
    elif gt_mask.dim() != 2:
        raise ValueError(f"Unexpected gt_mask shape: {gt_mask.shape}")
    
    # 确保形状匹配（高度和宽度）
    if pred_mask.shape != gt_mask.shape:
        # 如果高度或宽度不匹配，进行插值
        pred_mask = F.interpolate(
            pred_mask.unsqueeze(0).unsqueeze(0),
            size=gt_mask.shape,
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)
    
    # 获取统计信息（使用 (H, W) vs (H, W) 格式，与 wesam 原始代码保持一致）
    # pred_mask 是 logits，smp.metrics.get_stats 在 mode='binary' 时会先应用 sigmoid，然后与 threshold 比较
    # threshold=0.5 意味着 sigmoid(logits) >= 0.5，即 logits >= 0
    batch_stats = smp.metrics.get_stats(
        pred_mask,      # (H, W) 格式，logits
        gt_mask.int(),  # (H, W) 格式
        mode='binary',
        threshold=0.5,
    )
    
    # 计算 IoU
    iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
    
    return iou.item()


def inference_with_box_prompt(model, image, box, device, sam_input_size=1024):
    """
    使用 box prompt 进行推理
    
    Args:
        model: SAM 模型
        image: 输入图像，形状为 (3, H, W)
        box: 外接矩形框，形状为 (4,)，格式为 [x1, y1, x2, y2]（原始图像尺寸下的坐标）
        device: 设备
        sam_input_size: SAM 输入尺寸（默认 1024）
    
    Returns:
        pred_mask: 预测的 mask，形状为 (H, W)，与输入图像尺寸相同
    """
    model.eval()
    
    with torch.no_grad():
        # 获取原始图像尺寸
        if image.dim() == 3:
            img_h, img_w = image.shape[1], image.shape[2]
            image = image.unsqueeze(0)  # (1, 3, H, W)
        else:
            _, _, img_h, img_w = image.shape
        
        # 将图像移动到设备并缩放到 SAM 输入尺寸
        image = image.to(device)
        image_sam = F.interpolate(
            image,
            size=(sam_input_size, sam_input_size),
            mode='bilinear',
            align_corners=False
        )
        
        # 缩放 box 坐标到 SAM 输入尺寸（与 wesam_val.py 保持一致）
        if isinstance(box, torch.Tensor):
            box = box.to(device=device)
        else:
            box = torch.tensor(box, device=device, dtype=torch.float32)
        
        scale_x = sam_input_size / img_w
        scale_y = sam_input_size / img_h
        box_sam = torch.tensor([
            box[0] * scale_x,
            box[1] * scale_y,
            box[2] * scale_x,
            box[3] * scale_y
        ], device=device, dtype=torch.float32).unsqueeze(0)  # (1, 4)
        
        # 提取图像特征
        image_embeddings = model.image_encoder(image_sam)
        
        # 编码 prompt（只使用 big_box，不使用 small_box）
        sparse_p, dense_p = model.prompt_encoder(
            points=None,
            boxes=box_sam,
            masks=None
        )
        
        # 解码 mask
        low_res_masks, scores = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_p,
            dense_prompt_embeddings=dense_p,
            multimask_output=False
        )
        
        # low_res_masks 形状为 (1, 1, 256, 256)
        # 上采样到原始图像尺寸
        pred_mask = F.interpolate(
            low_res_masks,
            size=(img_h, img_w),
            mode='bilinear',
            align_corners=False
        )
        
        # 不应用 sigmoid，直接返回 logits（与 wesam 原始代码保持一致）
        # wesam 原始代码中 pred_mask 是 logits，get_stats 的 threshold=0.5 作用于 logits
        pred_mask = pred_mask.squeeze(0).squeeze(0)  # (H, W)，logits 格式
        
    return pred_mask


def visualize_prediction(image, pred_mask, gt_mask, box, save_path):
    """
    可视化预测结果
    
    Args:
        image: 原始图像，形状为 (3, H, W)，值在 [0, 1]
        pred_mask: 预测 mask，形状为 (H, W)，值在 [0, 1]
        gt_mask: 真实 mask，形状为 (H, W)，值为 0 或 1
        box: 边界框，形状为 (4,)，格式为 [x1, y1, x2, y2]
        save_path: 保存路径
    """
    # 转换为 numpy 格式用于可视化
    if isinstance(image, torch.Tensor):
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = image
    
    if isinstance(pred_mask, torch.Tensor):
        pred_mask_np = (pred_mask.cpu().numpy() >= 0.5).astype(np.uint8)
    else:
        pred_mask_np = (pred_mask >= 0.5).astype(np.uint8)
    
    if isinstance(gt_mask, torch.Tensor):
        gt_mask_np = gt_mask.cpu().numpy().astype(np.uint8)
    else:
        gt_mask_np = gt_mask.astype(np.uint8)
    
    # 创建可视化图像
    vis_img = img_np.copy()
    
    # 绘制边界框（红色）
    x1, y1, x2, y2 = box.int().cpu().numpy() if isinstance(box, torch.Tensor) else box
    cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    
    # 绘制预测 mask（绿色，半透明）
    pred_colored = np.zeros_like(vis_img)
    pred_colored[pred_mask_np > 0] = [0, 255, 0]
    vis_img = cv2.addWeighted(vis_img, 0.7, pred_colored, 0.3, 0)
    
    # 绘制真实 mask（蓝色边框）
    gt_contours, _ = cv2.findContours(gt_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_img, gt_contours, -1, (0, 0, 255), 2)
    
    # 保存图像
    cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))


def evaluate_model(model, dataloader, device, model_name="Model", save_vis=False, vis_dir=None):
    """
    评估模型性能（与 wesam 原始代码保持一致）
    
    Args:
        model: SAM 模型
        dataloader: 数据加载器
        device: 设备
        model_name: 模型名称（用于打印）
        save_vis: 是否保存可视化结果
        vis_dir: 可视化结果保存目录
    
    Returns:
        mean_iou: 平均 IoU
    """
    model.eval()
    ious = AverageMeter()
    
    # 创建可视化目录
    if save_vis and vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
    
    print(f"评估 {model_name}...")
    
    with torch.no_grad():
        # 使用 tqdm 添加进度条
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"评估 {model_name}")
        
        for batch_idx, batch in pbar:
            images = batch['image']  # (B, 3, H, W)
            big_boxes = batch['big_boxes']  # list of (4,)
            masks = batch['mask']  # (B, 1, H, W)
            img_names = batch.get('img_names', [f'img_{batch_idx}_{i}' for i in range(len(big_boxes))])
            
            B = images.size(0)
            
            for b in range(B):
                image = images[b]  # (3, H, W)
                box = torch.tensor(big_boxes[b], dtype=torch.float32)  # (4,)
                gt_mask = masks[b].squeeze(0)  # (H, W)
                # 获取图像名称，如果没有则使用默认名称
                img_name = img_names[b] if b < len(img_names) else f'img_{batch_idx}_{b}'
                # 清理文件名（移除扩展名和路径）
                img_name = os.path.splitext(os.path.basename(str(img_name)))[0]
                
                # 推理
                pred_mask = inference_with_box_prompt(
                    model, image, box, device
                )
                
                # 计算 IoU（每张图像的权重为 1，而不是 batch_size）
                # 修复：之前使用 num_images (batch_size) 作为权重，导致每张图像的 IoU 被重复累计 B 次
                iou = compute_miou(pred_mask.cpu(), gt_mask.cpu())
                ious.update(iou, n=1)  # 每张图像的权重为 1
                
                # 保存可视化结果
                if save_vis and vis_dir:
                    # 清理模型名称，移除特殊字符
                    safe_model_name = model_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                    vis_path = os.path.join(vis_dir, f"{safe_model_name}_{img_name}.png")
                    visualize_prediction(image.cpu(), pred_mask.cpu(), gt_mask.cpu(), box, vis_path)
            
            # 更新进度条
            pbar.set_postfix({'mIoU': f'{ious.avg:.4f}'})
            
            torch.cuda.empty_cache()
    
    print(f"{model_name}: mIoU = {ious.avg:.4f}")
    return ious.avg


def detect_model_type_from_checkpoint(checkpoint_path):
    """
    从 checkpoint 中检测模型类型
    
    Args:
        checkpoint_path: checkpoint 路径
    
    Returns:
        model_type: 'vit_b', 'vit_l', 或 'vit_h'
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'sam_image_encoder' not in checkpoint:
        raise ValueError("Checkpoint 中缺少 'sam_image_encoder' 键")
    
    encoder_state = checkpoint['sam_image_encoder']
    
    # 处理 DataParallel 的情况
    if any(k.startswith('module.') for k in encoder_state.keys()):
        encoder_state = {k.replace('module.', ''): v for k, v in encoder_state.items()}
    
    # 通过检查 embed_dim 来确定模型类型
    # 查找 pos_embed 或 patch_embed 来确定维度
    embed_dim = None
    
    # 方法1: 检查 pos_embed
    if 'pos_embed' in encoder_state:
        pos_embed = encoder_state['pos_embed']
        if pos_embed.dim() == 4:
            embed_dim = pos_embed.shape[-1]
    
    # 方法2: 检查 patch_embed.proj.weight
    if embed_dim is None and 'patch_embed.proj.weight' in encoder_state:
        # patch_embed.proj.weight 形状通常是 (embed_dim, 3, patch_size, patch_size)
        weight = encoder_state['patch_embed.proj.weight']
        if weight.dim() == 4:
            embed_dim = weight.shape[0]
    
    # 方法3: 检查 blocks 中的权重
    if embed_dim is None:
        for key in encoder_state.keys():
            if 'blocks.0.norm1.weight' in key:
                # norm1.weight 形状是 (embed_dim,)
                weight = encoder_state[key]
                if weight.dim() == 1:
                    embed_dim = weight.shape[0]
                    break
    
    # 根据 embed_dim 确定模型类型
    # 注意：SAM 模型的实际 embed_dim：
    # vit_b: 768
    # vit_l: 1024
    # vit_h: 1280
    if embed_dim == 768:
        return 'vit_b'
    elif embed_dim == 1024:
        return 'vit_l'
    elif embed_dim == 1280:
        return 'vit_h'
    else:
        raise ValueError(f"无法识别模型类型，embed_dim={embed_dim}。支持的 embed_dim: 768 (vit_b), 1024 (vit_l), 1280 (vit_h)")


def load_trained_model(checkpoint_path, sam_original_model, device, model_type=None, original_model_type='vit_h'):
    """
    加载训练好的模型（只包含 encoder 和 proj）
    
    Args:
        checkpoint_path: checkpoint 路径
        sam_original_model: SAM 原始模型（用于获取完整结构，用于对比的模型）
        device: 设备
        model_type: 训练模型类型，如果为 None 则自动检测
        original_model_type: 原始模型类型（用于对比的模型类型，默认 vit_h）
    
    Returns:
        trained_model: 加载了训练权重的 SAM 模型
    """
    print(f"\n加载训练好的模型: {checkpoint_path}")
    
    # 自动检测模型类型
    if model_type is None:
        model_type = detect_model_type_from_checkpoint(checkpoint_path)
    else:
        # 验证指定的类型是否正确
        detected_type = detect_model_type_from_checkpoint(checkpoint_path)
        if detected_type != model_type:
            print(f"  ⚠ 警告：指定的模型类型 ({model_type}) 与检测到的类型 ({detected_type}) 不一致")
            print(f"  将使用检测到的类型: {detected_type}")
            model_type = detected_type
    
    print(f"  模型类型: {model_type}")
    
    # 创建新的模型实例（使用训练模型的类型）
    sam_model_registry = setup_sam_path()
    trained_model = sam_model_registry[model_type](checkpoint=None)
    trained_model.to(device)
    trained_model.eval()
    
    # 冻结所有参数
    for p in trained_model.parameters():
        p.requires_grad = False
    
    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载 encoder 权重
    if 'sam_image_encoder' in checkpoint:
        encoder_state = checkpoint['sam_image_encoder']
        
        # 处理 DataParallel 的情况（如果 key 以 'module.' 开头）
        if any(k.startswith('module.') for k in encoder_state.keys()):
            # 移除 'module.' 前缀
            encoder_state = {k.replace('module.', ''): v for k, v in encoder_state.items()}
        
        # 验证模型类型是否匹配
        checkpoint_embed_dim = None
        if 'pos_embed' in encoder_state:
            pos_embed = encoder_state['pos_embed']
            if pos_embed.dim() == 4:
                checkpoint_embed_dim = pos_embed.shape[-1]
        elif 'patch_embed.proj.weight' in encoder_state:
            weight = encoder_state['patch_embed.proj.weight']
            if weight.dim() == 4:
                checkpoint_embed_dim = weight.shape[0]
        
        model_embed_dim = None
        if hasattr(trained_model.image_encoder, 'pos_embed'):
            pos_embed = trained_model.image_encoder.pos_embed
            if pos_embed.dim() == 4:
                model_embed_dim = pos_embed.shape[-1]
        elif hasattr(trained_model.image_encoder, 'patch_embed'):
            weight = trained_model.image_encoder.patch_embed.proj.weight
            if weight.dim() == 4:
                model_embed_dim = weight.shape[0]
        
        if checkpoint_embed_dim is not None and model_embed_dim is not None:
            if checkpoint_embed_dim != model_embed_dim:
                raise ValueError(
                    f"模型类型不匹配！checkpoint 的 embed_dim={checkpoint_embed_dim} "
                    f"（对应 {detect_model_type_from_checkpoint(checkpoint_path)}），"
                    f"但创建的模型 embed_dim={model_embed_dim}（对应 {model_type}）。"
                    f"请检查模型类型设置是否正确。"
                )
        
        trained_model.image_encoder.load_state_dict(encoder_state, strict=False)
    else:
        raise ValueError("Checkpoint 中缺少 'sam_image_encoder' 键")
    
    # 加载 prompt_encoder 和 mask_decoder
    # 注意：prompt_encoder 和 mask_decoder 的尺寸与模型类型相关
    # 训练模型使用自己类型的 decoder，而不是 huge 模型的 decoder
    # 因为不同模型类型的 decoder 尺寸不同
    
    # 加载 prompt_encoder 和 mask_decoder（使用相同类型的原始模型）
    trained_model.prompt_encoder.load_state_dict(sam_original_model.prompt_encoder.state_dict())
    trained_model.mask_decoder.load_state_dict(sam_original_model.mask_decoder.state_dict())
    
    return trained_model


def load_config(config_path):
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        config: 配置字典
    """
    # 检查文件是否存在
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    # 尝试多种编码方式读取配置文件
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin-1']
    config = None
    last_error = None
    
    for encoding in encodings:
        try:
            with open(config_path, 'r', encoding=encoding) as f:
                content = f.read()
                config = yaml.safe_load(content)
            if encoding != 'utf-8':
                print(f"  注意：配置文件使用 {encoding} 编码读取，建议转换为 UTF-8")
            break
        except UnicodeDecodeError as e:
            last_error = f"UnicodeDecodeError with {encoding}: {e}"
            continue
        except yaml.YAMLError as e:
            last_error = f"YAMLError with {encoding}: {e}"
            # YAML 解析错误，可能是编码问题，继续尝试下一个编码
            continue
        except Exception as e:
            last_error = f"Error with {encoding}: {e}"
            if encoding == encodings[-1]:  # 最后一次尝试
                raise ValueError(f"无法读取配置文件 {config_path}: {e}")
            continue
    
    if config is None:
        error_msg = f"无法读取配置文件 {config_path}，尝试了多种编码方式都失败"
        if last_error:
            error_msg += f"\n最后错误: {last_error}"
        raise ValueError(error_msg)
    
    # 将相对路径转换为绝对路径（相对于项目根目录）
    # 如果配置文件路径是绝对路径，直接使用；否则相对于当前工作目录
    if os.path.isabs(config_path):
        config_dir = os.path.dirname(config_path)
    else:
        config_dir = os.path.dirname(os.path.abspath(config_path))
    
    # 假设配置文件在 validate/ 目录下，项目根目录是 validate 的父目录
    # 如果配置文件在其他位置，需要调整
    if 'validate' in config_dir or os.path.basename(config_dir) == 'validate':
        project_root = os.path.dirname(config_dir)
    else:
        # 如果不在 validate 目录，假设配置文件就在项目根目录
        project_root = config_dir
    
    # 转换数据路径
    if 'data' in config:
        if 'data_root' in config['data']:
            config['data']['data_root'] = os.path.normpath(
                os.path.join(project_root, config['data']['data_root'])
            )
        if 'test_box_csv' in config['data']:
            config['data']['test_box_csv'] = os.path.normpath(
                os.path.join(project_root, config['data']['test_box_csv'])
            )
    
    # 转换模型路径
    if 'models' in config:
        if 'sam_checkpoint' in config['models']:
            config['models']['sam_checkpoint'] = os.path.normpath(
                os.path.join(project_root, config['models']['sam_checkpoint'])
            )
        if 'trained_checkpoint' in config['models']:
            config['models']['trained_checkpoint'] = os.path.normpath(
                os.path.join(project_root, config['models']['trained_checkpoint'])
            )
        if 'checkpoint_paths' in config['models']:
            for key, path in config['models']['checkpoint_paths'].items():
                config['models']['checkpoint_paths'][key] = os.path.normpath(
                    os.path.join(project_root, path)
                )
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description='对比 SAM huge 和训练后的模型性能',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用配置文件
  python validate/compare_models.py --config validate/config.yaml
  
  # 使用命令行参数（会覆盖配置文件中的对应参数）
  python validate/compare_models.py --config validate/config.yaml --batch_size 8
  
  # 不使用配置文件，直接使用命令行参数
  python validate/compare_models.py --data_root /path/to/data --test_box_csv /path/to/test.csv
        """
    )
    
    # 配置文件参数
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（YAML 格式），如果提供则从配置文件读取参数。'
                            '可以使用相对路径（相对于项目根目录），例如: validate/config.yaml')
    
    # 数据集参数
    parser.add_argument('--data_root', type=str, default=None, help='ISIC 数据集根目录')
    parser.add_argument('--test_box_csv', type=str, default=None, help='测试集 box CSV 文件路径')
    
    # 模型参数
    parser.add_argument('--sam_checkpoint', type=str, default=None,
                       help='SAM huge checkpoint 路径（默认使用 huge 模型）')
    parser.add_argument('--trained_checkpoint', type=str, default=None,
                       help='训练好的模型 checkpoint 路径')
    parser.add_argument('--trained_model_type', type=str, default=None,
                       choices=['vit_b', 'vit_l', 'vit_h'],
                       help='训练好的模型类型，如果为 None 则自动检测')
    
    # 评估参数
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小')
    parser.add_argument('--img_size', type=int, default=None, help='图像尺寸')
    parser.add_argument('--device', type=str, default=None, help='设备 (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=None, help='数据加载器工作进程数')
    parser.add_argument('--save_vis', action='store_true', help='是否保存可视化结果')
    parser.add_argument('--vis_dir', type=str, default=None, help='可视化结果保存目录')
    
    args = parser.parse_args()
    
    # 加载配置文件（如果提供）
    config = {}
    if args.config:
        config_path = args.config
        
        # 如果路径不存在，尝试多种方式查找
        if not os.path.exists(config_path):
            # 方法1: 尝试相对于当前工作目录
            cwd_path = os.path.join(os.getcwd(), config_path)
            if os.path.exists(cwd_path):
                config_path = cwd_path
            else:
                # 方法2: 尝试相对于项目根目录
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                project_path = os.path.join(project_root, config_path)
                if os.path.exists(project_path):
                    config_path = project_path
                else:
                    # 方法3: 尝试相对于脚本所在目录
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    script_path = os.path.join(script_dir, config_path)
                    if os.path.exists(script_path):
                        config_path = script_path
                    else:
                        # 如果都找不到，给出详细的错误信息
                        print(f"\n错误：配置文件未找到: {args.config}")
                        print(f"尝试过的路径：")
                        print(f"  1. {os.path.abspath(args.config)}")
                        print(f"  2. {cwd_path}")
                        print(f"  3. {project_path}")
                        print(f"  4. {script_path}")
                        print(f"\n提示：请使用相对于项目根目录的路径，例如：")
                        print(f"  --config validate/config.yaml")
                        print(f"  或使用绝对路径")
                        raise FileNotFoundError(f"配置文件未找到: {args.config}")
        
        # 确保使用绝对路径
        if not os.path.isabs(config_path):
            config_path = os.path.abspath(config_path)
        
        config = load_config(config_path)
        print(f"已加载配置文件: {config_path}")
    
    # 合并配置：命令行参数优先于配置文件
    def get_arg(key, config_section=None, default=None):
        """获取参数值，优先级：命令行 > 配置文件 > 默认值"""
        cmd_value = getattr(args, key, None)
        if cmd_value is not None:
            return cmd_value
        if config_section and config_section in config and key in config[config_section]:
            value = config[config_section][key]
            return value if value != 'null' else None
        return default
    
    # 获取所有参数
    data_root = get_arg('data_root', 'data')
    test_box_csv = get_arg('test_box_csv', 'data')
    sam_checkpoint = get_arg('sam_checkpoint', 'models', './checkpoints/sam_vit_h_4b8939.pth')
    trained_checkpoint = get_arg('trained_checkpoint', 'models', './outputs/checkpoint_epoch_30.pth')
    trained_model_type = get_arg('trained_model_type', 'models')
    batch_size = get_arg('batch_size', 'evaluation', 4)
    img_size = get_arg('img_size', 'evaluation', 1024)
    device_str = get_arg('device', 'evaluation', 'cuda')
    num_workers = get_arg('num_workers', 'evaluation', 4)
    save_vis = args.save_vis
    # 获取项目根目录（用于设置默认可视化目录）
    script_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vis_dir = args.vis_dir if args.vis_dir else (os.path.join(script_project_root, 'validate', 'visualizations') if save_vis else None)
    
    # 检查必需参数
    if data_root is None or test_box_csv is None:
        parser.error("必须提供 --data_root 和 --test_box_csv 参数，或通过 --config 配置文件提供")
    
    # 获取 checkpoint 路径映射（如果配置文件中有）
    checkpoint_paths = {}
    if 'models' in config and 'checkpoint_paths' in config['models']:
        checkpoint_paths = config['models']['checkpoint_paths']
    
    # 设置设备
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 打印配置信息
    print("\n" + "="*60)
    print("配置信息")
    print("="*60)
    print(f"数据根目录: {data_root}")
    print(f"测试集 CSV: {test_box_csv}")
    print(f"训练模型 checkpoint: {trained_checkpoint}")
    print(f"批次大小: {batch_size}, 图像尺寸: {img_size}")
    print("="*60)
    
    # 加载数据集
    print(f"\n加载测试数据集...")
    test_dataset = ISIC2016Dataset(
        root=data_root,
        box_csv=test_box_csv,
        img_size=img_size,
        split='test'
    )
    print(f"  测试集大小: {len(test_dataset)}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_isic
    )
    
    # 检测训练好的模型类型
    if trained_model_type is None:
        trained_model_type = detect_model_type_from_checkpoint(trained_checkpoint)
        print(f"\n自动检测到训练好的模型类型: {trained_model_type}")
    else:
        print(f"\n使用指定的训练模型类型: {trained_model_type}")
    
    # 根据训练模型类型选择对应的基座模型进行对比
    original_model_type = trained_model_type  # 使用与训练模型相同的类型
    original_checkpoint = sam_checkpoint
    
    # 根据模型类型确定checkpoint路径
    checkpoint_dir = os.path.dirname(sam_checkpoint) if os.path.dirname(sam_checkpoint) else './checkpoints'
    default_checkpoints = {
        'vit_b': os.path.join(checkpoint_dir, 'sam_vit_b_01ec64.pth'),
        'vit_l': os.path.join(checkpoint_dir, 'sam_vit_l_0b3195.pth'),
        'vit_h': os.path.join(checkpoint_dir, 'sam_vit_h_4b8939.pth')
    }
    
    # 优先使用配置文件中的路径，否则使用默认路径
    if checkpoint_paths and trained_model_type in checkpoint_paths:
        if os.path.exists(checkpoint_paths[trained_model_type]):
            original_checkpoint = checkpoint_paths[trained_model_type]
            print(f"  从配置文件找到 {trained_model_type} 模型 checkpoint: {original_checkpoint}")
        else:
            print(f"  警告：配置文件中的 {trained_model_type} checkpoint 不存在，使用默认路径")
            if trained_model_type in default_checkpoints and os.path.exists(default_checkpoints[trained_model_type]):
                original_checkpoint = default_checkpoints[trained_model_type]
            else:
                raise FileNotFoundError(f"未找到 {trained_model_type} 类型的基座模型 checkpoint")
    else:
        # 使用默认路径
        if trained_model_type in default_checkpoints and os.path.exists(default_checkpoints[trained_model_type]):
            original_checkpoint = default_checkpoints[trained_model_type]
            print(f"  使用默认 {trained_model_type} 模型 checkpoint: {original_checkpoint}")
        else:
            raise FileNotFoundError(f"未找到 {trained_model_type} 类型的基座模型 checkpoint: {default_checkpoints.get(trained_model_type, 'N/A')}")
    
    # 加载对应类型的基座模型
    model_name_map = {'vit_b': 'Base', 'vit_l': 'Large', 'vit_h': 'Huge'}
    print(f"\n加载 SAM {model_name_map[original_model_type]} ({original_model_type}) 原始模型...")
    sam_original = load_sam_model(
        checkpoint_path=original_checkpoint,
        model_type=original_model_type,
        device=device,
        unfreeze_last_k=0
    )
    for p in sam_original.parameters():
        p.requires_grad = False
    sam_original.eval()
    print(f"  ✓ SAM {model_name_map[original_model_type]} 原始模型加载完成")
    
    # 加载训练好的模型
    trained_model = load_trained_model(
        checkpoint_path=trained_checkpoint,
        sam_original_model=sam_original,
        device=device,
        model_type=trained_model_type,
        original_model_type=original_model_type  # 使用与训练模型相同的类型
    )
    
    # 评估基座模型和训练后的模型
    print("\n" + "="*60)
    vis_dir_original = os.path.join(vis_dir, 'original') if save_vis and vis_dir else None
    vis_dir_trained = os.path.join(vis_dir, 'trained') if save_vis and vis_dir else None
    
    miou_original = evaluate_model(
        sam_original, test_loader, device, 
        f"SAM {model_name_map[original_model_type]} 原始模型",
        save_vis=save_vis, vis_dir=vis_dir_original
    )
    miou_trained = evaluate_model(
        trained_model, test_loader, device, 
        "训练后模型",
        save_vis=save_vis, vis_dir=vis_dir_trained
    )
    
    # 打印对比结果
    print("\n" + "="*60)
    print("对比结果")
    print("="*60)
    print(f"基座模型 ({model_name_map[original_model_type]}):  mIoU = {miou_original:.4f}")
    print(f"训练后模型:              mIoU = {miou_trained:.4f}")
    improvement = miou_trained - miou_original
    improvement_pct = (improvement / miou_original * 100) if miou_original > 0 else 0
    print(f"性能变化:                {improvement:+.4f} ({improvement_pct:+.2f}%)")
    print("="*60)
    
    # 清理
    del sam_original, trained_model, test_loader
    torch.cuda.empty_cache()
    print("\n评估完成！")


if __name__ == '__main__':
    main()

