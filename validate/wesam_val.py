#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用 wesam 评估工具验证 mySAM 目录下的 ISIC 数据集
对比 SAM huge 和 base 模型的表现
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import argparse
import yaml
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 添加 wesam 目录到路径（如果需要导入 wesam 的工具）
wesam_root = os.path.join(os.path.dirname(project_root), 'wesam')
if os.path.exists(wesam_root):
    sys.path.insert(0, wesam_root)

from models.sam_wrapper import load_sam_model, setup_sam_path
from dataset.ISIC import ISIC2016Dataset
from training.data_utils import collate_fn_isic


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


class SAMModelAdapter(nn.Module):
    """
    将 mySAM 的 SAM 模型适配为类似 wesam Model 的接口
    """
    def __init__(self, sam_model, device, sam_input_size=1024):
        super().__init__()
        self.model = sam_model
        self.device = device
        self.sam_input_size = sam_input_size
        self.image_shape = None
        
    def encode(self, images):
        """编码图像"""
        _, _, H, W = images.shape
        self.image_shape = (H, W)
        
        # 将图像缩放到 SAM 输入尺寸
        images_sam = F.interpolate(
            images,
            size=(self.sam_input_size, self.sam_input_size),
            mode='bilinear',
            align_corners=False
        )
        
        # 提取图像特征
        image_embeddings = self.model.image_encoder(images_sam)
        return image_embeddings
    
    def decode(self, prompts, image_embeddings):
        """解码 mask"""
        pred_masks = []
        ious = []
        res_masks = []
        
        for prompt, embedding in zip(prompts, image_embeddings):
            # prompt 是 box，形状为 (4,)
            if isinstance(prompt, torch.Tensor):
                prompt = prompt.to(device=embedding.device)
            else:
                prompt = torch.tensor(prompt, device=embedding.device, dtype=torch.float32)
            
            # 缩放 box 坐标到 SAM 输入尺寸
            if self.image_shape:
                img_h, img_w = self.image_shape
                scale_x = self.sam_input_size / img_w
                scale_y = self.sam_input_size / img_h
                box_sam = torch.tensor([
                    prompt[0] * scale_x,
                    prompt[1] * scale_y,
                    prompt[2] * scale_x,
                    prompt[3] * scale_y
                ], device=embedding.device, dtype=torch.float32).unsqueeze(0)  # (1, 4)
            else:
                box_sam = prompt.unsqueeze(0) if prompt.dim() == 1 else prompt
            
            # 编码 prompt
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=box_sam,
                masks=None,
            )
            
            # 解码 mask
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            # 上采样到原始图像尺寸
            if self.image_shape:
                masks = F.interpolate(
                    low_res_masks,
                    self.image_shape,
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                masks = low_res_masks
            
            # 应用 sigmoid
            masks = torch.sigmoid(masks)
            pred_masks.append(masks.squeeze(1).squeeze(0))  # (H, W)
            ious.append(iou_predictions)
            res_masks.append(low_res_masks)
        
        return pred_masks, ious, res_masks
    
    def forward(self, images, prompts):
        """前向传播"""
        image_embeddings = self.encode(images)
        pred_masks, ious, res_masks = self.decode(prompts, image_embeddings)
        return image_embeddings, pred_masks, ious, res_masks
    
    def eval(self):
        """设置为评估模式"""
        super().eval()
        self.model.eval()
        return self
    
    def train(self, mode=True):
        """设置为训练模式"""
        super().train(mode)
        self.model.train(mode)
        return self


def get_prompts(cfg, bboxes, gt_masks):
    """
    获取 prompts（适配 wesam 的 get_prompts 函数）
    目前只支持 box prompt
    """
    if cfg.prompt == "box" or cfg.prompt == "coarse":
        prompts = bboxes
    elif cfg.prompt == "point":
        # 如果需要 point prompt，需要实现 get_point_prompts
        raise NotImplementedError("Point prompt not implemented yet")
    else:
        raise ValueError("Prompt Type Error! Only 'box' is supported")
    return prompts


def validate(model_adapter, dataloader, device, model_name="Model", prompt="box", verbose=True):
    """
    验证模型性能（适配 wesam 的 validate 函数）
    
    Args:
        model_adapter: SAMModelAdapter 实例
        dataloader: 数据加载器
        device: 设备
        model_name: 模型名称
        prompt: prompt 类型（'box' 或 'point'）
        verbose: 是否打印详细信息
    
    Returns:
        mean_iou: 平均 IoU
        mean_f1: 平均 F1
    """
    model_adapter.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()
    
    if verbose:
        print(f"评估 {model_name}...", end=' ')
    
    # 创建简单的配置对象
    class SimpleConfig:
        def __init__(self, prompt):
            self.prompt = prompt
    
    cfg = SimpleConfig(prompt)
    
    with torch.no_grad():
        for iter, batch in enumerate(dataloader):
            images = batch['image']  # (B, 3, H, W)
            big_boxes = batch['big_boxes']  # list of (4,)
            masks = batch['mask']  # (B, 1, H, W)
            
            num_images = images.size(0)
            
            # 获取 prompts
            prompts = get_prompts(cfg, big_boxes, masks)
            
            # 前向传播
            _, pred_masks, _, _ = model_adapter(images.to(device), prompts)
            
            # 计算指标（与 wesam 版本保持一致：使用 num_images 作为权重）
            for pred_mask, gt_mask in zip(pred_masks, masks):
                # 确保 pred_mask 在 CPU 上
                if isinstance(pred_mask, torch.Tensor):
                    pred_mask = pred_mask.cpu()
                else:
                    pred_mask = torch.tensor(pred_mask).cpu()
                
                # gt_mask 形状是 (1, H, W)，需要转换为 (H, W) 或 (1, H, W)
                if isinstance(gt_mask, torch.Tensor):
                    gt_mask = gt_mask.cpu()
                else:
                    gt_mask = torch.tensor(gt_mask).cpu()
                
                # 确保维度正确：pred_mask 和 gt_mask 都应该是 (H, W) 或 (1, H, W)
                if pred_mask.dim() == 2:
                    pred_mask = pred_mask.unsqueeze(0)  # (1, H, W)
                elif pred_mask.dim() == 3:
                    # 如果已经是 (1, H, W)，保持不变
                    pass
                else:
                    raise ValueError(f"Unexpected pred_mask shape: {pred_mask.shape}")
                
                # gt_mask 从 (1, H, W) 转换为 (H, W) 再转回 (1, H, W) 以确保一致性
                if gt_mask.dim() == 3:
                    gt_mask = gt_mask.squeeze(0)  # (H, W)
                if gt_mask.dim() == 2:
                    gt_mask = gt_mask.unsqueeze(0)  # (1, H, W)
                
                # 确保形状匹配
                if pred_mask.shape != gt_mask.shape:
                    # 如果高度或宽度不匹配，进行插值
                    if pred_mask.shape[1:] != gt_mask.shape[1:]:
                        pred_mask = F.interpolate(
                            pred_mask.unsqueeze(0),
                            size=gt_mask.shape[1:],
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)
                
                # 计算统计信息
                batch_stats = smp.metrics.get_stats(
                    pred_mask,
                    gt_mask.int(),
                    mode='binary',
                    threshold=0.5,
                )
                
                # 计算 IoU 和 F1（与 wesam 版本保持一致）
                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                
                # 使用 num_images 作为权重，与 wesam 版本保持一致
                ious.update(batch_iou.item(), num_images)
                f1_scores.update(batch_f1.item(), num_images)
            
            if verbose and (iter + 1) % 10 == 0:
                print(f'\n  [{iter+1}/{len(dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]', end='')
            
            torch.cuda.empty_cache()
    
    if verbose:
        print(f"\n{model_name}: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]")
    
    return ious.avg, f1_scores.avg


def load_config(config_path):
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin-1']
    config = None
    last_error = None
    
    for encoding in encodings:
        try:
            with open(config_path, 'r', encoding=encoding) as f:
                content = f.read()
                config = yaml.safe_load(content)
            if encoding != 'utf-8':
                print(f"  注意：配置文件使用 {encoding} 编码读取")
            break
        except Exception as e:
            last_error = str(e)
            continue
    
    if config is None:
        raise ValueError(f"无法读取配置文件 {config_path}: {last_error}")
    
    # 转换相对路径为绝对路径
    if os.path.isabs(config_path):
        config_dir = os.path.dirname(config_path)
    else:
        config_dir = os.path.dirname(os.path.abspath(config_path))
    
    if 'validate' in config_dir or os.path.basename(config_dir) == 'validate':
        project_root = os.path.dirname(config_dir)
    else:
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
        if 'checkpoint_paths' in config['models']:
            for key, path in config['models']['checkpoint_paths'].items():
                config['models']['checkpoint_paths'][key] = os.path.normpath(
                    os.path.join(project_root, path)
                )
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description='使用 wesam 评估工具验证 SAM huge 和 base 模型在 ISIC 数据集上的表现',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用配置文件
  python validate/wesam_val.py --config validate/config.yaml
  
  # 使用命令行参数
  python validate/wesam_val.py --data_root ./data/ISIC --test_box_csv ./data/ISIC/test_boxes.csv
        """
    )
    
    # 配置文件参数
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（YAML 格式）')
    
    # 数据集参数
    parser.add_argument('--data_root', type=str, default=None, help='ISIC 数据集根目录')
    parser.add_argument('--test_box_csv', type=str, default=None, help='测试集 box CSV 文件路径')
    
    # 模型参数
    parser.add_argument('--sam_huge_checkpoint', type=str, default=None,
                       help='SAM huge checkpoint 路径')
    parser.add_argument('--sam_base_checkpoint', type=str, default=None,
                       help='SAM base checkpoint 路径')
    
    # 评估参数
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小')
    parser.add_argument('--img_size', type=int, default=None, help='图像尺寸')
    parser.add_argument('--device', type=str, default=None, help='设备 (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=None, help='数据加载器工作进程数')
    parser.add_argument('--sam_input_size', type=int, default=1024, help='SAM 输入尺寸')
    
    args = parser.parse_args()
    
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 加载配置文件
    config = {}
    if args.config:
        config_path = args.config
        if not os.path.exists(config_path):
            # 尝试多种路径
            cwd_path = os.path.join(os.getcwd(), config_path)
            project_path = os.path.join(project_root, config_path)
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
            
            for path in [cwd_path, project_path, script_path]:
                if os.path.exists(path):
                    config_path = path
                    break
            else:
                raise FileNotFoundError(f"配置文件未找到: {args.config}")
        
        if not os.path.isabs(config_path):
            config_path = os.path.abspath(config_path)
        
        config = load_config(config_path)
        print(f"已加载配置文件: {config_path}")
    
    # 合并配置：命令行参数优先
    def get_arg(key, config_section=None, default=None):
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
    sam_huge_checkpoint = get_arg('sam_huge_checkpoint', 'models', './checkpoints/sam_vit_h_4b8939.pth')
    sam_base_checkpoint = get_arg('sam_base_checkpoint', 'models', './checkpoints/sam_vit_b_01ec64.pth')
    
    # 从配置文件获取 checkpoint 路径
    if 'models' in config and 'checkpoint_paths' in config['models']:
        checkpoint_paths = config['models']['checkpoint_paths']
        if 'vit_h' in checkpoint_paths:
            sam_huge_checkpoint = checkpoint_paths['vit_h']
        if 'vit_b' in checkpoint_paths:
            sam_base_checkpoint = checkpoint_paths['vit_b']
    
    batch_size = get_arg('batch_size', 'evaluation', 4)
    img_size = get_arg('img_size', 'evaluation', 1024)
    device_str = get_arg('device', 'evaluation', 'cuda')
    num_workers = get_arg('num_workers', 'evaluation', 4)
    sam_input_size = args.sam_input_size
    
    # 检查必需参数
    if data_root is None or test_box_csv is None:
        parser.error("必须提供 --data_root 和 --test_box_csv 参数，或通过 --config 配置文件提供")
    
    # 设置设备
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 打印配置信息
    print("\n" + "="*60)
    print("配置信息")
    print("="*60)
    print(f"数据根目录: {data_root}")
    print(f"测试集 CSV: {test_box_csv}")
    print(f"SAM Huge checkpoint: {sam_huge_checkpoint}")
    print(f"SAM Base checkpoint: {sam_base_checkpoint}")
    print(f"批次大小: {batch_size}, 图像尺寸: {img_size}, SAM输入尺寸: {sam_input_size}")
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
    
    # 加载 SAM Huge 模型
    print(f"\n加载 SAM Huge 模型...")
    sam_huge = load_sam_model(
        checkpoint_path=sam_huge_checkpoint,
        model_type='vit_h',
        device=device,
        unfreeze_last_k=0
    )
    for p in sam_huge.parameters():
        p.requires_grad = False
    sam_huge.eval()
    huge_adapter = SAMModelAdapter(sam_huge, device, sam_input_size=sam_input_size)
    print(f"  ? SAM Huge 模型加载完成")
    
    # 加载 SAM Base 模型
    print(f"\n加载 SAM Base 模型...")
    sam_base = load_sam_model(
        checkpoint_path=sam_base_checkpoint,
        model_type='vit_b',
        device=device,
        unfreeze_last_k=0
    )
    for p in sam_base.parameters():
        p.requires_grad = False
    sam_base.eval()
    base_adapter = SAMModelAdapter(sam_base, device, sam_input_size=sam_input_size)
    print(f"  ? SAM Base 模型加载完成")
    
    # 评估两个模型
    print("\n" + "="*60)
    print("开始评估")
    print("="*60)
    
    print("\n评估 SAM Huge 模型...")
    miou_huge, f1_huge = validate(huge_adapter, test_loader, device, "SAM Huge", prompt="box", verbose=True)
    
    print("\n评估 SAM Base 模型...")
    miou_base, f1_base = validate(base_adapter, test_loader, device, "SAM Base", prompt="box", verbose=True)
    
    # 打印对比结果
    print("\n" + "="*60)
    print("对比结果")
    print("="*60)
    print(f"SAM Huge:  mIoU = {miou_huge:.4f}, F1 = {f1_huge:.4f}")
    print(f"SAM Base:  mIoU = {miou_base:.4f}, F1 = {f1_base:.4f}")
    
    iou_diff = miou_huge - miou_base
    f1_diff = f1_huge - f1_base
    iou_pct = (iou_diff / miou_base * 100) if miou_base > 0 else 0
    f1_pct = (f1_diff / f1_base * 100) if f1_base > 0 else 0
    
    print(f"\n性能差异:")
    print(f"  mIoU: {iou_diff:+.4f} ({iou_pct:+.2f}%)")
    print(f"  F1:   {f1_diff:+.4f} ({f1_pct:+.2f}%)")
    print("="*60)
    
    # 清理
    del sam_huge, sam_base, huge_adapter, base_adapter, test_loader
    torch.cuda.empty_cache()
    print("\n评估完成！")


if __name__ == '__main__':
    main()

