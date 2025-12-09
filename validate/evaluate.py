#!/usr/bin/env python
# coding: utf-8

"""
模型评估脚本：基于 SAM 的像素级对比学习模型评估
- 使用 mIOU 作为主要评价指标
- 支持 box 和 point prompt
- 自动保存评估结果到 CSV 文件

用法示例：
python validate/evaluate.py \
    --checkpoint_path ./outputs/checkpoint_epoch_30.pth \
    --data_root ./data/ISIC2016 \
    --test_box_csv ./data/test_boxes.csv \
    --sam_checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
    --model_type vit_b \
    --img_size 1024 \
    --batch_size 8
"""

import os
import sys
import argparse
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# 尝试导入smp库（用于与wesam一致的评估方法）
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("[WARN] segmentation_models_pytorch not available. Install with: pip install segmentation-models-pytorch")

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dataset.ISIC import ISIC2016Dataset
from model import compute_miou, sample_points_in_ring, collate_fn_isic

# 添加 segment_anything 到 Python 路径
segment_anything_path = os.path.join(os.path.dirname(project_root), 'segment-anything')
if os.path.exists(segment_anything_path) and segment_anything_path not in sys.path:
    sys.path.insert(0, segment_anything_path)

try:
    from segment_anything import sam_model_registry
except ImportError as e:
    alternative_paths = [
        os.path.join(os.path.dirname(project_root), 'segment-anything'),
        '/root/workspace/segment-anything',
        './segment-anything',
        '../segment-anything',
    ]
    for alt_path in alternative_paths:
        if os.path.exists(alt_path) and alt_path not in sys.path:
            sys.path.insert(0, alt_path)
            try:
                from segment_anything import sam_model_registry
                break
            except ImportError:
                continue
    else:
        raise ImportError(f"无法找到 segment_anything 模块。错误: {e}")


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


def load_checkpoint(checkpoint_path, model, device):
    """加载 checkpoint"""
    print(f"[INFO] Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载 image_encoder 权重
    if 'sam_image_encoder' in checkpoint:
        model.image_encoder.load_state_dict(checkpoint['sam_image_encoder'])
        print("[INFO] Loaded sam_image_encoder weights")
    else:
        print("[WARN] No sam_image_encoder found in checkpoint")
    
    # 注意：评估时不需要 proj，因为只评估 mask 预测
    # 但如果 checkpoint 中有 proj，也可以加载（虽然不会用到）
    
    epoch = checkpoint.get('epoch', 0)
    print(f"[INFO] Checkpoint epoch: {epoch}")
    return epoch


def compute_miou_smp(pred_logits, gt_masks, threshold=0.5):
    """
    使用smp库计算mIOU（与wesam一致的方法）
    使用micro-imagewise reduction
    """
    if not SMP_AVAILABLE:
        raise ImportError("segmentation_models_pytorch is required for this method")
    
    pred_bin = (torch.sigmoid(pred_logits) > threshold).float()
    gt_bin = (gt_masks > threshold).float()
    
    # 对每个样本计算（与wesam一致）
    batch_ious = []
    for pred, gt in zip(pred_bin, gt_bin):
        # pred和gt需要是(H, W)格式
        pred_2d = pred.squeeze(0)  # (H, W)
        gt_2d = gt.squeeze(0).int()  # (H, W)
        
        batch_stats = smp.metrics.get_stats(
            pred_2d.unsqueeze(0),  # (1, H, W)
            gt_2d.unsqueeze(0),    # (1, H, W)
            mode='binary',
            threshold=0.5,
        )
        iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
        batch_ious.append(iou)
    
    return torch.tensor(batch_ious).mean().item()


def evaluate_with_box_prompt(model, dataloader, device, sam_input_size=1024, img_size=1024, use_smp=False):
    """使用 box prompt 进行评估"""
    model.eval()
    miou_meter = AverageMeter()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = batch['image'].to(device)  # (B, 3, H, W)
            boxes_list = batch['boxes']  # list of boxes
            gt_masks = batch['mask'].to(device)  # (B, 1, H, W)
            
            B = images.size(0)
            
            # SAM 的 image_encoder 期望输入是 1024x1024
            if images.shape[-1] != sam_input_size or images.shape[-2] != sam_input_size:
                images_sam = F.interpolate(images, size=(sam_input_size, sam_input_size), 
                                         mode='bilinear', align_corners=False)
                scale_factor = sam_input_size / img_size
                boxes_list_sam = []
                for box in boxes_list:
                    box_sam = [coord * scale_factor for coord in box]
                    boxes_list_sam.append(box_sam)
            else:
                images_sam = images
                boxes_list_sam = boxes_list
            
            # 获取图像特征
            image_embeddings = model.image_encoder(images_sam)
            
            # 对每个样本进行预测
            pred_masks = []
            for b in range(B):
                box = boxes_list_sam[b]
                box_tensor = torch.tensor([box], device=device, dtype=torch.float32)  # (1, 4)
                
                # 编码 prompt
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=None,
                    boxes=box_tensor,
                    masks=None,
                )
                
                # 解码生成 mask
                low_res_masks, iou_predictions = model.mask_decoder(
                    image_embeddings=image_embeddings[b:b+1],
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                
                # 上采样到原始图像尺寸
                masks_upsampled = F.interpolate(
                    low_res_masks,
                    size=(img_size, img_size),
                    mode='bilinear',
                    align_corners=False
                )
                pred_masks.append(masks_upsampled.squeeze(1))
            
            pred_logits = torch.stack(pred_masks, dim=0).unsqueeze(1)  # (B, 1, H, W)
            
            # 计算 mIOU
            if use_smp and SMP_AVAILABLE:
                batch_miou = compute_miou_smp(pred_logits, gt_masks)
            else:
                batch_miou = compute_miou(pred_logits, gt_masks)
            miou_meter.update(batch_miou, B)
            
            torch.cuda.empty_cache()
    
    return miou_meter.avg


def evaluate_with_point_prompt(model, dataloader, device, sam_input_size=1024, img_size=1024, num_points=10, use_smp=False):
    """使用 point prompt 进行评估"""
    model.eval()
    miou_meter = AverageMeter()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = batch['image'].to(device)  # (B, 3, H, W)
            boxes_list = batch['boxes']  # list of big_boxes
            small_boxes_list = batch.get('small_boxes', None)  # list of small_boxes
            gt_masks = batch['mask'].to(device)  # (B, 1, H, W)
            
            B = images.size(0)
            
            # SAM 的 image_encoder 期望输入是 1024x1024
            if images.shape[-1] != sam_input_size or images.shape[-2] != sam_input_size:
                images_sam = F.interpolate(images, size=(sam_input_size, sam_input_size), 
                                         mode='bilinear', align_corners=False)
                scale_factor = sam_input_size / img_size
                boxes_list_sam = []
                small_boxes_list_sam = []
                for big_box, small_box in zip(boxes_list, small_boxes_list if small_boxes_list else [None] * B):
                    big_box_sam = [coord * scale_factor for coord in big_box]
                    boxes_list_sam.append(big_box_sam)
                    if small_box is not None:
                        small_box_sam = [coord * scale_factor for coord in small_box]
                        small_boxes_list_sam.append(small_box_sam)
                    else:
                        small_boxes_list_sam.append(None)
            else:
                images_sam = images
                boxes_list_sam = boxes_list
                small_boxes_list_sam = small_boxes_list if small_boxes_list else [None] * B
            
            # 获取图像特征
            image_embeddings = model.image_encoder(images_sam)
            
            # 对每个样本进行预测
            pred_masks = []
            for b in range(B):
                big_box = boxes_list_sam[b]
                small_box = small_boxes_list_sam[b]
                
                # 在大小框中间区域采样离散点作为 prompt
                if small_box is not None:
                    points, point_labels = sample_points_in_ring(
                        small_box, big_box, 
                        num_points=num_points,
                        img_size=sam_input_size
                    )
                    points = points.to(device)  # (N, 2)
                    point_labels = point_labels.to(device)  # (N,)
                    points_tensor = points.unsqueeze(0)  # (1, N, 2)
                    labels_tensor = point_labels.unsqueeze(0)  # (1, N)
                else:
                    # 如果没有 small_box，使用 big_box 的中心点
                    cx = (big_box[0] + big_box[2]) / 2
                    cy = (big_box[1] + big_box[3]) / 2
                    points_tensor = torch.tensor([[[cx, cy]]], device=device, dtype=torch.float32)
                    labels_tensor = torch.tensor([[1]], device=device, dtype=torch.long)
                
                # 编码 prompt
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=(points_tensor, labels_tensor),
                    boxes=None,
                    masks=None,
                )
                
                # 解码生成 mask
                low_res_masks, iou_predictions = model.mask_decoder(
                    image_embeddings=image_embeddings[b:b+1],
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                
                # 上采样到原始图像尺寸
                masks_upsampled = F.interpolate(
                    low_res_masks,
                    size=(img_size, img_size),
                    mode='bilinear',
                    align_corners=False
                )
                pred_masks.append(masks_upsampled.squeeze(1))
            
            pred_logits = torch.stack(pred_masks, dim=0).unsqueeze(1)  # (B, 1, H, W)
            
            # 计算 mIOU
            if use_smp and SMP_AVAILABLE:
                batch_miou = compute_miou_smp(pred_logits, gt_masks)
            else:
                batch_miou = compute_miou(pred_logits, gt_masks)
            miou_meter.update(batch_miou, B)
            
            torch.cuda.empty_cache()
    
    return miou_meter.avg


def save_results(output_dir, results_dict):
    """保存评估结果到 CSV 文件"""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'evaluation_results.csv')
    
    # 检查文件是否存在，决定是否写入表头
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        fieldnames = ['timestamp', 'model_type', 'checkpoint_path', 'prompt_type', 'dataset', 
                     'mIOU', 'num_samples', 'img_size', 'batch_size', 'epoch', 
                     'improvement', 'improvement_pct']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(results_dict)
    
    print(f"[INFO] Results saved to {csv_path}")


def build_argparser():
    parser = argparse.ArgumentParser(description='Evaluate SAM-based model')
    
    # Checkpoint 相关
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='训练好的 checkpoint 路径')
    parser.add_argument('--sam_checkpoint', type=str, required=True,
                       help='SAM 预训练模型 checkpoint 路径')
    parser.add_argument('--model_type', type=str, default='vit_b',
                       choices=['vit_b', 'vit_l', 'vit_h'],
                       help='SAM 模型类型')
    
    # 数据相关
    parser.add_argument('--data_root', type=str, required=True,
                       help='数据集根目录')
    parser.add_argument('--test_box_csv', type=str, required=True,
                       help='测试集 box CSV 文件路径')
    parser.add_argument('--img_size', type=int, default=1024,
                       help='图像尺寸')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='评估时的 batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载线程数')
    
    # Prompt 相关
    parser.add_argument('--prompt_type', type=str, default='box',
                       choices=['box', 'point'],
                       help='Prompt 类型：box 或 point')
    parser.add_argument('--num_points', type=int, default=10,
                       help='使用 point prompt 时的采样点数')
    
    # 输出相关
    parser.add_argument('--output_dir', type=str, default='./validate/results',
                       help='结果保存目录')
    parser.add_argument('--device', type=str, default='cuda',
                       help='使用的设备')
    parser.add_argument('--use_smp', action='store_true',
                       help='使用smp库计算mIOU（与wesam一致的方法）')
    parser.add_argument('--compare_methods', action='store_true',
                       help='同时使用两种方法计算mIOU并对比')
    
    return parser


def main():
    args = build_argparser().parse_args()
    
    # 修复路径：如果路径是相对路径，转换为相对于项目根目录的路径
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent  # validate 的父目录是项目根目录
    
    # 处理相对路径
    if not Path(args.checkpoint_path).is_absolute():
        args.checkpoint_path = str(project_root / args.checkpoint_path)
    if not Path(args.sam_checkpoint).is_absolute():
        args.sam_checkpoint = str(project_root / args.sam_checkpoint)
    if not Path(args.data_root).is_absolute():
        args.data_root = str(project_root / args.data_root)
    if not Path(args.test_box_csv).is_absolute():
        args.test_box_csv = str(project_root / args.test_box_csv)
    if args.output_dir and not Path(args.output_dir).is_absolute():
        args.output_dir = str(project_root / args.output_dir)
    
    # 设备设置
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    # 加载测试数据集
    print(f"[INFO] Loading test dataset from {args.data_root}")
    test_dataset = ISIC2016Dataset(
        root=args.data_root,
        box_csv=args.test_box_csv,
        img_size=args.img_size,
        split='test'
    )
    print(f"[INFO] Test dataset size: {len(test_dataset)}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_isic
    )
    
    results_comparison = {}
    
    # ========== 1. 评估 SAM 预训练模型 ==========
    print("\n" + "="*70)
    print("Step 1: Evaluating SAM Pretrained Model (Baseline)")
    print("="*70)
    
    sam_pretrained = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam_pretrained.to(device)
    sam_pretrained.eval()
    
    print(f"[INFO] Using SAM pretrained encoder (baseline)")
    if args.use_smp:
        print(f"[INFO] Using smp library method (micro-imagewise, same as wesam)")
    elif args.compare_methods:
        print(f"[INFO] Comparing both methods (macro and micro)")
    
    # 评估预训练模型
    if args.compare_methods and SMP_AVAILABLE:
        # 同时使用两种方法
        if args.prompt_type == 'box':
            miou_pretrained_macro = evaluate_with_box_prompt(
                sam_pretrained, test_loader, device,
                sam_input_size=1024,
                img_size=args.img_size,
                use_smp=False
            )
            miou_pretrained_micro = evaluate_with_box_prompt(
                sam_pretrained, test_loader, device,
                sam_input_size=1024,
                img_size=args.img_size,
                use_smp=True
            )
        else:  # point
            miou_pretrained_macro = evaluate_with_point_prompt(
                sam_pretrained, test_loader, device,
                sam_input_size=1024,
                img_size=args.img_size,
                num_points=args.num_points,
                use_smp=False
            )
            miou_pretrained_micro = evaluate_with_point_prompt(
                sam_pretrained, test_loader, device,
                sam_input_size=1024,
                img_size=args.img_size,
                num_points=args.num_points,
                use_smp=True
            )
        miou_pretrained = miou_pretrained_micro if args.use_smp else miou_pretrained_macro
        results_comparison['pretrained_macro'] = miou_pretrained_macro
        results_comparison['pretrained_micro'] = miou_pretrained_micro
        print(f"\n[RESULT] SAM Pretrained mIOU:")
        print(f"  - Macro (mySAM method): {miou_pretrained_macro:.4f}")
        print(f"  - Micro (wesam method): {miou_pretrained_micro:.4f}")
    else:
        if args.prompt_type == 'box':
            miou_pretrained = evaluate_with_box_prompt(
                sam_pretrained, test_loader, device,
                sam_input_size=1024,
                img_size=args.img_size,
                use_smp=args.use_smp
            )
        else:  # point
            miou_pretrained = evaluate_with_point_prompt(
                sam_pretrained, test_loader, device,
                sam_input_size=1024,
                img_size=args.img_size,
                num_points=args.num_points,
                use_smp=args.use_smp
            )
        method_name = "Micro (wesam)" if args.use_smp else "Macro (mySAM)"
        print(f"\n[RESULT] SAM Pretrained mIOU ({method_name}): {miou_pretrained:.4f}")
    
    results_comparison['pretrained'] = {
        'miou': miou_pretrained,
        'encoder': 'SAM Pretrained',
        'checkpoint': args.sam_checkpoint
    }
    
    # 清理GPU内存
    del sam_pretrained
    torch.cuda.empty_cache()
    
    # ========== 2. 评估自己训练的模型 ==========
    print("\n" + "="*70)
    print("Step 2: Evaluating Your Trained Model")
    print("="*70)
    
    sam_trained = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam_trained.to(device)
    sam_trained.eval()
    
    # 加载训练的 checkpoint
    epoch = load_checkpoint(args.checkpoint_path, sam_trained, device)
    print(f"[INFO] Using trained encoder from epoch {epoch}")
    
    # 评估训练模型
    if args.compare_methods and SMP_AVAILABLE:
        # 同时使用两种方法
        if args.prompt_type == 'box':
            miou_trained_macro = evaluate_with_box_prompt(
                sam_trained, test_loader, device,
                sam_input_size=1024,
                img_size=args.img_size,
                use_smp=False
            )
            miou_trained_micro = evaluate_with_box_prompt(
                sam_trained, test_loader, device,
                sam_input_size=1024,
                img_size=args.img_size,
                use_smp=True
            )
        else:  # point
            miou_trained_macro = evaluate_with_point_prompt(
                sam_trained, test_loader, device,
                sam_input_size=1024,
                img_size=args.img_size,
                num_points=args.num_points,
                use_smp=False
            )
            miou_trained_micro = evaluate_with_point_prompt(
                sam_trained, test_loader, device,
                sam_input_size=1024,
                img_size=args.img_size,
                num_points=args.num_points,
                use_smp=True
            )
        miou_trained = miou_trained_micro if args.use_smp else miou_trained_macro
        results_comparison['trained_macro'] = miou_trained_macro
        results_comparison['trained_micro'] = miou_trained_micro
        print(f"\n[RESULT] Your Trained mIOU:")
        print(f"  - Macro (mySAM method): {miou_trained_macro:.4f}")
        print(f"  - Micro (wesam method): {miou_trained_micro:.4f}")
    else:
        if args.prompt_type == 'box':
            miou_trained = evaluate_with_box_prompt(
                sam_trained, test_loader, device,
                sam_input_size=1024,
                img_size=args.img_size,
                use_smp=args.use_smp
            )
        else:  # point
            miou_trained = evaluate_with_point_prompt(
                sam_trained, test_loader, device,
                sam_input_size=1024,
                img_size=args.img_size,
                num_points=args.num_points,
                use_smp=args.use_smp
            )
        method_name = "Micro (wesam)" if args.use_smp else "Macro (mySAM)"
        print(f"\n[RESULT] Your Trained mIOU ({method_name}): {miou_trained:.4f}")
    
    results_comparison['trained'] = {
        'miou': miou_trained,
        'encoder': 'Your Trained',
        'checkpoint': args.checkpoint_path,
        'epoch': epoch
    }
    
    # 清理GPU内存
    del sam_trained
    torch.cuda.empty_cache()
    
    # ========== 3. 对比结果 ==========
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"Dataset: {args.data_root}")
    print(f"Test Samples: {len(test_dataset)}")
    print(f"Prompt Type: {args.prompt_type}")
    print(f"Image Size: {args.img_size}")
    print("-"*70)
    print(f"{'Model':<30} {'mIOU':<15} {'Improvement':<15}")
    print("-"*70)
    print(f"{'SAM Pretrained (Baseline)':<30} {miou_pretrained:<15.4f} {'-':<15}")
    
    improvement = miou_trained - miou_pretrained
    improvement_pct = (improvement / miou_pretrained * 100) if miou_pretrained > 0 else 0
    
    if improvement > 0:
        improvement_str = f"+{improvement:.4f} (+{improvement_pct:.2f}%)"
    else:
        improvement_str = f"{improvement:.4f} ({improvement_pct:.2f}%)"
    
    print(f"{'Your Trained Model':<30} {miou_trained:<15.4f} {improvement_str:<15}")
    print("="*70)
    
    # 如果对比了两种方法，显示详细对比
    if args.compare_methods and SMP_AVAILABLE:
        print("\n" + "="*70)
        print("METHOD COMPARISON (Macro vs Micro)")
        print("="*70)
        print(f"{'Model':<30} {'Macro mIOU':<15} {'Micro mIOU':<15} {'Difference':<15}")
        print("-"*70)
        
        # 预训练模型对比
        if 'pretrained_macro' in results_comparison:
            pretrained_macro = results_comparison['pretrained_macro']
            pretrained_micro = results_comparison['pretrained_micro']
            diff_pretrained = pretrained_micro - pretrained_macro
            print(f"{'SAM Pretrained':<30} {pretrained_macro:<15.4f} {pretrained_micro:<15.4f} {diff_pretrained:+.4f}")
        
        # 训练模型对比
        if 'trained_macro' in results_comparison:
            trained_macro = results_comparison['trained_macro']
            trained_micro = results_comparison['trained_micro']
            diff_trained = trained_micro - trained_macro
            print(f"{'Your Trained':<30} {trained_macro:<15.4f} {trained_micro:<15.4f} {diff_trained:+.4f}")
        
        print("="*70)
        print("\nNote:")
        print("  - Macro: Sample-level average (mySAM method) - treats all samples equally")
        print("  - Micro: Pixel-level aggregation (wesam method) - favors large objects")
        print("  - If micro > macro: dataset has many large objects")
        print("  - If macro > micro: dataset has many small objects")
        print("="*70)
    
    # 详细对比
    print("\nDetailed Comparison:")
    print("-"*70)
    print(f"Baseline (SAM Pretrained):")
    print(f"  - mIOU: {miou_pretrained:.4f}")
    print(f"  - Encoder: SAM Pretrained ViT")
    print(f"  - Checkpoint: {args.sam_checkpoint}")
    print(f"\nYour Trained Model:")
    print(f"  - mIOU: {miou_trained:.4f}")
    print(f"  - Encoder: Fine-tuned ViT (Epoch {epoch})")
    print(f"  - Checkpoint: {args.checkpoint_path}")
    print(f"  - Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)")
    
    if improvement > 0:
        print(f"\n✓ Your trained model performs BETTER than baseline!")
        print(f"  Improvement: {improvement:.4f} ({improvement_pct:.2f}% relative improvement)")
    elif improvement < 0:
        print(f"\n⚠ Your trained model performs WORSE than baseline.")
        print(f"  Degradation: {abs(improvement):.4f} ({abs(improvement_pct):.2f}% relative decrease)")
    else:
        print(f"\n= Your trained model performs SIMILAR to baseline.")
    
    print("="*70)
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 保存预训练模型结果
    results_pretrained = {
        'timestamp': timestamp,
        'model_type': 'SAM Pretrained',
        'checkpoint_path': args.sam_checkpoint,
        'prompt_type': args.prompt_type,
        'dataset': args.data_root,
        'mIOU': f'{miou_pretrained:.4f}',
        'num_samples': len(test_dataset),
        'img_size': args.img_size,
        'batch_size': args.batch_size,
        'epoch': 'N/A'
    }
    save_results(args.output_dir, results_pretrained)
    
    # 保存训练模型结果
    results_trained = {
        'timestamp': timestamp,
        'model_type': 'Your Trained',
        'checkpoint_path': args.checkpoint_path,
        'prompt_type': args.prompt_type,
        'dataset': args.data_root,
        'mIOU': f'{miou_trained:.4f}',
        'num_samples': len(test_dataset),
        'img_size': args.img_size,
        'batch_size': args.batch_size,
        'epoch': epoch,
        'improvement': f'{improvement:+.4f}',
        'improvement_pct': f'{improvement_pct:+.2f}%'
    }
    save_results(args.output_dir, results_trained)
    
    print("\n[INFO] Evaluation completed! Results saved to CSV file.")


if __name__ == '__main__':
    main()

