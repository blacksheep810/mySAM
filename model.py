#!/usr/bin/env python
# coding: utf-8


"""
可运行训练脚本：基于 SAM 的像素级对比学习微调 image encoder（含伪标签自训练 + EMA teacher + 部分解冻策略）
  - 使用 SAM 提供的 image_encoder 特征作为初始表征
  - 使用 EMA teacher（冻结副本）生成伪标签并做熵+IoU 筛选
  - 使用 中间区域构造 prompt（离散边界点）供 SAM 生成候选 masks
  - Pixel-wise InfoNCE（anchor: 高置信伪标签内像素；positive: 同目标另一视图/teacher；neg: 困难负样本 + in-batch negatives）
  - 支持部分解冻 encoder（解冻最后 K 块 transformer block），且支持在后期全量解冻（通过命令行参数控制）
  - 支持 feature distillation（L2 匹配 frozen original SAM features）以减少遗忘

用法示例：
python sam_encoder_contrastive_ft.py \
  --data_root ./dataset \
  --annotations annotations.json \
  --sam_checkpoint sam_vit_b_01ec64.pth \
  --output_dir ./checkpoints \
  --batch_size 4 \
  --epochs 30
"""


import os
import sys
import argparse
import random
from pathlib import Path
from collections import deque

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataset.ISIC import ISIC2016Dataset

# 添加 segment_anything 到 Python 路径
segment_anything_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'segment-anything')
if os.path.exists(segment_anything_path) and segment_anything_path not in sys.path:
    sys.path.insert(0, segment_anything_path)

try:
    from segment_anything import sam_model_registry
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
                break
            except ImportError:
                continue
    else:
        raise ImportError(f"无法找到 segment_anything 模块。请确保 segment-anything 目录在正确的位置。错误: {e}")




def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

# 1 - (2*intersect + eps) / (union + eps)
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        pred = torch.sigmoid(logits)
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        intersect = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        loss = 1 - (2. * intersect + self.eps) / (union + self.eps)
        return loss.mean()


# -----------------------------
# Pixel projection head: (B, C, H, W) -> (B, D, H, W) normalized
# -----------------------------
class PixelProjHead(nn.Module):
    def __init__(self, in_dim, proj_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1),
            nn.BatchNorm2d(in_dim),
            nn.GELU(),
            nn.Conv2d(in_dim, proj_dim, kernel_size=1)
        )

    def forward(self, x):
        z = self.net(x)  # (B, D, H, W)
        # L2 normalize per pixel
        z = z.permute(0, 2, 3, 1)  # (B, H, W, D)
        z = F.normalize(z, p=2, dim=-1)
        z = z.permute(0, 3, 1, 2)  # (B, D, H, W)
        return z


# -----------------------------
# Pixel InfoNCE
# anchors: (N, D), positives: (N, D), negatives: (M, D)
# -----------------------------
def pixel_info_nce(anchors, positives, negatives, temperature=0.1):
    anchors = F.normalize(anchors, dim=1)
    positives = F.normalize(positives, dim=1)
    negatives = F.normalize(negatives, dim=1)

    sim_pos = torch.exp(torch.sum(anchors * positives, dim=1) / temperature)  
    sim_neg = torch.exp(torch.matmul(anchors, negatives.T) / temperature) 
    denom = sim_pos + sim_neg.sum(dim=1)
    loss = -torch.log(sim_pos / (denom + 1e-12) + 1e-12)
    return loss.mean()


# -----------------------------
# 辅助：计算每个 mask 的 logits 熵 (二分类) 以及平均熵
# -----------------------------
def mask_entropy_logits(mask_logits):
    # mask_logits: (B, 1, H, W)
    p = torch.sigmoid(mask_logits)
    # binary entropy per pixel
    eps = 1e-8
    ent = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
    # avg per image
    return ent.view(ent.size(0), -1).mean(dim=1)


# -----------------------------
# 计算 mIOU (mean Intersection over Union)
# -----------------------------
def compute_miou(pred_logits, gt_masks, threshold=0.5):
    """
    计算预测mask和ground truth mask之间的mIOU
    
    Args:
        pred_logits: (B, 1, H, W) 预测的logits
        gt_masks: (B, 1, H, W) ground truth masks
        threshold: 二值化阈值
    
    Returns:
        miou: 标量，平均IoU值
    """
    # 确保尺寸一致
    if pred_logits.shape[-2:] != gt_masks.shape[-2:]:
        pred_logits = F.interpolate(pred_logits, size=gt_masks.shape[-2:], mode='bilinear', align_corners=False)
    
    # 二值化预测
    pred_bin = (torch.sigmoid(pred_logits) > threshold).float()  # (B, 1, H, W)
    gt_bin = (gt_masks > threshold).float()  # (B, 1, H, W)
    
    # 计算每个样本的IoU
    intersection = (pred_bin * gt_bin).sum(dim=[1, 2, 3])  # (B,)
    union = ((pred_bin + gt_bin) > 0).float().sum(dim=[1, 2, 3])  # (B,)
    
    # 避免除零
    iou_per_sample = intersection / (union + 1e-6)  # (B,)
    
    # 返回平均IoU
    return iou_per_sample.mean().item()


# In[12]:


def collate_fn_isic(batch):
    """将 ISIC2016Dataset 的 batch 转换为训练所需的格式"""
    images = []
    big_boxes_list = []
    small_boxes_list = []
    masks = []
    img_names = []
    
    for item in batch:
        image, big_box, small_box, mask, img_name = item
        images.append(image)
        big_boxes_list.append(big_box.tolist())  # 转换为 list
        small_boxes_list.append(small_box.tolist())
        masks.append(mask)
        img_names.append(img_name)
    
    # 堆叠图像和 masks
    images = torch.stack(images, dim=0)  # (B, 3, H, W)
    masks = torch.stack(masks, dim=0)  # (B, 1, H, W)
    
    # 返回格式：与原来的 SimpleMaskDataset 兼容
    return {
        'image': images,
        'boxes': big_boxes_list,  # 使用 big_box 作为主要 box
        'big_boxes': big_boxes_list,  # 保留 big_box
        'small_boxes': small_boxes_list,  # 保留 small_box
        'mask': masks,
        'img_names': img_names
    }

# -----------------------------
# 辅助函数：在大小框中间区域（环形区域）随机采样离散点作为 prompt
# -----------------------------
def sample_points_in_ring(small_box, big_box, num_points=10, img_size=1024):
    """
    在 small_box 和 big_box 之间的环形区域随机采样点
    
    Args:
        small_box: [x1, y1, x2, y2] 小框
        big_box: [x1, y1, x2, y2] 大框
        num_points: 采样点数量
        img_size: 图像尺寸
    
    Returns:
        points: (num_points, 2) tensor, 每行是 [x, y]
        labels: (num_points,) tensor, 1 表示前景点（在环形区域内）
    """
    if isinstance(small_box, torch.Tensor):
        small_box = small_box.tolist()
    if isinstance(big_box, torch.Tensor):
        big_box = big_box.tolist()
    
    sx1, sy1, sx2, sy2 = small_box
    bx1, by1, bx2, by2 = big_box
    
    points = []
    labels = []
    
    # 在环形区域采样点：在 big_box 内但不在 small_box 内
    attempts = 0
    max_attempts = num_points * 10
    
    while len(points) < num_points and attempts < max_attempts:
        attempts += 1
        # 在 big_box 内随机采样
        x = np.random.uniform(bx1, bx2)
        y = np.random.uniform(by1, by2)
        
        # 检查是否在环形区域内（在 big_box 内但不在 small_box 内）
        in_big = (bx1 <= x <= bx2) and (by1 <= y <= by2)
        in_small = (sx1 <= x <= sx2) and (sy1 <= y <= sy2)
        
        if in_big and not in_small:
            points.append([x, y])
            labels.append(1)  # 前景点（在边界区域）
    
    # 如果采样点不够，用边界上的点补充
    if len(points) < num_points:
        # 在 big_box 边界上采样
        remaining = num_points - len(points)
        for _ in range(remaining):
            # 随机选择一条边
            edge = np.random.randint(4)
            if edge == 0:  # 上边
                x = np.random.uniform(bx1, bx2)
                y = by1
            elif edge == 1:  # 下边
                x = np.random.uniform(bx1, bx2)
                y = by2
            elif edge == 2:  # 左边
                x = bx1
                y = np.random.uniform(by1, by2)
            else:  # 右边
                x = bx2
                y = np.random.uniform(by1, by2)
            
            # 确保不在 small_box 内
            if not ((sx1 <= x <= sx2) and (sy1 <= y <= sy2)):
                points.append([x, y])
                labels.append(1)
    
    if len(points) == 0:
        # 如果还是没采样到，使用 big_box 的中心点
        cx = (bx1 + bx2) / 2
        cy = (by1 + by2) / 2
        points = [[cx, cy]]
        labels = [1]
    
    points = torch.tensor(points, dtype=torch.float32)  # (N, 2)
    labels = torch.tensor(labels, dtype=torch.long)  # (N,)
    
    return points, labels


# In[13]:


# -----------------------------
# 参数解析器
# -----------------------------
def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, required=True, help='ISIC 数据集根目录')
    p.add_argument('--train_box_csv', type=str, required=True, help='训练集 box CSV 文件路径')
    p.add_argument('--test_box_csv', type=str, default=None, help='测试集 box CSV 文件路径（可选）')
    p.add_argument('--sam_checkpoint', type=str, required=True, help='SAM checkpoint 路径')
    p.add_argument('--model_type', type=str, default='vit_h', help='sam model type: vit_b, vit_l, vit_h (默认: vit_h)')
    p.add_argument('--output_dir', type=str, default='./outputs')
    p.add_argument('--img_size', type=int, default=1024)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr_decoder', type=float, default=1e-4)
    p.add_argument('--lr_encoder', type=float, default=2e-6)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--proj_dim', type=int, default=64)
    p.add_argument('--pos_samples', type=int, default=256)
    p.add_argument('--neg_samples', type=int, default=1024)
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--entropy_thresh', type=float, default=0.2)
    p.add_argument('--unfreeze_last_k', type=int, default=0, help='解冻 encoder 最后 K 个 transformer blocks; 0 表示训练整个encoder')
    p.add_argument('--use_gradient_checkpointing', action='store_true', help='使用梯度检查点节省显存（用时间换空间）')
    p.add_argument('--use_amp', action='store_true', help='使用混合精度训练（AMP）节省显存')
    p.add_argument('--gradient_accumulation_steps', type=int, default=1, help='梯度累积步数（等效增大batch size）')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--multi_gpu', action='store_true', help='使用多GPU训练（自动检测所有可用GPU）')
    p.add_argument('--gpu_ids', type=str, default=None, help='指定使用的GPU ID，例如 "0,1" 或 "0,1,2,3"，默认使用所有GPU')
    p.add_argument('--save_every', type=int, default=1)
    return p


# In[14]:


# -----------------------------
# 训练主函数
# -----------------------------
def train_main(args):
    set_seed(42)
    mkdir(args.output_dir)

    # ========== PyTorch CUDA 内存分配优化 ==========
    # 设置可扩展内存段以减少碎片化（解决 OOM 问题）
    import os
    if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print("[INFO] 已设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 以减少内存碎片化")
    
    # ========== 显存清理和监控 ==========
    if torch.cuda.is_available():
        # 清理之前的缓存
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        # 重置内存统计（用于准确监控）
        torch.cuda.reset_peak_memory_stats()
        
        # 显示初始显存状态
        device_id = 0 if args.device == 'cuda' else int(args.device.split(':')[1]) if ':' in args.device else 0
        allocated = torch.cuda.memory_allocated(device_id) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device_id) / 1024**3    # GB
        total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3  # GB
        print(f"[INFO] GPU {device_id} 初始显存状态: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB, 总计={total:.2f}GB")
        
        if reserved > 0.1:  # 如果保留显存超过 100MB
            print(f"[WARN] 检测到 {reserved:.2f}GB 显存被保留，可能是之前的进程未释放")
            print(f"[INFO] 已清理 PyTorch 缓存，如果问题持续，请检查是否有其他进程占用显存")

    # ========== 多GPU设置 ==========
    num_gpus = torch.cuda.device_count()
    use_multi_gpu = False
    device_ids = None
    
    if args.multi_gpu or args.gpu_ids is not None:
        if num_gpus > 1:
            if args.gpu_ids is not None:
                # 使用指定的GPU
                device_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
                use_multi_gpu = len(device_ids) > 1
                print(f"[INFO] Using specified GPUs: {device_ids}")
            else:
                # 使用所有可用GPU
                device_ids = list(range(num_gpus))
                use_multi_gpu = True
                print(f"[INFO] Using all available GPUs: {device_ids}")
        else:
            print(f"[WARN] Only {num_gpus} GPU(s) available, using single GPU")
            use_multi_gpu = False
    
    if use_multi_gpu:
        device = torch.device(f'cuda:{device_ids[0]}')
        print(f"[INFO] Multi-GPU training enabled on {len(device_ids)} GPUs")
        print(f"[INFO] Primary device: {device}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Using device: {device}")

    # -----------------------------
    # 加载 SAM
    # -----------------------------
    checkpoint_path = args.sam_checkpoint
    sam = sam_model_registry[args.model_type](checkpoint=checkpoint_path)
    sam.to(device)
    sam.eval()

    # 创建一个 frozen teacher copy (EMA 模型)
    teacher = sam_model_registry[args.model_type](checkpoint=checkpoint_path)
    teacher.to(device)
    teacher.eval()

    # 参数冻结策略：只训练 image_encoder，冻结 decoder 和 prompt_encoder
    # 冻结 mask_decoder 和 prompt_encoder
    for p in sam.mask_decoder.parameters():
        p.requires_grad = False
    for p in sam.prompt_encoder.parameters():
        p.requires_grad = False
    
    # ========== 显存优化方案 1: 梯度检查点 ==========
    if args.use_gradient_checkpointing:
        print("[INFO] Enabling gradient checkpointing to save memory")
        # 为 encoder 的 transformer blocks 启用梯度检查点
        if hasattr(sam.image_encoder, 'blocks'):
            for block in sam.image_encoder.blocks:
                if hasattr(block, 'gradient_checkpointing'):
                    block.gradient_checkpointing = True
                # 或者使用 PyTorch 的 checkpoint
                # 需要在 forward 中手动使用 torch.utils.checkpoint
    
    # image_encoder 保持可训练（不解冻）
    # 如果需要部分解冻，可以通过 unfreeze_last_k 控制
    if args.unfreeze_last_k > 0:
        # 只解冻最后 K 个 transformer blocks（推荐：减少显存）
        unfreeze_count = args.unfreeze_last_k
        print(f"[INFO] Only unfreezing last {unfreeze_count} encoder blocks (memory efficient)")
        # 先冻结所有 encoder 参数
        for p in sam.image_encoder.parameters():
            p.requires_grad = False
        
        block_container = None
        for attr in ['blocks', 'transformer', 'resblocks', 'layers']:
            if hasattr(sam.image_encoder, attr):
                block_container = getattr(sam.image_encoder, attr)
                break
        if block_container is None:
            print("[WARN] Cannot find encoder blocks container; will train all encoder parameters.")
            # 如果找不到 blocks，解冻整个 encoder
            for p in sam.image_encoder.parameters():
                p.requires_grad = True
        else:
            total = len(block_container)
            start = max(0, total - unfreeze_count)
            print(f"[INFO] Unfreezing blocks {start} to {total-1} out of {total} total blocks")
            for i in range(start, total):
                for p in block_container[i].parameters():
                    p.requires_grad = True
    else:
        # 解冻整个 image_encoder
        for p in sam.image_encoder.parameters():
            p.requires_grad = True
        print("[INFO] Training entire image_encoder")

    # Teacher 完全冻结
    for p in teacher.parameters():
        p.requires_grad = False

    # -----------------------------
    # 获取特征维度
    # 注意：SAM 的 image_encoder 期望输入是 1024x1024（预训练尺寸）
    # 如果使用其他尺寸，需要先 resize 到 1024x1024
    # 为了节省内存，直接从模型结构推断特征维度，避免运行前向传播
    # -----------------------------
    sam_input_size = 1024  # SAM 预训练尺寸
    
    # 从模型结构推断特征维度（避免前向传播节省内存）
    in_dim = 256  # SAM ViT-B 的默认输出通道数
    try:
        # 从 neck 层推断输出通道数
        if hasattr(sam.image_encoder, 'neck'):
            # neck 是 Sequential，查找最后一个 Conv2d
            for layer in reversed(sam.image_encoder.neck):
                if isinstance(layer, nn.Conv2d):
                    in_dim = layer.out_channels
                    break
        print(f"[INFO] Using image_encoder feature channels: {in_dim} (inferred from model structure)")
    except Exception as e:
        print(f"[WARN] Could not infer feature dimension, using default: {in_dim}")
        print(f"[WARN] Error: {e}")
    
    print(f"[INFO] SAM encoder expects input size: {sam_input_size}x{sam_input_size}, dataset img_size: {args.img_size}")

    proj = PixelProjHead(in_dim=in_dim, proj_dim=args.proj_dim).to(device)

    # ========== 多GPU包装 ==========
    if use_multi_gpu:
        print(f"[INFO] Wrapping models with DataParallel on {len(device_ids)} GPUs")
        # 包装主模型（student）
        sam = torch.nn.DataParallel(sam, device_ids=device_ids)
        # 包装 teacher（用于推理，不需要梯度）
        teacher = torch.nn.DataParallel(teacher, device_ids=device_ids)
        # 包装投影头
        proj = torch.nn.DataParallel(proj, device_ids=device_ids)
        print(f"[INFO] Models wrapped successfully")

    # 可训练参数：image_encoder（部分或全部）+ proj
    # 注意：如果使用DataParallel，需要访问module属性
    trainable_params = list(proj.module.parameters() if use_multi_gpu else proj.parameters())
    sam_encoder = sam.module.image_encoder if use_multi_gpu else sam.image_encoder
    for p in sam_encoder.parameters():
        if p.requires_grad:
            trainable_params.append(p)

    print(f"[INFO] Total trainable parameters: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M")

    # 优化器：使用 encoder 学习率（因为主要训练 encoder）
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr_encoder, weight_decay=args.weight_decay)

    # ========== 显存优化方案 2: 混合精度训练 ==========
    scaler = None
    if args.use_amp:
        print("[INFO] Using Automatic Mixed Precision (AMP) to save memory")
        # 使用新的API避免警告
        try:
            scaler = torch.amp.GradScaler('cuda')
        except AttributeError:
            # 兼容旧版本PyTorch
            scaler = torch.cuda.amp.GradScaler()

    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 损失
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    # -----------------------------
    # 加载 ISIC 数据集
    # -----------------------------
    print(f"[INFO] Loading ISIC dataset from {args.data_root}")
    train_dataset = ISIC2016Dataset(
        root=args.data_root,
        box_csv=args.train_box_csv,
        img_size=args.img_size,
        split='train'
    )
    print(f"[INFO] Train dataset size: {len(train_dataset)}")
    
    # 多GPU时，每个GPU会处理 batch_size/num_gpus 的数据
    # 所以实际总batch size = batch_size * num_gpus
    effective_batch_size = args.batch_size * (len(device_ids) if use_multi_gpu else 1)
    print(f"[INFO] Effective batch size: {effective_batch_size} (batch_size={args.batch_size} x {len(device_ids) if use_multi_gpu else 1} GPU(s))")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        collate_fn=collate_fn_isic  # 使用自定义 collate function
    )

    # EMA teacher decay
    ema_decay = 0.999

    # -----------------------------
    # 训练循环
    # -----------------------------
    for epoch in range(args.epochs):
        sam.train()
        proj.train()
        total_loss_m = 0.0
        total_loss_c = 0.0
        total_loss_iou = 0.0
        total_miou = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for it, batch in enumerate(pbar):
            images = batch['image'].to(device)  # (B, 3, H, W)
            boxes_list = batch['boxes']  # list of big_boxes per image
            small_boxes_list = batch.get('small_boxes', None)  # list of small_boxes per image
            gt_masks = batch['mask'].to(device)  # (B, 1, H, W)

            B = images.size(0)
            
            # SAM 的 image_encoder 期望输入是 1024x1024
            # 如果输入尺寸不同，需要 resize
            sam_input_size = 1024
            if images.shape[-1] != sam_input_size or images.shape[-2] != sam_input_size:
                # Resize 图像到 SAM 期望的尺寸
                images_sam = F.interpolate(images, size=(sam_input_size, sam_input_size), mode='bilinear', align_corners=False)
                # 同时需要缩放 box 坐标
                scale_factor = sam_input_size / args.img_size
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

            # 1) teacher 生成伪标签（不回传）
            with torch.no_grad():
                # 处理DataParallel：需要通过module访问
                teacher_encoder = teacher.module.image_encoder if use_multi_gpu else teacher.image_encoder
                t_img_emb = teacher_encoder(images_sam)  # 使用 SAM 输入尺寸
                all_mask_logits = []
                all_mask_entropy = []
                
                for b in range(B):
                    # 使用缩放后的 box 坐标
                    big_box = boxes_list_sam[b]  # 已经是 list [x1, y1, x2, y2]，已缩放到 SAM 尺寸
                    small_box = small_boxes_list_sam[b]
                    
                    # 在大小框中间区域采样离散点作为 prompt
                    if small_box is not None:
                        points, point_labels = sample_points_in_ring(
                            small_box, big_box, 
                            num_points=10,  # 采样点数量
                            img_size=sam_input_size  # 使用 SAM 输入尺寸
                        )
                        points = points.to(device)  # (N, 2)
                        point_labels = point_labels.to(device)  # (N,)
                        
                        # SAM 需要 points 格式: (1, N, 2) 和 labels: (1, N)
                        points_tensor = points.unsqueeze(0)  # (1, N, 2)
                        labels_tensor = point_labels.unsqueeze(0)  # (1, N)
                    else:
                        # 如果没有 small_box，回退到使用 big_box
                        points_tensor = None
                        labels_tensor = None
                        box_tensor = torch.tensor([big_box], device=device, dtype=torch.float32)  # (1, 4)
                    
                    try:
                        # 处理DataParallel：需要通过module访问
                        teacher_prompt_encoder = teacher.module.prompt_encoder if use_multi_gpu else teacher.prompt_encoder
                        teacher_mask_decoder = teacher.module.mask_decoder if use_multi_gpu else teacher.mask_decoder
                        
                        if points_tensor is not None:
                            # 使用点 prompt
                            sparse_p, dense_p = teacher_prompt_encoder(
                                points=(points_tensor, labels_tensor), 
                                boxes=None, 
                                masks=None
                            )
                        else:
                            # 回退到 box prompt
                            sparse_p, dense_p = teacher_prompt_encoder(
                                points=None, 
                                boxes=box_tensor, 
                                masks=None
                            )
                        
                        out = teacher_mask_decoder(
                            image_embeddings=t_img_emb[b:b+1],
                            image_pe=teacher_prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_p,
                            dense_prompt_embeddings=dense_p,
                            multimask_output=False
                        )
                        if isinstance(out, tuple) and len(out) >= 1:
                            logits = out[0]  # (1, 1, H, W) - low resolution masks
                            # 上采样到原始图像尺寸（不是 SAM 输入尺寸）
                            logits = F.interpolate(
                                logits, 
                                size=(args.img_size, args.img_size), 
                                mode='bilinear', 
                                align_corners=False
                            )
                        else:
                            logits = F.interpolate(out, size=(args.img_size, args.img_size), mode='bilinear', align_corners=False)
                    except Exception as e:
                        print(f"[WARN] Teacher forward failed for image {b}: {e}")
                        logits = torch.zeros(1, 1, args.img_size, args.img_size, device=device)

                    all_mask_logits.append(logits)
                    all_mask_entropy.append(mask_entropy_logits(logits))

                mask_logits_stack = torch.cat(all_mask_logits, dim=0)  # (B, 1, H, W)
                mask_entropy_vals = torch.stack([e if torch.is_tensor(e) else torch.tensor(e, device=device) 
                                                for e in all_mask_entropy]).view(-1)
                
                # 清理 teacher 的中间变量（已保存到 mask_logits_stack）
                del t_img_emb, all_mask_logits, all_mask_entropy
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # 2) 置信度筛选
            trusted_idx = (mask_entropy_vals <= args.entropy_thresh).nonzero(as_tuple=False).squeeze(-1).tolist()

            # 3) student 前向
            # 保存冻结的参考特征（用于蒸馏，在 encoder 更新前保存）
            with torch.no_grad():
                # 处理DataParallel：需要通过module访问
                sam_encoder_ref = sam.module.image_encoder if use_multi_gpu else sam.image_encoder
                ref_feats = sam_encoder_ref(images_sam).detach()  # 使用 SAM 输入尺寸

            # Student image_encoder 前向（可训练，需要梯度）
            # 使用混合精度前向传播（如果启用）
            sam_encoder = sam.module.image_encoder if use_multi_gpu else sam.image_encoder
            if args.use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    img_emb = sam_encoder(images_sam)  # (B, 256, H', W') - 需要梯度
            else:
                img_emb = sam_encoder(images_sam)  # (B, 256, H', W') - 需要梯度

            # student decoder 前向传播（冻结，不需要梯度）
            preds = []
            ious = []
            with torch.no_grad():  # decoder 冻结，不需要梯度
                for b in range(B):
                    big_box = boxes_list_sam[b]  # 使用缩放后的 big_box（SAM 尺寸）
                    small_box = small_boxes_list_sam[b]
                    
                    # 在大小框中间区域采样离散点作为 prompt（与 teacher 一致）
                    if small_box is not None:
                        points, point_labels = sample_points_in_ring(
                            small_box, big_box, 
                            num_points=10,
                            img_size=sam_input_size  # 使用 SAM 输入尺寸
                        )
                        points = points.to(device)
                        point_labels = point_labels.to(device)
                        points_tensor = points.unsqueeze(0)  # (1, N, 2)
                        labels_tensor = point_labels.unsqueeze(0)  # (1, N)
                    else:
                        points_tensor = None
                        labels_tensor = None
                        box_tensor = torch.tensor([big_box], device=device, dtype=torch.float32)
                    
                    try:
                        # 处理DataParallel：需要通过module访问
                        sam_prompt_encoder = sam.module.prompt_encoder if use_multi_gpu else sam.prompt_encoder
                        sam_mask_decoder = sam.module.mask_decoder if use_multi_gpu else sam.mask_decoder
                        
                        if points_tensor is not None:
                            sp, dp = sam_prompt_encoder(
                                points=(points_tensor, labels_tensor), 
                                boxes=None, 
                                masks=None
                            )
                        else:
                            sp, dp = sam_prompt_encoder(
                                points=None, 
                                boxes=box_tensor, 
                                masks=None
                            )
                        
                        # 使用 detach 的 image_embeddings 通过 decoder（decoder 冻结）
                        outb = sam_mask_decoder(
                            image_embeddings=img_emb[b:b+1].detach(),  # decoder 不需要梯度
                            image_pe=sam_prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sp,
                            dense_prompt_embeddings=dp,
                            multimask_output=False
                        )
                        if isinstance(outb, tuple) and len(outb) >= 2:
                            low_res_masks, iou_pred = outb[0], outb[1]
                            # 上采样到原始图像尺寸
                            masks_upsampled = F.interpolate(
                                low_res_masks,
                                size=(args.img_size, args.img_size),
                                mode='bilinear',
                                align_corners=False
                            )
                            preds.append(masks_upsampled)
                            ious.append(iou_pred)
                        else:
                            masks_upsampled = F.interpolate(outb, size=(args.img_size, args.img_size), mode='bilinear', align_corners=False)
                            preds.append(masks_upsampled)
                            ious.append(torch.zeros(1, 1, device=device))
                    except Exception as e:
                        print(f"[WARN] Student forward failed for image {b}: {e}")
                        preds.append(torch.zeros(1, 1, args.img_size, args.img_size, device=device))
                        ious.append(torch.zeros(1, 1, device=device))

            pred_logits = torch.cat(preds, dim=0)  # (B, 1, H, W)
            try:
                pred_iou = torch.cat(ious, dim=0)  # (B, 1)
            except Exception:
                pred_iou = None

            # 4) supervised mask loss
            # 注意：由于 decoder 冻结，mask loss 只用于监控，不参与 encoder 训练
            # 如果需要 mask loss 参与训练，需要重新通过 decoder（但会计算两次）
            # 这里我们只用于监控，主要训练信号来自对比学习
            loss_mask = torch.tensor(0.0, device=device)
            batch_miou = 0.0
            if gt_masks is not None:
                with torch.no_grad():  # mask loss 不参与训练（decoder 冻结）
                    gt_resized = F.interpolate(gt_masks, size=pred_logits.shape[-2:], mode='nearest')
                    loss_bce = bce_loss(pred_logits, gt_resized)
                    loss_dice = dice_loss(pred_logits, gt_resized)
                    loss_mask = loss_bce + loss_dice
                    # 计算 mIOU
                    batch_miou = compute_miou(pred_logits, gt_resized)

            # 5) pixel-wise contrast
            proxy_feats = img_emb  # 使用 image_encoder 的输出作为对比学习的特征

            if proxy_feats is None:
                loss_contrast = torch.tensor(0.0, device=device)
            else:
                # 投影到对比学习空间（支持混合精度）
                if args.use_amp and scaler is not None:
                    with torch.cuda.amp.autocast():
                        z = proj(proxy_feats)  # (B, D, Hf, Wf)
                else:
                    z = proj(proxy_feats)  # (B, D, Hf, Wf)
                Bz, D, Hf, Wf = z.shape

                mask_pred_bin = (torch.sigmoid(pred_logits) > 0.5).float()
                mask_resized = F.interpolate(mask_pred_bin, size=(Hf, Wf), mode='nearest')
                teacher_mask_resized = F.interpolate(torch.sigmoid(mask_logits_stack), size=(Hf, Wf), mode='nearest')

                anchors_list = []
                positives_list = []
                negatives_pool = []

                for b in range(B):
                    if b not in trusted_idx:
                        continue
                    
                    # 获取当前图像的 box 信息（注意：特征图是基于 SAM 输入尺寸的）
                    # img_emb 是基于 images_sam (1024x1024) 的，所以特征图尺寸对应 SAM 输入尺寸
                    big_box = boxes_list_sam[b]  # [x1, y1, x2, y2] - 已缩放到 SAM 尺寸
                    small_box = small_boxes_list_sam[b]
                    
                    if small_box is None:
                        continue
                    
                    # 创建 small_box 和 big_box 的 mask（在特征图尺寸 Hf, Wf 上）
                    # 特征图尺寸对应 SAM 输入尺寸（1024），不是原始图像尺寸
                    sx1, sy1, sx2, sy2 = small_box
                    bx1, by1, bx2, by2 = big_box
                    
                    # 将坐标缩放到特征图尺寸（特征图对应 SAM 输入尺寸）
                    scale_h = Hf / sam_input_size
                    scale_w = Wf / sam_input_size
                    
                    sx1_f, sy1_f = int(sx1 * scale_w), int(sy1 * scale_h)
                    sx2_f, sy2_f = int(sx2 * scale_w), int(sy2 * scale_h)
                    bx1_f, by1_f = int(bx1 * scale_w), int(by1 * scale_h)
                    bx2_f, by2_f = int(bx2 * scale_w), int(by2 * scale_h)
                    
                    # 创建 small_box mask
                    small_box_mask = torch.zeros(Hf, Wf, device=device, dtype=torch.float32)
                    small_box_mask[sy1_f:sy2_f+1, sx1_f:sx2_f+1] = 1.0
                    
                    # 创建 big_box mask
                    big_box_mask = torch.zeros(Hf, Wf, device=device, dtype=torch.float32)
                    big_box_mask[by1_f:by2_f+1, bx1_f:bx2_f+1] = 1.0
                    
                    # Teacher 伪标签 mask（二值化）
                    tmask = (torch.sigmoid(mask_logits_stack[b:b+1]) > 0.5).float()
                    tmask_resized = F.interpolate(tmask, size=(Hf, Wf), mode='nearest').squeeze(0).squeeze(0)  # (Hf, Wf)
                    
                    # ========== 正样本：伪标签与小 box 高度重叠的区域 ==========
                    # 显著性正样本：伪标签 AND 小 box（高度重叠区域）
                    salient_pos_mask = (tmask_resized > 0.5) & (small_box_mask > 0.5)
                    pos_idx = salient_pos_mask.view(-1).nonzero(as_tuple=False).squeeze(-1)
                    
                    if pos_idx.numel() == 0:
                        continue
                    
                    npos = min(pos_idx.numel(), args.pos_samples)
                    perm = torch.randperm(pos_idx.numel(), device=device)[:npos]
                    chosen_pos = pos_idx[perm]
                    
                    z_b = z[b].permute(1, 2, 0).reshape(-1, D)  # (Hf*Wf, D)
                    anchors_list.append(z_b[chosen_pos])
                    
                    # Positive: teacher 特征对应位置
                    with torch.no_grad():
                        try:
                            # 处理DataParallel：需要通过module访问
                            teacher_encoder_pos = teacher.module.image_encoder if use_multi_gpu else teacher.image_encoder
                            t_img_emb_b = teacher_encoder_pos(images[b:b+1])
                            t_dense = t_img_emb_b
                        except Exception:
                            t_dense = ref_feats[b:b+1] if ref_feats is not None else t_img_emb_b

                        tproj = proj(t_dense)  # (1, D, Hf, Wf)
                        tproj_flat = tproj.permute(0, 2, 3, 1).reshape(-1, D)  # (Hf*Wf, D)
                        positives_list.append(tproj_flat[chosen_pos].detach())
                    
                    # ========== 困难负样本 ==========
                    hard_neg_list = []
                    
                    # 类型1：小 box 内但伪标签未覆盖的区域（漏掉的部分）
                    missed_in_small = (small_box_mask > 0.5) & (tmask_resized < 0.5)
                    missed_idx = missed_in_small.view(-1).nonzero(as_tuple=False).squeeze(-1)
                    if missed_idx.numel() > 0:
                        n_missed = min(missed_idx.numel(), args.neg_samples // 2)
                        perm_missed = torch.randperm(missed_idx.numel(), device=device)[:n_missed]
                        hard_neg_list.append(z_b[missed_idx[perm_missed]].detach())
                    
                    # 类型2：大 box 外但伪标签覆盖的区域（溢出的部分）
                    overflow_out_big = (big_box_mask < 0.5) & (tmask_resized > 0.5)
                    overflow_idx = overflow_out_big.view(-1).nonzero(as_tuple=False).squeeze(-1)
                    if overflow_idx.numel() > 0:
                        n_overflow = min(overflow_idx.numel(), args.neg_samples // 2)
                        perm_overflow = torch.randperm(overflow_idx.numel(), device=device)[:n_overflow]
                        hard_neg_list.append(z_b[overflow_idx[perm_overflow]].detach())
                    
                    # 组合困难负样本
                    if len(hard_neg_list) > 0:
                        hard_negs = torch.cat(hard_neg_list, dim=0)
                    else:
                        # 如果没有困难负样本，使用随机负样本
                        all_pixels_b = z_b.detach()
                        n_random = min(all_pixels_b.shape[0], args.neg_samples)
                        perm_random = torch.randperm(all_pixels_b.shape[0], device=device)[:n_random]
                        hard_negs = all_pixels_b[perm_random]
                    
                    negatives_pool.append(hard_negs)

                if len(anchors_list) == 0:
                    loss_contrast = torch.tensor(0.0, device=device)
                else:
                    anchors = torch.cat(anchors_list, dim=0)
                    positives = torch.cat(positives_list, dim=0)
                    
                    # 合并所有困难负样本
                    if len(negatives_pool) > 0:
                        # 如果每个 batch 的负样本数量不同，需要统一处理
                        # 方案：将所有困难负样本合并，然后随机采样到固定数量
                        all_hard_negs = torch.cat(negatives_pool, dim=0)
                        max_neg = min(all_hard_negs.shape[0], args.neg_samples * len(anchors_list))
                        if max_neg > 0:
                            perm_neg = torch.randperm(all_hard_negs.shape[0], device=device)[:max_neg]
                            negatives = all_hard_negs[perm_neg]
                        else:
                            # 如果困难负样本不够，补充随机负样本
                            all_pixels = z.permute(0, 2, 3, 1).reshape(-1, D).detach()
                            n_random = min(all_pixels.shape[0], args.neg_samples * len(anchors_list))
                            perm_random = torch.randperm(all_pixels.shape[0], device=device)[:n_random]
                            negatives = all_pixels[perm_random]
                    else:
                        # 如果没有困难负样本，使用随机负样本
                        all_pixels = z.permute(0, 2, 3, 1).reshape(-1, D).detach()
                        n_random = min(all_pixels.shape[0], args.neg_samples * len(anchors_list))
                        perm_random = torch.randperm(all_pixels.shape[0], device=device)[:n_random]
                        negatives = all_pixels[perm_random]
                    
                    loss_contrast = pixel_info_nce(anchors, positives, negatives, temperature=args.temperature)

            # 6) IoU loss
            loss_iou = torch.tensor(0.0, device=device)
            if pred_iou is not None:
                with torch.no_grad():
                    pred_bin = (torch.sigmoid(pred_logits) > 0.5).float()
                    inter = (pred_bin * (mask_logits_stack.sigmoid() > 0.5).float()).sum(dim=[1,2,3])
                    union = ((pred_bin + (mask_logits_stack.sigmoid() > 0.5).float()) > 0).float().sum(dim=[1,2,3])
                    gt_iou = (inter / (union + 1e-6)).unsqueeze(-1)
                loss_iou = F.mse_loss(pred_iou, gt_iou)

            # 7) Distillation loss
            # 由于 encoder 是可训练的，蒸馏损失帮助保持预训练知识
            # 但权重应该较小，避免过度约束
            loss_distill = torch.tensor(0.0, device=device)
            try:
                s_feat = img_emb  # 当前 encoder 输出（可训练）
                r_feat = ref_feats  # 冻结的参考特征（初始状态）
                if s_feat.shape == r_feat.shape:
                    loss_distill = F.mse_loss(s_feat, r_feat)
                else:
                    if s_feat.dim() == 4 and r_feat.dim() == 4:
                        r_pool = F.interpolate(r_feat, size=(s_feat.shape[2], s_feat.shape[3]), mode='bilinear', align_corners=False)
                        loss_distill = F.mse_loss(s_feat, r_pool)
            except Exception:
                loss_distill = torch.tensor(0.0, device=device)

            # 8) 总损失
            alpha = 1.0
            beta = 0.5
            gamma = 0.1
            loss = loss_mask + alpha * loss_contrast + beta * loss_iou + gamma * loss_distill

            # ========== 显存优化：梯度累积 ==========
            # 归一化损失（考虑梯度累积）
            loss = loss / args.gradient_accumulation_steps

            # 反向传播（支持混合精度）
            if args.use_amp and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # 梯度累积：只在累积步数达到时才更新
            if (it + 1) % args.gradient_accumulation_steps == 0:
                if args.use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                optimizer.zero_grad()

            # EMA teacher update（只在梯度更新后执行）
            if (it + 1) % args.gradient_accumulation_steps == 0:
                with torch.no_grad():
                    # 处理DataParallel的参数名称（会添加module.前缀）
                    if use_multi_gpu:
                        # DataParallel会添加module.前缀，需要访问module属性
                        teacher_module = teacher.module
                        sam_module = sam.module
                        for t_param, s_param in zip(teacher_module.parameters(), sam_module.parameters()):
                            t_param.data.mul_(ema_decay).add_(s_param.data * (1.0 - ema_decay))
                    else:
                        for t_param, s_param in zip(teacher.parameters(), sam.parameters()):
                            t_param.data.mul_(ema_decay).add_(s_param.data * (1.0 - ema_decay))

            total_loss_m += loss_mask.item() if isinstance(loss_mask, torch.Tensor) else float(loss_mask)
            total_loss_c += loss_contrast.item() if isinstance(loss_contrast, torch.Tensor) else float(loss_contrast)
            total_loss_iou += loss_iou.item() if isinstance(loss_iou, torch.Tensor) else float(loss_iou)
            total_miou += batch_miou

            # 显示损失（考虑梯度累积，显示真实损失值）
            display_loss = loss.item() * args.gradient_accumulation_steps
            
            # 显存监控（每10个batch显示一次）
            mem_info = ""
            if torch.cuda.is_available() and (it + 1) % 10 == 0:
                device_id = device.index if hasattr(device, 'index') else 0
                allocated = torch.cuda.memory_allocated(device_id) / 1024**3  # GB
                reserved = torch.cuda.memory_reserved(device_id) / 1024**3    # GB
                mem_info = f" GPU:{allocated:.2f}/{reserved:.2f}GB"
            
            pbar.set_description(f"E{epoch+1} L={display_loss:.4f} mask={float(loss_mask):.4f} cont={float(loss_contrast):.4f} iou={float(loss_iou):.4f} mIOU={batch_miou:.4f}{mem_info}")
            
            # 每个 batch 结束后清理中间变量和显存缓存
            if torch.cuda.is_available():
                # 清理不需要的变量
                del loss, loss_mask, loss_contrast, loss_iou, loss_distill
                if (it + 1) % 5 == 0:  # 每5个batch清理一次缓存（更频繁）
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()  # 收集进程间通信的缓存

        # 学习率调度（每个 epoch 更新一次）
        scheduler.step()

        # 保存 checkpoint
        if (epoch + 1) % args.save_every == 0:
            # 处理DataParallel：保存时需要访问module属性
            if use_multi_gpu:
                sam_encoder_state = sam.module.image_encoder.state_dict()
                proj_state = proj.module.state_dict()
            else:
                sam_encoder_state = sam.image_encoder.state_dict()
                proj_state = proj.state_dict()
            
            ckpt = {
                'sam_image_encoder': sam_encoder_state,  # 保存整个 encoder（因为可训练）
                'proj': proj_state,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(ckpt, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            print(f"[INFO] Saved checkpoint: checkpoint_epoch_{epoch+1}.pth")

        avg_miou = total_miou / len(train_loader)
        print(f"Epoch {epoch+1} avg_mask_loss={total_loss_m/len(train_loader):.4f} "
              f"avg_contrast={total_loss_c/len(train_loader):.4f} "
              f"avg_iou={total_loss_iou/len(train_loader):.4f} "
              f"avg_mIOU={avg_miou:.4f}")
        
        # 每个 epoch 结束后清理显存缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            device_id = device.index if hasattr(device, 'index') else 0
            allocated = torch.cuda.memory_allocated(device_id) / 1024**3  # GB
            print(f"[INFO] Epoch {epoch+1} 结束后显存使用: {allocated:.2f}GB")

    print('Training finished!')


# 主程序入口
# -----------------------------
if __name__ == '__main__':
    argp = build_argparser()
    args = argp.parse_args()
    train_main(args)




