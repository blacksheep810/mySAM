#!/usr/bin/env python
# coding: utf-8
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import cv2

from dataset.ISIC import ISIC2016Dataset
from models.sam_wrapper import (
    load_sam_model, create_teacher_model, get_encoder_feature_dim, setup_multi_gpu
)
from models.projection import PixelProjHead
from models.losses import DiceLoss, pixel_info_nce, weighted_dice_loss
from utils.metrics import compute_miou, compute_iou_per_sample, mask_entropy_logits, mask_entropy_map_logits
from utils.prompts import prepare_box_prompts
from utils.visualization import visualize_prediction_with_boxes, visualize_prediction_on_original_image
from utils.training_utils import set_seed, mkdir, setup_cuda_memory
from training.data_utils import collate_fn_isic
import pandas as pd


class Trainer:

    def __init__(self, args):
        self.args = args
        self.sam_input_size = 1024  # SAM 
        
        
        set_seed(42)
        mkdir(args.output_dir)
        
        device_id = 0 if args.device == 'cuda' else int(args.device.split(':')[1]) if ':' in args.device else 0
        mem_info = setup_cuda_memory(device_id)
        if mem_info:
            print(f"[INFO] GPU {mem_info['device_id']} åå§æ¾å­ç¶æ?: "
                  f"å·²åé?={mem_info['allocated']:.2f}GB, "
                  f"å·²ä¿ç?={mem_info['reserved']:.2f}GB, "
                  f"æ»è®¡={mem_info['total']:.2f}GB")
            if mem_info['reserved'] > 0.1:
                print(f"[WARN] æ£æµå° {mem_info['reserved']:.2f}GB æ¾å­è¢«ä¿ç?")
        
        self.device, self.use_multi_gpu, self.device_ids = self._setup_device()
        
        #
        self.sam, self.teacher, self.proj = self._setup_models()
       
        self.optimizer, self.scheduler, self.scaler = self._setup_optimizer()
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        
       
        self.train_loader, self.val_loader = self._setup_dataloaders()
        
        # EMA 参数
        self.ema_decay = 0.99
        
        self.epoch_skipped_batches = 0
        self.epoch_low_quality_pos_batches = 0
        self.epoch_total_batches = 0
        
        # 动态权重调整：记录历史mIoU和Teacher-Student一致性
        self.history_miou = []  # 记录每个epoch的平均mIoU
        self.history_ts_consistency = []  # 记录每个epoch的Teacher-Student一致性
        self.best_miou = 0.0  # 记录最佳mIoU
        self.best_miou_epoch = 0  # 达到最佳mIoU的epoch（用于早停）

        # 表现差样本检测相关
        self.detect_poor_samples = getattr(args, 'detect_poor_samples', False)
        # 多级分类阈值：<0.75严重, 0.75-0.8中等, >=0.8合理
        self.poor_sample_iou_threshold_severe = getattr(args, 'poor_sample_iou_threshold_severe', 0.75)  # 严重阈值
        self.poor_sample_iou_threshold_moderate = getattr(args, 'poor_sample_iou_threshold_moderate', 0.8)  # 中等阈值上限
        self.poor_sample_save_interval = getattr(args, 'poor_sample_save_interval', 1)  # 每N个epoch保存一次
        self.poor_samples_data = []  # 存储表现差样本信息
        
        # 强数据增强与伪标签 Dice
        self.student_strong_aug = getattr(args, 'student_strong_aug', False)
        self.pseudo_dice_weight = getattr(args, 'pseudo_dice_weight', 0.0)
        self.pseudo_dice_use_confidence_mask = getattr(args, 'pseudo_dice_use_confidence_mask', False)
        self.pseudo_dice_confidence_thresh = getattr(args, 'pseudo_dice_confidence_thresh', 0.8)
        self.pseudo_dice_boundary_weighted = getattr(args, 'pseudo_dice_boundary_weighted', False)
        self.pseudo_dice_boundary_alpha = getattr(args, 'pseudo_dice_boundary_alpha', 1.0)

        # 平台期优化：更慢 teacher 与分歧驱动困难样本
        self.ema_fixed_decay = getattr(args, 'ema_fixed_decay', 0.0)
        self.ema_update_interval = max(1, int(getattr(args, 'ema_update_interval', 1)))
        self.hard_sample_use_disagreement = getattr(args, 'hard_sample_use_disagreement', False)
        self.hardness_alpha = getattr(args, 'hardness_alpha', 0.5)
        self.hardness_beta = getattr(args, 'hardness_beta', 0.5)
        self.optimizer_step_count = 0

        if getattr(args, 'lr_decoder', None) is not None:
            print("[INFO] lr_decoder is currently ignored because decoder parameters are frozen.")
    
    def _augment_images_student(self, images: torch.Tensor) -> torch.Tensor:
        """
        对 Student 输入应用强颜色增强（仅颜色，不改变几何，避免 box 变换）
        Teacher 使用原图；Student 使用增强图，提升鲁棒性与对比学习难度
        增强：亮度 ±15%、对比度 ±15%
        """
        if not self.student_strong_aug:
            return images
        out = images.clone()
        B, C, H, W = out.shape
        for b in range(B):
            # 亮度：乘 [0.85, 1.15]
            br = 0.85 + 0.3 * random.random()
            out[b] = out[b] * br
            # 对比度：(x - mean) * factor + mean, factor in [0.85, 1.15]
            mean = out[b].mean()
            ct = 0.85 + 0.3 * random.random()
            out[b] = (out[b] - mean) * ct + mean
        out = torch.clamp(out, 0, 1)
        return out
    
    def _student_decoder_forward_for_dice(self, img_emb, boxes_list_sam, small_boxes_list_sam):
        """
        用 img_emb（不 detach）跑 decoder，得到 pred_logits 用于伪标签 Dice 损失
        梯度可回传到 encoder
        """
        B = img_emb.size(0)
        preds = []
        device = self.device
        for b in range(B):
            big_box = boxes_list_sam[b]
            small_box = small_boxes_list_sam[b]
            boxes_tensor = prepare_box_prompts(big_box, small_box, device)
            try:
                sam_prompt_encoder = self.sam.module.prompt_encoder if self.use_multi_gpu else self.sam.prompt_encoder
                sam_mask_decoder = self.sam.module.mask_decoder if self.use_multi_gpu else self.sam.mask_decoder
                sp, dp = sam_prompt_encoder(points=None, boxes=boxes_tensor, masks=None)
                outb = sam_mask_decoder(
                    image_embeddings=img_emb[b:b+1],  # 不 detach，保持梯度
                    image_pe=sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sp,
                    dense_prompt_embeddings=dp,
                    multimask_output=False
                )
                logits = outb[0] if isinstance(outb, tuple) and len(outb) >= 1 else outb
                if logits.shape[0] > 1:
                    logits = logits[0:1]
                logits = F.interpolate(logits, size=(self.args.img_size, self.args.img_size),
                                       mode='bilinear', align_corners=False)
                preds.append(logits)
            except Exception:
                preds.append(torch.zeros(1, 1, self.args.img_size, self.args.img_size, device=device))
        return torch.cat(preds, dim=0)

    def _sample_indices(self, candidate_mask, max_samples, device, hardness_map=None):
        """
        从候选像素中采样索引。若提供 hardness_map 且启用分歧驱动，则优先取 hardest 像素。
        """
        indices = candidate_mask.view(-1).nonzero(as_tuple=False).squeeze(-1)
        if indices.numel() == 0 or max_samples <= 0:
            return indices[:0]

        max_samples = min(max_samples, indices.numel())
        if hardness_map is not None and self.hard_sample_use_disagreement:
            hard_values = hardness_map.view(-1)[indices]
            _, order = torch.topk(hard_values, k=max_samples, largest=True)
            return indices[order]

        perm = torch.randperm(indices.numel(), device=device)[:max_samples]
        return indices[perm]

    def _build_hardness_map(self, teacher_prob, student_prob):
        """
        构建逐像素 hardness map:
        hardness = alpha * |p_t - p_s| + beta * entropy_t
        """
        teacher_prob = teacher_prob.clamp(1e-6, 1 - 1e-6)
        disagreement = torch.abs(teacher_prob - student_prob)
        entropy_map = -(teacher_prob * torch.log(teacher_prob) + (1 - teacher_prob) * torch.log(1 - teacher_prob))
        return self.hardness_alpha * disagreement + self.hardness_beta * entropy_map
    
    def _setup_device(self):
        """è®¾ç½®è®¾å¤åå¤GPUéç½®"""
        num_gpus = torch.cuda.device_count()
        use_multi_gpu = False
        device_ids = None
        
        if self.args.multi_gpu or self.args.gpu_ids is not None:
            if num_gpus > 1:
                if self.args.gpu_ids is not None:
                    device_ids = [int(x.strip()) for x in self.args.gpu_ids.split(',')]
                    use_multi_gpu = len(device_ids) > 1
                    print(f"[INFO] Using specified GPUs: {device_ids}")
                else:
                    device_ids = list(range(num_gpus))
                    use_multi_gpu = True
                    print(f"[INFO] Using all available GPUs: {device_ids}")
            else:
                print(f"[WARN] Only {num_gpus} GPU(s) available, using single GPU")
                use_multi_gpu = False
        
        if use_multi_gpu:
            device = torch.device(f'cuda:{device_ids[0]}')
            print(f"[INFO] Multi-GPU training enabled on {len(device_ids)} GPUs")
        else:
            device = torch.device(self.args.device if torch.cuda.is_available() else 'cpu')
            print(f"[INFO] Using device: {device}")
        
        return device, use_multi_gpu, device_ids
    
    def _get_autocast(self):
        """
        获取适合当前设备的 autocast 上下文管理器
        兼容新旧 API 和 CPU/CUDA 设备
        """
        if not (self.args.use_amp and self.scaler is not None):
            return None
        
        # 检查是否使用 CUDA
        use_cuda = self.device.type == 'cuda' and torch.cuda.is_available()
        
        # 使用新的 API（PyTorch 2.0+）
        try:
            if use_cuda:
                return torch.amp.autocast('cuda')
            else:
                return torch.amp.autocast('cpu')
        except AttributeError:
            # 回退到旧 API（PyTorch < 2.0）
            if use_cuda:
                return torch.cuda.amp.autocast()
            else:
                # CPU 模式下不使用 autocast
                return None
    
    def _setup_models(self):
        sam = load_sam_model(
            checkpoint_path=self.args.sam_checkpoint,
            model_type=self.args.model_type,
            device=self.device,
            unfreeze_last_k=self.args.unfreeze_last_k,
            use_gradient_checkpointing=self.args.use_gradient_checkpointing
        )
        
        # åå»º teacher æ¨¡å
        teacher = create_teacher_model(
            checkpoint_path=self.args.sam_checkpoint,
            model_type=self.args.model_type,
            device=self.device
        )
        
        # è·åç¹å¾ç»´åº¦å¹¶åå»ºæå½±å¤´
        in_dim = get_encoder_feature_dim(sam)
        print(f"[INFO] SAM encoder expects input size: {self.sam_input_size}x{self.sam_input_size}, "
              f"dataset img_size: {self.args.img_size}")
        
        proj = PixelProjHead(in_dim=in_dim, proj_dim=self.args.proj_dim).to(self.device)
        
        # å¤GPU åè£
        if self.use_multi_gpu:
            print(f"[INFO] Wrapping models with DataParallel on {len(self.device_ids)} GPUs")
            models_dict = {'sam': sam, 'teacher': teacher, 'proj': proj}
            wrapped = setup_multi_gpu(models_dict, self.device_ids)
            sam = wrapped['sam']
            teacher = wrapped['teacher']
            proj = wrapped['proj']
            print(f"[INFO] Models wrapped successfully")
        
        return sam, teacher, proj
    
    def _setup_optimizer(self):
        trainable_params = list(self.proj.module.parameters() if self.use_multi_gpu else self.proj.parameters())
        sam_encoder = self.sam.module.image_encoder if self.use_multi_gpu else self.sam.image_encoder
        for p in sam_encoder.parameters():
            if p.requires_grad:
                trainable_params.append(p)
        
        print(f"[INFO] Total trainable parameters: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M")
        
        # ä¼åå?
        optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=self.args.lr_encoder, 
            weight_decay=self.args.weight_decay
        )
        
        # æ··åç²¾åº¦
        scaler = None
        if self.args.use_amp:
            print("[INFO] Using Automatic Mixed Precision (AMP) to save memory")
            try:
                scaler = torch.amp.GradScaler('cuda')
            except AttributeError:
                scaler = torch.cuda.amp.GradScaler()
        
        # å­¦ä¹ çè°åº?
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        
        return optimizer, scheduler, scaler
    
    def _setup_dataloaders(self):
        """设置训练/验证数据加载器"""
        print(f"[INFO] Loading ISIC dataset from {self.args.data_root}")
        train_dataset = ISIC2016Dataset(
            root=self.args.data_root,
            box_csv=self.args.train_box_csv,
            img_size=self.args.img_size,
            split='train'
        )
        print(f"[INFO] Train dataset size: {len(train_dataset)}")
        
        effective_batch_size = self.args.batch_size * (len(self.device_ids) if self.use_multi_gpu else 1)
        print(f"[INFO] Effective batch size: {effective_batch_size} "
              f"(batch_size={self.args.batch_size} x {len(self.device_ids) if self.use_multi_gpu else 1} GPU(s))")
        
        # 只在 CUDA 可用时启用 pin_memory
        pin_memory = self.device.type == 'cuda' and torch.cuda.is_available()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=pin_memory,
            collate_fn=collate_fn_isic
        )

        val_loader = None
        if getattr(self.args, 'test_box_csv', None):
            val_dataset = ISIC2016Dataset(
                root=self.args.data_root,
                box_csv=self.args.test_box_csv,
                img_size=self.args.img_size,
                split='test'
            )
            print(f"[INFO] Validation dataset size: {len(val_dataset)}")
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=pin_memory,
                collate_fn=collate_fn_isic
            )
        else:
            print("[WARN] No test_box_csv provided; best checkpoint and early stop will fall back to train mIOU.")

        return train_loader, val_loader
    
    def _prepare_images_and_boxes(self, images, boxes_list, small_boxes_list):
        """åå¤å¾ååæ¡ï¼resize å? SAM è¾å¥å°ºå¯¸ï¼?"""
        B = images.size(0)
        
        if images.shape[-1] != self.sam_input_size or images.shape[-2] != self.sam_input_size:
            # Resize å¾åå? SAM ææçå°ºå¯?
            images_sam = F.interpolate(
                images, 
                size=(self.sam_input_size, self.sam_input_size), 
                mode='bilinear', 
                align_corners=False
            )
            # ç¼©æ¾ box åæ 
            scale_factor = self.sam_input_size / self.args.img_size
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
        
        return images_sam, boxes_list_sam, small_boxes_list_sam
    
    def _create_box_mask_from_corners(self, corners, H, W, device):
        """
        从四个角点创建框的mask
        
        Args:
            corners: 四个角点坐标，形状为 (4, 2) 或 list of [x, y]
            H, W: mask的高度和宽度
            device: 设备
        
        Returns:
            mask: (H, W) tensor，框内为1，外为0
        """
        if corners is None:
            return None
        
        # 转换为numpy数组
        if isinstance(corners, (list, tuple)):
            corners_np = np.array(corners, dtype=np.float32)
        else:
            corners_np = corners
        
        # 确保是 (4, 2) 形状
        if corners_np.shape != (4, 2):
            return None
        
        # 转换为整数坐标
        corners_int = corners_np.astype(np.int32)
        
        # 创建mask并填充
        mask_np = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask_np, [corners_int], 255)
        mask = torch.from_numpy(mask_np).float().to(device) / 255.0
        
        return mask
    
    def _teacher_forward(self, images_sam, boxes_list_sam, small_boxes_list_sam):
        """Teacher ååä¼ æ­çæä¼ªæ ç­?"""
        with torch.no_grad():
            teacher_encoder = self.teacher.module.image_encoder if self.use_multi_gpu else self.teacher.image_encoder
            t_img_emb = teacher_encoder(images_sam)
            
            all_mask_logits = []
            all_mask_entropy = []
            B = images_sam.size(0)
            
            for b in range(B):
                big_box = boxes_list_sam[b]
                small_box = small_boxes_list_sam[b]
                # 使用大框+小框作为 prompt（与 mySAM 一致；仅大框会导致 decoder 缺少细粒度约束，mIOU 难提升）
                boxes_tensor = prepare_box_prompts(big_box, small_box, self.device)
                
                try:
                    teacher_prompt_encoder = self.teacher.module.prompt_encoder if self.use_multi_gpu else self.teacher.prompt_encoder
                    teacher_mask_decoder = self.teacher.module.mask_decoder if self.use_multi_gpu else self.teacher.mask_decoder
                    
                    sparse_p, dense_p = teacher_prompt_encoder(
                        points=None,
                        boxes=boxes_tensor,
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
                        logits = out[0]
                        # Èç¹ûÊäÈëÁË¶à¸ö¿ò£¬SAM ¿ÉÄÜÎªÃ¿¸ö¿òÉú³ÉÒ»¸ö mask
                        # È·±£Ö»Ê¹ÓÃµÚÒ»¸ö mask£¨batch_size=1£©
                        if logits.shape[0] > 1:
                            logits = logits[0:1]  # Ö»È¡µÚÒ»¸ö
                        logits = F.interpolate(
                            logits,
                            size=(self.args.img_size, self.args.img_size),
                            mode='bilinear',
                            align_corners=False
                        )
                    else:
                        logits = F.interpolate(
                            out, 
                            size=(self.args.img_size, self.args.img_size), 
                            mode='bilinear', 
                            align_corners=False
                        )
                        # È·±£ batch_size=1
                        if logits.shape[0] > 1:
                            logits = logits[0:1]
                except Exception as e:
                    print(f"[WARN] Teacher forward failed for image {b}: {e}")
                    logits = torch.zeros(1, 1, self.args.img_size, self.args.img_size, device=self.device)
                
                all_mask_logits.append(logits)
                all_mask_entropy.append(mask_entropy_logits(logits))
            
            mask_logits_stack = torch.cat(all_mask_logits, dim=0)
            mask_entropy_vals = torch.stack([
                e if torch.is_tensor(e) else torch.tensor(e, device=self.device) 
                for e in all_mask_entropy
            ]).view(-1)

            teacher_img_emb = t_img_emb.detach()
            del all_mask_logits, all_mask_entropy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return mask_logits_stack, mask_entropy_vals, teacher_img_emb
    
    def _student_forward(self, images_sam, boxes_list_sam, small_boxes_list_sam):
        """Student ååä¼ æ­"""

        sam_encoder = self.sam.module.image_encoder if self.use_multi_gpu else self.sam.image_encoder
        autocast_ctx = self._get_autocast()
        if autocast_ctx is not None:
            with autocast_ctx:
                img_emb = sam_encoder(images_sam)
        else:
            img_emb = sam_encoder(images_sam)
        
        # Student decoder 前向（冻结，仅用于生成 pred 做监控；训练信号来自 contrastive + distill）
        preds = []
        ious = []
        B = images_sam.size(0)
        
        with torch.no_grad():
            for b in range(B):
                big_box = boxes_list_sam[b]
                small_box = small_boxes_list_sam[b]
                boxes_tensor = prepare_box_prompts(big_box, small_box, self.device)
                try:
                    sam_prompt_encoder = self.sam.module.prompt_encoder if self.use_multi_gpu else self.sam.prompt_encoder
                    sam_mask_decoder = self.sam.module.mask_decoder if self.use_multi_gpu else self.sam.mask_decoder
                    sp, dp = sam_prompt_encoder(
                        points=None,
                        boxes=boxes_tensor,
                        masks=None
                    )
                    outb = sam_mask_decoder(
                        image_embeddings=img_emb[b:b+1].detach(),
                        image_pe=sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sp,
                        dense_prompt_embeddings=dp,
                        multimask_output=False
                    )
                    if isinstance(outb, tuple) and len(outb) >= 2:
                        low_res_masks, iou_pred = outb[0], outb[1]
                        if low_res_masks.shape[0] > 1:
                            low_res_masks = low_res_masks[0:1]
                            if iou_pred.shape[0] > 1:
                                iou_pred = iou_pred[0:1]
                        masks_upsampled = F.interpolate(
                            low_res_masks,
                            size=(self.args.img_size, self.args.img_size),
                            mode='bilinear',
                            align_corners=False
                        )
                        preds.append(masks_upsampled)
                        ious.append(iou_pred)
                    else:
                        masks_upsampled = F.interpolate(
                            outb,
                            size=(self.args.img_size, self.args.img_size),
                            mode='bilinear',
                            align_corners=False
                        )
                        if masks_upsampled.shape[0] > 1:
                            masks_upsampled = masks_upsampled[0:1]
                        preds.append(masks_upsampled)
                        ious.append(torch.zeros(1, 1, device=self.device))
                except Exception as e:
                    print(f"[WARN] Student forward failed for image {b}: {e}")
                    preds.append(torch.zeros(1, 1, self.args.img_size, self.args.img_size, device=self.device))
                    ious.append(torch.zeros(1, 1, device=self.device))
        
        pred_logits = torch.cat(preds, dim=0)
        try:
            pred_iou = torch.cat(ious, dim=0)
        except Exception:
            pred_iou = None
        
        return img_emb, pred_logits, pred_iou
    
    def _compute_losses(self, pred_logits, pred_iou, gt_masks, img_emb, teacher_img_emb,
                        mask_logits_stack, mask_entropy_vals, boxes_list_sam, small_boxes_list_sam, images_sam,
                        small_box_corners_list=None, epoch=None):
        """è®¡ç®æææå¤?"""
        device = self.device
        
        # 1) Mask loss（仅用于监控，不参与反传；训练信号来自 contrastive + distill）
        loss_mask = torch.tensor(0.0, device=device)
        batch_miou = 0.0
        if gt_masks is not None:
            with torch.no_grad():
                gt_resized = F.interpolate(gt_masks, size=pred_logits.shape[-2:], mode='nearest')
                loss_bce = self.bce_loss(pred_logits, gt_resized)
                loss_dice = self.dice_loss(pred_logits, gt_resized)
                loss_mask = loss_bce + loss_dice
                batch_miou = compute_miou(pred_logits, gt_resized)
                iou_per_sample = None
                if self.detect_poor_samples:
                    iou_per_sample = compute_iou_per_sample(pred_logits, gt_resized)
        
        # 2) Contrastive loss（传入student预测和epoch用于自适应并集交集策略）
        loss_contrast, pos_count, neg_count = self._compute_contrastive_loss(
            img_emb, mask_logits_stack, mask_entropy_vals, 
            boxes_list_sam, small_boxes_list_sam, teacher_img_emb, small_box_corners_list,
            pred_logits_student=pred_logits, epoch=epoch
        )
        
        # 3) IoU loss
        loss_iou = torch.tensor(0.0, device=device)
        if pred_iou is not None:
            with torch.no_grad():
                pred_bin = (torch.sigmoid(pred_logits) > 0.5).float()
                inter = (pred_bin * (mask_logits_stack.sigmoid() > 0.5).float()).sum(dim=[1,2,3])
                union = ((pred_bin + (mask_logits_stack.sigmoid() > 0.5).float()) > 0).float().sum(dim=[1,2,3])
                gt_iou = (inter / (union + 1e-6)).unsqueeze(-1)
            loss_iou = F.mse_loss(pred_iou, gt_iou)
        
        # 4) Distillation loss
        loss_distill = torch.tensor(0.0, device=device)
        try:
            s_feat = img_emb
            r_feat = teacher_img_emb.detach()
            if s_feat.shape == r_feat.shape:
                loss_distill = F.mse_loss(s_feat, r_feat)
            else:
                if s_feat.dim() == 4 and r_feat.dim() == 4:
                    r_pool = F.interpolate(
                        r_feat, 
                        size=(s_feat.shape[2], s_feat.shape[3]), 
                        mode='bilinear', 
                        align_corners=False
                    )
                    loss_distill = F.mse_loss(s_feat, r_pool)
        except Exception:
            loss_distill = torch.tensor(0.0, device=device)
        
        # æ»æå¤?
        # 5) 伪标签 Dice 损失（支持置信度 mask 与边界加权）
        loss_dice_pseudo = torch.tensor(0.0, device=device)
        if self.pseudo_dice_weight > 0:
            pred_for_dice = self._student_decoder_forward_for_dice(
                img_emb, boxes_list_sam, small_boxes_list_sam
            )
            teacher_prob = torch.sigmoid(mask_logits_stack).detach()
            if pred_for_dice.shape[-2:] != teacher_prob.shape[-2:]:
                teacher_prob = F.interpolate(teacher_prob, size=pred_for_dice.shape[-2:], mode='nearest')

            weight_map = None
            if self.pseudo_dice_use_confidence_mask or self.pseudo_dice_boundary_weighted:
                weight_map = torch.ones_like(teacher_prob)

                if self.pseudo_dice_use_confidence_mask:
                    conf = self.pseudo_dice_confidence_thresh
                    confidence_mask = ((teacher_prob >= conf) | (teacher_prob <= (1.0 - conf))).float()
                    weight_map = weight_map * confidence_mask

                if self.pseudo_dice_boundary_weighted:
                    teacher_entropy_map = mask_entropy_map_logits(mask_logits_stack).detach()
                    if teacher_entropy_map.shape[-2:] != teacher_prob.shape[-2:]:
                        teacher_entropy_map = F.interpolate(
                            teacher_entropy_map, size=teacher_prob.shape[-2:],
                            mode='bilinear', align_corners=False
                        )
                    student_prob_for_weight = torch.sigmoid(pred_logits).detach()
                    if student_prob_for_weight.shape[-2:] != teacher_prob.shape[-2:]:
                        student_prob_for_weight = F.interpolate(
                            student_prob_for_weight, size=teacher_prob.shape[-2:],
                            mode='bilinear', align_corners=False
                        )
                    disagreement_map = torch.abs(teacher_prob - student_prob_for_weight)
                    boundary_boost = 1.0 + self.pseudo_dice_boundary_alpha * (teacher_entropy_map + disagreement_map)
                    weight_map = weight_map * boundary_boost

                if torch.all(weight_map <= 0):
                    weight_map = None

            loss_dice_pseudo = weighted_dice_loss(pred_for_dice, teacher_prob, weight=weight_map)
        
        alpha, gamma = 1.0, 0.1
        loss = alpha * loss_contrast + gamma * loss_distill
        if self.pseudo_dice_weight > 0:
            loss = loss + self.pseudo_dice_weight * loss_dice_pseudo
        
        return loss, loss_mask, loss_contrast, loss_iou, loss_distill, loss_dice_pseudo, batch_miou, iou_per_sample, pos_count, neg_count
    
    def _get_adaptive_weights(self, epoch, current_miou=None, ts_consistency=None):
        """
        根据训练进度和性能指标动态计算自适应权重（用于并集交集策略）
        优化：基于mIoU和Teacher-Student一致性动态调整困难样本权重
        
        Args:
            epoch: 当前epoch（从0开始）
            current_miou: 当前epoch的平均mIoU（可选，用于动态调整）
            ts_consistency: Teacher-Student一致性（可选，用于动态调整）
        
        Returns:
            intersection_weight: 交集（高置信度）权重 (0-1)
            union_weight: 并集（中等置信度）权重 (0-1)
            hard_sample_weight: 困难样本权重 (0-1)
            strategy: 当前策略名称
        """
        total_epochs = self.args.epochs
        progress = epoch / total_epochs if total_epochs > 0 else 0.0
        # 困难样本引入进度（调优建议：可设为 0.7 延后引入）
        start_hard = getattr(self.args, 'hard_sample_start_progress', 0.6)
        
        # 基础权重（收紧正样本以缓解 mask>GT 过分割：更高交集、更低并集/困难/过渡区）
        if progress < 0.15:
            base_intersection_weight = 0.85
            base_union_weight = 0.15
            base_hard_sample_weight = 0.0
            base_strategy = "early_union"
        elif progress < 0.3:
            alpha = (progress - 0.15) / 0.15
            base_intersection_weight = 0.85 - 0.15 * alpha
            base_union_weight = 0.15 + 0.08 * alpha
            base_hard_sample_weight = 0.0
            base_strategy = "intersection_union"
        elif progress < start_hard:
            alpha = (progress - 0.3) / (start_hard - 0.3) if start_hard > 0.3 else 0.0
            base_intersection_weight = 0.7 - 0.1 * alpha
            base_union_weight = 0.25
            base_hard_sample_weight = 0.03 * alpha
            base_strategy = "with_hard_samples"
        elif progress < 0.85:
            alpha = (progress - start_hard) / (0.85 - start_hard) if (0.85 - start_hard) > 0 else 1.0
            base_intersection_weight = 0.6 - 0.1 * alpha
            base_union_weight = 0.25
            base_hard_sample_weight = 0.03 + 0.04 * alpha
            base_strategy = "with_hard_samples"
        else:
            base_intersection_weight = 0.45
            base_union_weight = 0.25
            base_hard_sample_weight = getattr(self.args, 'hard_sample_max_weight', 0.12)
            base_strategy = "balanced"
        
        # 动态调整：基于mIoU和Teacher-Student一致性
        hard_sample_weight = base_hard_sample_weight
        strategy = base_strategy
        
        if current_miou is not None and len(self.history_miou) > 0:
            # 性能保护：如果mIoU下降，降低困难样本权重
            recent_miou = self.history_miou[-1] if len(self.history_miou) > 0 else current_miou
            
            # 如果当前mIoU低于历史最佳，降低困难样本权重
            if current_miou < self.best_miou - 0.02:  # 下降超过2%
                reduction_factor = max(0.3, 1.0 - (self.best_miou - current_miou) / 0.1)  # 最多降低70%
                hard_sample_weight = base_hard_sample_weight * reduction_factor
                strategy = f"{base_strategy}_miou_protect"
            
            # 如果mIoU持续下降（连续3个epoch），进一步降低困难样本权重
            if len(self.history_miou) >= 3:
                if all(self.history_miou[-i] < self.history_miou[-i-1] for i in range(1, min(3, len(self.history_miou)))):
                    hard_sample_weight = base_hard_sample_weight * 0.5  # 减半
                    strategy = f"{base_strategy}_miou_degrading"
        
        if ts_consistency is not None and len(self.history_ts_consistency) > 0:
            # Teacher-Student一致性保护：如果一致性低，降低困难样本权重
            recent_consistency = self.history_ts_consistency[-1] if len(self.history_ts_consistency) > 0 else ts_consistency
            
            if ts_consistency < 0.7:  # 一致性低于70%
                consistency_factor = max(0.5, ts_consistency / 0.7)  # 最多降低50%
                hard_sample_weight = hard_sample_weight * consistency_factor
                if "ts_protect" not in strategy:
                    strategy = f"{strategy}_ts_protect"
        
        # 确保权重在合理范围内（调优建议：上限从 0.3 降为 0.2，见 docs/训练日志分析与调优建议.md）
        max_hard = getattr(self.args, 'hard_sample_max_weight', 0.2)
        hard_sample_weight = max(0.0, min(max_hard, hard_sample_weight))
        
        return base_intersection_weight, base_union_weight, hard_sample_weight, strategy
    
    def _compute_contrastive_loss(self, img_emb, mask_logits_stack, mask_entropy_vals, 
                                  boxes_list_sam, small_boxes_list_sam, teacher_img_emb, small_box_corners_list=None,
                                  pred_logits_student=None, epoch=None):
        """è®¡ç®å¯¹æ¯å­¦ä¹ æå¤±（使用并集交集策略和自适应权重）"""
        device = self.device
        
        # 获取自适应权重
        # 获取动态权重（传入当前epoch的mIoU和TS一致性）
        current_miou = self.history_miou[-1] if len(self.history_miou) > 0 else None
        ts_consistency = self.history_ts_consistency[-1] if len(self.history_ts_consistency) > 0 else None
        inter_weight, union_weight, hard_weight, strategy = self._get_adaptive_weights(
            epoch if epoch is not None else 0, 
            current_miou, 
            ts_consistency
        )
        
        # ç½®ä¿¡åº¦ç­é?
        trusted_idx = (mask_entropy_vals <= self.args.entropy_thresh).nonzero(as_tuple=False).squeeze(-1).tolist()
        
        if len(trusted_idx) == 0:
            return torch.tensor(0.0, device=device), 0, 0
        
        # æå½±ç¹å¾
        autocast_ctx = self._get_autocast()
        if autocast_ctx is not None:
            with autocast_ctx:
                z = self.proj(img_emb)
        else:
            z = self.proj(img_emb)
        with torch.no_grad():
            teacher_proj = self.proj(teacher_img_emb).detach()
        
        Bz, D, Hf, Wf = z.shape
        
        # åå¤ mask
        teacher_mask_resized = F.interpolate(
            torch.sigmoid(mask_logits_stack), 
            size=(Hf, Wf), 
            mode='nearest'
        )
        teacher_entropy_resized = F.interpolate(
            mask_entropy_map_logits(mask_logits_stack).detach(),
            size=(Hf, Wf),
            mode='bilinear',
            align_corners=False
        )
        
        # 准备student mask（用于并集交集策略）
        student_mask_resized = None
        if pred_logits_student is not None:
            student_mask_resized = F.interpolate(
                torch.sigmoid(pred_logits_student),
                size=(Hf, Wf),
                mode='nearest'
            )
        
        anchors_list = []
        positives_list = []
        negatives_pool = []
        
        for b in range(Bz):
            if b not in trusted_idx:
                continue
            
            big_box = boxes_list_sam[b]
            small_box = small_boxes_list_sam[b]
            
            if small_box is None:
                continue
            
            # åå»º mask
            sx1, sy1, sx2, sy2 = small_box
            bx1, by1, bx2, by2 = big_box
            
            scale_h = Hf / self.sam_input_size
            scale_w = Wf / self.sam_input_size
            
            sx1_f, sy1_f = int(sx1 * scale_w), int(sy1 * scale_h)
            sx2_f, sy2_f = int(sx2 * scale_w), int(sy2 * scale_h)
            bx1_f, by1_f = int(bx1 * scale_w), int(by1 * scale_h)
            bx2_f, by2_f = int(bx2 * scale_w), int(by2 * scale_h)
            
            # 创建小框mask（优先使用四个角点，否则使用AABB）
            if small_box_corners_list is not None and b < len(small_box_corners_list) and small_box_corners_list[b] is not None:
                # 角点来自 dataset，为 img_size(512) 空间；特征图对应 SAM 输入 1024。
                # 需先映射到 1024 再映射到特征图：scale = (Hf/1024) 等价于 512->1024->feat 当角点为 512 时应用 scale_feat = Hf/img_size
                corners = np.array(small_box_corners_list[b], dtype=np.float32)
                corners_scaled = corners.copy()
                scale_corners_w = Wf / self.args.img_size  # 512 -> feature
                scale_corners_h = Hf / self.args.img_size
                corners_scaled[:, 0] *= scale_corners_w
                corners_scaled[:, 1] *= scale_corners_h
                # 创建旋转框mask
                small_box_mask = self._create_box_mask_from_corners(corners_scaled, Hf, Wf, device)
                if small_box_mask is None:
                    # 如果创建失败，回退到AABB
                    small_box_mask = torch.zeros(Hf, Wf, device=device, dtype=torch.float32)
                    small_box_mask[sy1_f:sy2_f+1, sx1_f:sx2_f+1] = 1.0
            else:
                # 使用轴对齐矩形（AABB）
                small_box_mask = torch.zeros(Hf, Wf, device=device, dtype=torch.float32)
                small_box_mask[sy1_f:sy2_f+1, sx1_f:sx2_f+1] = 1.0
            
            big_box_mask = torch.zeros(Hf, Wf, device=device, dtype=torch.float32)
            big_box_mask[by1_f:by2_f+1, bx1_f:bx2_f+1] = 1.0
            
            # 定义过渡区域：大框内但小框外（策略1：采样密度控制）
            # 需要转换为布尔类型才能使用 ~ 操作符
            transition_mask = (big_box_mask > 0.5) & (~(small_box_mask > 0.5))
            transition_mask = transition_mask.float()  # 转换回float类型以便后续使用
            
            tmask_resized = teacher_mask_resized[b].squeeze(0).squeeze(0)
            smask_resized = student_mask_resized[b].squeeze(0).squeeze(0) if student_mask_resized is not None else None
            hardness_map = None
            if smask_resized is not None and self.hard_sample_use_disagreement:
                disagreement_map = torch.abs(tmask_resized - smask_resized)
                entropy_map = teacher_entropy_resized[b].squeeze(0).squeeze(0)
                hardness_map = self.hardness_alpha * disagreement_map + self.hardness_beta * entropy_map
            
            # ========== 基于大小框和并集交集的自适应正样本选择 ==========
            # 核心原则：正样本主要在小框内，过渡区域作为补充（策略1：采样密度控制）
            pos_mask_list = []
            
            if smask_resized is not None:
                # 1. 【高置信度正样本】小框内且Teacher和Student都预测为前景（交集）
                intersection_pos = (tmask_resized > 0.5) & (smask_resized > 0.5) & (small_box_mask > 0.5)
                if inter_weight > 0 and intersection_pos.sum() > 0:
                    pos_mask_list.append((intersection_pos, inter_weight, "intersection"))
                
                # 2. 【中等置信度正样本】小框内且Teacher或Student预测为前景（并集，排除交集部分）
                union_pos = ((tmask_resized > 0.5) | (smask_resized > 0.5)) & (small_box_mask > 0.5)
                union_only_pos = union_pos & (~intersection_pos)
                if union_weight > 0 and union_only_pos.sum() > 0:
                    pos_mask_list.append((union_only_pos, union_weight, "union"))
                
                # 3. 【困难正样本】小框内且Teacher预测为前景但Student预测为背景（需要学习）
                # 添加置信度检查：只使用Teacher高置信度的困难样本（置信度>0.7）
                hard_pos = (tmask_resized > 0.5) & (smask_resized < 0.5) & (small_box_mask > 0.5)
                if hard_weight > 0 and hard_pos.sum() > 0:
                    # 质量检查：只使用Teacher高置信度的困难样本
                    hard_pos_confidences = tmask_resized.view(-1)[hard_pos.view(-1).nonzero(as_tuple=False).squeeze(-1)]
                    min_conf_thresh = getattr(self.args, 'hard_sample_min_confidence', 0.7)
                    high_conf_mask = hard_pos_confidences > min_conf_thresh
                    
                    if high_conf_mask.sum() > 0:
                        # 创建高置信度困难正样本mask
                        hard_pos_indices = hard_pos.view(-1).nonzero(as_tuple=False).squeeze(-1)
                        high_conf_indices = hard_pos_indices[high_conf_mask]
                        hard_pos_filtered = torch.zeros_like(small_box_mask, dtype=torch.bool)
                        hard_pos_filtered.view(-1)[high_conf_indices] = True
                        
                        pos_mask_list.append((hard_pos_filtered, hard_weight, "hard_teacher"))
                
                # 4. 【Student主导正样本】小框内且Student预测为前景但Teacher预测为背景
                # 重要：因为小框内一定是前景，所以Student预测前景是对的，Teacher预测背景是错的
                # 这种情况应该被当作正样本，而不是困难负样本
                # 提前引入（不等到hard_weight > 0.15），因为这是正确的预测
                student_pos = (smask_resized > 0.5) & (tmask_resized < 0.5) & (small_box_mask > 0.5)
                if student_pos.sum() > 0:
                    # 使用较小的权重，但比原来更早引入
                    student_weight = hard_weight * 0.5 if hard_weight > 0 else 0.1  # 如果hard_weight=0，使用固定权重0.1
                    if student_weight > 0:
                        pos_mask_list.append((student_pos, student_weight, "student_correct"))
                
                # 5. 【策略1：过渡区域正样本】大框内但小框外且Teacher预测为前景（边界扩展区域）
                # 目的：在过渡区域进行更高频率的采样，强化边界判别能力
                if getattr(self.args, 'transition_region_enabled', False) and transition_mask.sum() > 0:
                    transition_pos = (tmask_resized > 0.5) & (transition_mask > 0.5)
                    if transition_pos.sum() > 0:
                        # 置信度过滤：过渡区域不确定性更高，使用稍低的阈值
                        transition_confidences = tmask_resized.view(-1)[transition_pos.view(-1).nonzero(as_tuple=False).squeeze(-1)]
                        min_transition_conf = getattr(self.args, 'transition_region_min_confidence', 0.6)
                        high_conf_transition = transition_confidences > min_transition_conf
                        
                        if high_conf_transition.sum() > 0:
                            # 创建高置信度过渡区域正样本mask
                            transition_pos_indices = transition_pos.view(-1).nonzero(as_tuple=False).squeeze(-1)
                            high_conf_transition_indices = transition_pos_indices[high_conf_transition]
                            transition_pos_filtered = torch.zeros_like(transition_mask, dtype=torch.bool)
                            transition_pos_filtered.view(-1)[high_conf_transition_indices] = True
                            
                            # 过渡区域权重：使用union_weight作为基础，但会根据采样密度调整
                            transition_weight = union_weight * 0.8 if union_weight > 0 else 0.1
                            if transition_weight > 0:
                                pos_mask_list.append((transition_pos_filtered, transition_weight, "transition_region"))
            else:
                # 回退：只使用teacher预测（小框内且teacher预测为前景）
                teacher_pos = (tmask_resized > 0.5) & (small_box_mask > 0.5)
                pos_mask_list.append((teacher_pos, 1.0, "teacher_only"))
            
            # 根据权重组合正样本mask（策略1：过渡区域采样密度控制）
            if len(pos_mask_list) > 0:
                salient_pos_mask = torch.zeros_like(small_box_mask, dtype=torch.bool)
                transition_pos_selected = torch.zeros_like(small_box_mask, dtype=torch.bool)  # 过渡区域被选中的正样本（与 transition_mask 区域定义区分）
                
                # 先处理非过渡区域的样本
                for pos_mask, weight, mask_type in pos_mask_list:
                    if weight > 0 and mask_type != "transition_region":
                        pos_indices = pos_mask.view(-1).nonzero(as_tuple=False).squeeze(-1)
                        if pos_indices.numel() > 0:
                            n_samples = max(1, int(pos_indices.numel() * weight))
                            local_hardness = hardness_map if mask_type in {"hard_teacher", "student_correct"} else None
                            selected_indices = self._sample_indices(
                                pos_mask, n_samples, device, hardness_map=local_hardness
                            )
                            if selected_indices.numel() > 0:
                                salient_pos_mask.view(-1)[selected_indices] = True
                
                # 单独处理过渡区域样本（策略1：提高采样密度）
                transition_sampling_ratio = getattr(self.args, 'transition_region_sampling_ratio', 1.5)
                transition_max_ratio = getattr(self.args, 'transition_region_max_ratio', 0.3)
                
                for pos_mask, weight, mask_type in pos_mask_list:
                    if weight > 0 and mask_type == "transition_region":
                        transition_indices = pos_mask.view(-1).nonzero(as_tuple=False).squeeze(-1)
                        if transition_indices.numel() > 0:
                            # 策略1核心：提高过渡区域采样密度（采样率是小框内的transition_sampling_ratio倍）
                            base_n_samples = max(1, int(transition_indices.numel() * weight))
                            # 应用采样密度倍数
                            n_transition_samples = int(base_n_samples * transition_sampling_ratio)
                            n_transition_samples = min(n_transition_samples, transition_indices.numel())
                            
                            # 限制过渡区域样本不超过总正样本的最大比例（T/(P+T)<=r => T<=P*r/(1-r)）
                            total_pos_so_far = salient_pos_mask.sum().item()
                            ratio_cap = min(transition_max_ratio, 0.99)  # 避免 1-r=0 除零
                            max_transition_samples = int(total_pos_so_far * ratio_cap / (1 - ratio_cap)) if total_pos_so_far > 0 else n_transition_samples
                            n_transition_samples = min(n_transition_samples, max_transition_samples)
                            
                            if n_transition_samples > 0:
                                selected_transition_indices = self._sample_indices(
                                    pos_mask, n_transition_samples, device, hardness_map=hardness_map
                                )
                                transition_pos_selected.view(-1)[selected_transition_indices] = True
                
                # 合并过渡区域选中的正样本到总正样本 mask
                salient_pos_mask = salient_pos_mask | transition_pos_selected
            else:
                # 回退：使用teacher预测
                salient_pos_mask = (tmask_resized > 0.5) & (small_box_mask > 0.5)
            
            pos_idx = salient_pos_mask.view(-1).nonzero(as_tuple=False).squeeze(-1)
            
            if pos_idx.numel() == 0:
                # 尝试恢复策略：使用Teacher-only正样本
                teacher_pos = (tmask_resized > 0.5) & (small_box_mask > 0.5)
                if teacher_pos.sum() > 0:
                    pos_idx = teacher_pos.view(-1).nonzero(as_tuple=False).squeeze(-1)
                else:
                    # 最后回退：使用小框内所有像素
                    pos_idx = small_box_mask.view(-1).nonzero(as_tuple=False).squeeze(-1)
                    if pos_idx.numel() == 0:
                        # 完全无法恢复，统计跳过的batch
                        if hasattr(self, 'epoch_skipped_batches'):
                            self.epoch_skipped_batches += 1
                        continue  # 跳过batch
            
            # 正样本质量检查：检查置信度
            pos_confidences = tmask_resized.view(-1)[pos_idx]
            min_confidence = pos_confidences.min().item()
            mean_confidence = pos_confidences.mean().item()
            
            # 如果正样本置信度过低，记录警告（但不阻止训练）
            min_confidence_thresh = getattr(self.args, 'min_pos_confidence', 0.2)
            if min_confidence < min_confidence_thresh:
                # 标记为低质量，但继续训练（让模型学习）
                pass  # 可以在训练循环中统计
            
            # 初始正样本采样
            chosen_pos = self._sample_indices(
                salient_pos_mask, self.args.pos_samples, device, hardness_map=None
            )
            if chosen_pos.numel() == 0:
                continue
            
            # 保存正样本质量信息（用于统计）
            chosen_pos_confidences = tmask_resized.view(-1)[chosen_pos]
            chosen_min_conf = chosen_pos_confidences.min().item()
            chosen_mean_conf = chosen_pos_confidences.mean().item()
            
            z_b = z[b].permute(1, 2, 0).reshape(-1, D)
            
            # Positive: teacher 特征（必须成功才加入 anchor/positive 对，避免长度不一致导致整 batch 对比损失为 0）
            tproj_flat = teacher_proj[b].permute(1, 2, 0).reshape(-1, D)
            positives_list.append(tproj_flat[chosen_pos].detach())
            anchors_list.append(z_b[chosen_pos])
            
            # ========== 基于大小框和并集交集的自适应负样本选择 ==========
            hard_neg_list = []
            if smask_resized is not None:
                # 【负样本类型1】小框内为背景（高置信度负样本）
                # 1.1 交集：小框内且Teacher和Student都预测为背景（最可靠）
                intersection_neg_in_small = (tmask_resized < 0.5) & (smask_resized < 0.5) & (small_box_mask > 0.5)
                missed_idx_inter = intersection_neg_in_small.view(-1).nonzero(as_tuple=False).squeeze(-1)
                if missed_idx_inter.numel() > 0 and inter_weight > 0:
                    n_missed = min(missed_idx_inter.numel(), int(self.args.neg_samples // 3 * inter_weight))
                    if n_missed > 0:
                        chosen = self._sample_indices(
                            intersection_neg_in_small, n_missed, device, hardness_map=None
                        )
                        hard_neg_list.append(z_b[chosen].detach())
                
                # 1.2 并集：小框内且Teacher或Student预测为背景（排除交集部分）
                union_neg_in_small = ((tmask_resized < 0.5) | (smask_resized < 0.5)) & (small_box_mask > 0.5)
                union_neg_only = union_neg_in_small & (~intersection_neg_in_small)
                missed_idx_union = union_neg_only.view(-1).nonzero(as_tuple=False).squeeze(-1)
                if missed_idx_union.numel() > 0 and union_weight > 0:
                    n_missed = min(missed_idx_union.numel(), int(self.args.neg_samples // 3 * union_weight))
                    if n_missed > 0:
                        chosen = self._sample_indices(
                            union_neg_only, n_missed, device, hardness_map=None
                        )
                        hard_neg_list.append(z_b[chosen].detach())
                
                # 【负样本类型2】大框外为前景（溢出，高置信度负样本）
                # 2.1 交集：大框外且Teacher和Student都预测为前景（最可靠）
                intersection_neg_out_big = (tmask_resized > 0.5) & (smask_resized > 0.5) & (big_box_mask < 0.5)
                overflow_idx_inter = intersection_neg_out_big.view(-1).nonzero(as_tuple=False).squeeze(-1)
                if overflow_idx_inter.numel() > 0 and inter_weight > 0:
                    n_overflow = min(overflow_idx_inter.numel(), int(self.args.neg_samples // 3 * inter_weight))
                    if n_overflow > 0:
                        chosen = self._sample_indices(
                            intersection_neg_out_big, n_overflow, device, hardness_map=None
                        )
                        hard_neg_list.append(z_b[chosen].detach())
                
                # 2.2 并集：大框外且Teacher或Student预测为前景（排除交集部分）
                union_neg_out_big = ((tmask_resized > 0.5) | (smask_resized > 0.5)) & (big_box_mask < 0.5)
                union_neg_out_only = union_neg_out_big & (~intersection_neg_out_big)
                overflow_idx_union = union_neg_out_only.view(-1).nonzero(as_tuple=False).squeeze(-1)
                if overflow_idx_union.numel() > 0 and union_weight > 0:
                    n_overflow = min(overflow_idx_union.numel(), int(self.args.neg_samples // 3 * union_weight))
                    if n_overflow > 0:
                        chosen = self._sample_indices(
                            union_neg_out_only, n_overflow, device, hardness_map=None
                        )
                        hard_neg_list.append(z_b[chosen].detach())
                
                # 【负样本类型3】困难负样本：Student预测为前景但Teacher预测为背景（需要纠正）
                # 重要修正：基于大小框约束（小框内一定是前景，大框外一定是背景）
                # - 小框内：Student预测前景是对的，不应该被当作困难负样本
                # - 大框外：Student预测前景是错的，应该被纠正
                if hard_weight > 0:
                    # 3.1 小框内困难负样本 - 已移除
                    # 原因：小框内一定是前景，Student预测前景是对的，不应该被纠正
                    # 这种情况已经在正样本选择中处理（Student主导正样本）
                    
                    # 3.2 大框外困难负样本（保留，因为大框外一定是背景）
                    # Student预测前景但Teacher预测背景 → Student错误，需要纠正
                    # ⚠️ 重要：添加Teacher置信度过滤，避免Teacher在困难区域出错时误导训练
                    hard_neg_out_big = (tmask_resized < 0.5) & (smask_resized > 0.5) & (big_box_mask < 0.5)
                    if hard_neg_out_big.sum() > 0:
                        # 质量检查：只使用Teacher高置信度预测为背景的样本
                        # Teacher预测背景的置信度 = 1 - tmask_resized（值越小，预测背景的置信度越高）
                        hard_neg_confidences = 1.0 - tmask_resized.view(-1)[hard_neg_out_big.view(-1).nonzero(as_tuple=False).squeeze(-1)]
                        min_conf_thresh = getattr(self.args, 'hard_sample_min_confidence', 0.7)
                        # Teacher预测背景的置信度要足够高（即tmask_resized要足够低）
                        high_conf_mask = hard_neg_confidences > min_conf_thresh
                        
                        if high_conf_mask.sum() > 0:
                            # 创建高置信度困难负样本mask
                            hard_neg_indices = hard_neg_out_big.view(-1).nonzero(as_tuple=False).squeeze(-1)
                            high_conf_indices = hard_neg_indices[high_conf_mask]
                            hard_neg_filtered = torch.zeros_like(big_box_mask, dtype=torch.bool)
                            hard_neg_filtered.view(-1)[high_conf_indices] = True
                            
                            overflow_idx_hard = hard_neg_filtered.view(-1).nonzero(as_tuple=False).squeeze(-1)
                            if overflow_idx_hard.numel() > 0:
                                # 限制困难负样本数量：不超过总负样本的20%
                                max_hard_neg = int(self.args.neg_samples * 0.2)
                                n_overflow = min(overflow_idx_hard.numel(), max_hard_neg, int(self.args.neg_samples // 4 * hard_weight))
                                if n_overflow > 0:
                                    chosen = self._sample_indices(
                                        hard_neg_filtered, n_overflow, device, hardness_map=hardness_map
                                    )
                                    hard_neg_list.append(z_b[chosen].detach())
            else:
                # 回退：只使用teacher预测
                # 【负样本类型1】小框内为背景
                missed_in_small = (small_box_mask > 0.5) & (tmask_resized < 0.5)
                missed_idx = missed_in_small.view(-1).nonzero(as_tuple=False).squeeze(-1)
                if missed_idx.numel() > 0:
                    n_missed = min(missed_idx.numel(), self.args.neg_samples // 2)
                    chosen = self._sample_indices(missed_in_small, n_missed, device)
                    hard_neg_list.append(z_b[chosen].detach())
                
                # 【负样本类型2】大框外为前景
                overflow_out_big = (big_box_mask < 0.5) & (tmask_resized > 0.5)
                overflow_idx = overflow_out_big.view(-1).nonzero(as_tuple=False).squeeze(-1)
                if overflow_idx.numel() > 0:
                    n_overflow = min(overflow_idx.numel(), self.args.neg_samples // 2)
                    chosen = self._sample_indices(overflow_out_big, n_overflow, device)
                    hard_neg_list.append(z_b[chosen].detach())
            
            if len(hard_neg_list) > 0:
                hard_negs = torch.cat(hard_neg_list, dim=0)
            else:
                all_pixels_b = z_b.detach()
                n_random = min(all_pixels_b.shape[0], self.args.neg_samples)
                perm_random = torch.randperm(all_pixels_b.shape[0], device=device)[:n_random]
                hard_negs = all_pixels_b[perm_random]
            
            negatives_pool.append(hard_negs)
        
        if len(anchors_list) == 0:
            # 统计：所有batch都被跳过
            if hasattr(self, 'epoch_skipped_batches'):
                self.epoch_skipped_batches += 1
            return torch.tensor(0.0, device=device), 0, 0
        
        # 检查 positives_list 是否为空或长度不匹配
        if len(positives_list) == 0 or len(positives_list) != len(anchors_list):
            return torch.tensor(0.0, device=device), 0, 0
        
        anchors = torch.cat(anchors_list, dim=0)
        positives = torch.cat(positives_list, dim=0)
        
        # 统计正样本数量
        pos_count = anchors.shape[0]
        
        # 保存所有可用的负样本池（用于均衡调整）
        all_available_negs = None
        if len(negatives_pool) > 0:
            all_available_negs = torch.cat(negatives_pool, dim=0)
        
        # 初始负样本采样
        if len(negatives_pool) > 0:
            all_hard_negs = all_available_negs
            max_neg = min(all_hard_negs.shape[0], self.args.neg_samples * len(anchors_list))
            if max_neg > 0:
                perm_neg = torch.randperm(all_hard_negs.shape[0], device=device)[:max_neg]
                negatives = all_hard_negs[perm_neg]
            else:
                all_pixels = z.permute(0, 2, 3, 1).reshape(-1, D).detach()
                n_random = min(all_pixels.shape[0], self.args.neg_samples * len(anchors_list))
                if n_random > 0:
                    perm_random = torch.randperm(all_pixels.shape[0], device=device)[:n_random]
                    negatives = all_pixels[perm_random]
                else:
                    return torch.tensor(0.0, device=device), pos_count, 0
        else:
            all_pixels = z.permute(0, 2, 3, 1).reshape(-1, D).detach()
            n_random = min(all_pixels.shape[0], self.args.neg_samples * len(anchors_list))
            if n_random > 0:
                perm_random = torch.randperm(all_pixels.shape[0], device=device)[:n_random]
                negatives = all_pixels[perm_random]
            else:
                return torch.tensor(0.0, device=device), pos_count, 0
        
        # 最终检查 negatives 是否为空
        if negatives.shape[0] == 0:
            return torch.tensor(0.0, device=device), pos_count, 0
        
        # 统计初始负样本数量
        neg_count = negatives.shape[0]
        
        # ========== 正负样本均衡机制 ==========
        # 如果正负样本比例失衡，调整采样数量以保证均衡
        pos_neg_ratio = pos_count / (neg_count + 1e-6)
        target_ratio = getattr(self.args, 'pos_neg_ratio', 0.25)  # 默认正负样本比例 1:4
        
        if pos_neg_ratio > target_ratio * 1.5:
            # 正样本过多，优先增加负样本
            target_neg_count = int(pos_count / target_ratio)
            
            # 策略1：从负样本池中增加负样本（优先）
            if all_available_negs is not None and all_available_negs.shape[0] > neg_count:
                target_neg_count = min(target_neg_count, all_available_negs.shape[0])
                if target_neg_count > neg_count:
                    perm_neg = torch.randperm(all_available_negs.shape[0], device=device)[:target_neg_count]
                    negatives = all_available_negs[perm_neg]
                    neg_count = negatives.shape[0]
            # 策略2：如果负样本池不够，从所有像素中补充负样本
            elif neg_count < target_neg_count:
                all_pixels = z.permute(0, 2, 3, 1).reshape(-1, D).detach()
                # 计算需要补充的负样本数量
                needed_neg_count = target_neg_count - neg_count
                # 确保不超过可用像素数
                max_additional = min(all_pixels.shape[0], needed_neg_count)
                if max_additional > 0:
                    perm_additional = torch.randperm(all_pixels.shape[0], device=device)[:max_additional]
                    additional_negs = all_pixels[perm_additional]
                    # 合并负样本
                    negatives = torch.cat([negatives, additional_negs], dim=0)
                    neg_count = negatives.shape[0]
                    
        elif pos_neg_ratio < target_ratio * 0.5:
            # 负样本过多，减少负样本
            target_neg_count = int(pos_count / target_ratio)
            if neg_count > target_neg_count and target_neg_count > 0:
                perm_neg = torch.randperm(negatives.shape[0], device=device)[:target_neg_count]
                negatives = negatives[perm_neg]
                neg_count = negatives.shape[0]
        
        loss_contrast = pixel_info_nce(anchors, positives, negatives, temperature=self.args.temperature)
        return loss_contrast, pos_count, neg_count
    
    def _update_ema_teacher(self, epoch=None):
        """更新 EMA teacher（支持 warm-up、固定慢 EMA 与间隔更新）"""
        # Warm-up 阶段：前 N 个 epoch 不更新 Teacher
        warmup_epochs = getattr(self.args, 'teacher_warmup_epochs', 3)
        if epoch is not None and epoch < warmup_epochs:
            return  # 不更新 Teacher，让 Student 先学习

        self.optimizer_step_count += 1
        if self.optimizer_step_count % self.ema_update_interval != 0:
            return
        
        # 动态调整 EMA decay（训练初期更新更快）或使用固定慢 EMA
        if epoch is not None:
            ema_decay = self._get_ema_decay(epoch)
        else:
            ema_decay = self.ema_decay
        
        with torch.no_grad():
            if self.use_multi_gpu:
                teacher_module = self.teacher.module
                sam_module = self.sam.module
                for t_param, s_param in zip(teacher_module.parameters(), sam_module.parameters()):
                    t_param.data.mul_(ema_decay).add_(s_param.data * (1.0 - ema_decay))
            else:
                for t_param, s_param in zip(self.teacher.parameters(), self.sam.parameters()):
                    t_param.data.mul_(ema_decay).add_(s_param.data * (1.0 - ema_decay))
    
    def _get_ema_decay(self, epoch):
        """动态调整 EMA decay（减缓 Teacher 更新以缓解过分割：TS 过高时避免强化大 mask）"""
        if self.ema_fixed_decay and self.ema_fixed_decay > 0:
            return self.ema_fixed_decay

        total_epochs = self.args.epochs
        progress = epoch / total_epochs if total_epochs > 0 else 0.0
        
        if progress < 0.3:
            return 0.97
        elif progress < 0.6:
            return 0.98
        else:
            return 0.998

    def _compute_ts_consistency(self, teacher_logits, student_logits, boxes_list_sam):
        """
        计算更有意义的 TS 一致性：
        在大框区域内统计 teacher/student 前景 IoU，而不是全图像素一致率。
        """
        teacher_bin = (torch.sigmoid(teacher_logits) > 0.5)
        student_bin = (torch.sigmoid(student_logits) > 0.5)
        B, _, H, W = teacher_bin.shape
        scores = []
        for b in range(B):
            bx1, by1, bx2, by2 = boxes_list_sam[b]
            sx = W / self.sam_input_size
            sy = H / self.sam_input_size
            x1 = max(0, min(W, int(bx1 * sx)))
            y1 = max(0, min(H, int(by1 * sy)))
            x2 = max(x1 + 1, min(W, int(bx2 * sx)))
            y2 = max(y1 + 1, min(H, int(by2 * sy)))
            t_box = teacher_bin[b:b+1, :, y1:y2, x1:x2]
            s_box = student_bin[b:b+1, :, y1:y2, x1:x2]
            inter = (t_box & s_box).float().sum()
            union = (t_box | s_box).float().sum()
            if union > 0:
                scores.append((inter / (union + 1e-6)).item())
            else:
                scores.append((t_box == s_box).float().mean().item())
        return float(np.mean(scores)) if len(scores) > 0 else 0.0

    def _run_validation(self):
        """
        在验证集上评估当前 Student 的 mIoU。
        若未提供验证集，则返回 None。
        """
        if self.val_loader is None:
            return None

        self.sam.eval()
        total_iou = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                boxes_list = batch['boxes']
                small_boxes_list = batch.get('small_boxes', None)
                gt_masks = batch['mask'].to(self.device)

                images_sam, boxes_list_sam, small_boxes_list_sam = self._prepare_images_and_boxes(
                    images, boxes_list, small_boxes_list
                )
                img_emb, pred_logits, _ = self._student_forward(
                    images_sam, boxes_list_sam, small_boxes_list_sam
                )
                del img_emb

                gt_resized = F.interpolate(gt_masks, size=pred_logits.shape[-2:], mode='nearest')
                batch_iou = compute_iou_per_sample(pred_logits, gt_resized)
                total_iou += batch_iou.sum().item()
                total_samples += batch_iou.numel()

        self.sam.train()
        return total_iou / max(total_samples, 1)
    
    def _save_checkpoint(self, epoch, filename=None):
        """ä¿å­ checkpoint"""
        if self.use_multi_gpu:
            sam_encoder_state = self.sam.module.image_encoder.state_dict()
            proj_state = self.proj.module.state_dict()
        else:
            sam_encoder_state = self.sam.image_encoder.state_dict()
            proj_state = self.proj.state_dict()
        
        ckpt = {
            'sam_image_encoder': sam_encoder_state,
            'proj': proj_state,
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch
        }
        if filename is not None:
            ckpt['best_miou'] = self.best_miou
            path = os.path.join(self.args.output_dir, filename)
        else:
            path = os.path.join(self.args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(ckpt, path)
        print(f"[INFO] Saved checkpoint: {os.path.basename(path)}")
    
    def train(self):
        for epoch in range(self.args.epochs):
            self.sam.train()
            self.proj.train()
            
            # 获取当前epoch的mIoU和Teacher-Student一致性（用于动态权重调整）
            current_miou = self.history_miou[-1] if len(self.history_miou) > 0 else None
            
            # 计算Teacher-Student一致性（在epoch开始时使用上一epoch的值，epoch结束后更新）
            ts_consistency = self.history_ts_consistency[-1] if len(self.history_ts_consistency) > 0 else None
            
            # 打印当前epoch的自适应策略信息（使用动态权重）
            inter_weight, union_weight, hard_weight, strategy = self._get_adaptive_weights(epoch, current_miou, ts_consistency)
            if epoch == 0 or epoch % max(1, self.args.epochs // 10) == 0:
                print(f"\n[INFO] Epoch {epoch+1}/{self.args.epochs} - 自适应并集交集策略: {strategy}")
                print(f"[INFO]   交集权重: {inter_weight:.2f}, 并集权重: {union_weight:.2f}, 困难样本权重: {hard_weight:.2f}")
                if current_miou is not None:
                    print(f"[INFO]   当前mIoU: {current_miou:.4f}, 最佳mIoU: {self.best_miou:.4f}")
                if ts_consistency is not None:
                    print(f"[INFO]   Teacher-Student一致性: {ts_consistency:.4f}")
            
            total_loss_m = 0.0
            total_loss_c = 0.0
            total_loss_iou = 0.0
            total_loss_dice_pseudo = 0.0
            total_miou = 0.0
            
            # 统计正负样本数量和Teacher-Student一致性
            total_pos_count = 0
            total_neg_count = 0
            batch_count = 0
            total_ts_consistency = 0.0  # 用于计算平均TS一致性
            
            # 重置epoch统计
            self.epoch_skipped_batches = 0
            self.epoch_low_quality_pos_batches = 0
            self.epoch_total_batches = 0
            
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.args.epochs}')
            
            for it, batch in enumerate(pbar):
                self.epoch_total_batches += 1
                images = batch['image'].to(self.device)
                boxes_list = batch['boxes']
                small_boxes_list = batch.get('small_boxes', None)
                gt_masks = batch['mask'].to(self.device)
                
                # åå¤å¾ååæ¡
                images_sam, boxes_list_sam, small_boxes_list_sam = self._prepare_images_and_boxes(
                    images, boxes_list, small_boxes_list
                )
                
                # Teacher çæä¼ªæ ç­?
                mask_logits_stack, mask_entropy_vals, teacher_img_emb = self._teacher_forward(
                    images_sam, boxes_list_sam, small_boxes_list_sam
                )
                
                # Student ååä¼ æ­
                images_student = self._augment_images_student(images_sam)
                img_emb, pred_logits, pred_iou = self._student_forward(
                    images_student, boxes_list_sam, small_boxes_list_sam
                )
                
                # 获取小框四个角点（如果存在）
                small_box_corners_list = batch.get('small_box_corners', None)
                
                # 计算Teacher-Student一致性（用于动态权重调整）
                with torch.no_grad():
                    total_ts_consistency += self._compute_ts_consistency(
                        mask_logits_stack, pred_logits, boxes_list_sam
                    )
                
                # è®¡ç®æå¤±（传入epoch和动态权重用于自适应并集交集策略）
                loss, loss_mask, loss_contrast, loss_iou, loss_distill, loss_dice_pseudo, batch_miou, iou_per_sample, pos_count, neg_count = self._compute_losses(
                    pred_logits, pred_iou, gt_masks, img_emb, teacher_img_emb,
                    mask_logits_stack, mask_entropy_vals, boxes_list_sam, small_boxes_list_sam, images_sam,
                    small_box_corners_list=small_box_corners_list, epoch=epoch
                )
                # å½ä¸åæå¤±ï¼æ¢¯åº¦ç´¯ç§¯ï¼?
                loss = loss / self.args.gradient_accumulation_steps
                
                # ååä¼ æ­
                if self.args.use_amp and self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # æ¢¯åº¦æ´æ°
                if (it + 1) % self.args.gradient_accumulation_steps == 0:
                    trainable_params = list(self.proj.module.parameters() if self.use_multi_gpu else self.proj.parameters())
                    sam_encoder = self.sam.module.image_encoder if self.use_multi_gpu else self.sam.image_encoder
                    for p in sam_encoder.parameters():
                        if p.requires_grad:
                            trainable_params.append(p)
                    
                    if self.args.use_amp and self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # æ´æ° EMA teacher
                    self._update_ema_teacher(epoch)
                
                # ç´¯è®¡ç»è®¡
                total_loss_m += loss_mask.item() if isinstance(loss_mask, torch.Tensor) else float(loss_mask)
                total_loss_c += loss_contrast.item() if isinstance(loss_contrast, torch.Tensor) else float(loss_contrast)
                total_loss_iou += loss_iou.item() if isinstance(loss_iou, torch.Tensor) else float(loss_iou)
                total_loss_dice_pseudo += loss_dice_pseudo.item() if isinstance(loss_dice_pseudo, torch.Tensor) else float(loss_dice_pseudo)
                total_miou += batch_miou
                
                # 统计正负样本数量
                total_pos_count += pos_count
                total_neg_count += neg_count
                batch_count += 1
                
                # 检测并保存表现差的样本（每个batch实时保存）
                if self.detect_poor_samples and iou_per_sample is not None:
                    # 根据poor_sample_save_interval决定是否保存（每N个epoch保存一次）
                    if (epoch + 1) % self.poor_sample_save_interval == 0:
                        self._detect_and_save_poor_samples(
                            batch=batch,
                            pred_logits=pred_logits,
                            gt_masks=gt_masks,
                            iou_per_sample=iou_per_sample,
                            boxes_list=boxes_list,
                            small_boxes_list=small_boxes_list,
                            images=images,
                            epoch=epoch,
                            it=it
                        )
                
                # æ¾ç¤ºè¿åº¦
                display_loss = loss.item() * self.args.gradient_accumulation_steps
                mem_info = ""
                if torch.cuda.is_available() and (it + 1) % 10 == 0:
                    device_id = self.device.index if hasattr(self.device, 'index') else 0
                    allocated = torch.cuda.memory_allocated(device_id) / 1024**3
                    reserved = torch.cuda.memory_reserved(device_id) / 1024**3
                    mem_info = f" GPU:{allocated:.2f}/{reserved:.2f}GB"
                
                desc = f"E{epoch+1} L={display_loss:.4f} mask={float(loss_mask):.4f} cont={float(loss_contrast):.4f}"
                if self.pseudo_dice_weight > 0:
                    desc += f" dice_p={float(loss_dice_pseudo):.4f}"
                desc += f" mIOU={batch_miou:.4f}{mem_info}"
                pbar.set_description(desc)
                
                # æ¸çç¼å­
                if torch.cuda.is_available():
                    del loss, loss_mask, loss_contrast, loss_iou, loss_distill, loss_dice_pseudo, teacher_img_emb
                    if (it + 1) % 5 == 0:
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
            
            # å­¦ä¹ çè°åº?
            self.scheduler.step()

            val_miou = self._run_validation()
            
            # ä¿å­ checkpoint
            if (epoch + 1) % self.args.save_every == 0:
                self._save_checkpoint(epoch)
            
            # æå° epoch ç»è®¡
            avg_miou = total_miou / len(self.train_loader)
            avg_ts_consistency = total_ts_consistency / batch_count if batch_count > 0 else 0.0
            
            model_selection_miou = val_miou if val_miou is not None else avg_miou

            # 记录历史mIoU和TS一致性（用于下一epoch的动态权重调整）
            self.history_miou.append(model_selection_miou)
            self.history_ts_consistency.append(avg_ts_consistency)
            
            # 更新最佳mIoU，并可选保存最佳 checkpoint（调优建议：便于评估与部署）
            if model_selection_miou > self.best_miou:
                self.best_miou = model_selection_miou
                self.best_miou_epoch = epoch + 1
                if getattr(self.args, 'save_best_ckpt', False):
                    self._save_checkpoint(epoch, 'best_checkpoint.pth')
            
            avg_pos_count = total_pos_count / batch_count if batch_count > 0 else 0
            avg_neg_count = total_neg_count / batch_count if batch_count > 0 else 0
            pos_neg_ratio = avg_pos_count / (avg_neg_count + 1e-6)
            target_ratio = getattr(self.args, 'pos_neg_ratio', 0.25)
            
            # avg_mIOU 用 .5f 便于观察小幅提升（.4f 时 0.7578x 会长时间显示为 0.7578）
            n_loader = len(self.train_loader)
            print(f"Epoch {epoch+1} avg_mask_loss={total_loss_m/n_loader:.4f} "
                  f"avg_contrast={total_loss_c/n_loader:.4f} "
                  f"avg_iou_loss={total_loss_iou/n_loader:.4f} "
                  f"train_mIOU={avg_miou:.5f} "
                  f"{'val_mIOU=' + format(val_miou, '.5f') if val_miou is not None else 'val_mIOU=N/A'} "
                  f"(最佳: {self.best_miou:.5f}) "
                  f"TS一致性(Box-FgIoU)={avg_ts_consistency:.4f}")
            if self.pseudo_dice_weight > 0:
                print(f"  avg_dice_pseudo={total_loss_dice_pseudo/n_loader:.4f} (weight={self.pseudo_dice_weight})")
            print(f"  正负样本统计: 平均正样本={avg_pos_count:.1f}, 平均负样本={avg_neg_count:.1f}, "
                  f"正负比例={pos_neg_ratio:.3f} (目标比例≈{target_ratio:.3f}, 即1:{1/target_ratio:.1f})")
            
            # 如果比例失衡，给出警告
            if pos_neg_ratio > target_ratio * 2.0:
                print(f"  [WARN] 正样本过多！当前比例 {pos_neg_ratio:.3f} 远高于目标比例 {target_ratio:.3f}")
            elif pos_neg_ratio < target_ratio * 0.5:
                print(f"  [WARN] 负样本过多！当前比例 {pos_neg_ratio:.3f} 远低于目标比例 {target_ratio:.3f}")
            
            # 样本选择质量统计报告
            if self.epoch_total_batches > 0:
                skip_ratio = self.epoch_skipped_batches / self.epoch_total_batches
                if skip_ratio > 0.05:  # 如果跳过超过5%的batch
                    print(f"  [WARN] 样本选择警告: 跳过了 {self.epoch_skipped_batches}/{self.epoch_total_batches} batches ({skip_ratio*100:.1f}%)")
                    print(f"         这可能导致训练信号不足，建议检查：")
                    print(f"         1. Teacher模型预测质量")
                    print(f"         2. 小框标注准确性")
                    print(f"         3. 自适应权重策略设置")
                elif skip_ratio > 0:
                    print(f"  [INFO] 样本选择: 跳过了 {self.epoch_skipped_batches}/{self.epoch_total_batches} batches ({skip_ratio*100:.1f}%)")
            
            # 早停：连续 N 个 epoch 无 mIOU 提升则停止（调优建议：patience=5，见 docs/训练日志分析与调优建议.md）
            patience = getattr(self.args, 'early_stop_patience', 0)
            if patience > 0 and self.best_miou_epoch > 0:
                if (epoch + 1) - self.best_miou_epoch >= patience:
                    print(f"[INFO] Early stopping: no mIOU improvement for {patience} epochs (best at epoch {self.best_miou_epoch})")
                    break
            
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                device_id = self.device.index if hasattr(self.device, 'index') else 0
                allocated = torch.cuda.memory_allocated(device_id) / 1024**3
                print(f"[INFO] Epoch {epoch+1} 结束后显存使用: {allocated:.2f}GB")
        
        # 保存最终的表现差样本CSV
        if self.detect_poor_samples and len(self.poor_samples_data) > 0:
            self._save_poor_samples_csv()
        
        print('Training finished!')
    
    def _detect_and_save_poor_samples(self, batch, pred_logits, gt_masks, iou_per_sample, 
                                     boxes_list, small_boxes_list, images, epoch, it):
        """
        检测并保存表现差的样本（在原始图像上可视化）
        
        Args:
            batch: 数据batch
            pred_logits: 预测logits (B, 1, H, W) - 512x512尺寸
            gt_masks: 真实masks (B, 1, H, W) - 512x512尺寸
            iou_per_sample: 每个样本的IoU (B,)
            boxes_list: 大框列表（512x512坐标系）
            small_boxes_list: 小框列表（512x512坐标系）
            images: 图像 (B, 3, H, W) - 512x512尺寸
            epoch: 当前epoch
            it: 当前iteration
        """
        B = pred_logits.shape[0]
        img_names = batch.get('img_names', [f'img_epoch{epoch}_it{it}_{i}' for i in range(B)])
        
        # 创建epoch目录
        epoch_vis_dir = os.path.join(self.args.output_dir, f'poor_samples_epoch_{epoch+1}', 'visualization')
        mkdir(epoch_vis_dir)
        
        # 确定原始图像目录（优先使用update_ISIC目录，如果不存在则使用data_root）
        # 检查update_ISIC目录是否存在（用户期望在此目录下找到原始图像）
        data_root_parent = os.path.dirname(self.args.data_root)
        update_isic_dir = os.path.join(data_root_parent, 'update_ISIC', 'ISBI2016_ISIC_Part1_Training_Data')
        if os.path.exists(update_isic_dir):
            original_img_dir = update_isic_dir
        else:
            # 回退到data_root下的目录
            original_img_dir = os.path.join(self.args.data_root, 'ISBI2016_ISIC_Part1_Training_Data')
        
        for b in range(B):
            iou_value = iou_per_sample[b].item()
            
            # 多级分类：<0.75严重, 0.75-0.8中等, >=0.8合理（不保存）
            if iou_value < self.poor_sample_iou_threshold_moderate:
                pred_mask = pred_logits[b].squeeze(0).cpu()  # (H, W) - 512x512
                gt_mask = gt_masks[b].squeeze(0).cpu()  # (H, W) - 512x512
                big_box = boxes_list[b]  # 512x512坐标系
                small_box = small_boxes_list[b] if small_boxes_list and small_boxes_list[b] is not None else None
                img_name = img_names[b] if b < len(img_names) else f'img_epoch{epoch}_it{it}_{b}'
                small_box_corners = None
                if 'small_box_corners' in batch and batch['small_box_corners'] is not None:
                    small_box_corners = batch['small_box_corners'][b] if b < len(batch['small_box_corners']) else None
                
                # 清理文件名
                img_name_base = os.path.splitext(os.path.basename(str(img_name)))[0]
                img_name_ext = os.path.splitext(os.path.basename(str(img_name)))[1] or '.jpg'
                
                # 确定严重程度
                if iou_value < self.poor_sample_iou_threshold_severe:
                    severity = 'severe'  # 严重
                else:
                    severity = 'moderate'  # 中等
                
                # 确保box是list格式
                if isinstance(big_box, torch.Tensor):
                    big_box = big_box.tolist()
                if small_box is not None and isinstance(small_box, torch.Tensor):
                    small_box = small_box.tolist()
                
                # 构建原始图像路径
                original_img_path = os.path.join(original_img_dir, img_name)
                
                # 保存可视化（按严重程度分类保存）
                severity_dir = os.path.join(epoch_vis_dir, severity)
                mkdir(severity_dir)
                # 保存的文件名和原始图像文件名一致，只加IoU后缀
                vis_path = os.path.join(severity_dir, f'{img_name_base}_iou{iou_value:.4f}{img_name_ext}')
                
                try:
                    # 在原始图像上可视化（坐标会自动映射回原始尺寸）
                    visualize_prediction_on_original_image(
                        original_img_path=original_img_path,
                        pred_mask_512=pred_mask,
                        gt_mask_512=gt_mask,
                        big_box_512=big_box,
                        small_box_512=small_box,
                        save_path=vis_path,
                        img_name=img_name_base,
                        iou_value=iou_value,
                        small_box_corners_512=small_box_corners,  # 传递四个角点（512x512坐标系）
                        img_size=self.args.img_size
                    )
                    
                    # 记录样本信息
                    sample_info = {
                        'epoch': epoch + 1,
                        'iteration': it,
                        'image_file': img_name,
                        'iou': iou_value,
                        'severity': severity,  # 严重程度：severe或moderate
                        'big_box_x1': big_box[0],
                        'big_box_y1': big_box[1],
                        'big_box_x2': big_box[2],
                        'big_box_y2': big_box[3],
                    }
                    
                    if small_box is not None:
                        sample_info.update({
                            'small_box_x1': small_box[0],
                            'small_box_y1': small_box[1],
                            'small_box_x2': small_box[2],
                            'small_box_y2': small_box[3],
                        })
                    else:
                        sample_info.update({
                            'small_box_x1': None,
                            'small_box_y1': None,
                            'small_box_x2': None,
                            'small_box_y2': None,
                        })
                    
                    self.poor_samples_data.append(sample_info)
                except Exception as e:
                    print(f"[WARN] 保存可视化失败 {img_name}: {e}")
    
    def _save_poor_samples_csv(self):
        """保存表现差样本的CSV文件"""
        if not self.poor_samples_data:
            return
        
        csv_dir = os.path.join(self.args.output_dir, 'poor_samples_csv')
        mkdir(csv_dir)
        
        df = pd.DataFrame(self.poor_samples_data)
        csv_path = os.path.join(csv_dir, 'poor_samples_all_epochs.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"\n[INFO] 训练过程中共发现 {len(self.poor_samples_data)} 个表现差的样本")
        print(f"[INFO] CSV文件已保存到: {csv_path}")
        print(f"[INFO] 平均IoU: {df['iou'].mean():.4f}, 最小IoU: {df['iou'].min():.4f}, 最大IoU: {df['iou'].max():.4f}")
        
        # 按严重程度统计
        if 'severity' in df.columns:
            severe_count = len(df[df['severity'] == 'severe'])
            moderate_count = len(df[df['severity'] == 'moderate'])
            print(f"[INFO] 严重样本(IoU<{self.poor_sample_iou_threshold_severe}): {severe_count} 个")
            print(f"[INFO] 中等样本(IoU {self.poor_sample_iou_threshold_severe}-{self.poor_sample_iou_threshold_moderate}): {moderate_count} 个")
        
        # 按epoch保存
        for epoch in df['epoch'].unique():
            epoch_df = df[df['epoch'] == epoch]
            epoch_csv_path = os.path.join(csv_dir, f'poor_samples_epoch_{epoch}.csv')
            epoch_df.to_csv(epoch_csv_path, index=False)

