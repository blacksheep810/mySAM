#!/usr/bin/env python
# coding: utf-8

"""
è®­ç»ä¸»é»è¾æ¨¡å
å®ç°å®æ´çè®­ç»æµç¨ï¼åæ¬è®­ç»å¾ªç¯ãæå¤±è®¡ç®ãcheckpoint ä¿å­ç­?
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset.ISIC import ISIC2016Dataset
from models.sam_wrapper import (
    load_sam_model, create_teacher_model, get_encoder_feature_dim, setup_multi_gpu
)
from models.projection import PixelProjHead
from models.losses import DiceLoss, pixel_info_nce
from utils.metrics import compute_miou, mask_entropy_logits
from utils.prompts import prepare_box_prompts
from utils.training_utils import set_seed, mkdir, setup_cuda_memory
from training.data_utils import collate_fn_isic


class Trainer:
    """
    è®­ç»å¨ç±»
    ç®¡çæ´ä¸ªè®­ç»æµç¨
    """
    
    def __init__(self, args):
        """
        åå§åè®­ç»å¨
        
        Args:
            args: éç½®åæ°
        """
        self.args = args
        self.sam_input_size = 1024  # SAM é¢è®­ç»å°ºå¯?
        
        # è®¾ç½®éæºç§å­ååå»ºè¾åºç®å½?
        set_seed(42)
        mkdir(args.output_dir)
        
        # è®¾ç½® CUDA åå­
        device_id = 0 if args.device == 'cuda' else int(args.device.split(':')[1]) if ':' in args.device else 0
        mem_info = setup_cuda_memory(device_id)
        if mem_info:
            print(f"[INFO] GPU {mem_info['device_id']} åå§æ¾å­ç¶æ?: "
                  f"å·²åé?={mem_info['allocated']:.2f}GB, "
                  f"å·²ä¿ç?={mem_info['reserved']:.2f}GB, "
                  f"æ»è®¡={mem_info['total']:.2f}GB")
            if mem_info['reserved'] > 0.1:
                print(f"[WARN] æ£æµå° {mem_info['reserved']:.2f}GB æ¾å­è¢«ä¿ç?")
        
        # è®¾ç½®è®¾å¤åå¤GPU
        self.device, self.use_multi_gpu, self.device_ids = self._setup_device()
        
        # å è½½æ¨¡å
        self.sam, self.teacher, self.proj = self._setup_models()
        
        # è®¾ç½®ä¼åå¨åè°åº¦å?
        self.optimizer, self.scheduler, self.scaler = self._setup_optimizer()
        
        # è®¾ç½®æå¤±å½æ°
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        
        # å è½½æ°æ®é?
        self.train_loader = self._setup_dataloader()
        
        # EMA åæ°
        self.ema_decay = 0.999
    
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
    
    def _setup_models(self):
        """å è½½åéç½®æ¨¡å?"""
        # å è½½ SAM æ¨¡åï¼studentï¼?
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
        """è®¾ç½®ä¼åå¨åå­¦ä¹ çè°åº¦å¨"""
        # æ¶éå¯è®­ç»åæ?
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
    
    def _setup_dataloader(self):
        """è®¾ç½®æ°æ®å è½½å?"""
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
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn_isic
        )
        
        return train_loader
    
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
                
                # åå¤æ¡? prompt
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
            
            # æ¸çä¸­é´åé
            del t_img_emb, all_mask_logits, all_mask_entropy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return mask_logits_stack, mask_entropy_vals
    
    def _student_forward(self, images_sam, boxes_list_sam, small_boxes_list_sam):
        """Student ååä¼ æ­"""
        # ä¿å­åèç¹å¾ï¼ç¨äºè¸é¦ï¼?
        with torch.no_grad():
            sam_encoder_ref = self.sam.module.image_encoder if self.use_multi_gpu else self.sam.image_encoder
            ref_feats = sam_encoder_ref(images_sam).detach()
        
        # Student encoder ååï¼å¯è®­ç»ï¼?
        sam_encoder = self.sam.module.image_encoder if self.use_multi_gpu else self.sam.image_encoder
        if self.args.use_amp and self.scaler is not None:
            with torch.cuda.amp.autocast():
                img_emb = sam_encoder(images_sam)
        else:
            img_emb = sam_encoder(images_sam)
        
        # Student decoder ååï¼å»ç»ï¼
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
                        # Èç¹ûÊäÈëÁË¶à¸ö¿ò£¬SAM ¿ÉÄÜÎªÃ¿¸ö¿òÉú³ÉÒ»¸ö mask
                        # È·±£Ö»Ê¹ÓÃµÚÒ»¸ö mask£¨batch_size=1£©
                        if low_res_masks.shape[0] > 1:
                            low_res_masks = low_res_masks[0:1]  # Ö»È¡µÚÒ»¸ö
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
                        # È·±£ batch_size=1
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
        
        return img_emb, ref_feats, pred_logits, pred_iou
    
    def _compute_losses(self, pred_logits, pred_iou, gt_masks, img_emb, ref_feats, 
                        mask_logits_stack, mask_entropy_vals, boxes_list_sam, small_boxes_list_sam, images_sam):
        """è®¡ç®æææå¤?"""
        device = self.device
        
        # 1) Mask lossï¼ä»ç¨äºçæ§ï¼?
        loss_mask = torch.tensor(0.0, device=device)
        batch_miou = 0.0
        if gt_masks is not None:
            with torch.no_grad():
                gt_resized = F.interpolate(gt_masks, size=pred_logits.shape[-2:], mode='nearest')
                loss_bce = self.bce_loss(pred_logits, gt_resized)
                loss_dice = self.dice_loss(pred_logits, gt_resized)
                loss_mask = loss_bce + loss_dice
                batch_miou = compute_miou(pred_logits, gt_resized)
        
        # 2) Contrastive loss
        loss_contrast = self._compute_contrastive_loss(
            img_emb, mask_logits_stack, mask_entropy_vals, 
            boxes_list_sam, small_boxes_list_sam, images_sam, ref_feats
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
            r_feat = ref_feats
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
        alpha, beta, gamma = 1.0, 0.5, 0.1
        loss = loss_mask + alpha * loss_contrast + beta * loss_iou + gamma * loss_distill
        
        return loss, loss_mask, loss_contrast, loss_iou, loss_distill, batch_miou
    
    def _compute_contrastive_loss(self, img_emb, mask_logits_stack, mask_entropy_vals, 
                                  boxes_list_sam, small_boxes_list_sam, images_sam, ref_feats):
        """è®¡ç®å¯¹æ¯å­¦ä¹ æå¤±"""
        device = self.device
        
        # ç½®ä¿¡åº¦ç­é?
        trusted_idx = (mask_entropy_vals <= self.args.entropy_thresh).nonzero(as_tuple=False).squeeze(-1).tolist()
        
        if len(trusted_idx) == 0:
            return torch.tensor(0.0, device=device)
        
        # æå½±ç¹å¾
        if self.args.use_amp and self.scaler is not None:
            with torch.cuda.amp.autocast():
                z = self.proj(img_emb)
        else:
            z = self.proj(img_emb)
        
        Bz, D, Hf, Wf = z.shape
        
        # åå¤ mask
        teacher_mask_resized = F.interpolate(
            torch.sigmoid(mask_logits_stack), 
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
            
            small_box_mask = torch.zeros(Hf, Wf, device=device, dtype=torch.float32)
            small_box_mask[sy1_f:sy2_f+1, sx1_f:sx2_f+1] = 1.0
            
            big_box_mask = torch.zeros(Hf, Wf, device=device, dtype=torch.float32)
            big_box_mask[by1_f:by2_f+1, bx1_f:bx2_f+1] = 1.0
            
            tmask_resized = teacher_mask_resized[b].squeeze(0).squeeze(0)
            
            # æ­£æ ·æ?
            salient_pos_mask = (tmask_resized > 0.5) & (small_box_mask > 0.5)
            pos_idx = salient_pos_mask.view(-1).nonzero(as_tuple=False).squeeze(-1)
            
            if pos_idx.numel() == 0:
                continue
            
            npos = min(pos_idx.numel(), self.args.pos_samples)
            perm = torch.randperm(pos_idx.numel(), device=device)[:npos]
            chosen_pos = pos_idx[perm]
            
            z_b = z[b].permute(1, 2, 0).reshape(-1, D)
            anchors_list.append(z_b[chosen_pos])
            
            # Positive: teacher ç¹å¾
            with torch.no_grad():
                try:
                    teacher_encoder_pos = self.teacher.module.image_encoder if self.use_multi_gpu else self.teacher.image_encoder
                    t_img_emb_b = teacher_encoder_pos(images_sam[b:b+1])
                    t_dense = t_img_emb_b
                except Exception:
                    t_dense = ref_feats[b:b+1] if ref_feats is not None else None
                
                if t_dense is not None:
                    tproj = self.proj(t_dense)
                    tproj_flat = tproj.permute(0, 2, 3, 1).reshape(-1, D)
                    positives_list.append(tproj_flat[chosen_pos].detach())
            
            # å°é¾è´æ ·æ?
            hard_neg_list = []
            
            missed_in_small = (small_box_mask > 0.5) & (tmask_resized < 0.5)
            missed_idx = missed_in_small.view(-1).nonzero(as_tuple=False).squeeze(-1)
            if missed_idx.numel() > 0:
                n_missed = min(missed_idx.numel(), self.args.neg_samples // 2)
                perm_missed = torch.randperm(missed_idx.numel(), device=device)[:n_missed]
                hard_neg_list.append(z_b[missed_idx[perm_missed]].detach())
            
            overflow_out_big = (big_box_mask < 0.5) & (tmask_resized > 0.5)
            overflow_idx = overflow_out_big.view(-1).nonzero(as_tuple=False).squeeze(-1)
            if overflow_idx.numel() > 0:
                n_overflow = min(overflow_idx.numel(), self.args.neg_samples // 2)
                perm_overflow = torch.randperm(overflow_idx.numel(), device=device)[:n_overflow]
                hard_neg_list.append(z_b[overflow_idx[perm_overflow]].detach())
            
            if len(hard_neg_list) > 0:
                hard_negs = torch.cat(hard_neg_list, dim=0)
            else:
                all_pixels_b = z_b.detach()
                n_random = min(all_pixels_b.shape[0], self.args.neg_samples)
                perm_random = torch.randperm(all_pixels_b.shape[0], device=device)[:n_random]
                hard_negs = all_pixels_b[perm_random]
            
            negatives_pool.append(hard_negs)
        
        if len(anchors_list) == 0:
            return torch.tensor(0.0, device=device)
        
        anchors = torch.cat(anchors_list, dim=0)
        positives = torch.cat(positives_list, dim=0)
        
        if len(negatives_pool) > 0:
            all_hard_negs = torch.cat(negatives_pool, dim=0)
            max_neg = min(all_hard_negs.shape[0], self.args.neg_samples * len(anchors_list))
            if max_neg > 0:
                perm_neg = torch.randperm(all_hard_negs.shape[0], device=device)[:max_neg]
                negatives = all_hard_negs[perm_neg]
            else:
                all_pixels = z.permute(0, 2, 3, 1).reshape(-1, D).detach()
                n_random = min(all_pixels.shape[0], self.args.neg_samples * len(anchors_list))
                perm_random = torch.randperm(all_pixels.shape[0], device=device)[:n_random]
                negatives = all_pixels[perm_random]
        else:
            all_pixels = z.permute(0, 2, 3, 1).reshape(-1, D).detach()
            n_random = min(all_pixels.shape[0], self.args.neg_samples * len(anchors_list))
            perm_random = torch.randperm(all_pixels.shape[0], device=device)[:n_random]
            negatives = all_pixels[perm_random]
        
        loss_contrast = pixel_info_nce(anchors, positives, negatives, temperature=self.args.temperature)
        return loss_contrast
    
    def _update_ema_teacher(self):
        """æ´æ° EMA teacher"""
        with torch.no_grad():
            if self.use_multi_gpu:
                teacher_module = self.teacher.module
                sam_module = self.sam.module
                for t_param, s_param in zip(teacher_module.parameters(), sam_module.parameters()):
                    t_param.data.mul_(self.ema_decay).add_(s_param.data * (1.0 - self.ema_decay))
            else:
                for t_param, s_param in zip(self.teacher.parameters(), self.sam.parameters()):
                    t_param.data.mul_(self.ema_decay).add_(s_param.data * (1.0 - self.ema_decay))
    
    def _save_checkpoint(self, epoch):
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
        torch.save(ckpt, os.path.join(self.args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        print(f"[INFO] Saved checkpoint: checkpoint_epoch_{epoch+1}.pth")
    
    def train(self):
        """ä¸»è®­ç»å¾ªç?"""
        for epoch in range(self.args.epochs):
            self.sam.train()
            self.proj.train()
            
            total_loss_m = 0.0
            total_loss_c = 0.0
            total_loss_iou = 0.0
            total_miou = 0.0
            
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.args.epochs}')
            
            for it, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                boxes_list = batch['boxes']
                small_boxes_list = batch.get('small_boxes', None)
                gt_masks = batch['mask'].to(self.device)
                
                # åå¤å¾ååæ¡
                images_sam, boxes_list_sam, small_boxes_list_sam = self._prepare_images_and_boxes(
                    images, boxes_list, small_boxes_list
                )
                
                # Teacher çæä¼ªæ ç­?
                mask_logits_stack, mask_entropy_vals = self._teacher_forward(
                    images_sam, boxes_list_sam, small_boxes_list_sam
                )
                
                # Student ååä¼ æ­
                img_emb, ref_feats, pred_logits, pred_iou = self._student_forward(
                    images_sam, boxes_list_sam, small_boxes_list_sam
                )
                
                # è®¡ç®æå¤±
                loss, loss_mask, loss_contrast, loss_iou, loss_distill, batch_miou = self._compute_losses(
                    pred_logits, pred_iou, gt_masks, img_emb, ref_feats,
                    mask_logits_stack, mask_entropy_vals, boxes_list_sam, small_boxes_list_sam, images_sam
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
                    self._update_ema_teacher()
                
                # ç´¯è®¡ç»è®¡
                total_loss_m += loss_mask.item() if isinstance(loss_mask, torch.Tensor) else float(loss_mask)
                total_loss_c += loss_contrast.item() if isinstance(loss_contrast, torch.Tensor) else float(loss_contrast)
                total_loss_iou += loss_iou.item() if isinstance(loss_iou, torch.Tensor) else float(loss_iou)
                total_miou += batch_miou
                
                # æ¾ç¤ºè¿åº¦
                display_loss = loss.item() * self.args.gradient_accumulation_steps
                mem_info = ""
                if torch.cuda.is_available() and (it + 1) % 10 == 0:
                    device_id = self.device.index if hasattr(self.device, 'index') else 0
                    allocated = torch.cuda.memory_allocated(device_id) / 1024**3
                    reserved = torch.cuda.memory_reserved(device_id) / 1024**3
                    mem_info = f" GPU:{allocated:.2f}/{reserved:.2f}GB"
                
                pbar.set_description(
                    f"E{epoch+1} L={display_loss:.4f} mask={float(loss_mask):.4f} "
                    f"cont={float(loss_contrast):.4f} iou={float(loss_iou):.4f} "
                    f"mIOU={batch_miou:.4f}{mem_info}"
                )
                
                # æ¸çç¼å­
                if torch.cuda.is_available():
                    del loss, loss_mask, loss_contrast, loss_iou, loss_distill
                    if (it + 1) % 5 == 0:
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
            
            # å­¦ä¹ çè°åº?
            self.scheduler.step()
            
            # ä¿å­ checkpoint
            if (epoch + 1) % self.args.save_every == 0:
                self._save_checkpoint(epoch)
            
            # æå° epoch ç»è®¡
            avg_miou = total_miou / len(self.train_loader)
            print(f"Epoch {epoch+1} avg_mask_loss={total_loss_m/len(self.train_loader):.4f} "
                  f"avg_contrast={total_loss_c/len(self.train_loader):.4f} "
                  f"avg_iou={total_loss_iou/len(self.train_loader):.4f} "
                  f"avg_mIOU={avg_miou:.4f}")
            
            # æ¸çæ¾å­
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                device_id = self.device.index if hasattr(self.device, 'index') else 0
                allocated = torch.cuda.memory_allocated(device_id) / 1024**3
                print(f"[INFO] Epoch {epoch+1} ç»æåæ¾å­ä½¿ç?: {allocated:.2f}GB")
        
        print('Training finished!')

