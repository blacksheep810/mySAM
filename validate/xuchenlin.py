import os
import torch
import lightning as L
import segmentation_models_pytorch as smp
from box import Box
from torch.utils.data import DataLoader
from model import Model
from utils.sample_utils import get_point_prompts
from utils.tools import write_csv
import os
import numpy as np
from PIL import Image

def calculate_dice_torch(gt_mask: torch.Tensor, pred_mask: torch.Tensor, smooth=1e-8):
    """
    自定义Dice，输入都是float二值Tensor，shape (H, W) 或 (batch, H, W)
    """
    gt_mask = gt_mask.float()
    pred_mask = pred_mask.float()

    intersection = torch.sum(gt_mask * pred_mask, dim=(-2, -1))
    union = torch.sum(gt_mask, dim=(-2, -1)) + torch.sum(pred_mask, dim=(-2, -1))

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

class AverageMeter:
    """Computes and stores the average and current value."""

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


def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred_prob = torch.sigmoid(pred_mask)
    pred_bin = (pred_prob >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)

    batch_iou = batch_iou.unsqueeze(1)
    return batch_iou


def get_prompts(cfg: Box, bboxes, gt_masks):
    if cfg.prompt == "box" or cfg.prompt == "coarse":
        prompts = bboxes
    elif cfg.prompt == "point":
        prompts = get_point_prompts(gt_masks, cfg.num_points)
    else:
        raise ValueError("Prompt Type Error!")
    return prompts


def validate(fabric: L.Fabric, cfg: Box, model: Model, load_datasets, name: str, iters: int = 0):
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()
    dice_scores = AverageMeter()  # 新增Dice指标
    saved_flag = False  # 控制只保存一张掩码图像
    train_dataloader, val_dataloader = load_datasets(cfg, model.model.image_encoder.img_size)

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            images, bboxes, gt_masks = data
            num_images = images.size(0)

            prompts = get_prompts(cfg, bboxes, gt_masks)
            _, pred_masks, _, _ = model(images, prompts)
            print("Predicted mask min/max:", pred_masks[0].min().item(), pred_masks[0].max().item())

            for i, (pred_mask, gt_mask) in enumerate(zip(pred_masks, gt_masks)):
                pred_bin = (pred_mask >= 0.5).float()

                # 保存一张预测掩码图
                if not saved_flag:
                    pred_np = pred_bin.squeeze().cpu().numpy().astype(np.uint8) * 255
                    save_dir = os.path.join(cfg.out_dir, "pred_masks")
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, "val_pred_mask.png")
                    Image.fromarray(pred_np).save(save_path)
                    fabric.print(f"? 已保存验证集预测掩码图像: {save_path}")
                    saved_flag = True  # 只保存一次

                if gt_mask.max() > 1:
                    gt_bin = (gt_mask == 255).float()
                else:
                    gt_bin = gt_mask.float()

                # 自定义Dice
                batch_dice = calculate_dice_torch(gt_bin, pred_bin).item()

                # IoU 和 F1
                batch_stats = smp.metrics.get_stats(
                    pred_mask,
                    gt_mask.int(),
                    mode='binary',
                    threshold=0.5,
                )
                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")

                ious.update(batch_iou, num_images)
                f1_scores.update(batch_f1, num_images)
                dice_scores.update(batch_dice, num_images)

            fabric.print(
                f'Val: [{iters}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}] -- Mean Dice: [{dice_scores.avg:.4f}]'
            )
            torch.cuda.empty_cache()

    fabric.print(f'Validation [{iters}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}] -- Mean Dice: [{dice_scores.avg:.4f}]')
    csv_dict = {
        "Name": name,
        "Prompt": cfg.prompt,
        "Mean Dice": f"{dice_scores.avg:.4f}",
        "Mean IoU": f"{ious.avg:.4f}",
        "Mean F1": f"{f1_scores.avg:.4f}",
        "Mean Dice": f"{dice_scores.avg:.4f}",
        "iters": iters
    }

    if fabric.global_rank == 0:
        write_csv(os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}.csv"), csv_dict, csv_head=cfg.csv_keys)

    model.train()
    return ious.avg, f1_scores.avg, dice_scores.avg
