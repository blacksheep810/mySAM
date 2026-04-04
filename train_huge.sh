#!/bin/bash
# ÑµÁ· SAM Huge (vit_h) Ä£ÐÍµÄ½Å±¾

python train.py \
  --data_root ./data/ISIC \
  --train_box_csv ./data/ISIC/train_boxes.csv \
  --test_box_csv ./data/ISIC/test_boxes.csv \
  --sam_checkpoint ./checkpoints/sam_vit_h_4b8939.pth \
  --model_type vit_h \
  --output_dir ./outputs_huge \
  --batch_size 2 \
  --epochs 30 \
  --img_size 1024 \
  --lr_encoder 2e-6 \
  --lr_decoder 1e-4 \
  --weight_decay 1e-4 \
  --proj_dim 64 \
  --pos_samples 256 \
  --neg_samples 1024 \
  --temperature 0.1 \
  --entropy_thresh 0.2 \
  --unfreeze_last_k 2 \
  --use_amp \
  --use_gradient_checkpointing \
  --gradient_accumulation_steps 2 \
  --device cuda \
  --save_every 1

