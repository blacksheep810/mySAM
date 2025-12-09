# Quick Start Guide

## Quick Evaluation Command

### Evaluate Latest Trained Model (Epoch 30)

```bash
cd /root/workspace/mySAM/validate

python evaluate.py \
    --checkpoint_path ./outputs/checkpoint_epoch_30.pth \
    --data_root ./data/ISIC \
    --test_box_csv ./data/ISIC/test_boxes.csv \
    --sam_checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
    --model_type vit_b \
    --img_size 1024 \
    --batch_size 8
```

### Evaluate Specific Epoch Model

```bash
cd /root/workspace/mySAM/validate

# Example: Evaluate Epoch 20
python evaluate.py \
    --checkpoint_path ./outputs/checkpoint_epoch_20.pth \
    --data_root ./data/ISIC \
    --test_box_csv ./data/ISIC/test_boxes.csv \
    --sam_checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
    --model_type vit_b \
    --img_size 1024 \
    --batch_size 8
```

### Evaluate with Point Prompt

```bash
cd /root/workspace/mySAM/validate

python evaluate.py \
    --checkpoint_path ./outputs/checkpoint_epoch_30.pth \
    --data_root ./data/ISIC \
    --test_box_csv ./data/ISIC/test_boxes.csv \
    --sam_checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
    --model_type vit_b \
    --prompt_type point \
    --num_points 10 \
    --img_size 1024 \
    --batch_size 8
```

## Parameter Description

### Required Parameters
- `--checkpoint_path`: Trained model checkpoint path (e.g., `./outputs/checkpoint_epoch_30.pth`)
- `--data_root`: Dataset root directory (`./data/ISIC`)
- `--test_box_csv`: Test set box CSV file path (`./data/ISIC/test_boxes.csv`)
- `--sam_checkpoint`: SAM pretrained model path (`./checkpoints/sam_vit_b_01ec64.pth`)

### Common Optional Parameters
- `--model_type`: SAM model type, default `vit_b`
- `--img_size`: Image size, default `1024`
- `--batch_size`: Batch size, default `8` (can use larger value for evaluation)
- `--prompt_type`: Prompt type, `box` or `point`, default `box`
- `--device`: Device to use, `cuda` or `cpu`, default `cuda`

## Output Results

After evaluation completes, you will see:
```
==================================================
Evaluation Results:
==================================================
Checkpoint: /root/workspace/mySAM/outputs/checkpoint_epoch_30.pth
Epoch: 29
Prompt Type: box
Dataset: /root/workspace/mySAM/data/ISIC
Number of Samples: 379
mIOU: 0.2231
==================================================
```

Results are automatically saved to `validate/results/evaluation_results.csv`

## Common Issues

### Q: File not found?
A: The script automatically handles paths. Make sure to run from `validate` directory, or use absolute paths.

### Q: CUDA out of memory?
A: Reduce `--batch_size` or use `--device cpu`

### Q: How to evaluate multiple checkpoints?
A: See batch evaluation script example in `README.md`
