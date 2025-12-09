import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

class ISIC2016Dataset(Dataset):
    def __init__(self, root, box_csv, transform=None, img_size=512, split='train'):
        """
        root: ISIC 根目录
        box_csv: 包含 box 信息的 CSV 文件路径（如 train_boxes.csv 或 test_boxes.csv）
        transform: 可选的图像变换
        img_size: 目标图像尺寸
        split: 'train' 或 'test'，用于确定数据目录
        """
        self.root = root
        self.df = pd.read_csv(box_csv)
        self.transform = transform
        self.img_size = img_size
        self.split = split

        # 根据 split 确定目录名
        if split == 'train':
            self.img_dir = os.path.join(root, "ISBI2016_ISIC_Part1_Training_Data")
            self.mask_dir = os.path.join(root, "ISBI2016_ISIC_Part1_Training_GroundTruth")
        elif split == 'test':
            self.img_dir = os.path.join(root, "ISBI2016_ISIC_Part1_Test_Data")
            self.mask_dir = os.path.join(root, "ISBI2016_ISIC_Part1_Test_GroundTruth")
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 从 CSV 读取信息
        row = self.df.iloc[idx]
        img_name = row['image_file']
        mask_name = row['mask_file']
        
        # 从 CSV 读取 box 坐标（原始图像尺寸下的坐标）
        max_x1 = row['max_boxes_x1']
        max_y1 = row['max_boxes_y1']
        max_x2 = row['max_boxes_x2']
        max_y2 = row['max_boxes_y2']
        
        min_x1 = row['min_boxes_x1']
        min_y1 = row['min_boxes_y1']
        min_x2 = row['min_boxes_x2']
        min_y2 = row['min_boxes_y2']

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # --- read image ---
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --- read mask ---
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        mask = (mask > 0).astype(np.uint8)

        # 获取原始图像尺寸
        orig_h, orig_w = image.shape[:2]
        
        # 计算缩放比例
        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h

        # resize to fixed size
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        # --- 缩放 box 坐标到新的图像尺寸 ---
        # big box (max_boxes): [x1, y1, x2, y2]
        big_box = torch.tensor([
            max_x1 * scale_x,
            max_y1 * scale_y,
            max_x2 * scale_x,
            max_y2 * scale_y
        ]).float()
        
        # small box (min_boxes): [x1, y1, x2, y2]
        small_box = torch.tensor([
            min_x1 * scale_x,
            min_y1 * scale_y,
            min_x2 * scale_x,
            min_y2 * scale_y
        ]).float()
        
        # 确保 box 坐标在有效范围内
        big_box = torch.clamp(big_box, 0, self.img_size)
        small_box = torch.clamp(small_box, 0, self.img_size)

        return image, big_box, small_box, mask, img_name


# In[11]:


def test_dataset():
    """测试数据集类的功能"""
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    
    # 设置路径
    root = "../data/ISIC"
    train_box_file = "../data/ISIC/train_boxes.csv"
    test_box_file = "../data/ISIC/test_boxes.csv"
    
    print("=" * 60)
    print("测试 ISIC2016Dataset 数据集类")
    print("=" * 60)
    
    # 测试训练集
    print("\n1. 创建训练集...")
    try:
        train_dataset = ISIC2016Dataset(
            root=root,
            box_csv=train_box_file,
            img_size=512,
            split='train'
        )
        print(f"   ✓ 训练集创建成功，共 {len(train_dataset)} 个样本")
    except Exception as e:
        print(f"   ✗ 训练集创建失败: {e}")
        return
    
    # 测试测试集
    print("\n2. 创建测试集...")
    try:
        test_dataset = ISIC2016Dataset(
            root=root,
            box_csv=test_box_file,
            img_size=512,
            split='test'
        )
        print(f"   ✓ 测试集创建成功，共 {len(test_dataset)} 个样本")
    except Exception as e:
        print(f"   ✗ 测试集创建失败: {e}")
        return
    
    # 测试获取单个样本
    print("\n3. 测试获取单个样本...")
    try:
        idx = 0
        image, big_box, small_box, mask, img_name = train_dataset[idx]
        
        print(f"   ✓ 成功获取样本 {idx}")
        print(f"   - 图像名称: {img_name}")
        print(f"   - 图像形状: {image.shape} (C, H, W)")
        print(f"   - Mask 形状: {mask.shape} (1, H, W)")
        print(f"   - 大 Box: {big_box.tolist()} (x1, y1, x2, y2)")
        print(f"   - 小 Box: {small_box.tolist()} (x1, y1, x2, y2)")
        
        # 验证 box 坐标有效性
        x1, y1, x2, y2 = big_box
        if x1 < x2 and y1 < y2 and all(0 <= coord <= 512 for coord in big_box):
            print(f"   ✓ 大 Box 坐标有效")
        else:
            print(f"   ✗ 大 Box 坐标无效!")
            
        x1, y1, x2, y2 = small_box
        if x1 < x2 and y1 < y2 and all(0 <= coord <= 512 for coord in small_box):
            print(f"   ✓ 小 Box 坐标有效")
        else:
            print(f"   ✗ 小 Box 坐标无效!")
            
    except Exception as e:
        print(f"   ✗ 获取样本失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试 DataLoader
    print("\n4. 测试 DataLoader...")
    try:
        train_loader = DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0  # 设置为 0 避免多进程问题
        )
        
        # 获取一个 batch
        batch = next(iter(train_loader))
        images, big_boxes, small_boxes, masks, img_names = batch
        
        print(f"   ✓ DataLoader 创建成功")
        print(f"   - Batch 图像形状: {images.shape} (B, C, H, W)")
        print(f"   - Batch Mask 形状: {masks.shape} (B, 1, H, W)")
        print(f"   - Batch 大 Box 形状: {big_boxes.shape} (B, 4)")
        print(f"   - Batch 小 Box 形状: {small_boxes.shape} (B, 4)")
        print(f"   - 图像名称: {list(img_names)}")
        
    except Exception as e:
        print(f"   ✗ DataLoader 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 可视化测试（可选）
    print("\n5. 可视化测试（前3个样本）...")
    try:
        fig, axes = plt.subplots(3, 2, figsize=(12, 15))
        
        for i in range(min(3, len(train_dataset))):
            image, big_box, small_box, mask, img_name = train_dataset[i]
            
            # 转换为 numpy 用于显示
            img_np = image.permute(1, 2, 0).numpy()
            mask_np = mask.squeeze(0).numpy()
            
            # 绘制原图
            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title(f'Image {i+1}: {img_name}', fontsize=10)
            axes[i, 0].axis('off')
            
            # 绘制带大小框的图像（在同一张图上）
            img_with_boxes = img_np.copy()
            
            # 绘制大框（红色）
            bx1, by1, bx2, by2 = big_box.int().tolist()
            cv2.rectangle(img_with_boxes, (bx1, by1), (bx2, by2), (1, 0, 0), 2)  # 红色，线宽2
            
            # 绘制小框（绿色）
            sx1, sy1, sx2, sy2 = small_box.int().tolist()
            cv2.rectangle(img_with_boxes, (sx1, sy1), (sx2, sy2), (0, 1, 0), 2)  # 绿色，线宽2
            
            # 添加图例文本
            axes[i, 1].imshow(img_with_boxes)
            title = f'Big Box (red): [{bx1}, {by1}, {bx2}, {by2}]\nSmall Box (green): [{sx1}, {sy1}, {sx2}, {sy2}]'
            axes[i, 1].set_title(title, fontsize=9)
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('isic_dataset_test.png', dpi=150, bbox_inches='tight')
        print(f"   ✓ 可视化完成，已保存到 isic_dataset_test.png")
        print(f"   - 红色框：大框 (Big Box)")
        print(f"   - 绿色框：小框 (Small Box)")
        plt.close()
        
    except Exception as e:
        print(f"   ⚠ 可视化测试失败（可能缺少 matplotlib）: {e}")
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    test_dataset()





