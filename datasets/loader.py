import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class OreDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        # 标签文件名通常与原图一致
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", ".png"))
        
        # 读取图像 (RGB)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 读取掩码 (灰度)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 数据归一化与预处理
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # 基础转换
            image = torch.from_numpy(image).transpose(0, 2).transpose(1, 2).float() / 255.0
            mask = torch.from_numpy(mask).long()
            # 确保 mask 只有 0 和 1（如果原始 mask 是 255）
            mask[mask > 0] = 1 
            
        return image, mask