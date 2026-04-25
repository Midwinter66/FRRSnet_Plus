import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.frrsnet_plus import FRRSnetPlus
from datasets.loader import OreDataset
from utils.metrics import validate  # 引用你写在其他地方的验证函数

def train():
    # --- 1. 超参数与环境配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_channels = 40
    batch_size = 4
    epochs = 50
    lr = 1e-4
    save_dir = 'weights'
    os.makedirs(save_dir, exist_ok=True) # 确保权重文件夹存在

    # --- 2. 模型、损失函数与优化器 ---
    model = FRRSnetPlus(in_channels=3, out_channels=2, base_channels=base_channels).to(device)
    
    # 针对矿石分割，如果背景太多，可以给类别1（矿石）更高的权重，例如 [1.0, 3.0]
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- 3. 数据加载 ---
    # 建议将数据集分为 train 和 val 两个子目录
    train_ds = OreDataset(image_dir='data/train/images', mask_dir='data/train/masks')
    val_ds = OreDataset(image_dir='data/val/images', mask_dir='data/val/masks')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"配置完毕。设备: {device}, 训练样本: {len(train_ds)}, 验证样本: {len(val_ds)}")

    # --- 4. 训练与验证循环 ---
    best_miou = 0.0

    for epoch in range(epochs):
        model.train() # 切换训练模式
        epoch_loss = 0
        
        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # --- 每个 Epoch 结束进行一次验证 ---
        print("\n正在验证...")
        val_metrics = validate(model, val_loader, criterion, device)
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] 总结:")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['acc']:.4f} | Val MIoU: {val_metrics['miou']:.4f}")

        # --- 保存最优模型 ---
        if val_metrics['miou'] > best_miou:
            best_miou = val_metrics['miou']
            save_path = os.path.join(save_dir, 'model1_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"⭐⭐ 发现更好的 MIoU: {best_miou:.4f}, 已保存至 {save_path}")
        
        # 也可以定期保存一个最新的权重
        torch.save(model.state_dict(), os.path.join(save_dir, 'model1_latest.pth'))
        print("-" * 30)

    print("训练完成！")

if __name__ == "__main__":
    train()