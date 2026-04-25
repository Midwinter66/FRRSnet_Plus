import torch
import numpy as np

def get_metrics(pred, target, num_classes=2):
    """
    一键获取所有指标
    """
    pred_class = torch.argmax(pred, dim=1)
    
    # 1. 计算 Accuracy
    correct = (pred_class == target).float().sum()
    acc = correct / target.numel()
    
    # 2. 计算 MIoU
    ious = []
    # 忽略索引通常为背景或特定类，这里按 num_classes 循环
    for cls in range(num_classes):
        inter = ((pred_class == cls) & (target == cls)).float().sum()
        union = ((pred_class == cls) | (target == cls)).float().sum()
        
        if union == 0:
            # 如果标签和预测都没有这一类，IoU 设为 1
            ious.append(1.0)
        else:
            ious.append((inter / union).item())
    
    miou = sum(ious) / len(ious)
    
    return acc.item(), miou

def validate(model, dataloader, criterion, device):
    """
    完整的验证逻辑函数
    """
    model.eval()  # 切换到评估模式
    val_loss = 0
    total_acc = 0
    total_miou = 0
    
    with torch.no_grad(): # 验证时不计算梯度，节省显存
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            val_loss += loss.item()
            
            # 调用上面的指标函数
            acc, miou = get_metrics(outputs, masks)
            total_acc += acc
            total_miou += miou
            
    # 计算平均值
    num_batches = len(dataloader)
    return {
        'loss': val_loss / num_batches,
        'acc': total_acc / num_batches,
        'miou': total_miou / num_batches
    }