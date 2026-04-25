import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """基础卷积单元:Conv -> BN -> ReLU"""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, dilation=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    """
    符合 Fig. 6 (a) 和 (b) 的残差块。
    逻辑：输入输出通道一致 (C -> C)，内部不改变维度。
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity  # 恒等映射相加
        return self.relu(out)

class ASPP_Plus(nn.Module):
    """ASPP 瓶颈层：C1=160, C2=640 (在 base=40 时, C1 对应 enc3 输出层级)"""
    def __init__(self, in_ch, out_ch=640):
        super(ASPP_Plus, self).__init__()
        mid_ch = 160  # 论文指定的 C1
        self.stages = nn.ModuleList([
            ConvBlock(in_ch, mid_ch, kernel_size=1, padding=0),
            ConvBlock(in_ch, mid_ch, kernel_size=3, padding=6, dilation=6),
            ConvBlock(in_ch, mid_ch, kernel_size=3, padding=12, dilation=12),
            ConvBlock(in_ch, mid_ch, kernel_size=3, padding=18, dilation=18)
        ])
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True)
        )
        # 融合层使用 3x3 卷积
        self.bottleneck = nn.Sequential(
            nn.Conv2d(mid_ch * 5, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        res = [stage(x) for stage in self.stages]
        gp = self.global_pool(x)
        gp = torch.nn.functional.interpolate(gp, size=x.size()[2:], mode='bilinear', align_corners=True)
        res.append(gp)
        return self.bottleneck(torch.cat(res, dim=1))