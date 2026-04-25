import torch
import torch.nn as nn
from layers import ResidualBlock, ASPP_Plus, ConvBlock

class FRRSnetPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, base_channels=40):
        super(FRRSnetPlus, self).__init__()
        
        c = base_channels # 40

        # --- 初始卷积 (图中顶部浅绿色箭头) ---
        self.first_conv = ConvBlock(in_channels, c) # 3 -> 40

        # --- 收缩路径 (Contracting Path) ---
        # 红色下箭头：负责改变通道和尺寸。
        # 蓝色方块：Fig 6(a) 残差块，通道不变。
        
        self.enc1 = ResidualBlock(c)                # 40x512
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c, c*2, 1)                    # 红色箭头：40 -> 80
        )
        
        self.enc2 = ResidualBlock(c*2)              # 80x256
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c*2, c*4, 1)                  # 红色箭头：80 -> 160
        )
        
        self.enc3 = ResidualBlock(c*4)              # 160x128
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c*4, c*8, 1)                  # 红色箭头：160 -> 320
        )
        
        self.enc4 = ResidualBlock(c*8)              # 320x64
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c*8, c*16, 1)                 # 红色箭头：320 -> 640
        )

        self.enc5 = ResidualBlock(c*16)             # 640x32

        # --- Bottleneck (图中底部粗绿色箭头) ---
        self.aspp = ASPP_Plus(in_ch=c*16, out_ch=c*16) # 640 -> 640

        # --- 扩张路径 (Expansive Path) ---
        # 红色上箭头：Deconv 负责尺寸翻倍 & 通道减半。
        # 蓝色方块：Fig 6(b) 残差块，输入输出通道一致。

        self.up4 = nn.ConvTranspose2d(c*16, c*8, 2, stride=2) # 红色：640 -> 320
        self.dec4 = ResidualBlock(c*8)                        # 320x64

        self.up3 = nn.ConvTranspose2d(c*8, c*4, 2, stride=2)  # 红色：320 -> 160
        self.dec3 = ResidualBlock(c*4)                        # 160x128

        self.up2 = nn.ConvTranspose2d(c*4, c*2, 2, stride=2)  # 红色：160 -> 80
        self.dec2 = ResidualBlock(c*2)                        # 80x256

        self.up1 = nn.ConvTranspose2d(c*2, c, 2, stride=2)    # 红色：80 -> 40
        self.dec1 = ResidualBlock(c)                          # 40x512

        # --- 输出层 (图中顶部深绿色箭头) ---
        self.final_conv = nn.Conv2d(c, out_channels, kernel_size=1)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        x1_0 = self.first_conv(x)
        x1 = self.enc1(x1_0)           # Level 1 (40)
        
        x2_0 = self.down1(x1)
        x2 = self.enc2(x2_0)           # Level 2 (80)
        
        x3_0 = self.down2(x2)
        x3 = self.enc3(x3_0)           # Level 3 (160)
        
        x4_0 = self.down3(x3)
        x4 = self.enc4(x4_0)           # Level 4 (320)
        
        x5_0 = self.down4(x4)
        x5 = self.enc5(x5_0)           # Level 5 (640)
        
        # Bottleneck
        bottleneck = self.aspp(x5)     # 640
        
        # Decoder
        d4 = self.up4(bottleneck)      # 红色箭头：变维到 320
        d4 = d4 + x4                   # 蓝色箭头：直接相加 (Identity)
        d4 = self.dec4(d4)             # Fig 6b (320)
        
        d3 = self.up3(d4)              # 红色箭头：变维到 160
        d3 = d3 + x3                   # 蓝色箭头：直接相加
        d3 = self.dec3(d3)             # Fig 6b (160)
        
        d2 = self.up2(d3)              # 红色箭头：变维到 80
        d2 = d2 + x2                   # 蓝色箭头：直接相加
        d2 = self.dec2(d2)             # Fig 6b (80)
        
        d1 = self.up1(d2)              # 红色箭头：变维到 40
        d1 = d1 + x1                   # 蓝色箭头：直接相加
        d1 = self.dec1(d1)             # Fig 6b (40)
        
        return self.final_conv(d1)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FRRSnetPlus(base_channels=40).to(device)
    model.eval() # 避免单 Batch 时 BN 报错
    
    with torch.no_grad():
        test_in = torch.randn(1, 3, 512, 512).to(device)
        test_out = model(test_in)
    
    print(f"✅ 维度验证成功！\n输入: {test_in.shape}\n输出: {test_out.shape}")