from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    """Swish激活函数"""

    def forward(self, x):
        return x * torch.sigmoid(x)


class DoubleConv(nn.Sequential):
    """双层卷积模块，保持特征图尺寸不变（添加dropout）"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0.2):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),  # 保留BatchNorm层
            Swish(),
            nn.Dropout2d(dropout),  # 添加空间dropout
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),  # 保留BatchNorm层
            Swish(),
            nn.Dropout2d(dropout)  # 添加空间dropout
        )


class Down(nn.Sequential):
    """下采样模块（最大池化+双层卷积）"""

    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),  # 尺寸减半
            DoubleConv(in_channels, out_channels, dropout=dropout)  # 传入dropout参数
        )


class Up(nn.Module):
    """上采样模块（转置卷积+特征融合+双层卷积）"""

    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)  # 尺寸翻倍
        self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)  # 传入dropout参数

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # 处理尺寸不匹配（边缘对齐）
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)  # 按通道维度融合
        return self.conv(x)


class OutConv(nn.Sequential):
    """输出层，映射到温度场通道"""

    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)  # 1x1卷积调整通道数
        )


class TemperatureUNet(nn.Module):
    def __init__(self,
                 # 输入输出通道配置
                 sdf_channels=1,  # SDF通道数
                 heat_source_channels=1,  # 热源矩阵通道数
                 heatsink_channels=1,  # 散热口掩码通道数
                 out_channels=1,  # 输出温度场通道数
                 # 架构参数（核心可配置部分）
                 base_dim=64,  # 基础通道数（第一层卷积输出通道）
                 downsample_layers=4,  # 下采样层数（决定U-Net深度）
                 image_size=256,  # 输入输出尺寸
                 dropout=0.2):  # 添加dropout参数
        super(TemperatureUNet, self).__init__()

        # 验证参数合理性
        assert image_size % (2 ** downsample_layers) == 0, \
            f"输入尺寸{image_size}必须能被2^{downsample_layers}整除（确保下采样后尺寸为整数）"
        assert downsample_layers >= 1, "下采样层数至少为1"

        # 总输入通道数 = 各输入通道之和
        self.total_in_channels = sdf_channels + heat_source_channels + heatsink_channels
        self.downsample_layers = downsample_layers  # 保存下采样层数，方便forward中使用
        self.dropout = dropout  # 保存dropout参数

        # 输入层（融合多通道输入）
        self.inc = DoubleConv(self.total_in_channels, base_dim, dropout=dropout)

        # 动态生成编码器（下采样模块）
        self.downs = nn.ModuleList()
        current_channels = base_dim
        for i in range(downsample_layers):
            next_channels = current_channels * 2
            self.downs.append(Down(current_channels, next_channels, dropout=dropout))
            current_channels = next_channels

        # 动态生成解码器（上采样模块）
        self.ups = nn.ModuleList()
        for i in range(downsample_layers):
            next_channels = current_channels // 2
            self.ups.append(Up(current_channels, next_channels, dropout=dropout))
            current_channels = next_channels

        # 输出层
        self.outc = OutConv(base_dim, out_channels)

    def forward(self, sdf, heat_source, heatsink_mask):
        """
        输入：
            sdf: [batch_size, sdf_channels, H, W]
            heat_source: [batch_size, heat_source_channels, H, W]
            heatsink_mask: [batch_size, heatsink_channels, H, W]
        输出：
            temperature: [batch_size, out_channels, H, W]
        """
        # 融合所有输入（按通道拼接）
        x = torch.cat([sdf, heat_source, heatsink_mask], dim=1)

        # 编码器（下采样）
        x1 = self.inc(x)
        skips = [x1]  # 缓存跳跃连接特征
        for down in self.downs:
            x1 = down(x1)
            skips.append(x1)

        # 解码器（上采样+跳跃连接融合）
        for i, up in enumerate(self.ups):
            # 跳跃连接索引：从倒数第二个开始向前取（跳过最后一个下采样特征）
            x1 = up(x1, skips[-(i + 2)])

        # 输出温度场
        return self.outc(x1)


# --------------------------- 使用示例 ---------------------------
if __name__ == "__main__":
    # 测试不同配置的模型
    def test_model_config(downsample_layers=4, image_size=256):
        print(f"\n测试配置：下采样层数={downsample_layers}, 输入尺寸={image_size}")
        batch_size = 2

        # 生成模拟输入
        sdf = torch.randn(batch_size, 1, image_size, image_size)
        heat_source = torch.randn(batch_size, 1, image_size, image_size)
        heatsink_mask = torch.randint(0, 2, (batch_size, 1, image_size, image_size)).float()

        # 初始化模型（指定dropout=0.2）
        model = TemperatureUNet(
            downsample_layers=downsample_layers,
            image_size=image_size,
            base_dim=32,
            dropout=0.2
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型总参数数量: {total_params:,}")
        # 前向传播
        output = model(sdf, heat_source, heatsink_mask)
        print(f"输入尺寸: {sdf.shape[2:]} → 输出尺寸: {output.shape[2:]} (预期一致)")
        print(f"输出形状验证: {output.shape}")  # 应满足 [batch_size, 1, image_size, image_size]


    # 测试默认配置（4层下采样，256x256）
    test_model_config(downsample_layers=4)

