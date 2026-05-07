"""
注意力机制模块
用于增强模型对血管特征的关注
包含：通道注意力、空间注意力、CBAM（通道+空间注意力）、自注意力
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    通道注意力模块 (Channel Attention)
    通过学习每个通道的重要性来增强血管相关的特征通道
    
    修复：添加残差连接和更好的初始化策略
    """
    def __init__(self, in_channels, reduction=16, use_residual=True):
        super(ChannelAttention, self).__init__()
        self.use_residual = use_residual
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
        # 初始化最后一个卷积层的权重，使其初始输出接近0
        # 这样sigmoid后接近0.5，即初始时不做改变（恒等映射）
        nn.init.zeros_(self.fc[2].weight)

    def forward(self, x):
        # 平均池化和最大池化
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        # 相加后通过sigmoid得到通道权重
        attention_weight = self.sigmoid(avg_out + max_out)
        
        # 使用残差连接：x = x + alpha * (x * attention_weight - x)
        # 等价于：x = x * (1 + alpha * (attention_weight - 1))
        # 当alpha=1时，就是标准的注意力；当alpha=0时，就是恒等映射
        if self.use_residual:
            # 残差形式：output = input + attention(input) - input
            # 初始时attention_weight接近0.5，所以接近恒等映射
            out = x + (attention_weight - 0.5) * x
        else:
            out = x * attention_weight
        return out


class SpatialAttention(nn.Module):
    """
    空间注意力模块 (Spatial Attention)
    通过学习空间位置的重要性来关注血管的空间分布
    
    修复：添加残差连接和更好的初始化策略
    """
    def __init__(self, kernel_size=7, use_residual=True):
        super(SpatialAttention, self).__init__()
        self.use_residual = use_residual
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        # 初始化卷积层权重，使其初始输出接近0
        # 这样sigmoid后接近0.5，即初始时不做改变（恒等映射）
        nn.init.zeros_(self.conv.weight)

    def forward(self, x):
        # 在通道维度上计算平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接后通过卷积得到空间权重
        x_attention = torch.cat([avg_out, max_out], dim=1)
        x_attention = self.conv(x_attention)
        attention_weight = self.sigmoid(x_attention)
        
        # 使用残差连接
        if self.use_residual:
            # 残差形式：output = input + attention(input) - input
            out = x + (attention_weight - 0.5) * x
        else:
            out = x * attention_weight
        return out


class CBAM(nn.Module):
    """
    CBAM (Convolutional Block Attention Module)
    结合通道注意力和空间注意力，全面增强血管特征
    先应用通道注意力，再应用空间注意力
    
    修复：添加残差连接和更好的初始化策略
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7, use_residual=True):
        super(CBAM, self).__init__()
        self.use_residual = use_residual
        self.channel_attention = ChannelAttention(in_channels, reduction, use_residual=use_residual)
        self.spatial_attention = SpatialAttention(kernel_size, use_residual=use_residual)

    def forward(self, x):
        # 先应用通道注意力
        x = self.channel_attention(x)
        # 再应用空间注意力
        x = self.spatial_attention(x)
        return x


class SelfAttention(nn.Module):
    """
    自注意力机制 (Self-Attention)
    每个位置关注所有其他位置，学习长距离依赖和位置间关系
    
    特点：
    - 全局感受野：不受卷积核大小限制
    - 长距离依赖：可以捕获血管的全局结构
    - 动态权重：根据输入内容动态计算注意力
    
    适用场景：
    - 需要捕获长距离依赖（如血管的全局结构）
    - 需要学习位置间关系（如血管分叉点、交叉点）
    """
    def __init__(self, in_channels, reduction=4, use_residual=True):
        """
        Args:
            in_channels: 输入通道数
            reduction: 降维比例，用于减少计算量（Q和K的通道数 = in_channels // reduction）
            use_residual: 是否使用残差连接
        """
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.reduced_channels = max(1, in_channels // reduction)  # 至少为1
        self.use_residual = use_residual
        
        # Query, Key, Value 投影
        self.query_conv = nn.Conv2d(in_channels, self.reduced_channels, 1, bias=False)
        self.key_conv = nn.Conv2d(in_channels, self.reduced_channels, 1, bias=False)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        
        # 可学习的缩放因子（初始化为0，使初始时接近恒等映射）
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # 初始化：使初始输出接近0（通过gamma=0实现恒等映射）
        nn.init.xavier_uniform_(self.query_conv.weight)
        nn.init.xavier_uniform_(self.key_conv.weight)
        nn.init.xavier_uniform_(self.value_conv.weight)
    
    def forward(self, x):
        """
        Args:
            x: 输入特征 [B, C, H, W]
        Returns:
            out: 增强后的特征 [B, C, H, W]
        """
        B, C, H, W = x.size()
        N = H * W  # 空间位置数
        
        # 计算 Q, K, V
        Q = self.query_conv(x).view(B, self.reduced_channels, N).permute(0, 2, 1)  # [B, N, C']
        K = self.key_conv(x).view(B, self.reduced_channels, N)  # [B, C', N]
        V = self.value_conv(x).view(B, C, N).permute(0, 2, 1)  # [B, N, C]
        
        # 计算注意力分数：Q^T × K / √d
        attention = torch.bmm(Q, K)  # [B, N, N]
        attention = attention / (self.reduced_channels ** 0.5)  # 缩放
        attention = F.softmax(attention, dim=-1)  # [B, N, N]
        
        # 加权求和：attention × V
        out = torch.bmm(attention, V)  # [B, N, C]
        out = out.permute(0, 2, 1).view(B, C, H, W)  # [B, C, H, W]
        
        # 残差连接（gamma初始为0，初始时接近恒等映射）
        if self.use_residual:
            out = x + self.gamma * out
        else:
            out = self.gamma * out
        
        return out
