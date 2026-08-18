"""Local-detail and long-range-context retinal feature extractor.

This module is deliberately independent of SuperRetina.  It is the first
prototype for learning a point identity from both its local appearance and
the surrounding retinal anatomy.  It does not contain PKE, vessel masks, or
keypoint pseudo-label feedback.
"""

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels):
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return groups


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        padding = kernel_size // 2
        groups = _group_count(out_channels)
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            ConvNormAct(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.body(x))


class AnatomyContextFeatureNet(nn.Module):
    """Produce context-conditioned dense descriptors at 1/8 resolution.

    ``local_feature`` retains high-resolution detail. ``context_feature`` is
    constructed from 1/16 and 1/32 receptive fields. A spatially varying gate
    controls their contribution instead of applying a single global residual
    scalar. Repeatability and reliability are separate predictions by design.
    """

    def __init__(self, in_channels=1, descriptor_dim=64, feature_dim=128):
        super().__init__()
        if descriptor_dim < 8 or feature_dim < 16:
            raise ValueError("descriptor_dim and feature_dim are too small")

        self.stem = nn.Sequential(
            ConvNormAct(in_channels, 32, stride=2),
            ResidualBlock(32),
        )
        self.stage_quarter = nn.Sequential(
            ConvNormAct(32, 64, stride=2),
            ResidualBlock(64),
        )
        self.stage_eighth = nn.Sequential(
            ConvNormAct(64, 96, stride=2),
            ResidualBlock(96),
        )
        self.stage_sixteenth = nn.Sequential(
            ConvNormAct(96, 128, stride=2),
            ResidualBlock(128),
        )
        self.stage_thirty_second = nn.Sequential(
            ConvNormAct(128, 160, stride=2),
            ResidualBlock(160),
        )

        self.local_projection = ConvNormAct(96, feature_dim, kernel_size=1)
        self.context_sixteenth = ConvNormAct(
            128, feature_dim, kernel_size=1
        )
        self.context_thirty_second = ConvNormAct(
            160, feature_dim, kernel_size=1
        )
        self.context_fusion = nn.Sequential(
            ConvNormAct(feature_dim * 2, feature_dim),
            ResidualBlock(feature_dim),
        )
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim // 2, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(feature_dim // 2, 1, 1),
            nn.Sigmoid(),
        )
        self.descriptor_head = nn.Sequential(
            ConvNormAct(feature_dim, feature_dim),
            nn.Conv2d(feature_dim, descriptor_dim, 1),
        )
        self.repeatability_head = nn.Sequential(
            ConvNormAct(feature_dim, feature_dim // 2),
            nn.Conv2d(feature_dim // 2, 1, 1),
            nn.Sigmoid(),
        )
        self.reliability_head = nn.Sequential(
            ConvNormAct(feature_dim * 2, feature_dim // 2),
            nn.Conv2d(feature_dim // 2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, image):
        if image.ndim != 4:
            raise ValueError("image must have shape (B,C,H,W)")
        if image.shape[-2] < 32 or image.shape[-1] < 32:
            raise ValueError("image height and width must be at least 32")

        x = self.stem(image)
        x = self.stage_quarter(x)
        eighth = self.stage_eighth(x)
        sixteenth = self.stage_sixteenth(eighth)
        thirty_second = self.stage_thirty_second(sixteenth)

        local = self.local_projection(eighth)
        output_size = local.shape[-2:]
        context_16 = F.interpolate(
            self.context_sixteenth(sixteenth),
            size=output_size, mode="bilinear", align_corners=False,
        )
        context_32 = F.interpolate(
            self.context_thirty_second(thirty_second),
            size=output_size, mode="bilinear", align_corners=False,
        )
        context = self.context_fusion(torch.cat([context_16, context_32], 1))
        joint = torch.cat([local, context], 1)
        local_gate = self.fusion_gate(joint)
        fused = local_gate * local + (1.0 - local_gate) * context
        descriptor = F.normalize(self.descriptor_head(fused), p=2, dim=1)

        return {
            "descriptor": descriptor,
            "local_feature": local,
            "context_feature": context,
            "local_gate": local_gate,
            "repeatability": self.repeatability_head(local),
            "reliability": self.reliability_head(joint),
        }
