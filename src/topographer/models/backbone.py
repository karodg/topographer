"""
Multi-Scale Feature Pyramid Backbone for TopoGrapher.

Implements a modern, lightweight ConvNeXt-FPN hybrid encoder in pure PyTorch.
Extracts rich multi-scale spatial representations fusing wide receptive fields
(for long-range road continuity and occlusion reasoning) with sub-pixel localization
(for precise intersection and curve placement).
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style residual block with depthwise 7x7 conv and inverted bottleneck."""

    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)
        self.norm = nn.GroupNorm(1, dim)  # LayerNorm equivalent for 2D channels
        self.pwconv1 = nn.Conv2d(dim, dim * expansion, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim * expansion, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x + res


class FeaturePyramidEncoder(nn.Module):
    """
    4-stage multi-scale pyramid encoder fusing fine-grained and global spatial cues.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        out_channels: int = 64
    ):
        super().__init__()
        c = base_channels

        # Stem: Downsamples 2x with 4x4 stride-2 conv
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(1, c),
            nn.GELU()
        )

        # Stage 1 (H/2, W/2)
        self.stage1 = nn.Sequential(
            ConvNeXtBlock(c),
            ConvNeXtBlock(c)
        )

        # Downsample 1 -> 2
        self.down1 = nn.Sequential(
            nn.Conv2d(c, c * 2, kernel_size=2, stride=2),
            nn.GroupNorm(1, c * 2)
        )
        # Stage 2 (H/4, W/4)
        self.stage2 = nn.Sequential(
            ConvNeXtBlock(c * 2),
            ConvNeXtBlock(c * 2)
        )

        # Downsample 2 -> 3
        self.down2 = nn.Sequential(
            nn.Conv2d(c * 2, c * 4, kernel_size=2, stride=2),
            nn.GroupNorm(1, c * 4)
        )
        # Stage 3 (H/8, W/8)
        self.stage3 = nn.Sequential(
            ConvNeXtBlock(c * 4),
            ConvNeXtBlock(c * 4)
        )

        # Downsample 3 -> 4
        self.down3 = nn.Sequential(
            nn.Conv2d(c * 4, c * 8, kernel_size=2, stride=2),
            nn.GroupNorm(1, c * 8)
        )
        # Stage 4 (H/16, W/16) - Deep semantic bottleneck
        self.stage4 = nn.Sequential(
            ConvNeXtBlock(c * 8),
            ConvNeXtBlock(c * 8)
        )

        # FPN Lateral Projections to unified out_channels
        self.lat4 = nn.Conv2d(c * 8, out_channels, kernel_size=1)
        self.lat3 = nn.Conv2d(c * 4, out_channels, kernel_size=1)
        self.lat2 = nn.Conv2d(c * 2, out_channels, kernel_size=1)
        self.lat1 = nn.Conv2d(c, out_channels, kernel_size=1)

        # Smooth convolutions
        self.smooth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns
        -------
        fused_features : torch.Tensor
            High-resolution unified feature map (B, out_channels, H/2, W/2).
        pyramid_stages : List[torch.Tensor]
            List of multi-scale stage features [C1, C2, C3, C4].
        """
        s0 = self.stem(x)
        c1 = self.stage1(s0)            # H/2, W/2
        c2 = self.stage2(self.down1(c1))  # H/4, W/4
        c3 = self.stage3(self.down2(c2))  # H/8, W/8
        c4 = self.stage4(self.down3(c3))  # H/16, W/16

        # Top-down FPN pathway
        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[2:], mode='bilinear', align_corners=False)
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[2:], mode='bilinear', align_corners=False)
        p1 = self.lat1(c1) + F.interpolate(p2, size=c1.shape[2:], mode='bilinear', align_corners=False)

        fused = self.smooth(p1)
        return fused, [c1, c2, c3, c4]
