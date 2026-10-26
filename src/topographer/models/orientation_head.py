"""
Continuous Tangent Orientation and Road Segmentation Head for TopoGrapher.

Predicts:
1. Road presence probability mask P_road(x, y) in [0, 1];
2. Double-angle orientation tensor field (cos 2theta, sin 2theta) to eliminate
   180-degree ambiguity along undirected road corridors;
3. Metric road width and lanes estimation.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class OrientationHead(nn.Module):
    """
    Predicts road segmentation, tangent field vectors, and road width.
    """

    def __init__(self, in_channels: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, in_channels), in_channels),
            nn.GELU()
        )

        # Output channels:
        # 0: Road binary mask logit
        # 1: cos(2 * theta)
        # 2: sin(2 * theta)
        # 3: Road width (log-scale)
        self.out_conv = nn.Conv2d(in_channels, 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        road_prob : torch.Tensor
            (B, 1, H, W) in [0.0, 1.0].
        tangent_vector : torch.Tensor
            (B, 2, H, W) normalized unit vector (cos 2theta, sin 2theta).
        road_width : torch.Tensor
            (B, 1, H, W) predicted road width in pixels.
        """
        feat = self.conv(x)
        raw = self.out_conv(feat)

        road_prob = torch.sigmoid(raw[:, 0:1])

        # Normalize 2-angle orientation vector to unit length
        cos_sin = raw[:, 1:3]
        tangent_vector = F.normalize(cos_sin, p=2, dim=1, eps=1e-6)

        # Road width via softplus (guarantees positive values > 0)
        road_width = F.softplus(raw[:, 3:4]) * 2.0

        return road_prob, tangent_vector, road_width
