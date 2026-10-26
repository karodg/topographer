"""
Cognitive Graph Transformer Edge Connection Classifier with Multi-Modal Occlusion Reasoning.

Fuses:
1. Multi-scale ConvNeXt spatial feature maps;
2. Continuous unit tangent vector field (cos 2theta, sin 2theta);
3. Road surface presence probability;
4. Directional alignment tensor cos(2(theta - phi));
5. Distance and displacement geometric Fourier embeddings.

Processes path tokens through bidirectional Multi-Head Self-Attention layers,
enabling the network to autonomously reason about tree canopies, building shadows,
sand drifts, and road curvature without rigid manual heuristics.
"""

from typing import Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeTransformerClassifier(nn.Module):
    """
    Cognitive Deep Transformer evaluating topological connectivity between graph vertices.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        num_samples: int = 16,
        num_heads: int = 4,
        num_layers: int = 3
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_samples = num_samples

        # Input dimension: Feature(64) + Tangent(2) + RoadProb(1) + Alignment(1) + Geometry(8) = 76
        raw_dim = feature_dim + 2 + 1 + 1 + 8
        self.input_proj = nn.Linear(raw_dim, feature_dim)

        # 1D Positional embedding along line segment
        self.pos_embed = nn.Parameter(torch.randn(1, num_samples, feature_dim) * 0.02)

        # Transformer encoder layers with GELU and Pre-LN
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification heads
        # 1. Edge existence probability
        self.edge_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1)
        )
        # 2. Road class (Highway=0, Arterial=1, Residential=2, Track=3)
        self.type_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 4)
        )

    def sample_multimodal_path_features(
        self,
        feature_map: torch.Tensor,
        orient_map: Optional[torch.Tensor],
        road_prob: Optional[torch.Tensor],
        node_pairs: torch.Tensor
    ) -> torch.Tensor:
        """
        Samples K multi-modal profile tokens along segments connecting node pairs.
        """
        N = node_pairs.shape[0]
        if N == 0:
            return torch.zeros((0, self.num_samples, self.feature_dim), device=feature_map.device)

        device = feature_map.device
        t = torch.linspace(0.0, 1.0, self.num_samples, device=device).view(1, self.num_samples, 1)

        p1 = node_pairs[:, :2].unsqueeze(1)  # (N, 1, 2)
        p2 = node_pairs[:, 2:].unsqueeze(1)  # (N, 1, 2)

        # Interpolated grid points: p(t) = (1 - t) * p1 + t * p2
        pts = (1.0 - t) * p1 + t * p2  # (N, num_samples, 2)
        grid = pts.unsqueeze(0)        # (1, N, num_samples, 2)

        # 1. Sample FPN visual features
        sampled_feat = F.grid_sample(
            feature_map, grid, mode='bilinear', padding_mode='border', align_corners=True
        ).squeeze(0).permute(1, 2, 0)  # (N, num_samples, C)

        # 2. Sample or fallback orientation field
        if orient_map is not None:
            sampled_orient = F.grid_sample(
                orient_map, grid, mode='bilinear', padding_mode='border', align_corners=True
            ).squeeze(0).permute(1, 2, 0)  # (N, num_samples, 2)
        else:
            sampled_orient = torch.zeros((N, self.num_samples, 2), device=device)

        # 3. Sample or fallback road probability
        if road_prob is not None:
            sampled_prob = F.grid_sample(
                road_prob, grid, mode='bilinear', padding_mode='border', align_corners=True
            ).squeeze(0).permute(1, 2, 0)  # (N, num_samples, 1)
        else:
            sampled_prob = torch.ones((N, self.num_samples, 1), device=device) * 0.5

        # 4. Compute Geometric Alignment & Displacement Embeddings
        dx = node_pairs[:, 2] - node_pairs[:, 0]  # (N,)
        dy = node_pairs[:, 3] - node_pairs[:, 1]  # (N,)
        length = torch.sqrt(dx**2 + dy**2 + 1e-6)  # (N,)
        phi = torch.atan2(dy, dx)                 # (N,)
        cos_2phi = torch.cos(2.0 * phi).view(N, 1, 1)
        sin_2phi = torch.sin(2.0 * phi).view(N, 1, 1)

        # Continuous dot-product alignment cos(2(theta - phi))
        dot_align = (
            sampled_orient[:, :, 0:1] * cos_2phi +
            sampled_orient[:, :, 1:2] * sin_2phi
        )  # (N, num_samples, 1)

        # Fourier geometric positional vector: [dx, dy, length, cos_2phi, sin_2phi, sin(len), cos(len), length^2]
        geom_tokens = torch.stack([
            dx, dy, length,
            cos_2phi.squeeze(-1).squeeze(-1),
            sin_2phi.squeeze(-1).squeeze(-1),
            torch.sin(length * float(np.pi)),
            torch.cos(length * float(np.pi)),
            length**2
        ], dim=-1).unsqueeze(1).repeat(1, self.num_samples, 1)  # (N, num_samples, 8)

        # Fuse multi-modal tokens
        fused_raw = torch.cat([
            sampled_feat,
            sampled_orient,
            sampled_prob,
            dot_align,
            geom_tokens
        ], dim=-1)  # (N, num_samples, 76)

        # Linear projection to feature_dim
        tokens = self.input_proj(fused_raw)
        return tokens

    def forward(
        self,
        feature_map: torch.Tensor,
        node_pairs: torch.Tensor,
        orient_map: Optional[torch.Tensor] = None,
        road_prob: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        feature_map : torch.Tensor
            (1, C, H, W).
        node_pairs : torch.Tensor
            (N, 4) in [-1, 1] normalized coordinates.
        orient_map : Optional[torch.Tensor]
            (1, 2, H, W).
        road_prob : Optional[torch.Tensor]
            (1, 1, H, W).

        Returns
        -------
        edge_probs : torch.Tensor
            (N, 1) sigmoid edge probabilities.
        class_logits : torch.Tensor
            (N, 4) road functional classification logits.
        edge_embeddings : torch.Tensor
            (N, feature_dim) pooled edge context embeddings for GNN refinement.
        """
        N = node_pairs.shape[0]
        if N == 0:
            dev = feature_map.device
            return (
                torch.zeros((0, 1), device=dev),
                torch.zeros((0, 4), device=dev),
                torch.zeros((0, self.feature_dim), device=dev)
            )

        tokens = self.sample_multimodal_path_features(feature_map, orient_map, road_prob, node_pairs)

        # Add 1D positional embedding
        tokens = tokens + self.pos_embed

        # Multi-Head Self-Attention Transformer
        context = self.transformer(tokens)  # (N, num_samples, feature_dim)

        # Attentive segment pooling
        pooled = torch.mean(context, dim=1)  # (N, feature_dim)

        edge_logits = self.edge_head(pooled)
        edge_probs = torch.sigmoid(edge_logits)
        class_logits = self.type_head(pooled)

        return edge_probs, class_logits, pooled
