"""
Sub-Pixel Keypoint & Intersection Detector for TopoGrapher.

Predicts continuous spatial heatmaps for:
- Junctions (3-way, 4-way, and roundabout intersections)
- Terminal points (dead-ends, cul-de-sacs)
- Curvature anchors (sharp turns along winding roads)

Includes soft-NMS peak extraction with parabolic sub-pixel refinement.
"""

from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import maximum_filter


class KeypointHead(nn.Module):
    """
    Predicts multi-type topological vertex heatmaps from FPN feature maps.
    """

    def __init__(self, in_channels: int = 64, num_keypoint_types: int = 3):
        super().__init__()
        # Types: 0 = Junction, 1 = Terminal/DeadEnd, 2 = CurvatureAnchor
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(min(8, in_channels), in_channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels, num_keypoint_types, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns
        -------
        heatmaps : torch.Tensor
            Sigmoid-activated probabilities (B, num_types, H, W).
        """
        h = self.act(self.norm(self.conv1(x)))
        logits = self.conv2(h)
        return torch.sigmoid(logits)


def extract_peaks_from_heatmap(
    heatmap: np.ndarray,
    threshold: float = 0.35,
    min_distance: int = 5,
    max_peaks: int = 500
) -> List[Tuple[float, float, float, int]]:
    """
    Extracts local maximum keypoints with sub-pixel parabolic interpolation.

    Parameters
    ----------
    heatmap : np.ndarray
        2D probability array (H, W) or 3D array (C, H, W).
    threshold : float
        Minimum activation threshold.
    min_distance : int
        Minimum distance in pixels between distinct peaks.
    max_peaks : int
        Maximum number of vertices to retain.

    Returns
    -------
    peaks : List[Tuple[x, y, score, type_id]]
        Sub-pixel coordinates (x, y), confidence score, and keypoint class id.
    """
    if heatmap.ndim == 2:
        heatmap = heatmap[np.newaxis, ...]

    num_types, h, w = heatmap.shape
    peaks = []

    for t_idx in range(num_types):
        chan = heatmap[t_idx]
        neighborhood = 2 * min_distance + 1
        local_max = maximum_filter(chan, size=neighborhood) == chan
        detected_mask = local_max & (chan >= threshold)

        y_coords, x_coords = np.where(detected_mask)

        for y, x in zip(y_coords, x_coords):
            # Skip borders to allow 3x3 parabolic fitting
            if 1 <= x < w - 1 and 1 <= y < h - 1:
                # 1D parabolic sub-pixel offset along X
                dx = (chan[y, x + 1] - chan[y, x - 1]) / (2.0 * (2.0 * chan[y, x] - chan[y, x - 1] - chan[y, x + 1]) + 1e-6)
                dx = float(np.clip(dx, -0.5, 0.5))

                # 1D parabolic sub-pixel offset along Y
                dy = (chan[y + 1, x] - chan[y - 1, x]) / (2.0 * (2.0 * chan[y, x] - chan[y - 1, x] - chan[y + 1, x]) + 1e-6)
                dy = float(np.clip(dy, -0.5, 0.5))

                sub_x = float(x + dx)
                sub_y = float(y + dy)
            else:
                sub_x, sub_y = float(x), float(y)

            score = float(chan[y, x])
            peaks.append((sub_x, sub_y, score, t_idx))

    # Sort descending by confidence score
    peaks.sort(key=lambda p: p[2], reverse=True)
    return peaks[:max_peaks]
