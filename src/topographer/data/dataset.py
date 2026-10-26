"""
PyTorch Dataset and DataLoaders for Topological Road Network Training.

Provides high-throughput generation and batch collation of aerial scenes with:
- Multi-scale feature-aligned ground-truth heatmaps (downsampled 2x to match FPN stride);
- Unit-norm tangent vector fields (cos 2theta, sin 2theta);
- Balanced positive and negative candidate edge pairs for Graph Transformer supervision;
- Photometric and geometric data augmentations (flips, 90-degree rotations, brightness/contrast jitter).
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from topographer.data.procedural_generator import ProceduralRoadGenerator, ProceduralRoadScene


class ProceduralRoadDataset(Dataset):
    """
    On-the-fly procedural synthetic dataset providing infinite training variations
    with guaranteed mathematical ground truth.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        image_size: int = 512,
        max_edge_dist: float = 120.0,
        neg_to_pos_ratio: float = 2.0,
        patterns: Optional[List[str]] = None,
        augment: bool = True
    ):
        """
        Parameters
        ----------
        num_samples : int
            Virtual epoch length.
        image_size : int
            Height and width of generated square scenes.
        max_edge_dist : float
            Maximum pixel distance between vertices to consider as candidate edge pair.
        neg_to_pos_ratio : float
            Ratio of negative (unconnected) edge pairs to positive edge pairs.
        patterns : Optional[List[str]]
            Allowed road patterns ('grid', 'radial', 'organic', 'highway').
        augment : bool
            Whether to apply random flips and 90-deg rotations.
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.max_edge_dist = max_edge_dist
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.patterns = patterns or ["grid", "radial", "organic", "highway", "alpine", "desert"]
        self.augment = augment
        self.generator = ProceduralRoadGenerator(width=image_size, height=image_size)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pattern = self.patterns[idx % len(self.patterns)]
        scene = self.generator.generate_scene(pattern=pattern)

        img = scene.image.astype(np.float32) / 255.0  # (H, W, 3)
        kp = scene.keypoints                         # (3, H, W)
        mask = scene.road_mask.astype(np.float32)    # (H, W)
        orient = scene.orientation                   # (2, H, W)

        # 1. Augmentations
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img = np.fliplr(img).copy()
                kp = np.flip(kp, axis=2).copy()
                mask = np.fliplr(mask).copy()
                # Tangent: flip x negates cos 2theta? theta' = pi - theta -> 2theta' = 2pi - 2theta
                # cos(2pi - 2theta) = cos(2theta), sin(2pi - 2theta) = -sin(2theta)
                orient[1] = -orient[1]
                orient = np.flip(orient, axis=2).copy()
                # Update vertex positions
                scene.vertices = [(self.image_size - 1.0 - vx, vy, vt) for vx, vy, vt in scene.vertices]

            # Random vertical flip
            if np.random.rand() > 0.5:
                img = np.flipud(img).copy()
                kp = np.flip(kp, axis=1).copy()
                mask = np.flipud(mask).copy()
                # Tangent: theta' = -theta -> 2theta' = -2theta -> sin(2theta') = -sin(2theta)
                orient[1] = -orient[1]
                orient = np.flip(orient, axis=1).copy()
                scene.vertices = [(vx, self.image_size - 1.0 - vy, vt) for vx, vy, vt in scene.vertices]

            # Photometric jitter
            brightness = np.random.uniform(0.85, 1.15)
            contrast = np.random.uniform(0.85, 1.15)
            img = np.clip((img - 0.5) * contrast + 0.5 * brightness, 0.0, 1.0)

        # 2. Downsample dense targets by 2x using max-pooling or interpolation to align with FPN feature map
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()  # (3, H, W)
        kp_tensor = torch.from_numpy(kp).unsqueeze(0).float()        # (1, 3, H, W)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float() # (1, 1, H, W)
        orient_tensor = torch.from_numpy(orient).unsqueeze(0).float() # (1, 2, H, W)

        target_size = (self.image_size // 2, self.image_size // 2)
        # Max-pool keypoints to preserve exact peak magnitudes
        kp_down = F.adaptive_max_pool2d(kp_tensor, target_size).squeeze(0)
        mask_down = F.interpolate(mask_tensor, size=target_size, mode='nearest').squeeze(0)
        orient_down = F.interpolate(orient_tensor, size=target_size, mode='bilinear', align_corners=False).squeeze(0)

        # Normalize orientation unit vector
        norm = torch.sqrt(orient_down[0]**2 + orient_down[1]**2 + 1e-6)
        orient_down = orient_down / norm

        # 3. Build candidate edge pairs (Positives + Negatives)
        vertices = scene.vertices
        edges_set = set()
        for u, v, r_class in scene.edges:
            edges_set.add((min(u, v), max(u, v)))

        pos_pairs = []
        pos_classes = []
        pos_idx_pairs = []
        for u, v, r_class in scene.edges:
            if u < len(vertices) and v < len(vertices):
                x1, y1, _ = vertices[u]
                x2, y2, _ = vertices[v]
                norm_x1 = (x1 / (self.image_size - 1.0)) * 2.0 - 1.0
                norm_y1 = (y1 / (self.image_size - 1.0)) * 2.0 - 1.0
                norm_x2 = (x2 / (self.image_size - 1.0)) * 2.0 - 1.0
                norm_y2 = (y2 / (self.image_size - 1.0)) * 2.0 - 1.0
                pos_pairs.append([norm_x1, norm_y1, norm_x2, norm_y2])
                pos_classes.append(r_class)
                pos_idx_pairs.append([u, v])

        # Sample negative pairs within max_edge_dist
        neg_pairs = []
        neg_idx_pairs = []
        num_v = len(vertices)
        max_negs = int(len(pos_pairs) * self.neg_to_pos_ratio) + 4

        potential_negs = []
        potential_neg_indices = []
        for i in range(num_v):
            x1, y1, _ = vertices[i]
            for j in range(i + 1, num_v):
                if (i, j) not in edges_set:
                    x2, y2, _ = vertices[j]
                    dist = np.hypot(x1 - x2, y1 - y2)
                    if dist <= self.max_edge_dist:
                        norm_x1 = (x1 / (self.image_size - 1.0)) * 2.0 - 1.0
                        norm_y1 = (y1 / (self.image_size - 1.0)) * 2.0 - 1.0
                        norm_x2 = (x2 / (self.image_size - 1.0)) * 2.0 - 1.0
                        norm_y2 = (y2 / (self.image_size - 1.0)) * 2.0 - 1.0
                        potential_negs.append([norm_x1, norm_y1, norm_x2, norm_y2])
                        potential_neg_indices.append([i, j])

        if potential_negs:
            shuffled_idx = np.random.permutation(len(potential_negs))[:max_negs]
            neg_pairs = [potential_negs[k] for k in shuffled_idx]
            neg_idx_pairs = [potential_neg_indices[k] for k in shuffled_idx]

        # Combine edge candidates
        all_pairs = pos_pairs + neg_pairs
        all_idx_pairs = pos_idx_pairs + neg_idx_pairs
        edge_labels = [1.0] * len(pos_pairs) + [0.0] * len(neg_pairs)
        edge_classes = pos_classes + [0] * len(neg_pairs)

        if not all_pairs:
            all_pairs = [[0.0, 0.0, 0.1, 0.1]]
            all_idx_pairs = [[0, 0]]
            edge_labels = [0.0]
            edge_classes = [0]

        return {
            "image": img_tensor,
            "keypoints": kp_down,
            "road_mask": mask_down,
            "orientation": orient_down,
            "edge_pairs": torch.tensor(all_pairs, dtype=torch.float32),
            "edge_indices": torch.tensor(all_idx_pairs, dtype=torch.long),
            "edge_labels": torch.tensor(edge_labels, dtype=torch.float32).unsqueeze(1),
            "edge_classes": torch.tensor(edge_classes, dtype=torch.long)
        }


def collate_topographer_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collation function handling variable numbers of edge pairs per sample.
    """
    images = torch.stack([b["image"] for b in batch], dim=0)
    keypoints = torch.stack([b["keypoints"] for b in batch], dim=0)
    road_masks = torch.stack([b["road_mask"] for b in batch], dim=0)
    orientations = torch.stack([b["orientation"] for b in batch], dim=0)

    # For edge pairs, we retain list format or concatenate with sample index pointers
    edge_pairs_list = [b["edge_pairs"] for b in batch]
    edge_indices_list = [b["edge_indices"] for b in batch]
    edge_labels_list = [b["edge_labels"] for b in batch]
    edge_classes_list = [b["edge_classes"] for b in batch]

    return {
        "images": images,
        "keypoints": keypoints,
        "road_masks": road_masks,
        "orientations": orientations,
        "edge_pairs_list": edge_pairs_list,
        "edge_indices_list": edge_indices_list,
        "edge_labels_list": edge_labels_list,
        "edge_classes_list": edge_classes_list
    }
