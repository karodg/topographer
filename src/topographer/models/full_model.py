"""
TopoGrapherNet: Unified End-to-End Deep Topological Road Extraction Network.

Fuses:
1. Multi-scale ConvNeXt-FPN feature pyramid backbone;
2. Sub-pixel keypoint & intersection head;
3. Tangent orientation & width field head;
4. Graph Transformer edge connectivity classifier with occlusion gap-bridging.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from topographer.models.backbone import FeaturePyramidEncoder
from topographer.models.keypoint_head import KeypointHead, extract_peaks_from_heatmap
from topographer.models.orientation_head import OrientationHead
from topographer.models.graph_transformer import EdgeTransformerClassifier
from topographer.models.graph_refiner import GraphTopologyRefiner


class TopoGrapherNet(nn.Module):
    """
    End-to-end neural network for extracting topological road networks.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        feature_dim: int = 64,
        num_keypoint_types: int = 3
    ):
        super().__init__()
        self.encoder = FeaturePyramidEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            out_channels=feature_dim
        )
        self.keypoint_head = KeypointHead(
            in_channels=feature_dim,
            num_keypoint_types=num_keypoint_types
        )
        self.orientation_head = OrientationHead(
            in_channels=feature_dim
        )
        self.edge_transformer = EdgeTransformerClassifier(
            feature_dim=feature_dim,
            num_samples=16,
            num_heads=4,
            num_layers=3
        )
        self.graph_refiner = GraphTopologyRefiner(
            embed_dim=feature_dim,
            num_heads=4
        )

    def forward_dense(
        self,
        images: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Computes spatial dense predictions (features, keypoint heatmaps, orientation fields).

        Parameters
        ----------
        images : torch.Tensor
            Batch of aerial/satellite images (B, 3, H, W).

        Returns
        -------
        outputs : Dict[str, torch.Tensor]
            Dictionary containing 'features', 'keypoints', 'road_prob', 'tangent', 'width'.
        """
        features, _ = self.encoder(images)
        keypoint_maps = self.keypoint_head(features)
        road_prob, tangent, width = self.orientation_head(features)

        return {
            "features": features,
            "keypoints": keypoint_maps,
            "road_prob": road_prob,
            "tangent": tangent,
            "width": width
        }

    def predict_graph_elements(
        self,
        image_tensor: torch.Tensor,
        keypoint_threshold: float = 0.30,
        max_edge_distance_px: float = 120.0,
        edge_threshold: float = 0.45
    ) -> Dict[str, Any]:
        """
        Full inference from raw image to discrete vertices and connected edges.

        Parameters
        ----------
        image_tensor : torch.Tensor
            (1, 3, H, W) normalized image.
        keypoint_threshold : float
            Confidence threshold for peak vertex extraction.
        max_edge_distance_px : float
            Maximum pixel radius to search for candidate connections.
        edge_threshold : float
            Minimum transformer probability to confirm an edge.

        Returns
        -------
        graph_data : Dict[str, Any]
            Extracted vertices, connected edge indices, road classes, and dense maps.
        """
        self.eval()
        device = image_tensor.device
        _, _, h, w = image_tensor.shape

        with torch.no_grad():
            dense = self.forward_dense(image_tensor)
            feat = dense["features"]
            kp_map = dense["keypoints"].squeeze(0).cpu().numpy()  # (3, H/2, W/2)
            road_prob = dense["road_prob"].squeeze(0).cpu().numpy()[0]
            tangent = dense["tangent"].squeeze(0).cpu().numpy()

            _, _, feat_h, feat_w = feat.shape
            scale_x = w / float(feat_w)
            scale_y = h / float(feat_h)

            # 1. Extract candidate vertices (sub-pixel peaks in feature space)
            peaks = extract_peaks_from_heatmap(
                kp_map,
                threshold=keypoint_threshold,
                min_distance=4,
                max_peaks=450
            )

            # Map peaks back to full image coordinate space
            vertices = []
            for px, py, score, ktype in peaks:
                orig_x = px * scale_x
                orig_y = py * scale_y
                vertices.append({
                    "x": float(orig_x),
                    "y": float(orig_y),
                    "score": float(score),
                    "type": int(ktype)
                })

            # 1b. Sample intermediate corridor waypoints along long continuous roads
            try:
                from skimage.morphology import skeletonize
                bin_road = (road_prob > 0.40)
                if bin_road.sum() > 150:
                    skel = skeletonize(bin_road)
                    y_skel, x_skel = np.where(skel)
                    if len(x_skel) > 0:
                        existing_coords = np.array([[v["x"], v["y"]] for v in vertices]) if len(vertices) > 0 else np.zeros((0, 2))
                        step = max(10, int(len(x_skel) / 50))
                        for s_i in range(0, len(x_skel), step):
                            sx_px = float(x_skel[s_i] * scale_x)
                            sy_px = float(y_skel[s_i] * scale_y)
                            if len(existing_coords) > 0:
                                dists = np.hypot(existing_coords[:, 0] - sx_px, existing_coords[:, 1] - sy_px)
                                if np.min(dists) > 55.0:
                                    vertices.append({
                                        "x": sx_px,
                                        "y": sy_px,
                                        "score": float(road_prob[y_skel[s_i], x_skel[s_i]]),
                                        "type": 2
                                    })
                                    existing_coords = np.vstack([existing_coords, [sx_px, sy_px]])
                            else:
                                vertices.append({
                                    "x": sx_px,
                                    "y": sy_px,
                                    "score": float(road_prob[y_skel[s_i], x_skel[s_i]]),
                                    "type": 2
                                })
                                existing_coords = np.array([[sx_px, sy_px]])
            except Exception:
                pass

            num_v = len(vertices)
            if num_v < 2:
                return {
                    "vertices": vertices,
                    "edges": [],
                    "dense": dense
                }

            # 2. Formulate candidate edge pairs within max_edge_distance
            candidate_pairs = []
            pair_indices = []

            for i in range(num_v):
                v1 = vertices[i]
                for j in range(i + 1, num_v):
                    v2 = vertices[j]
                    dist = np.hypot(v1["x"] - v2["x"], v1["y"] - v2["y"])
                    if dist <= max_edge_distance_px:
                        # Normalize to [-1, 1] for grid_sample in feature map space
                        norm_x1 = (v1["x"] / (w - 1.0)) * 2.0 - 1.0
                        norm_y1 = (v1["y"] / (h - 1.0)) * 2.0 - 1.0
                        norm_x2 = (v2["x"] / (w - 1.0)) * 2.0 - 1.0
                        norm_y2 = (v2["y"] / (h - 1.0)) * 2.0 - 1.0
                        candidate_pairs.append([norm_x1, norm_y1, norm_x2, norm_y2])
                        pair_indices.append((i, j, float(dist)))

            if not candidate_pairs:
                return {
                    "vertices": vertices,
                    "edges": [],
                    "dense": dense
                }

            # 3. Predict edge connectivity with Cognitive Multi-Modal Transformer in chunks
            chunk_size = 512
            all_edge_probs = []
            all_class_logits = []
            all_pooled = []

            for k in range(0, len(candidate_pairs), chunk_size):
                chunk = candidate_pairs[k : k + chunk_size]
                chunk_tensor = torch.tensor(chunk, dtype=torch.float32, device=device)
                e_prob, c_logits, p_embed = self.edge_transformer(
                    feat, chunk_tensor, orient_map=dense["tangent"], road_prob=dense["road_prob"]
                )
                all_edge_probs.append(e_prob)
                all_class_logits.append(c_logits)
                all_pooled.append(p_embed)

            edge_probs = torch.cat(all_edge_probs, dim=0)
            class_logits = torch.cat(all_class_logits, dim=0)
            pooled_embed = torch.cat(all_pooled, dim=0)

            # 4. Filter promising candidate edges for Relational GNN Refinement (max 600 edges)
            prob_flat = edge_probs.squeeze(1)
            promising_mask = (prob_flat >= 0.15)
            promising_count = int(promising_mask.sum().item())

            if promising_count > 0:
                if promising_count > 600:
                    _, top_idxs = torch.topk(prob_flat, k=600)
                    active_indices = top_idxs
                else:
                    active_indices = torch.nonzero(promising_mask).squeeze(1)

                sub_embed = pooled_embed[active_indices]
                active_np = active_indices.cpu().numpy()
                sub_pairs = torch.tensor(
                    [(pair_indices[i][0], pair_indices[i][1]) for i in active_np],
                    dtype=torch.long, device=device
                )
                sub_refined = self.graph_refiner(sub_embed, sub_pairs)

                final_probs_tensor = edge_probs.clone()
                final_probs_tensor[active_indices] = 0.5 * edge_probs[active_indices] + 0.5 * sub_refined
            else:
                final_probs_tensor = edge_probs

            final_probs_np = final_probs_tensor.squeeze(1).cpu().numpy()
            class_preds_np = torch.argmax(class_logits, dim=1).cpu().numpy()

            confirmed_edges = []
            for idx, prob in enumerate(final_probs_np):
                if prob >= edge_threshold:
                    u, v, length = pair_indices[idx]
                    r_class = int(class_preds_np[idx])
                    confirmed_edges.append({
                        "source": u,
                        "target": v,
                        "probability": float(prob),
                        "length_px": length,
                        "road_class": r_class
                    })

        return {
            "vertices": vertices,
            "edges": confirmed_edges,
            "dense": {
                "road_prob": road_prob,
                "tangent": tangent
            }
        }
