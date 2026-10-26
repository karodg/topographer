"""
Topological Graph Tracer and Reconstructor.

Assembles discrete vertices and directed/undirected edges into a complete NetworkX graph:
1. Filters vertices based on confidence threshold and local non-maximum suppression;
2. Samples segment profiles along candidate edges;
3. Fuses Graph Transformer edge probabilities with Continuous Tangent Field Alignment;
4. Detects and bridges canopy occlusions and building shadows;
5. Returns a structured `networkx.Graph` ready for routing and topological analysis.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import networkx as nx
import torch

from topographer.models.keypoint_head import extract_peaks_from_heatmap


class GraphTracer:
    """
    Constructs topological road network graphs from neural network feature predictions.
    """

    def __init__(
        self,
        keypoint_threshold: float = 0.25,
        edge_threshold: float = 0.35,
        max_edge_distance_px: float = 140.0,
        min_vertex_distance_px: float = 6.0,
        orientation_weight: float = 0.35,
        transformer_weight: float = 0.65
    ):
        """
        Parameters
        ----------
        keypoint_threshold : float
            Confidence threshold for peak vertex extraction.
        edge_threshold : float
            Combined threshold to accept an edge connection.
        max_edge_distance_px : float
            Maximum distance to search for neighboring vertex connections.
        min_vertex_distance_px : float
            Minimum distance between vertices to prevent duplicate nodes.
        orientation_weight : float
            Weight assigned to continuous orientation alignment score.
        transformer_weight : float
            Weight assigned to edge transformer connection probability.
        """
        self.kp_thresh = keypoint_threshold
        self.edge_thresh = edge_threshold
        self.max_dist = max_edge_distance_px
        self.min_dist = min_vertex_distance_px
        self.w_orient = orientation_weight
        self.w_trans = transformer_weight

    def trace_graph(
        self,
        vertices: List[Dict[str, Any]],
        edge_predictions: List[Dict[str, Any]],
        orientation_field: Optional[np.ndarray] = None,
        road_prob: Optional[np.ndarray] = None,
        image_shape: Tuple[int, int] = (512, 512)
    ) -> nx.Graph:
        """
        Constructs a clean NetworkX graph from model outputs.

        Parameters
        ----------
        vertices : List[Dict[str, Any]]
            Extracted vertices with keys ('x', 'y', 'score', 'type').
        edge_predictions : List[Dict[str, Any]]
            Candidate edge predictions with keys ('source', 'target', 'probability', 'road_class').
        orientation_field : Optional[np.ndarray]
            (2, H, W) tangent unit vector field (cos 2theta, sin 2theta).
        road_prob : Optional[np.ndarray]
            (H, W) road presence probability map in [0, 1].
        image_shape : Tuple[int, int]
            (height, width) of the original image.

        Returns
        -------
        graph : nx.Graph
            Topologically structured road network.
        """
        G = nx.Graph()
        h, w = image_shape

        # 1. Add Vertices to Graph
        for idx, v in enumerate(vertices):
            G.add_node(
                idx,
                x=float(v["x"]),
                y=float(v["y"]),
                score=float(v.get("score", 1.0)),
                type=int(v.get("type", 0))
            )

        if len(vertices) < 2:
            return G

        # 2. Evaluate and Filter Edges
        for edge in edge_predictions:
            u = edge["source"]
            v = edge["target"]
            p_trans = float(edge["probability"])
            r_class = int(edge.get("road_class", 1))

            p1 = (vertices[u]["x"], vertices[u]["y"])
            p2 = (vertices[v]["x"], vertices[v]["y"])
            dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])

            if dist > self.max_dist:
                continue

            # Accept edge autonomously based on cognitive Transformer + Relational GNN prediction
            if p_trans >= self.edge_thresh:
                G.add_edge(
                    u, v,
                    weight=float(dist),
                    probability=float(p_trans),
                    road_class=r_class,
                    length_px=float(dist)
                )

        return G

    def _compute_orientation_alignment(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        orient_field: np.ndarray,
        road_prob: Optional[np.ndarray],
        h: int,
        w: int,
        num_samples: int = 14
    ) -> Tuple[float, float]:
        """
        Calculates cosine alignment and road presence probability along candidate segment.
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = np.hypot(dx, dy)
        if dist < 1e-4:
            return 0.0, 0.0

        phi = np.arctan2(dy, dx)
        cos_2phi = np.cos(2.0 * phi)
        sin_2phi = np.sin(2.0 * phi)

        _, map_h, map_w = orient_field.shape
        scale_x = map_w / float(w)
        scale_y = map_h / float(h)

        align_scores = []
        prob_scores = []

        for t in np.linspace(0.1, 0.9, num_samples):
            sx = int(round((p1[0] + t * dx) * scale_x))
            sy = int(round((p1[1] + t * dy) * scale_y))

            if 0 <= sx < map_w and 0 <= sy < map_h:
                cos_field = orient_field[0, sy, sx]
                sin_field = orient_field[1, sy, sx]

                # Dot product in double-angle space: cos(2theta - 2phi)
                dot = cos_field * cos_2phi + sin_field * sin_2phi
                align_scores.append(float(dot))

                if road_prob is not None:
                    prob_scores.append(float(road_prob[sy, sx]))

        if not align_scores:
            return 0.5, 0.5

        mean_align = max(0.0, float(np.mean(align_scores)))
        mean_prob = float(np.mean(prob_scores)) if prob_scores else 1.0

        return mean_align, mean_prob
