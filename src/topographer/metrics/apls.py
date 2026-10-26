"""
SpaceNet Average Path Length Similarity (APLS) Metric Engine.

Computes the industry-standard SpaceNet APLS metric to evaluate graph topology and routability.
Unlike pixel-wise IoU (which fails to detect disconnected roads), APLS measures whether an autonomous
vehicle or routing algorithm can navigate between all origin-destination pairs with the same shortest-path
distances on the extracted graph as on the true ground-truth graph.

Reference:
    Van Etten, Adam. "City-Scale Road Extraction from Satellite Imagery v2: Road Speed and Travel Time."
    arXiv:1805.01168 (CosmiQ Works / SpaceNet).
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import networkx as nx


class APLSMetric:
    """
    Computes bidirectional SpaceNet Average Path Length Similarity between Ground-Truth and Extracted graphs.
    """

    def __init__(
        self,
        max_buffer_px: float = 16.0,
        max_pairs_sample: int = 500
    ):
        """
        Parameters
        ----------
        max_buffer_px : float
            Maximum distance (in pixels) for a node to be snapped to a counterpart in the other graph.
            Nodes further than this threshold are considered unmapped (disconnected).
        max_pairs_sample : int
            Maximum number of random origin-destination pairs to sample for computationally efficient evaluation.
        """
        self.buffer = max_buffer_px
        self.max_pairs = max_pairs_sample

    def compute_apls(self, gt_graph: nx.Graph, prop_graph: nx.Graph) -> Dict[str, float]:
        """
        Calculates bidirectional APLS score:
            APLS = 0.5 * (APLS(GT -> Prop) + APLS(Prop -> GT))

        Returns
        -------
        scores : Dict[str, float]
            Contains 'apls' (in [0, 1]), 'apls_gt_to_prop' (recall), 'apls_prop_to_gt' (precision).
        """
        if gt_graph.number_of_nodes() < 2:
            return {"apls": 1.0 if prop_graph.number_of_nodes() < 2 else 0.0, "apls_recall": 1.0, "apls_precision": 1.0}

        if prop_graph.number_of_nodes() < 2:
            return {"apls": 0.0, "apls_recall": 0.0, "apls_precision": 0.0}

        # Densify both graphs along polylines/edges so node-to-node snapping evaluates the entire road length
        gt_dense = self._densify_graph(gt_graph, max_dist_px=35.0)
        prop_dense = self._densify_graph(prop_graph, max_dist_px=35.0)

        score_recall = self._directed_apls(gt_dense, prop_dense)
        score_precision = self._directed_apls(prop_dense, gt_dense)

        if score_recall + score_precision < 1e-6:
            harmonic = 0.0
        else:
            harmonic = 2.0 * (score_recall * score_precision) / (score_recall + score_precision)

        mean_apls = 0.5 * (score_recall + score_precision)

        return {
            "apls": float(mean_apls),
            "apls_harmonic": float(harmonic),
            "apls_recall": float(score_recall),
            "apls_precision": float(score_precision)
        }

    def _densify_graph(self, G: nx.Graph, max_dist_px: float = 35.0) -> nx.Graph:
        """
        Subdivides long edges and interpolates polylines so that both graphs have
        comparable vertex spacing along corridors for fair SpaceNet node-to-node snapping.
        """
        if G.number_of_nodes() < 2:
            return G.copy()

        H = nx.Graph()
        for n, d in G.nodes(data=True):
            H.add_node(n, **d)

        int_nodes = [int(n) for n in G.nodes() if isinstance(n, (int, np.integer))]
        next_id = max(int_nodes) + 1 if int_nodes else len(G.nodes()) + 1

        for u, v, data in G.edges(data=True):
            poly = data.get("polyline")
            if poly and len(poly) >= 2:
                prev = u
                for pt in poly[1:-1]:
                    new_n = next_id
                    next_id += 1
                    H.add_node(new_n, x=float(pt[0]), y=float(pt[1]), type=2)
                    d_step = float(np.hypot(H.nodes[prev]["x"] - pt[0], H.nodes[prev]["y"] - pt[1]))
                    H.add_edge(prev, new_n, weight=d_step, length_px=d_step)
                    prev = new_n
                d_final = float(np.hypot(H.nodes[prev]["x"] - H.nodes[v]["x"], H.nodes[prev]["y"] - H.nodes[v]["y"]))
                H.add_edge(prev, v, weight=d_final, length_px=d_final)
            else:
                p1 = (G.nodes[u]["x"], G.nodes[u]["y"])
                p2 = (G.nodes[v]["x"], G.nodes[v]["y"])
                dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
                if dist <= max_dist_px:
                    H.add_edge(u, v, **data)
                else:
                    num_steps = int(np.ceil(dist / max_dist_px))
                    prev = u
                    for s in range(1, num_steps):
                        t = float(s) / num_steps
                        new_n = next_id
                        next_id += 1
                        nx_pos = float(p1[0] + t * (p2[0] - p1[0]))
                        ny_pos = float(p1[1] + t * (p2[1] - p1[1]))
                        H.add_node(new_n, x=nx_pos, y=ny_pos, type=2)
                        d_step = dist / num_steps
                        H.add_edge(prev, new_n, weight=d_step, length_px=d_step)
                        prev = new_n
                    H.add_edge(prev, v, weight=dist / num_steps, length_px=dist / num_steps)
        return H

    def _directed_apls(self, G_true: nx.Graph, G_test: nx.Graph) -> float:
        """
        Evaluates routing path discrepancies from G_true to G_test.
        """
        nodes_true = list(G_true.nodes(data=True))
        nodes_test = list(G_test.nodes(data=True))

        # 1. Map each node in G_true to the nearest node in G_test within buffer
        mapping: Dict[int, Optional[int]] = {}
        for u, d_u in nodes_true:
            pu = (d_u["x"], d_u["y"])
            best_v = None
            min_dist = self.buffer

            for v, d_v in nodes_test:
                pv = (d_v["x"], d_v["y"])
                dist = np.hypot(pu[0] - pv[0], pu[1] - pv[1])
                if dist <= min_dist:
                    min_dist = dist
                    best_v = v

            mapping[u] = best_v

        # 2. Select origin-destination pairs in G_true
        all_pairs = []
        u_list = [u for u, _ in nodes_true]
        for i in range(len(u_list)):
            for j in range(i + 1, len(u_list)):
                all_pairs.append((u_list[i], u_list[j]))

        if not all_pairs:
            return 1.0

        if len(all_pairs) > self.max_pairs:
            indices = np.random.choice(len(all_pairs), size=self.max_pairs, replace=False)
            sampled_pairs = [all_pairs[idx] for idx in indices]
        else:
            sampled_pairs = all_pairs

        # 3. Evaluate shortest-path distances
        penalties = []

        for u, v in sampled_pairs:
            # Shortest path in G_true
            if not nx.has_path(G_true, u, v):
                continue

            try:
                d_true = nx.shortest_path_length(G_true, u, v, weight='weight')
            except nx.NetworkXNoPath:
                continue

            if d_true < 1e-4:
                continue

            # Counterpart nodes in G_test
            u_prime = mapping.get(u)
            v_prime = mapping.get(v)

            # If either node could not be mapped within max_buffer, maximum penalty
            if u_prime is None or v_prime is None:
                penalties.append(1.0)
                continue

            # Check if reachable in G_test
            if not nx.has_path(G_test, u_prime, v_prime):
                penalties.append(1.0)
                continue

            try:
                d_test = nx.shortest_path_length(G_test, u_prime, v_prime, weight='weight')
            except nx.NetworkXNoPath:
                penalties.append(1.0)
                continue

            # SpaceNet normalized route diff: min(1.0, |d_true - d_test| / d_true)
            diff = abs(d_true - d_test) / d_true
            penalties.append(min(1.0, float(diff)))

        if not penalties:
            return 1.0

        directed_score = max(0.0, 1.0 - float(np.mean(penalties)))
        return directed_score
