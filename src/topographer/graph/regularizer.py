"""
Graph Topological Regularizer and Geometric Cleanup Engine.

Performs robust post-processing on extracted road graphs:
1. Cluster & Snap: Merges redundant close-proximity vertices within epsilon_snap;
2. Dangling Spur Pruning: Removes spurious dead-end spurs shorter than min_spur_length;
3. Gap Bridging: Snaps colinear disconnected terminal endpoints;
4. Degree-2 Simplification: Compresses redundant intermediate nodes along linear corridors via Ramer-Douglas-Peucker (RDP);
5. Component Filtering: Retains significant connected components.
"""

from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np
import networkx as nx
from shapely.geometry import LineString


class GraphRegularizer:
    """
    Applies topological invariants and geometric smoothing to raw road graphs.
    """

    def __init__(
        self,
        snap_distance_px: float = 24.0,
        min_spur_length_px: float = 12.0,
        min_component_nodes: int = 3,
        rdp_epsilon: float = 2.0
    ):
        """
        Parameters
        ----------
        snap_distance_px : float
            Maximum distance to merge adjacent nodes or connect facing endpoints.
        min_spur_length_px : float
            Spurs shorter than this with degree=1 are pruned.
        min_component_nodes : int
            Isolated disconnected subgraphs with fewer nodes are discarded.
        rdp_epsilon : float
            Tolerance in pixels for Ramer-Douglas-Peucker edge simplification.
        """
        self.snap_dist = snap_distance_px
        self.min_spur_len = min_spur_length_px
        self.min_nodes = min_component_nodes
        self.rdp_eps = rdp_epsilon

    def regularize(self, graph: nx.Graph) -> nx.Graph:
        """
        Executes complete regularizer pipeline on the graph.
        """
        G = graph.copy()
        if G.number_of_nodes() < 2:
            return G

        # 1. Merge close-proximity vertices (sub-pixel / duplicate clusters)
        G = self._merge_close_vertices(G, self.snap_dist * 0.5)

        # 2. Prune false corner-cutting diagonal chords through city blocks
        G = self._remove_corner_cutting_chords(G)

        # 3. Resolve planar edge crossings into shared 4-way intersections
        G = self._resolve_planar_crossings(G)

        # 4. Connect facing dangling endpoints (bridge small gaps)
        G = self._bridge_dangling_endpoints(G, self.snap_dist)

        # 5. Prune short spurious dead-ends (degree 1 spurs)
        G = self._prune_short_spurs(G, self.min_spur_len)

        # 6. Remove microscopic disconnected components
        G = self._filter_small_components(G, self.min_nodes)

        # 7. Simplify collinear degree-2 chains using RDP
        G = self._simplify_degree_2_chains(G, self.rdp_eps)

        return G

    def _merge_close_vertices(self, G: nx.Graph, threshold: float) -> nx.Graph:
        """Merges vertices located closer than threshold."""
        H = G.copy()
        nodes = list(H.nodes(data=True))

        merged_into = {}
        for i in range(len(nodes)):
            u, d_u = nodes[i]
            if u in merged_into:
                continue
            pu = (d_u["x"], d_u["y"])

            for j in range(i + 1, len(nodes)):
                v, d_v = nodes[j]
                if v in merged_into:
                    continue
                pv = (d_v["x"], d_v["y"])
                dist = np.hypot(pu[0] - pv[0], pu[1] - pv[1])

                if dist < threshold:
                    merged_into[v] = u

        for v, u in merged_into.items():
            if H.has_node(v) and H.has_node(u):
                # Transfer edges from v to u
                for neighbor, edata in list(H[v].items()):
                    if neighbor != u and not H.has_edge(u, neighbor):
                        H.add_edge(u, neighbor, **edata)
                H.remove_node(v)

        return H

    def _remove_corner_cutting_chords(self, G: nx.Graph) -> nx.Graph:
        """Removes diagonal corner-cutting shortcut edges forming triangles across 90-deg turns."""
        H = G.copy()
        if H.number_of_nodes() < 3 or H.number_of_edges() < 3:
            return H

        edges_to_remove = set()
        for u, v in list(H.edges()):
            common_neighbors = set(H.neighbors(u)).intersection(H.neighbors(v))
            for corner in common_neighbors:
                pu = (H.nodes[u]["x"], H.nodes[u]["y"])
                pv = (H.nodes[v]["x"], H.nodes[v]["y"])
                pc = (H.nodes[corner]["x"], H.nodes[corner]["y"])

                d_uv = float(np.hypot(pu[0] - pv[0], pu[1] - pv[1]))
                d_uc = float(np.hypot(pu[0] - pc[0], pu[1] - pc[1]))
                d_vc = float(np.hypot(pv[0] - pc[0], pv[1] - pc[1]))

                # The chord (u, v) must be the hypotenuse
                if d_uv > d_uc and d_uv > d_vc:
                    v1 = (pu[0] - pc[0], pu[1] - pc[1])
                    v2 = (pv[0] - pc[0], pv[1] - pc[1])
                    dot = v1[0] * v2[0] + v1[1] * v2[1]
                    norm = d_uc * d_vc
                    if norm > 1e-4:
                        cos_angle = dot / norm
                        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                        if 55.0 <= angle_deg <= 125.0:
                            edges_to_remove.add((min(u, v), max(u, v)))

        for u, v in edges_to_remove:
            if H.has_edge(u, v):
                H.remove_edge(u, v)

        return H

    def _resolve_planar_crossings(self, G: nx.Graph, max_splits: int = 50) -> nx.Graph:
        """Resolves overlapping crossing lines into proper 4-way intersections."""
        from shapely.geometry import LineString, Point
        H = G.copy()
        splits_count = 0

        while splits_count < max_splits:
            edge_list = list(H.edges(data=True))
            found_intersection = False

            for i in range(len(edge_list)):
                u1, v1, d1 = edge_list[i]
                p1a = (H.nodes[u1]["x"], H.nodes[u1]["y"])
                p1b = (H.nodes[v1]["x"], H.nodes[v1]["y"])

                min_x1, max_x1 = min(p1a[0], p1b[0]), max(p1a[0], p1b[0])
                min_y1, max_y1 = min(p1a[1], p1b[1]), max(p1a[1], p1b[1])

                for j in range(i + 1, len(edge_list)):
                    u2, v2, d2 = edge_list[j]
                    if u1 in (u2, v2) or v1 in (u2, v2):
                        continue

                    p2a = (H.nodes[u2]["x"], H.nodes[u2]["y"])
                    p2b = (H.nodes[v2]["x"], H.nodes[v2]["y"])

                    min_x2, max_x2 = min(p2a[0], p2b[0]), max(p2a[0], p2b[0])
                    min_y2, max_y2 = min(p2a[1], p2b[1]), max(p2a[1], p2b[1])

                    # Fast AABB rejection filter
                    if max_x1 < min_x2 or min_x1 > max_x2 or max_y1 < min_y2 or min_y1 > max_y2:
                        continue

                    line1 = LineString([p1a, p1b])
                    line2 = LineString([p2a, p2b])

                    if line1.crosses(line2):
                        inter = line1.intersection(line2)
                        if isinstance(inter, Point):
                            new_node = max(H.nodes()) + 1
                            H.add_node(new_node, x=float(inter.x), y=float(inter.y), score=0.9, type=0)

                            if H.has_edge(u1, v1):
                                H.remove_edge(u1, v1)
                            if H.has_edge(u2, v2):
                                H.remove_edge(u2, v2)

                            d_u1 = float(np.hypot(p1a[0] - inter.x, p1a[1] - inter.y))
                            d_v1 = float(np.hypot(p1b[0] - inter.x, p1b[1] - inter.y))
                            d_u2 = float(np.hypot(p2a[0] - inter.x, p2a[1] - inter.y))
                            d_v2 = float(np.hypot(p2b[0] - inter.x, p2b[1] - inter.y))

                            H.add_edge(u1, new_node, weight=d_u1, length_px=d_u1, road_class=d1.get("road_class", 1))
                            H.add_edge(new_node, v1, weight=d_v1, length_px=d_v1, road_class=d1.get("road_class", 1))
                            H.add_edge(u2, new_node, weight=d_u2, length_px=d_u2, road_class=d2.get("road_class", 1))
                            H.add_edge(new_node, v2, weight=d_v2, length_px=d_v2, road_class=d2.get("road_class", 1))

                            splits_count += 1
                            found_intersection = True
                            break
                if found_intersection:
                    break

            if not found_intersection:
                break

        return H

    def _bridge_dangling_endpoints(self, G: nx.Graph, threshold: float) -> nx.Graph:
        """Connects dangling dead-end nodes that face each other."""
        H = G.copy()
        dangling = [n for n, deg in H.degree() if deg == 1]

        for i in range(len(dangling)):
            u = dangling[i]
            if not H.has_node(u) or H.degree(u) != 1:
                continue
            pu = (H.nodes[u]["x"], H.nodes[u]["y"])

            best_v = None
            min_dist = threshold

            for j in range(i + 1, len(dangling)):
                v = dangling[j]
                if not H.has_node(v) or H.degree(v) != 1:
                    continue
                pv = (H.nodes[v]["x"], H.nodes[v]["y"])
                dist = np.hypot(pu[0] - pv[0], pu[1] - pv[1])

                if dist < min_dist:
                    min_dist = dist
                    best_v = v

            if best_v is not None and not H.has_edge(u, best_v):
                H.add_edge(
                    u, best_v,
                    weight=float(min_dist),
                    probability=0.85,
                    road_class=1,
                    length_px=float(min_dist),
                    bridged=True
                )

        return H

    def _prune_short_spurs(self, G: nx.Graph, threshold: float) -> nx.Graph:
        """Iteratively trims dead-end leaves that are shorter than threshold."""
        H = G.copy()
        changed = True
        while changed:
            changed = False
            for n in list(H.nodes()):
                if H.degree(n) == 1:
                    neighbor = list(H.neighbors(n))[0]
                    pn = (H.nodes[n]["x"], H.nodes[n]["y"])
                    p_nb = (H.nodes[neighbor]["x"], H.nodes[neighbor]["y"])
                    dist = np.hypot(pn[0] - p_nb[0], pn[1] - p_nb[1])

                    # Prune if short and neighbor has other connections
                    if dist < threshold and H.degree(neighbor) >= 2:
                        H.remove_node(n)
                        changed = True
        return H

    def _filter_small_components(self, G: nx.Graph, min_nodes: int) -> nx.Graph:
        """Discards disconnected components with fewer than min_nodes."""
        H = G.copy()
        components = list(nx.connected_components(H))
        for comp in components:
            if len(comp) < min_nodes and len(comp) < len(H):
                for node in comp:
                    H.remove_node(node)
        return H

    def _simplify_degree_2_chains(self, G: nx.Graph, epsilon: float) -> nx.Graph:
        """Simplifies straight chains of degree-2 nodes using Ramer-Douglas-Peucker."""
        H = G.copy()
        degree_2_nodes = {n for n, deg in H.degree() if deg == 2}
        visited_nodes = set()

        for start_n in list(degree_2_nodes):
            if start_n in visited_nodes or not H.has_node(start_n):
                continue

            # Trace chain in both directions until encountering junction (deg != 2)
            chain = [start_n]
            visited_nodes.add(start_n)

            # Direction A
            curr = start_n
            prev = None
            while True:
                neighbors = [nb for nb in H.neighbors(curr) if nb != prev]
                if not neighbors:
                    break
                next_n = neighbors[0]
                if H.degree(next_n) == 2 and next_n not in visited_nodes:
                    chain.append(next_n)
                    visited_nodes.add(next_n)
                    prev = curr
                    curr = next_n
                else:
                    chain.append(next_n)  # Anchor junction
                    break

            # If chain has enough vertices, check RDP simplification
            if len(chain) >= 3:
                # Get coordinates
                coords = [(H.nodes[n]["x"], H.nodes[n]["y"]) for n in chain]
                line = LineString(coords)
                simplified = line.simplify(epsilon, preserve_topology=True)
                simp_coords = list(simplified.coords)

                if len(simp_coords) < len(coords):
                    # We can remove intermediate vertices that were smoothed away
                    endpoints = (chain[0], chain[-1])
                    total_dist = sum(
                        np.hypot(coords[k+1][0] - coords[k][0], coords[k+1][1] - coords[k][1])
                        for k in range(len(coords) - 1)
                    )
                    # Remove interior nodes of this segment
                    for n in chain[1:-1]:
                        if H.has_node(n) and H.degree(n) == 2:
                            H.remove_node(n)

                    if H.has_node(endpoints[0]) and H.has_node(endpoints[1]):
                        if not H.has_edge(endpoints[0], endpoints[1]):
                            H.add_edge(
                                endpoints[0], endpoints[1],
                                weight=float(total_dist),
                                length_px=float(total_dist),
                                road_class=1
                            )

        return H
