"""
Graph Topology Simplifier and Semantic Sanitization Engine for TopoGrapher.

Transforms raw neural network candidate predictions into physically and topologically
valid vector road graphs:
1. Visual Corridor Negative Evidence Validation (rejecting paths through buildings and water);
2. Angular Sector Non-Maximum Suppression (cone NMS, eliminating multi-degree spiderwebs);
3. Parallel Road Disambiguation & Ladder-Rung Chord Pruning;
4. Grade-Separation and Overpass Resolution (preventing false 4-way junctions on flyovers);
5. True Degree-2 Polyline Chain Collapsing (with closed roundabout cycle safety);
6. Dangling Micro-Spur Trimming and Spatial Clustering.
"""

from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np
import networkx as nx
from shapely.geometry import LineString, Point


class GraphTopologySimplifier:
    """
    Autonomous post-neural graph topology refiner and structural simplifier.
    """

    def __init__(
        self,
        min_road_prob: float = 0.22,
        max_negative_pixel_ratio: float = 0.25,
        negative_prob_threshold: float = 0.18,
        min_angular_alignment: float = 0.15,
        angular_cone_deg: float = 28.0,
        max_node_degree: int = 5,
        snap_distance_px: float = 10.0,
        min_spur_length_px: float = 16.0,
        min_component_nodes: int = 2,
        min_component_length_px: float = 75.0,
        rdp_epsilon: float = 2.0
    ):
        self.min_road_prob = min_road_prob
        self.max_neg_ratio = max_negative_pixel_ratio
        self.neg_thresh = negative_prob_threshold
        self.min_align = min_angular_alignment
        self.cone_deg = angular_cone_deg
        self.max_deg = max_node_degree
        self.snap_dist = snap_distance_px
        self.min_spur = min_spur_length_px
        self.min_nodes = min_component_nodes
        self.min_comp_len = min_component_length_px
        self.rdp_eps = rdp_epsilon

    def simplify_and_refine(
        self,
        vertices: List[Dict[str, Any]],
        raw_edges: List[Dict[str, Any]],
        road_prob_map: Optional[np.ndarray] = None,
        tangent_field: Optional[np.ndarray] = None,
        image_shape: Tuple[int, int] = (1024, 1024)
    ) -> nx.Graph:
        """
        Executes end-to-end topological refinement pipeline.
        """
        h, w = image_shape

        # Step 1: Multi-hypothesis visual corridor validation (Negative Evidence Filter)
        valid_edges = self._validate_corridors_negative_evidence(
            vertices, raw_edges, road_prob_map, tangent_field, h, w
        )

        # Step 1b: Intermediate-node chord pruning (Gabriel corridor check)
        non_skipping_edges = self._prune_intermediate_vertex_skipping_chords(vertices, valid_edges)

        # Step 2: Proximity-aware Angular Sector NMS at each node (kills degree 8-20 spiderwebs)
        nms_edges = self._angular_sector_nms(vertices, non_skipping_edges)

        # Step 3: Construct Initial Graph
        G = nx.Graph()
        for idx, v in enumerate(vertices):
            G.add_node(
                idx,
                x=float(v["x"]),
                y=float(v["y"]),
                score=float(v.get("score", 1.0)),
                type=int(v.get("type", 0))
            )

        for e in nms_edges:
            u, v = e["source"], e["target"]
            d = float(np.hypot(vertices[u]["x"] - vertices[v]["x"], vertices[u]["y"] - vertices[v]["y"]))
            G.add_edge(
                u, v,
                weight=d,
                length_px=d,
                probability=float(e.get("probability", 1.0)),
                road_class=int(e.get("road_class", 1))
            )

        if G.number_of_nodes() < 2:
            return G

        # Step 4: Merge close duplicate clusters
        G = self._merge_close_nodes(G, self.snap_dist * 0.5)

        # Step 5: Prune triangular corner-cutting diagonal chords
        G = self._remove_corner_cutting_chords(G)

        # Step 6: Prune ladder rungs between parallel roads
        G = self._prune_parallel_ladder_rungs(G)

        # Step 7: Resolve planar edge crossings vs overpasses
        G = self._resolve_crossings_or_overpasses(G)

        # Step 8: Prune short dead-end micro-spurs (< 16px)
        G = self._prune_short_spurs(G, self.min_spur)

        # Step 9: Filter microscopic disconnected noise components
        G = self._filter_small_components(G, self.min_nodes, self.min_comp_len)

        # Step 10: Bidirectional degree-2 chain collapsing into polyline edges
        G = self._collapse_degree_2_chains(G)

        return G

    def _validate_corridors_negative_evidence(
        self,
        vertices: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        road_prob: Optional[np.ndarray],
        tangent: Optional[np.ndarray],
        h: int,
        w: int,
        num_samples: int = 14
    ) -> List[Dict[str, Any]]:
        """
        Samples the line segment of each edge. Discards edges that traverse non-road
        terrain (buildings, water, parks) or conflict with continuous heading.
        """
        if road_prob is None:
            return edges

        map_h, map_w = road_prob.shape
        scale_x = map_w / float(w)
        scale_y = map_h / float(h)

        passed = []
        for e in edges:
            u = e["source"]
            v = e["target"]
            p1 = (vertices[u]["x"], vertices[u]["y"])
            p2 = (vertices[v]["x"], vertices[v]["y"])

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dist = np.hypot(dx, dy)
            if dist < 6.0:
                continue

            phi = np.arctan2(dy, dx)
            cos_2phi = np.cos(2.0 * phi)
            sin_2phi = np.sin(2.0 * phi)

            prob_vals = []
            align_vals = []

            for t in np.linspace(0.1, 0.9, num_samples):
                sx = int(round((p1[0] + t * dx) * scale_x))
                sy = int(round((p1[1] + t * dy) * scale_y))

                if 0 <= sx < map_w and 0 <= sy < map_h:
                    prob_vals.append(road_prob[sy, sx])
                    if tangent is not None:
                        dot = tangent[0, sy, sx] * cos_2phi + tangent[1, sy, sx] * sin_2phi
                        align_vals.append(dot)

            if not prob_vals:
                continue

            mean_p = float(np.mean(prob_vals))
            low_p_ratio = float(sum(1 for p in prob_vals if p < self.neg_thresh) / len(prob_vals))

            # Discard edge if it traverses buildings or empty space
            if low_p_ratio > self.max_neg_ratio or mean_p < self.min_road_prob:
                continue

            # Discard edge if direction conflicts with road tangent
            if align_vals:
                mean_align = float(np.mean(align_vals))
                if mean_align < self.min_align and float(e.get("probability", 0)) < 0.85:
                    continue

            passed.append(e)

        return passed

    def _prune_intermediate_vertex_skipping_chords(
        self,
        vertices: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        max_corridor_deviation_px: float = 12.0
    ) -> List[Dict[str, Any]]:
        """
        Gabriel / RNG corridor filter: Prunes candidate chords that bypass intermediate vertices
        along the same road corridor.
        If an intermediate vertex w lies strictly between u and v within max_corridor_deviation_px
        of the line segment (u, v), then (u, v) is a redundant multi-hop chord and is eliminated.
        """
        if len(vertices) < 3 or not edges:
            return edges

        coords = np.array([[float(v["x"]), float(v["y"])] for v in vertices], dtype=np.float32)
        passed_edges = []

        for e in edges:
            u, v = e["source"], e["target"]
            pu = coords[u]
            pv = coords[v]
            vec = pv - pu
            L2 = float(np.dot(vec, vec))
            if L2 < 225.0:  # Edges shorter than 15px cannot skip intermediate vertices
                passed_edges.append(e)
                continue

            L = np.sqrt(L2)
            diff = coords - pu
            t = np.dot(diff, vec) / L2  # (N,)

            # Interior vertices along segment span
            mask = (t > 0.15) & (t < 0.85)
            mask[u] = False
            mask[v] = False

            if not np.any(mask):
                passed_edges.append(e)
                continue

            # Perpendicular distance for interior vertices
            perp_dist = np.abs(vec[0] * diff[mask, 1] - vec[1] * diff[mask, 0]) / L
            if np.any(perp_dist < max_corridor_deviation_px):
                # An intermediate road node exists along this span; eliminate the bypassing chord!
                continue

            passed_edges.append(e)

        return passed_edges

    def _angular_sector_nms(
        self,
        vertices: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        At every node, enforces angular non-maximum suppression: within any cone of
        cone_deg (e.g. 28 degrees), at most ONE candidate edge can exit.
        Uses proximity-weighted priority scoring:
            score = prob / (1.0 + 0.005 * distance)
        to favor connecting adjacent sequential neighbors over distant leaps.
        Directly eliminates multi-degree spiderwebs.
        """
        node_edges = {i: [] for i in range(len(vertices))}
        for idx, e in enumerate(edges):
            u, v = e["source"], e["target"]
            prob = float(e.get("probability", 1.0))
            dist = float(e.get("length_px", np.hypot(vertices[u]["x"] - vertices[v]["x"], vertices[u]["y"] - vertices[v]["y"])))
            priority_score = prob / (1.0 + 0.005 * dist)
            node_edges[u].append((v, priority_score, idx, dist))
            node_edges[v].append((u, priority_score, idx, dist))

        rejected_edge_indices = set()

        for u in range(len(vertices)):
            incident = node_edges[u]
            if not incident:
                continue

            # Sort by descending priority score (high confidence, low distance)
            incident.sort(key=lambda item: item[1], reverse=True)
            pu = (vertices[u]["x"], vertices[u]["y"])
            accepted_bearings = []

            for v, score, e_idx, dist in incident:
                pv = (vertices[v]["x"], vertices[v]["y"])
                bearing = np.degrees(np.arctan2(pv[1] - pu[1], pv[0] - pu[0])) % 360.0

                conflict = False
                for existing in accepted_bearings:
                    diff = abs(bearing - existing)
                    if diff > 180.0:
                        diff = 360.0 - diff
                    if diff < self.cone_deg:
                        conflict = True
                        break

                if conflict:
                    rejected_edge_indices.add(e_idx)
                else:
                    accepted_bearings.append(bearing)

        return [e for idx, e in enumerate(edges) if idx not in rejected_edge_indices]

    def _merge_close_nodes(self, G: nx.Graph, threshold: float) -> nx.Graph:
        """Merges adjacent nodes located closer than threshold."""
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

    def _prune_parallel_ladder_rungs(self, G: nx.Graph) -> nx.Graph:
        """
        Detects and removes transverse ladder rungs connecting two parallel road lanes.
        """
        H = G.copy()
        edges_to_remove = set()

        for u, v in list(H.edges()):
            pu = (H.nodes[u]["x"], H.nodes[u]["y"])
            pv = (H.nodes[v]["x"], H.nodes[v]["y"])
            rung_vec = (pv[0] - pu[0], pv[1] - pu[1])
            rung_len = float(np.hypot(rung_vec[0], rung_vec[1]))

            # Ladder rungs are typically short (15-55 px)
            if rung_len > 55.0 or rung_len < 12.0:
                continue

            # Check neighbors of u and v
            u_nbrs = [n for n in H.neighbors(u) if n != v]
            v_nbrs = [n for n in H.neighbors(v) if n != u]

            if not u_nbrs or not v_nbrs:
                continue

            # For each neighbor of u and v, compute their corridor vectors
            for nu in u_nbrs:
                p_nu = (H.nodes[nu]["x"], H.nodes[nu]["y"])
                vec_u = (p_nu[0] - pu[0], p_nu[1] - pu[1])
                len_u = np.hypot(vec_u[0], vec_u[1])
                if len_u < 1e-4:
                    continue

                for nv in v_nbrs:
                    p_nv = (H.nodes[nv]["x"], H.nodes[nv]["y"])
                    vec_v = (p_nv[0] - pv[0], p_nv[1] - pv[1])
                    len_v = np.hypot(vec_v[0], vec_v[1])
                    if len_v < 1e-4:
                        continue

                    # Dot product between the two road corridors (must be roughly parallel: |dot| > 0.85)
                    corr_dot = abs((vec_u[0] * vec_v[0] + vec_u[1] * vec_v[1]) / (len_u * len_v))

                    # Dot product between rung and corridors (must be roughly perpendicular: |dot| < 0.40)
                    rung_dot_u = abs((rung_vec[0] * vec_u[0] + rung_vec[1] * vec_u[1]) / (rung_len * len_u))

                    if corr_dot > 0.85 and rung_dot_u < 0.40:
                        # Rung is perpendicular to two parallel streets!
                        # If rung has no further continuation across the streets, prune it!
                        if H.degree(u) <= 3 and H.degree(v) <= 3:
                            edges_to_remove.add((min(u, v), max(u, v)))

        for u, v in edges_to_remove:
            if H.has_edge(u, v):
                H.remove_edge(u, v)

        return H

    def _resolve_crossings_or_overpasses(self, G: nx.Graph, max_splits: int = 40) -> nx.Graph:
        """
        Detects intersecting lines. Distinguishes overpasses (grade-separated bridges)
        from true at-grade 4-way intersections.
        """
        H = G.copy()
        splits = 0

        while splits < max_splits:
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

                    # Fast AABB prefilter
                    if max_x1 < min_x2 or min_x1 > max_x2 or max_y1 < min_y2 or min_y1 > max_y2:
                        continue

                    line1 = LineString([p1a, p1b])
                    line2 = LineString([p2a, p2b])

                    if line1.crosses(line2):
                        inter = line1.intersection(line2)
                        if isinstance(inter, Point):
                            # Check for overpass vs at-grade intersection:
                            # If both roads are high-speed freeways/arterials (class 0 or 1) and crossing angle is ~90 deg
                            # with neither edge having branch ramps at the crossing point, mark as overpass!
                            v1_dir = (p1b[0] - p1a[0], p1b[1] - p1a[1])
                            v2_dir = (p2b[0] - p2a[0], p2b[1] - p2a[1])
                            l1 = np.hypot(v1_dir[0], v1_dir[1])
                            l2 = np.hypot(v2_dir[0], v2_dir[1])
                            dot = (v1_dir[0] * v2_dir[0] + v1_dir[1] * v2_dir[1]) / (l1 * l2 + 1e-6)
                            crossing_angle = np.degrees(np.arccos(np.clip(abs(dot), 0.0, 1.0)))

                            is_overpass = (
                                crossing_angle >= 70.0 and
                                d1.get("road_class", 1) == 0 and
                                d2.get("road_class", 1) <= 1 and
                                l1 > 80.0 and l2 > 80.0
                            )

                            if is_overpass:
                                # Retain grade separation: tag bridge/layer, DO NOT create 4-way node!
                                d1["layer"] = 1
                                d1["bridge"] = "yes"
                                d2["layer"] = 0
                                continue

                            # True at-grade intersection: create shared vertex
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

                            splits += 1
                            found_intersection = True
                            break

                if found_intersection:
                    break

            if not found_intersection:
                break

        return H

    def _collapse_degree_2_chains(self, G: nx.Graph) -> nx.Graph:
        """
        Collapses chains of degree-2 nodes into single edges with attached polyline geometry.
        Only structural junctions (degree >= 3) and terminals (degree == 1) remain as graph vertices!
        Includes cycle safety for roundabouts and ring roads.
        """
        H = G.copy()
        degree_2_nodes = {n for n, deg in H.degree() if deg == 2}
        visited_nodes = set()

        for start_n in list(degree_2_nodes):
            if start_n in visited_nodes or not H.has_node(start_n):
                continue

            nbrs = list(H.neighbors(start_n))
            if len(nbrs) != 2:
                continue

            # Bidirectional trace
            # 1. Walk in Direction A
            chain_a = [start_n]
            prev = start_n
            curr = nbrs[0]
            chain_nodes_set = {start_n}

            while H.has_node(curr) and H.degree(curr) == 2:
                if curr in chain_nodes_set:
                    # Circular ring road detected (roundabout)!
                    break
                chain_nodes_set.add(curr)
                chain_a.append(curr)
                next_candidates = [nb for nb in H.neighbors(curr) if nb != prev]
                if not next_candidates:
                    break
                prev = curr
                curr = next_candidates[0]

            chain_a.append(curr)  # Anchor junction A

            # 2. Walk in Direction B
            chain_b = []
            prev = start_n
            curr = nbrs[1]

            while H.has_node(curr) and H.degree(curr) == 2:
                if curr in chain_nodes_set:
                    break
                chain_nodes_set.add(curr)
                chain_b.append(curr)
                next_candidates = [nb for nb in H.neighbors(curr) if nb != prev]
                if not next_candidates:
                    break
                prev = curr
                curr = next_candidates[0]

            chain_b.append(curr)  # Anchor junction B

            # Full chain from Anchor A to Anchor B: chain_a[::-1] + chain_b
            full_chain = chain_a[::-1] + chain_b
            u_anchor = full_chain[0]
            v_anchor = full_chain[-1]

            # Closed circle check (e.g. isolated roundabout with no junctions)
            if u_anchor == v_anchor and len(full_chain) > 3:
                # Keep 3 equidistant anchor points to maintain the loop
                continue

            # Extract polyline coordinates
            polyline_coords = [(H.nodes[n]["x"], H.nodes[n]["y"]) for n in full_chain]
            total_dist = sum(
                np.hypot(polyline_coords[k + 1][0] - polyline_coords[k][0], polyline_coords[k + 1][1] - polyline_coords[k][1])
                for k in range(len(polyline_coords) - 1)
            )

            # Mark mid nodes as visited
            mid_nodes = full_chain[1:-1]
            for n in mid_nodes:
                visited_nodes.add(n)
                if H.has_node(n):
                    H.remove_node(n)

            if H.has_node(u_anchor) and H.has_node(v_anchor) and u_anchor != v_anchor:
                # Add single unified edge with full polyline geometry
                H.add_edge(
                    u_anchor, v_anchor,
                    weight=float(total_dist),
                    length_px=float(total_dist),
                    polyline=polyline_coords,
                    road_class=1
                )

        return H

    def _prune_short_spurs(self, G: nx.Graph, threshold: float) -> nx.Graph:
        """Trims dead-end leaf spurs shorter than threshold."""
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
                    if dist < threshold and H.degree(neighbor) >= 2:
                        H.remove_node(n)
                        changed = True
        return H

    def _filter_small_components(self, G: nx.Graph, min_nodes: int, min_total_length_px: float = 75.0) -> nx.Graph:
        """
        Discards microscopic disconnected noise components.
        Preserves components with total road length >= min_total_length_px (e.g. a long 2-node highway stretch or bridge).
        """
        H = G.copy()
        for comp in list(nx.connected_components(H)):
            sub = H.subgraph(comp)
            comp_len = sum(
                d.get("length_px", np.hypot(H.nodes[u]["x"] - H.nodes[v]["x"], H.nodes[u]["y"] - H.nodes[v]["y"]))
                for u, v, d in sub.edges(data=True)
            )
            if len(comp) < min_nodes and comp_len < min_total_length_px and len(comp) < len(H):
                for node in comp:
                    H.remove_node(node)
        return H
