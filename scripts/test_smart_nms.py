"""
Test Relative Neighborhood / Non-skipping Angular NMS
Compare standard probability-sorted NMS vs distance-weighted / empty-corridor NMS.
"""

import numpy as np
import networkx as nx
from topographer.data.world_benchmark import WorldBenchmarkSuite
from topographer.metrics.apls import APLSMetric
from topographer.pipeline import TopoGrapherEngine
import torch


def test_nms_fix():
    suite = WorldBenchmarkSuite(1024)
    engine = TopoGrapherEngine()
    apls_metric = APLSMetric(max_buffer_px=24.0, max_pairs_sample=200)

    for arch in ["sahara_corridor", "swiss_alpine", "paris_etoile", "manhattan_grid"]:
        scene = suite.generate_world_scene(arch, seed=2026)

        gt_graph = nx.Graph()
        for v_i, (x, y, t) in enumerate(scene.vertices):
            gt_graph.add_node(v_i, x=x, y=y, type=t)
        for u, v, c in scene.edges:
            p1 = (scene.vertices[u][0], scene.vertices[u][1])
            p2 = (scene.vertices[v][0], scene.vertices[v][1])
            d = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
            gt_graph.add_edge(u, v, weight=d, length_px=d, road_class=c)

        img_f = scene.image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).to(engine.device)

        with torch.no_grad():
            raw_preds = engine.model.predict_graph_elements(tensor, keypoint_threshold=0.30, edge_threshold=0.45)

        vertices = raw_preds["vertices"]
        raw_edges = raw_preds["edges"]
        road_prob = raw_preds["dense"]["road_prob"]
        tangent = raw_preds["dense"]["tangent"]

        # Strategy A: Current NMS (sort by probability only)
        # Strategy B: Prune edges that have an intermediate vertex along the segment FIRST,
        # then in each angular cone keep the closest valid vertex!
        
        # Intermediate vertex filter:
        # If vertex w lies within 10px of segment uv, and projection of w onto uv lies between 0.15 and 0.85,
        # then (u, v) is a chord skipping w! Remove (u, v)!
        coords = np.array([[v["x"], v["y"]] for v in vertices])
        
        non_skipping_edges = []
        for e in raw_edges:
            u, v = e["source"], e["target"]
            pu = coords[u]
            pv = coords[v]
            vec = pv - pu
            L2 = np.dot(vec, vec)
            if L2 < 1e-4:
                continue
            
            # Check all other vertices w
            # projection t = dot(pw - pu, vec) / L2
            diff = coords - pu  # (N, 2)
            t = np.dot(diff, vec) / L2  # (N,)
            
            # Find vertices strictly between u and v
            between = (t > 0.15) & (t < 0.85)
            if np.any(between):
                # Perpendicular distance
                proj = pu + t[:, None] * vec[None, :]
                perp_dist = np.hypot(coords[:, 0] - proj[:, 0], coords[:, 1] - proj[:, 1])
                # If any vertex is within 10px of the corridor line, uv is skipping it!
                if np.any(between & (perp_dist < 10.0)):
                    continue
            
            non_skipping_edges.append(e)

        print(f"\n[{arch.upper()}] Raw edges: {len(raw_edges)} -> Non-skipping edges: {len(non_skipping_edges)}")

        # Now apply Angular Sector NMS on non_skipping_edges, sorting by PROXIMITY (prefer closer adjacent node):
        node_edges = {i: [] for i in range(len(vertices))}
        for idx, e in enumerate(non_skipping_edges):
            u, v = e["source"], e["target"]
            prob = float(e.get("probability", 1.0))
            dist = float(e.get("length_px", np.hypot(vertices[u]["x"] - vertices[v]["x"], vertices[u]["y"] - vertices[v]["y"])))
            # Priority score: high probability, short distance
            score = prob / (1.0 + 0.005 * dist)
            node_edges[u].append((v, score, idx, dist))
            node_edges[v].append((u, score, idx, dist))

        rejected_indices = set()
        for u in range(len(vertices)):
            incident = node_edges[u]
            if not incident:
                continue
            # Sort by priority score
            incident.sort(key=lambda item: item[1], reverse=True)
            pu = coords[u]
            accepted_bearings = []

            for v, sc, e_idx, dist in incident:
                pv = coords[v]
                bearing = np.degrees(np.arctan2(pv[1] - pu[1], pv[0] - pu[0])) % 360.0
                conflict = False
                for existing in accepted_bearings:
                    diff = abs(bearing - existing)
                    if diff > 180.0:
                        diff = 360.0 - diff
                    if diff < 28.0:  # 28 deg cone
                        conflict = True
                        break
                if conflict:
                    rejected_indices.add(e_idx)
                else:
                    accepted_bearings.append(bearing)

        refined_edges = [e for idx, e in enumerate(non_skipping_edges) if idx not in rejected_indices]
        print(f"[{arch.upper()}] Refined edges after Smart NMS: {len(refined_edges)}")

        G_test = nx.Graph()
        for idx, v in enumerate(vertices):
            G_test.add_node(idx, x=v["x"], y=v["y"])
        for e in refined_edges:
            u, v = e["source"], e["target"]
            d = np.hypot(vertices[u]["x"] - vertices[v]["x"], vertices[u]["y"] - vertices[v]["y"])
            G_test.add_edge(u, v, weight=d, length_px=d)

        res = apls_metric.compute_apls(gt_graph, G_test)
        print(f"[{arch.upper()}] Smart NMS APLS: {res['apls']:.4f} (Prec: {res['apls_precision']:.4f}, Rec: {res['apls_recall']:.4f})")


if __name__ == "__main__":
    test_nms_fix()
