"""
Step-by-step ablation diagnostic for GraphTopologySimplifier.
Pinpoints exactly which stage prunes genuine roads in Swiss Alpine and Sahara.
"""

import os
import numpy as np
import networkx as nx
import torch

from topographer.pipeline import TopoGrapherEngine
from topographer.data.world_benchmark import WorldBenchmarkSuite
from topographer.metrics.apls import APLSMetric
from topographer.graph.simplifier import GraphTopologySimplifier


def diagnose_step_by_step():
    suite = WorldBenchmarkSuite(1024)
    engine = TopoGrapherEngine()
    apls_metric = APLSMetric(max_buffer_px=24.0, max_pairs_sample=200)

    for arch in ["swiss_alpine", "sahara_corridor", "paris_etoile"]:
        print(f"\n==================== {arch.upper()} ====================")
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
            raw_preds = engine.model.predict_graph_elements(
                tensor, keypoint_threshold=0.30, edge_threshold=0.45
            )

        vertices = raw_preds["vertices"]
        raw_edges = raw_preds["edges"]
        road_prob = raw_preds["dense"]["road_prob"]
        tangent = raw_preds["dense"]["tangent"]

        print(f"Initial raw predictions: {len(vertices)} vertices, {len(raw_edges)} edges")

        # Evaluate raw graph APLS
        G_raw = nx.Graph()
        for idx, v in enumerate(vertices):
            G_raw.add_node(idx, x=v["x"], y=v["y"])
        for e in raw_edges:
            u, v = e["source"], e["target"]
            d = np.hypot(vertices[u]["x"] - vertices[v]["x"], vertices[u]["y"] - vertices[v]["y"])
            G_raw.add_edge(u, v, weight=d, length_px=d)
        res_raw = apls_metric.compute_apls(gt_graph, G_raw)
        print(f"RAW: APLS={res_raw['apls']:.4f}, Prec={res_raw['apls_precision']:.4f}, Rec={res_raw['apls_recall']:.4f}")

        simplifier = GraphTopologySimplifier()

        # Step 1: Negative Evidence
        valid_edges = simplifier._validate_corridors_negative_evidence(
            vertices, raw_edges, road_prob, tangent, 1024, 1024
        )
        print(f"Step 1 (Negative evidence): {len(valid_edges)} edges remaining (dropped {len(raw_edges) - len(valid_edges)})")

        G1 = nx.Graph()
        for idx, v in enumerate(vertices):
            G1.add_node(idx, x=v["x"], y=v["y"])
        for e in valid_edges:
            u, v = e["source"], e["target"]
            d = np.hypot(vertices[u]["x"] - vertices[v]["x"], vertices[u]["y"] - vertices[v]["y"])
            G1.add_edge(u, v, weight=d, length_px=d)
        res1 = apls_metric.compute_apls(gt_graph, G1)
        print(f"  --> APLS={res1['apls']:.4f}, Prec={res1['apls_precision']:.4f}, Rec={res1['apls_recall']:.4f}")

        # Step 2: Angular Sector NMS
        nms_edges = simplifier._angular_sector_nms(vertices, valid_edges)
        print(f"Step 2 (Angular NMS): {len(nms_edges)} edges remaining (dropped {len(valid_edges) - len(nms_edges)})")

        G2 = nx.Graph()
        for idx, v in enumerate(vertices):
            G2.add_node(idx, x=v["x"], y=v["y"])
        for e in nms_edges:
            u, v = e["source"], e["target"]
            d = np.hypot(vertices[u]["x"] - vertices[v]["x"], vertices[u]["y"] - vertices[v]["y"])
            G2.add_edge(u, v, weight=d, length_px=d)
        res2 = apls_metric.compute_apls(gt_graph, G2)
        print(f"  --> APLS={res2['apls']:.4f}, Prec={res2['apls_precision']:.4f}, Rec={res2['apls_recall']:.4f}")

        # Step 4: Merge close nodes
        G4 = simplifier._merge_close_nodes(G2, simplifier.snap_dist * 0.5)
        print(f"Step 4 (Merge close nodes): {G4.number_of_nodes()} nodes, {G4.number_of_edges()} edges")

        # Step 5: Corner cutting chords
        G5 = simplifier._remove_corner_cutting_chords(G4)
        print(f"Step 5 (Corner cutting): {G5.number_of_nodes()} nodes, {G5.number_of_edges()} edges")

        # Step 6: Parallel ladder rungs
        G6 = simplifier._prune_parallel_ladder_rungs(G5)
        print(f"Step 6 (Ladder rungs): {G6.number_of_nodes()} nodes, {G6.number_of_edges()} edges")

        # Step 7: Resolve crossings
        G7 = simplifier._resolve_crossings_or_overpasses(G6)
        print(f"Step 7 (Crossings): {G7.number_of_nodes()} nodes, {G7.number_of_edges()} edges")

        # Step 8: Spur pruning
        G8 = simplifier._prune_short_spurs(G7, simplifier.min_spur)
        print(f"Step 8 (Spur pruning): {G8.number_of_nodes()} nodes, {G8.number_of_edges()} edges")

        # Step 9: Small components
        G9 = simplifier._filter_small_components(G8, simplifier.min_nodes)
        print(f"Step 9 (Small components): {G9.number_of_nodes()} nodes, {G9.number_of_edges()} edges")

        # Step 10: Degree-2 collapsing
        G10 = simplifier._collapse_degree_2_chains(G9)
        print(f"Step 10 (Degree-2 collapse): {G10.number_of_nodes()} nodes, {G10.number_of_edges()} edges")

        res_final = apls_metric.compute_apls(gt_graph, G10)
        print(f"FINAL: APLS={res_final['apls']:.4f}, Prec={res_final['apls_precision']:.4f}, Rec={res_final['apls_recall']:.4f}")


if __name__ == "__main__":
    diagnose_step_by_step()
