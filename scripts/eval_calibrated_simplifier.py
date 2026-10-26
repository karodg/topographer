"""
Evaluation script for Calibrated Simplifier across all 10 World Archetypes.
Calculates SpaceNet APLS, Precision, Recall, Harmonic F1, Reachability, and Routing.
"""

import time
import numpy as np
import networkx as nx
import torch

from topographer.data.world_benchmark import WorldBenchmarkSuite
from topographer.pipeline import TopoGrapherEngine
from topographer.metrics.apls import APLSMetric
from topographer.graph.simplifier import GraphTopologySimplifier


def evaluate_calibrated():
    suite = WorldBenchmarkSuite(1024)
    engine = TopoGrapherEngine()
    apls_metric = APLSMetric(max_buffer_px=24.0, max_pairs_sample=350)

    # Instantiate calibrated simplifier
    simplifier = GraphTopologySimplifier(
        min_road_prob=0.24,
        max_negative_pixel_ratio=0.25,
        negative_prob_threshold=0.18,
        min_angular_alignment=0.15,
        angular_cone_deg=28.0,
        max_node_degree=5,
        snap_distance_px=10.0,
        min_spur_length_px=15.0,
        min_component_nodes=2,
        rdp_epsilon=2.0
    )

    archetypes = WorldBenchmarkSuite.ARCHETYPES
    records = []

    print("====================================================================================================")
    print(" Archetype             | Nodes | Edges | Km     | Reach % | APLS   | Prec % | Rec %  | F1     | A*   ")
    print("====================================================================================================")

    for arch in archetypes:
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
                tensor, keypoint_threshold=0.28, edge_threshold=0.40
            )

        vertices = raw_preds["vertices"]
        raw_edges = raw_preds["edges"]
        road_prob = raw_preds["dense"]["road_prob"]
        tangent = raw_preds["dense"]["tangent"]

        # Run simplifier
        G_prop = simplifier.simplify_and_refine(
            vertices=vertices,
            raw_edges=raw_edges,
            road_prob_map=road_prob,
            tangent_field=tangent,
            image_shape=(1024, 1024)
        )

        apls_res = apls_metric.compute_apls(gt_graph, G_prop)

        # Telemetry
        num_n = G_prop.number_of_nodes()
        num_e = G_prop.number_of_edges()
        tot_km = sum(d.get("length_px", 0.0) for _, _, d in G_prop.edges(data=True)) * 0.5 / 1000.0

        if num_n > 0:
            comps = sorted(nx.connected_components(G_prop), key=len, reverse=True)
            reach_pct = (len(comps[0]) / num_n) * 100.0 if comps else 0.0
        else:
            reach_pct = 0.0

        # Routing test
        route_ok = False
        if num_n >= 2 and comps and len(comps[0]) >= 2:
            lc = list(comps[0])
            try:
                path = nx.shortest_path(G_prop, lc[0], lc[-1], weight="weight")
                route_ok = len(path) >= 2
            except nx.NetworkXNoPath:
                route_ok = False

        prec = apls_res["apls_precision"]
        rec = apls_res["apls_recall"]
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 1e-6 else 0.0

        records.append({
            "arch": arch,
            "nodes": num_n,
            "edges": num_e,
            "km": tot_km,
            "reach": reach_pct,
            "apls": apls_res["apls"],
            "prec": prec * 100.0,
            "rec": rec * 100.0,
            "f1": f1,
            "route": route_ok
        })

        print(f" {arch:<21} | {num_n:<5} | {num_e:<5} | {tot_km:<6.2f} | {reach_pct:<7.1f} | {apls_res['apls']:<6.4f} | {prec*100:<6.1f} | {rec*100:<6.1f} | {f1:<6.4f} | {'OK' if route_ok else 'FAIL'}")

    mean_apls = float(np.mean([r["apls"] for r in records]))
    mean_prec = float(np.mean([r["prec"] for r in records]))
    mean_rec = float(np.mean([r["rec"] for r in records]))
    mean_f1 = float(np.mean([r["f1"] for r in records]))
    mean_reach = float(np.mean([r["reach"] for r in records]))

    print("====================================================================================================")
    print(f" MEAN (10 Archetypes)  | ----- | ----- | ------ | {mean_reach:<7.1f} | {mean_apls:<6.4f} | {mean_prec:<6.1f} | {mean_rec:<6.1f} | {mean_f1:<6.4f} |")
    print("====================================================================================================")


if __name__ == "__main__":
    evaluate_calibrated()
