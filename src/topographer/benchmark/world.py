"""
Global Multi-Continent World Benchmark Runner.
"""

import os
import sys
import time
import json
from typing import Dict, Any, List
import numpy as np
from PIL import Image
import networkx as nx

from topographer.pipeline import TopoGrapherEngine
from topographer.data.world_benchmark import WorldBenchmarkSuite
from topographer.metrics.apls import APLSMetric


def run_world_benchmark(
    weights_path: str = "d:/Workspace/checkpoints/topographer_weights.pt",
    output_dir: str = "benchmark_results",
    resolution: int = 1024
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    print("======================================================================")
    print("     TopoGrapher Global Multi-Continent World Benchmark Suite")
    print(f"     Resolution: {resolution}x{resolution} | {len(WorldBenchmarkSuite.ARCHETYPES)} Global Archetypes")
    print("======================================================================")

    suite = WorldBenchmarkSuite(resolution=resolution)
    engine = TopoGrapherEngine(model_path=weights_path if os.path.exists(weights_path) else None)
    apls_evaluator = APLSMetric(max_buffer_px=24.0, max_pairs_sample=350)

    results_table = []
    archetypes = WorldBenchmarkSuite.ARCHETYPES

    for idx, arch in enumerate(archetypes, 1):
        print(f"\n[{idx}/{len(archetypes)}] Evaluating Archetype: {arch.upper()}...")
        arch_dir = os.path.join(output_dir, arch)
        os.makedirs(arch_dir, exist_ok=True)

        # 1. Synthesize Scene & GT Graph
        scene = suite.generate_world_scene(arch, seed=2026 + idx)
        raw_img_path = os.path.join(arch_dir, "input_scene.png")
        Image.fromarray(scene.image).save(raw_img_path)

        gt_graph = nx.Graph()
        for v_i, (x, y, t) in enumerate(scene.vertices):
            gt_graph.add_node(v_i, x=x, y=y, type=t)
        for u, v, c in scene.edges:
            p1 = (scene.vertices[u][0], scene.vertices[u][1])
            p2 = (scene.vertices[v][0], scene.vertices[v][1])
            d = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
            gt_graph.add_edge(u, v, weight=d, length_px=d, road_class=c)

        # 2. Run Inference
        t0 = time.time()
        preds = engine.extract_road_network(scene.image, regularize=True)
        t_infer = time.time() - t0

        prop_graph = preds["graph"]
        telemetry = preds["telemetry"]

        # 3. SpaceNet APLS Score
        apls_res = apls_evaluator.compute_apls(gt_graph, prop_graph)

        # 4. A* Shortest Path Routing Test
        routing = preds["routing"]
        route_ok = False
        route_len = 0.0
        route_coords = []
        if prop_graph.number_of_nodes() >= 2:
            components = sorted(nx.connected_components(prop_graph), key=len, reverse=True)
            if components and len(components[0]) >= 2:
                largest_comp = list(components[0])
                n_start = largest_comp[0]
                n_end = largest_comp[-1]
                p_start = (prop_graph.nodes[n_start]["x"], prop_graph.nodes[n_start]["y"])
                p_end = (prop_graph.nodes[n_end]["x"], prop_graph.nodes[n_end]["y"])
                route_res = routing.shortest_path(p_start, p_end)
                route_ok = route_res["success"]
                route_len = route_res["distance_m"]
                route_coords = route_res.get("path_coords", [])

        # 5. Render Visual Overlay
        overlay_path = os.path.join(arch_dir, "vector_overlay.png")
        vis = engine.render_overlay_visualization(
            scene.image,
            prop_graph,
            shortest_path_coords=route_coords if route_ok else None
        )
        vis.save(overlay_path)

        # 6. Export GeoJSON
        geojson_path = os.path.join(arch_dir, "extracted_roads.geojson")
        with open(geojson_path, "w", encoding="utf-8") as f:
            json.dump(preds["geojson"], f, indent=2)

        record = {
            "archetype": arch,
            "gt_nodes": gt_graph.number_of_nodes(),
            "gt_edges": gt_graph.number_of_edges(),
            "extracted_nodes": prop_graph.number_of_nodes(),
            "extracted_edges": prop_graph.number_of_edges(),
            "total_km": round(telemetry["total_length_km"], 2),
            "reachability_pct": round(telemetry["largest_component_ratio"] * 100.0, 1),
            "apls": round(apls_res["apls"], 4),
            "apls_harmonic": round(apls_res.get("apls_harmonic", 0.0), 4),
            "apls_precision": round(apls_res["apls_precision"], 4),
            "apls_recall": round(apls_res["apls_recall"], 4),
            "inference_sec": round(t_infer, 3),
            "routing_success": route_ok,
            "route_dist_m": round(route_len, 1)
        }
        results_table.append(record)

        print(f"  --> Extracted: {prop_graph.number_of_nodes()} nodes, {prop_graph.number_of_edges()} edges ({telemetry['total_length_km']:.2f} km)")
        print(f"  --> Reachability: {telemetry['largest_component_ratio']*100:.1f}% | Inference: {t_infer:.3f}s")
        print(f"  --> SpaceNet APLS: {apls_res['apls']:.4f} (F1: {apls_res.get('apls_harmonic', 0.0):.4f}, Prec: {apls_res['apls_precision']:.4f}, Rec: {apls_res['apls_recall']:.4f})")
        print(f"  --> A* Routing: {'SUCCESS' if route_ok else 'NO_PATH'} ({route_len:.1f}m)")

    # Global Aggregate Metrics
    mean_apls = float(np.mean([r["apls"] for r in results_table]))
    mean_f1 = float(np.mean([r["apls_harmonic"] for r in results_table]))
    mean_prec = float(np.mean([r["apls_precision"] for r in results_table]))
    mean_rec = float(np.mean([r["apls_recall"] for r in results_table]))
    mean_reach = float(np.mean([r["reachability_pct"] for r in results_table]))
    mean_time = float(np.mean([r["inference_sec"] for r in results_table]))

    summary = {
        "benchmark_archetypes_count": len(archetypes),
        "mean_apls": round(mean_apls, 4),
        "mean_f1": round(mean_f1, 4),
        "mean_precision": round(mean_prec, 4),
        "mean_recall": round(mean_rec, 4),
        "mean_reachability_pct": round(mean_reach, 1),
        "mean_inference_sec": round(mean_time, 3),
        "archetypes": results_table
    }

    summary_file = os.path.join(output_dir, "world_benchmark_report.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n======================================================================")
    print("                     GLOBAL WORLD BENCHMARK SUMMARY                   ")
    print("======================================================================")
    print(f" Archetypes Evaluated  : {len(archetypes)}")
    print(f" Mean SpaceNet APLS    : {mean_apls:.4f}")
    print(f" Mean Harmonic F1      : {mean_f1:.4f}")
    print(f" Mean APLS Precision   : {mean_prec*100:.1f}%")
    print(f" Mean APLS Recall      : {mean_rec*100:.1f}%")
    print(f" Mean Reachability     : {mean_reach:.1f}%")
    print(f" Mean Inference Time   : {mean_time:.3f}s")
    print(f" Report Saved          : {summary_file}")
    print("======================================================================")
    return summary
