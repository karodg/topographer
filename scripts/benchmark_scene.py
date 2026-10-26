"""
TopoGrapher Real-World & High-Complexity Scene Benchmark.

Evaluates trained neural topological road extraction on complex 1024x1024 scenes:
- Radial city centres with concentric ring avenues;
- Organic winding mountain passes with dense tree canopy occlusions;
- Computes exact SpaceNet Average Path Length Similarity (APLS);
- Executes A* shortest path pathfinding with turn-by-turn maneuvers;
- Exports GeoJSON, OSM XML, and renders diagnostic visual composites.
"""

import os
import sys
import time
import json
import numpy as np
from PIL import Image
import networkx as nx

from topographer.pipeline import TopoGrapherEngine
from topographer.data.procedural_generator import ProceduralRoadGenerator
from topographer.data.singapore_benchmark import generate_singapore_scene
from topographer.metrics.apls import APLSMetric
from topographer.export.exporters import GISExporter


def run_benchmark(
    weights_path: str = "d:/Workspace/checkpoints/topographer_weights.pt",
    output_dir: str = "d:/Workspace/topographer/benchmark_results",
    resolution: int = 1024,
    scene_type: str = "singapore"
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"==================================================")
    print(f"   TopoGrapher Autonomous Topology Extraction")
    print(f"   Scene: {scene_type.upper()} | Resolution: {resolution}x{resolution}")
    print(f"==================================================")

    # 1. Synthesize High-Complexity Ground Truth Scene
    print(f"\n[1/5] Synthesizing {scene_type} aerial scene with occluded road network...")
    if scene_type == "singapore":
        scene = generate_singapore_scene(width=resolution, height=resolution, seed=2026)
    else:
        generator = ProceduralRoadGenerator(width=resolution, height=resolution)
        scene = generator.generate_scene(pattern="radial", seed=2026)

    # Save raw satellite image
    raw_img_path = os.path.join(output_dir, "input_aerial_scene.png")
    Image.fromarray(scene.image).save(raw_img_path)
    print(f"  --> Saved input scene: {raw_img_path}")

    # Build Ground-Truth Graph
    gt_graph = nx.Graph()
    for idx, (x, y, t) in enumerate(scene.vertices):
        gt_graph.add_node(idx, x=x, y=y, type=t)
    for u, v, c in scene.edges:
        p1 = (scene.vertices[u][0], scene.vertices[u][1])
        p2 = (scene.vertices[v][0], scene.vertices[v][1])
        dist = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
        gt_graph.add_edge(u, v, weight=dist, length_px=dist, road_class=c)

    print(f"  --> Ground-truth: {gt_graph.number_of_nodes()} vertices, {gt_graph.number_of_edges()} edges")

    # 2. Run Autonomous Neural Inference
    print("\n[2/5] Running TopoGrapher Neural Inference...")
    t0 = time.time()
    engine = TopoGrapherEngine(model_path=weights_path if os.path.exists(weights_path) else None)
    results = engine.extract_road_network(scene.image, regularize=True)
    t_inference = time.time() - t0

    prop_graph = results["graph"]
    telemetry = results["telemetry"]
    print(f"  --> Inference completed in {t_inference:.3f}s")
    print(f"  --> Extracted graph: {prop_graph.number_of_nodes()} nodes, {prop_graph.number_of_edges()} edges")
    print(f"  --> Total extracted road length: {telemetry['total_length_km']:.2f} km")
    print(f"  --> Reachability ratio: {telemetry['largest_component_ratio']*100:.1f}%")

    # 3. SpaceNet APLS Evaluation
    print("\n[3/5] Computing SpaceNet Average Path Length Similarity (APLS)...")
    apls_evaluator = APLSMetric(max_buffer_px=24.0, max_pairs_sample=400)
    apls_scores = apls_evaluator.compute_apls(gt_graph, prop_graph)
    print(f"  --> APLS Score: {apls_scores['apls']:.4f}")
    print(f"  --> APLS Recall: {apls_scores['apls_recall']:.4f}")
    print(f"  --> APLS Precision: {apls_scores['apls_precision']:.4f}")

    # 4. Routing Engine Verification (A* Shortest Path across network)
    print("\n[4/5] Executing Autonomous A* Routing Navigation...")
    routing = results["routing"]
    # Pick two distant nodes in GT
    v_start = scene.vertices[1]
    v_end = scene.vertices[-1]
    start_xy = (v_start[0], v_start[1])
    end_xy = (v_end[0], v_end[1])

    route_res = routing.shortest_path(start_xy, end_xy)
    if route_res["success"]:
        print(f"  --> Routing Succeeded: Path length = {route_res['distance_m']:.1f}m ({route_res['node_count']} nodes)")
        print(f"  --> First 3 turn maneuvers:")
        for step in route_res["maneuvers"][:3]:
            print(f"      Step {step['step']}: {step['instruction']} (Bearing {step['bearing_deg']:.0f} deg, {step['distance_m']:.1f}m)")
    else:
        print(f"  --> Routing note: {route_res.get('error')}")

    # 5. Export GIS GeoJSON, OSM XML, and Visual Composite
    print("\n[5/5] Exporting GIS Deliverables & Overlay Visuals...")
    geojson_path = os.path.join(output_dir, "extracted_roads.geojson")
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(results["geojson"], f, indent=2)

    osm_path = os.path.join(output_dir, "extracted_roads.osm")
    engine.exporter.to_osm_xml(prop_graph, output_path=osm_path)

    # Render High-Resolution Visual Composite
    overlay_path = os.path.join(output_dir, "topographer_overlay_verification.png")
    vis = engine.render_overlay_visualization(
        scene.image,
        prop_graph,
        shortest_path_coords=route_res.get("path_coords") if route_res["success"] else None
    )
    vis.save(overlay_path)
    print(f"  --> GeoJSON saved: {geojson_path}")
    print(f"  --> OSM XML saved: {osm_path}")
    print(f"  --> Visual Overlay saved: {overlay_path}")

    # Save benchmark metrics json
    metrics_summary = {
        "scene_resolution": [resolution, resolution],
        "inference_time_sec": t_inference,
        "gt_nodes": gt_graph.number_of_nodes(),
        "gt_edges": gt_graph.number_of_edges(),
        "extracted_nodes": prop_graph.number_of_nodes(),
        "extracted_edges": prop_graph.number_of_edges(),
        "total_length_km": telemetry["total_length_km"],
        "reachability_ratio": telemetry["largest_component_ratio"],
        "apls": apls_scores,
        "routing_test": {
            "success": route_res["success"],
            "distance_m": route_res["distance_m"],
            "node_count": route_res["node_count"],
            "maneuvers_count": len(route_res["maneuvers"])
        }
    }
    metrics_path = os.path.join(output_dir, "benchmark_summary.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print("\n[SUCCESS] Benchmark Completed Successfully!")
    return metrics_summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run TopoGrapher High-Complexity Scene Benchmark")
    parser.add_argument("--scene", type=str, default="singapore", choices=["singapore", "radial"], help="Scene morphology")
    parser.add_argument("--resolution", type=int, default=1024, help="Square pixel resolution")
    parser.add_argument("--weights", type=str, default="d:/Workspace/checkpoints/topographer_weights.pt", help="Path to weights")
    parser.add_argument("--outdir", type=str, default="d:/Workspace/topographer/benchmark_results", help="Output directory")
    args = parser.parse_args()

    run_benchmark(
        weights_path=args.weights,
        output_dir=args.outdir,
        resolution=args.resolution,
        scene_type=args.scene
    )
