"""
Diagnostics and Taxonomy of False Positives in TopoGrapher.
Categorizes false positive edges across world benchmark scenes:
- Cross-block diagonal shortcuts
- Parallel-road ladder rungs
- Micro-spurs and dead-ends
- Spiderweb multi-degree clusters
"""

import os
import json
import numpy as np
import networkx as nx
import torch

from topographer.data.world_benchmark import WorldBenchmarkSuite
from topographer.pipeline import TopoGrapherEngine
from topographer.metrics.apls import APLSMetric


def diagnose_scene(engine: TopoGrapherEngine, suite: WorldBenchmarkSuite, archetype: str, seed: int = 2026):
    scene = suite.generate_world_scene(archetype, seed=seed)
    img_np = scene.image
    h, w, _ = img_np.shape
    img_f = img_np.astype(np.float32) / 255.0

    with torch.no_grad():
        tensor = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).to(engine.device)
        dense = engine.model.forward_dense(tensor)
        road_prob = dense["road_prob"].squeeze(0).cpu().numpy()[0]
        tangent = dense["tangent"].squeeze(0).cpu().numpy()
        raw = engine.model.predict_graph_elements(tensor, keypoint_threshold=0.25, edge_threshold=0.35)

    vertices = raw["vertices"]
    edges = raw["edges"]
    
    map_h, map_w = road_prob.shape
    scale_x = map_w / float(w)
    scale_y = map_h / float(h)

    # Categories
    cross_block_chords = []
    parallel_ladder_rungs = []
    short_spurs = []
    valid_corridors = []

    for e in edges:
        u, v = e["source"], e["target"]
        p1 = (vertices[u]["x"], vertices[u]["y"])
        p2 = (vertices[v]["x"], vertices[v]["y"])
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = np.hypot(dx, dy)

        if dist < 18.0:
            short_spurs.append((u, v, dist))
            continue

        phi = np.arctan2(dy, dx)
        cos_2phi = np.cos(2.0 * phi)
        sin_2phi = np.sin(2.0 * phi)

        samples_prob = []
        samples_align = []
        for t in np.linspace(0.1, 0.9, 14):
            sx = int(round((p1[0] + t * dx) * scale_x))
            sy = int(round((p1[1] + t * dy) * scale_y))
            if 0 <= sx < map_w and 0 <= sy < map_h:
                samples_prob.append(road_prob[sy, sx])
                samples_align.append(tangent[0, sy, sx] * cos_2phi + tangent[1, sy, sx] * sin_2phi)

        if not samples_prob:
            continue

        mean_p = float(np.mean(samples_prob))
        low_p_ratio = float(sum(1 for p in samples_prob if p < 0.25) / len(samples_prob))
        mean_align = float(np.mean(samples_align))

        # Check cross-block chord (cuts across buildings / low road prob)
        if low_p_ratio > 0.15 or mean_p < 0.35:
            cross_block_chords.append((u, v, dist, mean_p, low_p_ratio))
        # Check angle alignment (perpendicular or acute shortcut)
        elif mean_align < 0.20:
            parallel_ladder_rungs.append((u, v, dist, mean_align))
        else:
            valid_corridors.append((u, v, dist, mean_p, mean_align))

    # Graph degree diagnostics
    G_raw = nx.Graph()
    for idx, v in enumerate(vertices):
        G_raw.add_node(idx, x=v["x"], y=v["y"])
    for e in edges:
        G_raw.add_edge(e["source"], e["target"])

    high_degree_nodes = [(n, deg) for n, deg in G_raw.degree() if deg > 5]

    return {
        "archetype": archetype,
        "total_vertices": len(vertices),
        "total_edges": len(edges),
        "cross_block_chords_count": len(cross_block_chords),
        "parallel_ladder_rungs_count": len(parallel_ladder_rungs),
        "short_spurs_count": len(short_spurs),
        "valid_corridors_count": len(valid_corridors),
        "high_degree_nodes_count": len(high_degree_nodes),
        "max_node_degree": max([deg for _, deg in G_raw.degree()] or [0]),
        "mean_node_degree": float(np.mean([deg for _, deg in G_raw.degree()] or [0]))
    }


def run_diagnostics():
    suite = WorldBenchmarkSuite(1024)
    engine = TopoGrapherEngine()
    
    print("=" * 70)
    print("      TopoGrapher False Positive Diagnostic Analysis")
    print("=" * 70)

    scenes = ["manhattan_grid", "paris_etoile", "amsterdam_canals", "tokyo_organic", "barcelona_eixample"]
    summary = []
    for s in scenes:
        res = diagnose_scene(engine, suite, s)
        summary.append(res)
        print(f"\nArchetype: {s.upper()}")
        print(f"  Raw Edges: {res['total_edges']} | Vertices: {res['total_vertices']}")
        print(f"  - Cross-Block Building Chords : {res['cross_block_chords_count']} ({res['cross_block_chords_count']/res['total_edges']*100:.1f}%)")
        print(f"  - Angular/Ladder Mismatches   : {res['parallel_ladder_rungs_count']} ({res['parallel_ladder_rungs_count']/res['total_edges']*100:.1f}%)")
        print(f"  - Micro-Spurs (<18px)         : {res['short_spurs_count']} ({res['short_spurs_count']/res['total_edges']*100:.1f}%)")
        print(f"  - Valid Road Corridors        : {res['valid_corridors_count']} ({res['valid_corridors_count']/res['total_edges']*100:.1f}%)")
        print(f"  - High-Degree Nodes (deg > 5) : {res['high_degree_nodes_count']} (Max Degree: {res['max_node_degree']})")

    with open("d:/Workspace/topographer/false_positive_diagnostics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[DONE] Diagnostics saved to d:/Workspace/topographer/false_positive_diagnostics.json")


if __name__ == "__main__":
    run_diagnostics()
