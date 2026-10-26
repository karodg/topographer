"""
Unit tests for GraphTopologySimplifier.
"""

import numpy as np
import networkx as nx
from topographer.graph.simplifier import GraphTopologySimplifier


def test_angular_sector_nms_caps_degree():
    simplifier = GraphTopologySimplifier(angular_cone_deg=35.0)
    
    # Node 0 at center, 8 surrounding nodes at 45 degree intervals
    vertices = [{"x": 100.0, "y": 100.0}]
    edges = []
    
    for i in range(8):
        angle = i * (2.0 * np.pi / 8.0)
        # Add two nodes very close in angle (0 deg and 10 deg)
        vx = 100.0 + 50.0 * np.cos(angle)
        vy = 100.0 + 50.0 * np.sin(angle)
        vertices.append({"x": vx, "y": vy})
        edges.append({"source": 0, "target": i + 1, "probability": 0.90 - i * 0.05})

    # Add a duplicate edge in the same 10-degree cone
    vx_dup = 100.0 + 50.0 * np.cos(10.0 * np.pi / 180.0)
    vy_dup = 100.0 + 50.0 * np.sin(10.0 * np.pi / 180.0)
    vertices.append({"x": vx_dup, "y": vy_dup})
    edges.append({"source": 0, "target": 9, "probability": 0.40})  # Lower prob in same cone

    nms_edges = simplifier._angular_sector_nms(vertices, edges)
    # The duplicate in the 10-degree cone should be suppressed
    assert len(nms_edges) == 8


def test_corridor_negative_evidence_filter():
    simplifier = GraphTopologySimplifier(min_road_prob=0.35, max_negative_pixel_ratio=0.15)
    
    # Road prob map with road at y=50, and zero elsewhere
    road_prob = np.zeros((100, 100), dtype=np.float32)
    road_prob[50, :] = 0.95
    
    vertices = [
        {"x": 10.0, "y": 50.0},  # 0
        {"x": 90.0, "y": 50.0},  # 1 (valid road)
        {"x": 10.0, "y": 10.0},  # 2
        {"x": 90.0, "y": 90.0},  # 3 (diagonal cutting through empty space)
    ]
    edges = [
        {"source": 0, "target": 1, "probability": 0.95},
        {"source": 2, "target": 3, "probability": 0.90}
    ]
    
    valid = simplifier._validate_corridors_negative_evidence(
        vertices, edges, road_prob, tangent=None, h=100, w=100
    )
    assert len(valid) == 1
    assert valid[0]["source"] == 0 and valid[0]["target"] == 1


def test_degree_2_chain_collapsing_to_polyline():
    simplifier = GraphTopologySimplifier()
    G = nx.Graph()
    # Path: 0 (deg 3 junction) - 1 (deg 2) - 2 (deg 2) - 3 (deg 3 junction)
    G.add_node(0, x=0.0, y=0.0)
    G.add_node(1, x=20.0, y=5.0)
    G.add_node(2, x=40.0, y=2.0)
    G.add_node(3, x=60.0, y=0.0)
    # Branches at 0 and 3 to make them structural junctions
    G.add_node(4, x=0.0, y=30.0)
    G.add_node(5, x=0.0, y=-30.0)
    G.add_node(6, x=60.0, y=30.0)
    G.add_node(7, x=60.0, y=-30.0)
    
    G.add_edge(0, 1, weight=20.0)
    G.add_edge(1, 2, weight=20.0)
    G.add_edge(2, 3, weight=20.0)
    G.add_edge(0, 4, weight=30.0)
    G.add_edge(0, 5, weight=30.0)
    G.add_edge(3, 6, weight=30.0)
    G.add_edge(3, 7, weight=30.0)
    
    H = simplifier._collapse_degree_2_chains(G)
    # Nodes 1 and 2 should be collapsed into the single edge between 0 and 3
    assert not H.has_node(1)
    assert not H.has_node(2)
    assert H.has_edge(0, 3)
    assert "polyline" in H[0][3]
    assert len(H[0][3]["polyline"]) == 4


def test_overpass_grade_separation_resolution():
    simplifier = GraphTopologySimplifier()
    G = nx.Graph()
    # Freeway along x: (0, 50) to (100, 50) [class 0]
    G.add_node(0, x=0.0, y=50.0)
    G.add_node(1, x=100.0, y=50.0)
    G.add_edge(0, 1, weight=100.0, road_class=0)
    
    # Arterial along y: (50, 0) to (50, 100) [class 1]
    G.add_node(2, x=50.0, y=0.0)
    G.add_node(3, x=50.0, y=100.0)
    G.add_edge(2, 3, weight=100.0, road_class=1)
    
    H = simplifier._resolve_crossings_or_overpasses(G)
    # Since class 0 freeway crosses class 1 arterial at 90 deg with length > 80px,
    # it must NOT create an intersection node at (50, 50)!
    assert H.number_of_nodes() == 4
    assert H.has_edge(0, 1)
    assert H.has_edge(2, 3)
    assert H[0][1].get("layer") == 1
    assert H[0][1].get("bridge") == "yes"


def test_prune_intermediate_vertex_skipping_chords():
    simplifier = GraphTopologySimplifier()
    # Three nodes in a line: 0 -> (0, 0), 1 -> (50, 0), 2 -> (100, 0)
    vertices = [
        {"x": 0.0, "y": 0.0},
        {"x": 50.0, "y": 0.0},
        {"x": 100.0, "y": 0.0}
    ]
    # Edges: (0, 1), (1, 2) and chord (0, 2) which skips 1
    edges = [
        {"source": 0, "target": 1, "probability": 0.9},
        {"source": 1, "target": 2, "probability": 0.9},
        {"source": 0, "target": 2, "probability": 0.95}  # chord
    ]
    non_skipping = simplifier._prune_intermediate_vertex_skipping_chords(vertices, edges)
    # The direct chord (0, 2) must be pruned because 1 is between them
    assert len(non_skipping) == 2
    pairs = {(e["source"], e["target"]) for e in non_skipping}
    assert (0, 1) in pairs
    assert (1, 2) in pairs
    assert (0, 2) not in pairs


def test_filter_small_components_length_preserving():
    simplifier = GraphTopologySimplifier(min_component_nodes=3, min_component_length_px=75.0)
    G = nx.Graph()
    # Component A: tiny 2-node spur with length 20px (should be removed)
    G.add_node(0, x=0.0, y=0.0)
    G.add_node(1, x=20.0, y=0.0)
    G.add_edge(0, 1, length_px=20.0)

    # Component B: 2-node long highway stretch with length 150px (should be preserved!)
    G.add_node(2, x=200.0, y=0.0)
    G.add_node(3, x=350.0, y=0.0)
    G.add_edge(2, 3, length_px=150.0)

    H = simplifier._filter_small_components(G, min_nodes=3, min_total_length_px=75.0)
    assert not H.has_node(0)
    assert not H.has_node(1)
    assert H.has_node(2)
    assert H.has_node(3)
    assert H.has_edge(2, 3)

