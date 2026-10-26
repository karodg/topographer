"""
Unit tests for Graph Tracer, Regularizer, and Routing Engine.
"""

import pytest
import numpy as np
import networkx as nx
from topographer.graph.tracer import GraphTracer
from topographer.graph.regularizer import GraphRegularizer
from topographer.graph.routing import RoadRoutingEngine


def test_graph_tracer_and_edges():
    tracer = GraphTracer(keypoint_threshold=0.3, edge_threshold=0.4)
    vertices = [
        {"x": 10.0, "y": 10.0, "score": 0.9, "type": 0},
        {"x": 50.0, "y": 10.0, "score": 0.85, "type": 0},
        {"x": 50.0, "y": 50.0, "score": 0.8, "type": 0}
    ]
    edges = [
        {"source": 0, "target": 1, "probability": 0.85, "road_class": 1},
        {"source": 1, "target": 2, "probability": 0.75, "road_class": 1}
    ]

    G = tracer.trace_graph(vertices, edges, image_shape=(100, 100))
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2
    assert G.has_edge(0, 1)
    assert G.has_edge(1, 2)


def test_graph_regularizer_spur_pruning():
    G = nx.Graph()
    # Main street
    G.add_node(0, x=0.0, y=0.0)
    G.add_node(1, x=50.0, y=0.0)
    G.add_node(2, x=100.0, y=0.0)
    G.add_edge(0, 1, weight=50.0, length_px=50.0)
    G.add_edge(1, 2, weight=50.0, length_px=50.0)

    # Short noisy spur of 5px off node 1
    G.add_node(3, x=50.0, y=5.0)
    G.add_edge(1, 3, weight=5.0, length_px=5.0)

    regularizer = GraphRegularizer(min_spur_length_px=10.0)
    G_clean = regularizer.regularize(G)

    # Spur should be pruned
    assert not G_clean.has_node(3)
    assert G_clean.number_of_nodes() == 3
    assert G_clean.has_edge(0, 1)
    assert G_clean.has_edge(1, 2)


def test_routing_engine_shortest_path():
    G = nx.Graph()
    G.add_node(0, x=0.0, y=0.0)
    G.add_node(1, x=100.0, y=0.0)
    G.add_node(2, x=100.0, y=100.0)
    G.add_node(3, x=0.0, y=100.0)

    G.add_edge(0, 1, weight=100.0, length_px=100.0)
    G.add_edge(1, 2, weight=100.0, length_px=100.0)
    G.add_edge(2, 3, weight=100.0, length_px=100.0)
    G.add_edge(0, 3, weight=100.0, length_px=100.0)

    engine = RoadRoutingEngine(G, pixel_to_meter_scale=1.0)
    res = engine.shortest_path((0.0, 0.0), (100.0, 100.0))

    assert res["success"] is True
    assert res["distance_m"] == 200.0
    assert len(res["path_coords"]) == 3
    assert len(res["maneuvers"]) > 0

    telemetry = engine.compute_network_telemetry()
    assert telemetry["nodes"] == 4
    assert telemetry["edges"] == 4
    assert telemetry["total_length_km"] == 0.4
    assert telemetry["connected_components"] == 1
