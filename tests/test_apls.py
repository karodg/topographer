"""
Unit tests for SpaceNet Average Path Length Similarity (APLS).
"""

import pytest
import networkx as nx
from topographer.metrics.apls import APLSMetric


def test_apls_identical_graphs():
    G = nx.Graph()
    G.add_node(0, x=10.0, y=10.0)
    G.add_node(1, x=60.0, y=10.0)
    G.add_node(2, x=60.0, y=60.0)
    G.add_edge(0, 1, weight=50.0)
    G.add_edge(1, 2, weight=50.0)

    metric = APLSMetric(max_buffer_px=10.0)
    score = metric.compute_apls(G, G)

    # Identical graphs must yield perfect 1.0 APLS
    assert pytest.approx(score["apls"], abs=1e-3) == 1.0
    assert pytest.approx(score["apls_recall"], abs=1e-3) == 1.0
    assert pytest.approx(score["apls_precision"], abs=1e-3) == 1.0


def test_apls_disconnected_graph_penalty():
    G_gt = nx.Graph()
    G_gt.add_node(0, x=0.0, y=0.0)
    G_gt.add_node(1, x=100.0, y=0.0)
    G_gt.add_edge(0, 1, weight=100.0)

    # Proposed graph has missing edge (disconnected nodes)
    G_prop = nx.Graph()
    G_prop.add_node(0, x=0.0, y=0.0)
    G_prop.add_node(1, x=100.0, y=0.0)

    metric = APLSMetric(max_buffer_px=10.0)
    score = metric.compute_apls(G_gt, G_prop)

    # Disconnected path receives penalty
    assert score["apls"] <= 0.5
    assert score["apls_recall"] == 0.0
