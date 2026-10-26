"""
Unit tests for End-to-End TopoGrapher Pipeline, Exporters, and Visualizer.
"""

import os
import pytest
import numpy as np
from PIL import Image
from topographer.pipeline import TopoGrapherEngine
from topographer.data.procedural_generator import ProceduralRoadGenerator
from topographer.export.exporters import GISExporter


def test_gis_exporters_geojson_and_osm(tmp_path):
    gen = ProceduralRoadGenerator(width=200, height=200)
    scene = gen.generate_scene("grid", seed=10)

    # Convert scene to graph
    import networkx as nx
    G = nx.Graph()
    for idx, (x, y, t) in enumerate(scene.vertices):
        G.add_node(idx, x=x, y=y, score=0.95, type=t)
    for u, v, c in scene.edges:
        p1 = (scene.vertices[u][0], scene.vertices[u][1])
        p2 = (scene.vertices[v][0], scene.vertices[v][1])
        d = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
        G.add_edge(u, v, weight=d, road_class=c, length_px=d)

    exporter = GISExporter(pixel_to_meter_scale=0.5)

    # 1. GeoJSON export
    geojson_file = str(tmp_path / "roads.geojson")
    doc = exporter.to_geojson(G, output_path=geojson_file)
    assert doc["type"] == "FeatureCollection"
    assert len(doc["features"]) > 0
    assert os.path.exists(geojson_file)

    # 2. OSM XML export
    osm_file = str(tmp_path / "roads.osm")
    osm_str = exporter.to_osm_xml(G, output_path=osm_file)
    assert "<osm" in osm_str
    assert "<node" in osm_str
    assert "<way" in osm_str
    assert os.path.exists(osm_file)


def test_topographer_engine_pipeline():
    engine = TopoGrapherEngine()
    gen = ProceduralRoadGenerator(width=256, height=256)
    scene = gen.generate_scene("grid", seed=7)

    results = engine.extract_road_network(scene.image)
    assert "graph" in results
    assert "telemetry" in results
    assert "geojson" in results
    assert "routing" in results

    assert results["telemetry"]["nodes"] > 0
    assert results["telemetry"]["edges"] >= 0

    # Test visualization overlay
    vis_img = engine.render_overlay_visualization(scene.image, results["graph"])
    assert isinstance(vis_img, Image.Image)
    assert vis_img.size == (256, 256)
