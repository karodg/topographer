"""
Example 01: Basic Road Network Extraction from Aerial Imagery.

This minimal example demonstrates how to load an aerial image, run
the TopoGrapher neural engine, and export the extracted road network
as a standard GeoJSON FeatureCollection.
"""

import os
from topographer import TopoGrapherEngine
from topographer.data.world_benchmark import WorldBenchmarkSuite
from PIL import Image

# 1. Initialize TopoGrapher Engine (automatically utilizes CUDA if available)
engine = TopoGrapherEngine(gsd_meters_per_pixel=0.5)

# 2. Acquire a sample aerial scene (or provide your own "path/to/aerial_image.png")
sample_path = "sample_aerial.png"
if not os.path.exists(sample_path):
    print("Generating sample Manhattan urban scene...")
    suite = WorldBenchmarkSuite(resolution=1024)
    scene = suite.generate_world_scene("manhattan_grid", seed=42)
    Image.fromarray(scene.image).save(sample_path)

# 3. Extract Topologically Closed Vector Road Graph
print(f"Extracting road network from {sample_path}...")
results = engine.extract_road_network(sample_path, regularize=True)

graph = results["graph"]
telemetry = results["telemetry"]
geojson_data = results["geojson"]

print("\n--- Extraction Results ---")
print(f"Total Nodes Extracted : {graph.number_of_nodes()}")
print(f"Total Edges Extracted : {graph.number_of_edges()}")
print(f"Total Road Length     : {telemetry['total_length_km']:.2f} km")
print(f"Largest Component     : {telemetry['largest_component_ratio']*100:.1f}%")

# 4. Save Vector Network and Cartographic Overlay
os.makedirs("output", exist_ok=True)
import json
with open("output/manhattan_roads.geojson", "w", encoding="utf-8") as f:
    json.dump(geojson_data, f, indent=2)

overlay = engine.render_overlay_visualization(sample_path, graph)
overlay.save("output/manhattan_overlay.png")

print(f"\n[+] Saved GeoJSON vector roads to: output/manhattan_roads.geojson")
print(f"[+] Saved visual overlay to      : output/manhattan_overlay.png")
