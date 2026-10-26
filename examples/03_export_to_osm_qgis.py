"""
Example 03: Exporting Extracted Topology to OpenStreetMap (.osm) & QGIS.

Demonstrates converting an extracted vector graph into standard OSM XML format,
enabling direct ingestion into QGIS, JOSM, ArcGIS, and routing engines like OSRM / GraphHopper.
"""

import os
from topographer import TopoGrapherEngine
from topographer.data.world_benchmark import WorldBenchmarkSuite
from PIL import Image

engine = TopoGrapherEngine(gsd_meters_per_pixel=0.5)

sample_path = "sample_singapore.png"
if not os.path.exists(sample_path):
    print("Generating sample Singapore expressway scene...")
    suite = WorldBenchmarkSuite(resolution=1024)
    scene = suite.generate_world_scene("singapore_coastal", seed=77)
    Image.fromarray(scene.image).save(sample_path)

print("Extracting Singapore vector road network...")
results = engine.extract_road_network(sample_path, regularize=True)
graph = results["graph"]

os.makedirs("output", exist_ok=True)
osm_path = "output/singapore_network.osm"

# Export directly to OpenStreetMap XML
engine.exporter.to_osm_xml(
    graph,
    output_path=osm_path,
    origin_lat=1.290270,  # Singapore Marina Bay latitude
    origin_lon=103.851959  # Singapore Marina Bay longitude
)

print(f"\n[+] Successfully exported to OpenStreetMap XML: {osm_path}")
print("    This file can be opened directly in QGIS (Vector Layer -> OSM) or JOSM.")
