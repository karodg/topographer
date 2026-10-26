"""
Example 02: Turn-by-Turn Routing & Navigation on Extracted Road Networks.

Demonstrates extracting a routable vector graph, finding origin and destination
coordinates, running A* shortest path search, and inspecting turn-by-turn maneuvers.
"""

import os
from topographer import TopoGrapherEngine
from topographer.data.world_benchmark import WorldBenchmarkSuite
from PIL import Image

# 1. Initialize Engine
engine = TopoGrapherEngine(gsd_meters_per_pixel=0.5)

# 2. Prepare Sample Scene
sample_path = "sample_barcelona.png"
if not os.path.exists(sample_path):
    print("Generating sample Barcelona Eixample district...")
    suite = WorldBenchmarkSuite(resolution=1024)
    scene = suite.generate_world_scene("barcelona_eixample", seed=101)
    Image.fromarray(scene.image).save(sample_path)

# 3. Extract Road Network
print("Extracting routable vector road network...")
results = engine.extract_road_network(sample_path, regularize=True)
router = results["routing"]
graph = results["graph"]

# 4. Plan Route Between Two Coordinates
# Choose coordinates from two opposite corners of the scene
start_coord = (100.0, 100.0)
end_coord = (900.0, 900.0)

print(f"\nComputing A* shortest path from {start_coord} to {end_coord}...")
route = router.shortest_path(start_coord, end_coord)

if route["success"]:
    print(f"[+] Route Successfully Calculated!")
    print(f"    Total Distance   : {route['distance_m']:.1f} meters")
    print(f"    Estimated Time   : {route['travel_time_sec']:.1f} seconds")
    print(f"    Waypoints Count  : {len(route['path_coords'])}")

    # 5. Render Navigation Overlay with Highlighted Route
    os.makedirs("output", exist_ok=True)
    overlay = engine.render_overlay_visualization(
        sample_path,
        graph,
        shortest_path_coords=route["path_coords"]
    )
    overlay.save("output/barcelona_navigation_route.png")
    print(f"[+] Saved routing overlay to: output/barcelona_navigation_route.png")
else:
    print("[-] Route calculation failed: points are in disconnected components.")
