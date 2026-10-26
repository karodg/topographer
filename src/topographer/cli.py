"""
TopoGrapher Command-Line Interface (CLI).

Provides production-grade command-line entry points for:
- Vector road network extraction from aerial and satellite imagery
- Global multi-continent and SpaceNet benchmark execution
- Turn-by-turn routing and navigation analysis
- GIS vector export (GeoJSON, OpenStreetMap XML)
"""

import sys
import os
import argparse
import json
import time
import torch
import numpy as np
from PIL import Image

import topographer
from topographer.pipeline import TopoGrapherEngine
from topographer.graph.routing import RoadRoutingEngine


BANNER = r"""
  _____                 ____                 _               
 |_   _|___  _ __   ___ / ___|_ __ __ _ _ __ | |__   ___ _ __ 
   | | / _ \| '_ \ / _ \ |  _| '__/ _` | '_ \| '_ \ / _ \ '__|
   | || (_) | |_) | (_) | |_| | | | (_| | |_) | | | |  __/ |   
   |_| \___/| .__/ \___/ \____|_|  \__,_| .__/|_| |_|\___|_|   
            |_|                         |_|                    
 Autonomous Neural Road Network & Topology Graph Extraction Engine
"""


def cmd_extract(args):
    print(BANNER)
    if not os.path.exists(args.input):
        print(f"[Error] Input image not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    weights_path = args.weights if args.weights and os.path.exists(args.weights) else None
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    print(f"[*] Initializing TopoGrapher on [{device.upper()}]...")
    if weights_path:
        print(f"[*] Loaded Model Weights: {weights_path}")
    else:
        print("[*] Running in Native Inference Mode (default procedural checkpoint)")

    engine = TopoGrapherEngine(
        model_path=weights_path,
        device=device,
        gsd_meters_per_pixel=args.gsd
    )

    print(f"[*] Extracting vector topology from: {args.input}")
    t0 = time.time()
    results = engine.extract_road_network(args.input, regularize=not args.no_regularize)
    t_elapsed = time.time() - t0

    g = results["graph"]
    telemetry = results["telemetry"]
    base_name = os.path.splitext(os.path.basename(args.input))[0]

    # Save GeoJSON
    if args.format in ("all", "geojson"):
        geojson_path = os.path.join(args.output_dir, f"{base_name}_roads.geojson")
        with open(geojson_path, "w", encoding="utf-8") as f:
            json.dump(results["geojson"], f, indent=2)
        print(f"[+] GeoJSON Vector Road Network: {geojson_path}")

    # Save OSM XML
    if args.format in ("all", "osm"):
        osm_path = os.path.join(args.output_dir, f"{base_name}_roads.osm")
        engine.exporter.to_osm_xml(g, output_path=osm_path)
        print(f"[+] OpenStreetMap XML Network:   {osm_path}")

    # Render Visual Composite Overlay
    if args.visualize or args.format == "all":
        vis_path = os.path.join(args.output_dir, f"{base_name}_overlay.png")
        vis = engine.render_overlay_visualization(args.input, g)
        vis.save(vis_path)
        print(f"[+] Cartographic Vector Overlay: {vis_path}")

    # Summary
    print("\n" + "=" * 50)
    print("           EXTRACTION SUMMARY")
    print("=" * 50)
    print(f" Nodes Extracted       : {g.number_of_nodes()}")
    print(f" Edges Extracted       : {g.number_of_edges()}")
    print(f" Total Road Network    : {telemetry['total_length_km']:.2f} km")
    print(f" Largest Component     : {telemetry['largest_component_ratio']*100:.1f}%")
    print(f" Inference Time        : {t_elapsed:.3f} seconds")
    print("=" * 50)


def cmd_benchmark(args):
    print(BANNER)
    if args.suite == "world":
        from topographer.benchmark import run_world_benchmark
        run_world_benchmark(output_dir=args.output_dir, resolution=args.resolution)
    elif args.suite == "spacenet":
        from topographer.benchmark import run_spacenet_benchmark
        run_spacenet_benchmark(output_dir=args.output_dir)
        run_spacenet_benchmark(output_dir=args.output_dir)
    else:
        print(f"[Error] Unknown benchmark suite: {args.suite}", file=sys.stderr)
        sys.exit(1)


def cmd_route(args):
    print(BANNER)
    if not os.path.exists(args.input):
        print(f"[Error] Input image or GeoJSON not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[*] Extracting routing graph from: {args.input}")
    engine = TopoGrapherEngine(device=device, gsd_meters_per_pixel=args.gsd)
    results = engine.extract_road_network(args.input, regularize=True)

    routing: RoadRoutingEngine = results["routing"]
    start_pt = (args.start[0], args.start[1])
    end_pt = (args.end[0], args.end[1])

    print(f"[*] Computing A* shortest path from {start_pt} to {end_pt}...")
    res = routing.shortest_path(start_pt, end_pt)

    if res["success"]:
        print(f"[+] Route Found! Distance: {res['distance_m']:.1f} meters | Travel Time: {res['travel_time_sec']:.1f}s")
        print(f"[+] Waypoints Count: {len(res['path_coords'])}")
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            print(f"[+] Route details saved to: {args.output}")
    else:
        print("[-] No valid topological path found between the specified points.", file=sys.stderr)
        sys.exit(1)


def cmd_version(args):
    print(f"TopoGrapher v{topographer.__version__}")
    print(f"PyTorch Version  : {torch.__version__}")
    print(f"CUDA Available   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device       : {torch.cuda.get_device_name(0)}")


def main():
    parser = argparse.ArgumentParser(
        prog="topographer",
        description="TopoGrapher: Autonomous Neural Vector Road Network & Topology Graph Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: extract
    p_extract = subparsers.add_parser("extract", help="Extract vector road graph from an aerial or satellite image")
    p_extract.add_argument("input", type=str, help="Path to input image (.png, .jpg, .tif, GeoTIFF)")
    p_extract.add_argument("-o", "--output-dir", type=str, default="output", help="Directory to save extracted outputs")
    p_extract.add_argument("-w", "--weights", type=str, default=None, help="Path to trained PyTorch weights checkpoint")
    p_extract.add_argument("-f", "--format", type=str, choices=["all", "geojson", "osm"], default="all", help="Export format")
    p_extract.add_argument("--gsd", type=float, default=0.5, help="Ground sample distance in meters/pixel")
    p_extract.add_argument("-v", "--visualize", action="store_true", default=True, help="Render visual cartographic overlay")
    p_extract.add_argument("--no-regularize", action="store_true", help="Disable topological regularization and simplifier")
    p_extract.add_argument("--cpu", action="store_true", help="Force CPU inference")

    # Command: benchmark
    p_bench = subparsers.add_parser("benchmark", help="Execute benchmark suite")
    p_bench.add_argument("--suite", type=str, choices=["world", "spacenet"], default="world", help="Benchmark suite to execute")
    p_bench.add_argument("-o", "--output-dir", type=str, default="benchmark_results", help="Directory to save benchmark reports")
    p_bench.add_argument("-r", "--resolution", type=int, default=1024, help="Image resolution for synthetic benchmark")

    # Command: route
    p_route = subparsers.add_parser("route", help="Calculate turn-by-turn navigation shortest path")
    p_route.add_argument("input", type=str, help="Path to aerial image")
    p_route.add_argument("--start", nargs=2, type=float, required=True, metavar=("X", "Y"), help="Start coordinate (pixel X Y)")
    p_route.add_argument("--end", nargs=2, type=float, required=True, metavar=("X", "Y"), help="End coordinate (pixel X Y)")
    p_route.add_argument("-o", "--output", type=str, default=None, help="Save route JSON to file")
    p_route.add_argument("--gsd", type=float, default=0.5, help="Ground sample distance in meters/pixel")
    p_route.add_argument("--cpu", action="store_true", help="Force CPU inference")

    # Command: version
    subparsers.add_parser("version", help="Print version and environment diagnostic information")

    args = parser.parse_args()

    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "route":
        cmd_route(args)
    elif args.command == "version":
        cmd_version(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
