"""
Real-World SpaceNet 3 Satellite Benchmark Evaluator for TopoGrapher.

Evaluates TopoGrapher on authentic, high-resolution DigitalGlobe WorldView-3 satellite imagery
(0.3m ground sample distance) from the official CosmiQ Works SpaceNet 3 Challenge:
- AOI_2_Vegas: Las Vegas suburban grid & desert arterial
- AOI_3_Paris: Paris dense historical radial boulevard network
- AOI_4_Shanghai: Shanghai high-density Asian megacity corridor
- AOI_5_Khartoum: Khartoum Nile basin arid irregular road network

Uses official SpaceNet 3 Speed & Topology GeoJSON ground truth with geographic WGS84 coordinates
transformed to image pixel space via exact GeoTIFF affine metadata.
"""

import os
import json
import time
import ssl
import urllib.request
from typing import Dict, List, Tuple, Any
import numpy as np
from PIL import Image
import tifffile
import networkx as nx

from topographer.pipeline import TopoGrapherEngine
from topographer.metrics.apls import APLSMetric


SPACENET_CITIES = {
    "las_vegas": {
        "aoi": "AOI_2_Vegas",
        "img_key": "spacenet/SN3_roads/train/AOI_2_Vegas/PS-RGB/SN3_roads_train_AOI_2_Vegas_PS-RGB_img1001.tif",
        "json_key": "spacenet/SN3_roads/train/AOI_2_Vegas/geojson_roads_speed/SN3_roads_train_AOI_2_Vegas_geojson_roads_speed_img1001.geojson",
        "description": "Suburban Desert Grid (USA)"
    },
    "paris": {
        "aoi": "AOI_3_Paris",
        "img_key": "spacenet/SN3_roads/train/AOI_3_Paris/PS-RGB/SN3_roads_train_AOI_3_Paris_PS-RGB_img10.tif",
        "json_key": "spacenet/SN3_roads/train/AOI_3_Paris/geojson_roads_speed/SN3_roads_train_AOI_3_Paris_geojson_roads_speed_img10.geojson",
        "description": "Dense Historical Radial (France)"
    },
    "shanghai": {
        "aoi": "AOI_4_Shanghai",
        "img_key": "spacenet/SN3_roads/train/AOI_4_Shanghai/PS-RGB/SN3_roads_train_AOI_4_Shanghai_PS-RGB_img10.tif",
        "json_key": "spacenet/SN3_roads/train/AOI_4_Shanghai/geojson_roads_speed/SN3_roads_train_AOI_4_Shanghai_geojson_roads_speed_img10.geojson",
        "description": "High-Density Megacity (China)"
    },
    "khartoum": {
        "aoi": "AOI_5_Khartoum",
        "img_key": "spacenet/SN3_roads/train/AOI_5_Khartoum/PS-RGB/SN3_roads_train_AOI_5_Khartoum_PS-RGB_img100.tif",
        "json_key": "spacenet/SN3_roads/train/AOI_5_Khartoum/geojson_roads_speed/SN3_roads_train_AOI_5_Khartoum_geojson_roads_speed_img100.geojson",
        "description": "Arid Irregular Basin (Sudan)"
    }
}


def download_file_if_missing(url: str, dest_path: str):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        return
    print(f"Downloading {os.path.basename(dest_path)} from SpaceNet S3...")
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(url, headers={"User-Agent": "TopoGrapher-SpaceNet-Evaluator/1.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=60) as response, open(dest_path, "wb") as out_file:
        out_file.write(response.read())


def parse_spacenet_ground_truth(tif_path: str, geojson_path: str) -> Tuple[nx.Graph, np.ndarray]:
    """
    Reads SpaceNet GeoTIFF, extracts affine geotransform, converts GT GeoJSON
    WGS84 LineStrings into a networkx Graph in image pixel coordinates, and
    converts uint16 satellite bands into 8-bit RGB using 2%-98% percentile stretch.
    """
    with tifffile.TiffFile(tif_path) as tif:
        scale = tif.pages[0].tags["ModelPixelScaleTag"].value
        tiepoint = tif.pages[0].tags["ModelTiepointTag"].value
        dx, dy = scale[0], scale[1]
        x0, y0 = tiepoint[3], tiepoint[4]
        img_raw = tif.asarray()

    # Remote sensing percentile stretch to uint8 RGB
    if img_raw.ndim == 3 and img_raw.shape[2] >= 3:
        rgb = img_raw[:, :, :3]
    elif img_raw.ndim == 3 and img_raw.shape[0] in (3, 4, 8):
        rgb = np.transpose(img_raw[:3], (1, 2, 0))
    else:
        rgb = np.repeat(img_raw[:, :, None], 3, axis=2)

    p2, p98 = np.percentile(rgb, (2, 98))
    img_stretched = np.clip((rgb.astype(np.float32) - p2) / (p98 - p2 + 1e-5), 0.0, 1.0)
    img_uint8 = (img_stretched * 255.0).astype(np.uint8)

    h, w = img_uint8.shape[:2]

    # Parse GeoJSON into Graph
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    G_gt = nx.Graph()
    node_id_map = {}
    next_node_id = 0

    for feat in gj.get("features", []):
        geom = feat.get("geometry", {})
        if geom.get("type") != "LineString":
            continue
        coords = geom.get("coordinates", [])
        if len(coords) < 2:
            continue

        pts_px = []
        for lon, lat in coords:
            px = (lon - x0) / dx
            py = (y0 - lat) / dy
            pts_px.append((float(px), float(py)))

        prev_id = None
        for px, py in pts_px:
            # Snap close nodes (< 4px) to share identical vertex
            key = (round(px / 4.0), round(py / 4.0))
            if key not in node_id_map:
                nid = next_node_id
                next_node_id += 1
                node_id_map[key] = nid
                G_gt.add_node(nid, x=px, y=py, type=1)
            else:
                nid = node_id_map[key]

            if prev_id is not None and prev_id != nid:
                p_prev = (G_gt.nodes[prev_id]["x"], G_gt.nodes[prev_id]["y"])
                d = float(np.hypot(px - p_prev[0], py - p_prev[1]))
                if d > 0.5:
                    G_gt.add_edge(prev_id, nid, weight=d, length_px=d, road_class=1)

            prev_id = nid

    return G_gt, img_uint8


def run_spacenet_benchmark(
    weights_path: str = "d:/Workspace/checkpoints/topographer_weights.pt",
    output_dir: str = "d:/Workspace/topographer/real_world_data/benchmark_results"
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    cache_dir = "d:/Workspace/topographer/real_world_data/spacenet_cache"
    os.makedirs(cache_dir, exist_ok=True)

    print("=======================================================================================")
    print("      TopoGrapher Real-World Satellite Benchmark (SpaceNet 3 Challenge)")
    print("      Sensors: DigitalGlobe WorldView-3 | Ground Sample Distance: 0.30 m/pixel")
    print("=======================================================================================")

    engine = TopoGrapherEngine(model_path=weights_path if os.path.exists(weights_path) else None)
    apls_evaluator = APLSMetric(max_buffer_px=22.0, max_pairs_sample=400)

    results = []

    for city_id, meta in SPACENET_CITIES.items():
        print(f"\n[City: {city_id.upper()} - {meta['description']}]")
        tif_filename = os.path.basename(meta["img_key"])
        json_filename = os.path.basename(meta["json_key"])
        tif_path = os.path.join(cache_dir, tif_filename)
        json_path = os.path.join(cache_dir, json_filename)

        img_url = f"https://spacenet-dataset.s3.amazonaws.com/{meta['img_key']}"
        json_url = f"https://spacenet-dataset.s3.amazonaws.com/{meta['json_key']}"

        try:
            download_file_if_missing(img_url, tif_path)
            download_file_if_missing(json_url, json_path)
        except Exception as e:
            print(f"  Warning: Failed to fetch {city_id} from S3 ({e}). Skipping...")
            continue

        # Parse Ground Truth
        gt_graph, img_uint8 = parse_spacenet_ground_truth(tif_path, json_path)
        print(f"  Loaded Image: {img_uint8.shape} | Ground Truth: {gt_graph.number_of_nodes()} nodes, {gt_graph.number_of_edges()} edges")

        city_out = os.path.join(output_dir, city_id)
        os.makedirs(city_out, exist_ok=True)
        Image.fromarray(img_uint8).save(os.path.join(city_out, "real_satellite_input.png"))

        # Run TopoGrapher Model Inference
        t0 = time.time()
        preds = engine.extract_road_network(img_uint8, regularize=True)
        t_infer = time.time() - t0

        prop_graph = preds["graph"]
        telemetry = preds["telemetry"]

        # Compute SpaceNet APLS
        apls_res = apls_evaluator.compute_apls(gt_graph, prop_graph)

        # Test A* Navigation
        routing = preds["routing"]
        route_ok = False
        route_len = 0.0
        route_coords = []
        if prop_graph.number_of_nodes() >= 2:
            comps = sorted(nx.connected_components(prop_graph), key=len, reverse=True)
            if comps and len(comps[0]) >= 2:
                lc = list(comps[0])
                n_start, n_end = lc[0], lc[-1]
                p_start = (prop_graph.nodes[n_start]["x"], prop_graph.nodes[n_start]["y"])
                p_end = (prop_graph.nodes[n_end]["x"], prop_graph.nodes[n_end]["y"])
                route_res = routing.shortest_path(p_start, p_end)
                route_ok = route_res["success"]
                route_len = route_res["distance_m"]
                route_coords = route_res.get("path_coords", [])

        # Render Visual Composite Overlay
        vis = engine.render_overlay_visualization(
            img_uint8,
            prop_graph,
            shortest_path_coords=route_coords if route_ok else None
        )
        vis_path = os.path.join(city_out, "spacenet_vector_overlay.png")
        vis.save(vis_path)

        # Export GeoJSON
        with open(os.path.join(city_out, "extracted_roads.geojson"), "w", encoding="utf-8") as f:
            json.dump(preds["geojson"], f, indent=2)

        prec = apls_res["apls_precision"]
        rec = apls_res["apls_recall"]
        f1 = apls_res.get("apls_harmonic", 0.0)

        record = {
            "city": city_id,
            "aoi": meta["aoi"],
            "description": meta["description"],
            "gt_nodes": gt_graph.number_of_nodes(),
            "gt_edges": gt_graph.number_of_edges(),
            "extracted_nodes": prop_graph.number_of_nodes(),
            "extracted_edges": prop_graph.number_of_edges(),
            "total_km": round(telemetry["total_length_km"], 2),
            "reachability_pct": round(telemetry["largest_component_ratio"] * 100.0, 1),
            "apls": round(apls_res["apls"], 4),
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "inference_sec": round(t_infer, 3),
            "routing_success": route_ok,
            "route_dist_m": round(route_len, 1)
        }
        results.append(record)

        print(f"  --> Extracted: {prop_graph.number_of_nodes()} nodes, {prop_graph.number_of_edges()} edges ({telemetry['total_length_km']:.2f} km)")
        print(f"  --> Reachability: {telemetry['largest_component_ratio']*100:.1f}% | Inference: {t_infer:.3f}s")
        print(f"  --> SpaceNet APLS: {apls_res['apls']:.4f} (F1: {f1:.4f}, Prec: {prec*100:.1f}%, Rec: {rec*100:.1f}%)")
        print(f"  --> A* Shortest Path: {'SUCCESS' if route_ok else 'FAIL'} ({route_len:.1f}m)")

    summary_path = os.path.join(output_dir, "spacenet_evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=======================================================================================")
    print("                     REAL SATELLITE BENCHMARK SUMMARY (SPACENET 3)                    ")
    print("=======================================================================================")
    if results:
        print(f" Total Cities Evaluated : {len(results)}")
        print(f" Mean SpaceNet APLS     : {np.mean([r['apls'] for r in results]):.4f}")
        print(f" Mean Precision         : {np.mean([r['precision'] for r in results])*100:.1f}%")
        print(f" Mean Recall            : {np.mean([r['recall'] for r in results])*100:.1f}%")
        print(f" Mean Harmonic F1       : {np.mean([r['f1'] for r in results]):.4f}")
        print(f" Mean Reachability      : {np.mean([r['reachability_pct'] for r in results]):.1f}%")
        print(f" Summary Saved To       : {summary_path}")
    print("=======================================================================================")
    return {"results": results}


if __name__ == "__main__":
    run_spacenet_benchmark()
