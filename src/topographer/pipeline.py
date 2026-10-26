"""
TopoGrapher Master Autonomous Pipeline.

Unified interface orchestrating:
1. Neural feature extraction & dense orientation/keypoint prediction;
2. Candidate edge graph attention verification with tree-canopy bridging;
3. Graph tracing and topological regularization;
4. Routing analytics and GIS GeoJSON / OSM XML generation;
5. Diagnostic visualization rendering.
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import os
import numpy as np
from PIL import Image, ImageDraw
import torch
import networkx as nx

from topographer.models.full_model import TopoGrapherNet
from topographer.graph.tracer import GraphTracer
from topographer.graph.regularizer import GraphRegularizer
from topographer.graph.simplifier import GraphTopologySimplifier
from topographer.graph.routing import RoadRoutingEngine
from topographer.export.exporters import GISExporter


class TopoGrapherEngine:
    """
    End-to-End Autonomous Vector Road Extraction Engine.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        keypoint_threshold: float = 0.25,
        edge_threshold: float = 0.35,
        max_edge_dist_px: float = 140.0,
        gsd_meters_per_pixel: float = 0.5
    ):
        """
        Parameters
        ----------
        model_path : Optional[str]
            Path to pretrained PyTorch checkpoint (.pt / .pth).
        device : Optional[str]
            Target compute device ('cuda' or 'cpu').
        keypoint_threshold : float
            Confidence threshold for intersection and terminal detection.
        edge_threshold : float
            Acceptance threshold for road edge connection.
        max_edge_dist_px : float
            Maximum distance for candidate edge consideration.
        gsd_meters_per_pixel : float
            Ground Sampling Distance scale (e.g. 0.5m/px).
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = TopoGrapherNet().to(self.device)
        default_ckpt = "d:/Workspace/checkpoints/topographer_weights.pt"
        if model_path is None and os.path.exists(default_ckpt):
            model_path = default_ckpt

        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(checkpoint, dict):
                self.model.load_state_dict(checkpoint)
            self.model.eval()

        self.tracer = GraphTracer(
            keypoint_threshold=keypoint_threshold,
            edge_threshold=edge_threshold,
            max_edge_distance_px=max_edge_dist_px
        )
        self.regularizer = GraphRegularizer()
        self.simplifier = GraphTopologySimplifier()
        self.gsd = gsd_meters_per_pixel
        self.exporter = GISExporter(pixel_to_meter_scale=self.gsd)

    def extract_road_network(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        regularize: bool = True
    ) -> Dict[str, Any]:
        """
        Processes an aerial or satellite raster image and extracts a topologically closed vector graph.

        Parameters
        ----------
        image_input : Union[str, np.ndarray, Image.Image]
            File path, numpy uint8 array (H, W, 3), or PIL Image.
        regularize : bool
            Whether to run geometric and topological regularization.

        Returns
        -------
        results : Dict[str, Any]
            Extracted graph, GIS GeoJSON, routing engine, and telemetry summary.
        """
        # 1. Standardize Image Input to RGB numpy array
        if isinstance(image_input, str):
            img_pil = Image.open(image_input).convert("RGB")
            img_np = np.array(img_pil)
        elif isinstance(image_input, Image.Image):
            img_np = np.array(image_input.convert("RGB"))
        else:
            img_np = image_input

        h, w, c = img_np.shape

        # For very large aerial scenes (>1200px), execute memory-efficient tiled sliding window
        if max(h, w) > 1200:
            return self._extract_tiled_large_network(img_np, regularize=regularize)

        # 2. Prepare PyTorch Tensor
        if img_np.dtype == np.uint16 or img_np.max() > 255:
            # Standard remote sensing 2% - 98% percentile stretch for 16-bit satellite imagery
            p2, p98 = np.percentile(img_np, (2, 98))
            img_float = np.clip((img_np.astype(np.float32) - p2) / (p98 - p2 + 1e-5), 0.0, 1.0)
        else:
            img_float = img_np.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # 3. Predict graph elements via TopoGrapherNet
        raw_preds = self.model.predict_graph_elements(
            tensor,
            keypoint_threshold=self.tracer.kp_thresh,
            max_edge_distance_px=self.tracer.max_dist,
            edge_threshold=self.tracer.edge_thresh
        )

        orient_map = raw_preds["dense"]["tangent"] if "dense" in raw_preds else None
        road_prob = raw_preds["dense"]["road_prob"] if "dense" in raw_preds else None

        # 4. Refine, simplify and sanitize topological graph
        if regularize:
            G_final = self.simplifier.simplify_and_refine(
                vertices=raw_preds["vertices"],
                raw_edges=raw_preds["edges"],
                road_prob_map=road_prob,
                tangent_field=orient_map,
                image_shape=(h, w)
            )
            G_raw = G_final
        else:
            G_raw = self.tracer.trace_graph(
                vertices=raw_preds["vertices"],
                edge_predictions=raw_preds["edges"],
                orientation_field=orient_map,
                road_prob=road_prob,
                image_shape=(h, w)
            )
            G_final = G_raw

        # 6. Routing & Telemetry
        routing_engine = RoadRoutingEngine(G_final, pixel_to_meter_scale=self.gsd)
        telemetry = routing_engine.compute_network_telemetry()

        # 7. GIS GeoJSON
        geojson = self.exporter.to_geojson(G_final)

        return {
            "graph": G_final,
            "raw_graph": G_raw,
            "telemetry": telemetry,
            "geojson": geojson,
            "routing": routing_engine,
            "input_shape": (h, w),
            "dense_maps": raw_preds.get("dense", {})
        }

    def _extract_tiled_large_network(
        self,
        img_np: np.ndarray,
        tile_size: int = 768,
        stride: int = 640,
        regularize: bool = True
    ) -> Dict[str, Any]:
        """
        Executes sliding-window tiled graph extraction for massive city-scale GeoTIFFs (up to 16Kx16K).
        """
        h, w, _ = img_np.shape
        G_merged = nx.Graph()
        global_node_id = 0

        y_steps = list(range(0, max(1, h - tile_size + 1), stride))
        if y_steps[-1] + tile_size < h:
            y_steps.append(h - tile_size)
        x_steps = list(range(0, max(1, w - tile_size + 1), stride))
        if x_steps[-1] + tile_size < w:
            x_steps.append(w - tile_size)

        for y0 in y_steps:
            for x0 in x_steps:
                tile = img_np[y0 : y0 + tile_size, x0 : x0 + tile_size]
                tile_float = tile.astype(np.float32) / 255.0
                tensor = torch.from_numpy(tile_float).permute(2, 0, 1).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    preds = self.model.predict_graph_elements(
                        tensor,
                        keypoint_threshold=self.tracer.kp_thresh,
                        max_edge_distance_px=self.tracer.max_dist,
                        edge_threshold=self.tracer.edge_thresh
                    )

                orient_map = preds["dense"]["tangent"] if "dense" in preds else None
                road_prob = preds["dense"]["road_prob"] if "dense" in preds else None

                G_tile = self.tracer.trace_graph(
                    vertices=preds["vertices"],
                    edge_predictions=preds["edges"],
                    orientation_field=orient_map,
                    road_prob=road_prob,
                    image_shape=(tile_size, tile_size)
                )

                # Offset tile nodes to global space and insert into G_merged
                local_to_global = {}
                for n, d in G_tile.nodes(data=True):
                    gx = float(d["x"] + x0)
                    gy = float(d["y"] + y0)
                    G_merged.add_node(global_node_id, x=gx, y=gy, score=d.get("score", 1.0), type=d.get("type", 0))
                    local_to_global[n] = global_node_id
                    global_node_id += 1

                for u, v, d in G_tile.edges(data=True):
                    gu = local_to_global[u]
                    gv = local_to_global[v]
                    G_merged.add_edge(gu, gv, **d)

        if regularize:
            G_final = self.regularizer.regularize(G_merged)
        else:
            G_final = G_merged

        routing_engine = RoadRoutingEngine(G_final, pixel_to_meter_scale=self.gsd)
        telemetry = routing_engine.compute_network_telemetry()
        geojson = self.exporter.to_geojson(G_final)

        return {
            "graph": G_final,
            "raw_graph": G_merged,
            "telemetry": telemetry,
            "geojson": geojson,
            "routing": routing_engine,
            "input_shape": (h, w),
            "tiled": True
        }

    def render_overlay_visualization(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        graph: nx.Graph,
        shortest_path_coords: Optional[List[Tuple[float, float]]] = None
    ) -> Image.Image:
        """
        Renders a diagnostic GIS visualization of the vector graph overlaid on the aerial image.
        """
        if isinstance(image_input, str):
            base_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            base_img = image_input.convert("RGB").copy()
        else:
            base_img = Image.fromarray(image_input).convert("RGB")

        overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Draw Edges (Cyan for regular roads, Orange for bridged gap segments)
        for u, v, d in graph.edges(data=True):
            p1 = (graph.nodes[u]["x"], graph.nodes[u]["y"])
            p2 = (graph.nodes[v]["x"], graph.nodes[v]["y"])

            is_bridged = d.get("bridged", False)
            edge_color = (255, 165, 0, 220) if is_bridged else (0, 220, 255, 200)
            edge_width = 3 if not is_bridged else 2

            draw.line([p1, p2], fill=edge_color, width=edge_width)

        # Draw Shortest Path Route (Glowing Neon Green/Lime)
        if shortest_path_coords and len(shortest_path_coords) >= 2:
            draw.line(shortest_path_coords, fill=(50, 255, 50, 240), width=6)

        # Draw Nodes (Amber for junctions deg >= 3, White for waypoints, Red for dead-ends)
        for n, d in graph.nodes(data=True):
            px, py = d["x"], d["y"]
            deg = graph.degree(n)

            if deg >= 3:
                node_color = (255, 200, 0, 255)  # Amber Junction
                radius = 4.5
            elif deg == 1:
                node_color = (255, 70, 70, 255)   # Red Dead-End
                radius = 3.5
            else:
                node_color = (200, 230, 255, 255) # Light blue waypoint
                radius = 2.5

            draw.ellipse(
                [(px - radius, py - radius), (px + radius, py + radius)],
                fill=node_color,
                outline=(20, 20, 20, 255),
                width=1
            )

        # Composite overlay on base image
        result = Image.alpha_composite(base_img.convert("RGBA"), overlay)
        return result.convert("RGB")


def main():
    """Command-line interface for TopoGrapher."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="topographer",
        description="TopoGrapher: Autonomous Neural Road Network & Topology Graph Extraction"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: extract
    p_extract = subparsers.add_parser("extract", help="Extract vector road graph from satellite/aerial image")
    p_extract.add_argument("image", type=str, help="Path to input image (GeoTIFF, PNG, JPG)")
    p_extract.add_argument("--weights", type=str, default="checkpoints/topographer_weights.pt", help="Path to model weights")
    p_extract.add_argument("--outdir", type=str, default="output", help="Directory to save exports")
    p_extract.add_argument("--gsd", type=float, default=0.5, help="Ground sampling distance (meters/pixel)")

    # Command: benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run benchmark on synthetic Singapore or radial scene")
    p_bench.add_argument("--scene", type=str, default="singapore", choices=["singapore", "radial"], help="Scene morphology")
    p_bench.add_argument("--resolution", type=int, default=1024, help="Resolution in pixels")
    p_bench.add_argument("--weights", type=str, default="checkpoints/topographer_weights.pt", help="Path to weights")
    p_bench.add_argument("--outdir", type=str, default="benchmark_results", help="Directory to save benchmark")

    args = parser.parse_args()

    if args.command == "benchmark":
        from topographer.scripts.benchmark_scene import run_benchmark
        run_benchmark(
            weights_path=args.weights,
            output_dir=args.outdir,
            resolution=args.resolution,
            scene_type=args.scene
        )
    elif args.command == "extract":
        os.makedirs(args.outdir, exist_ok=True)
        print(f"--> Extracting vector road graph from: {args.image}")
        engine = TopoGrapherEngine(
            model_path=args.weights if os.path.exists(args.weights) else None,
            gsd_meters_per_pixel=args.gsd
        )
        results = engine.extract_road_network(args.image, regularize=True)
        g = results["graph"]
        t = results["telemetry"]

        # Save GeoJSON
        geojson_out = os.path.join(args.outdir, "roads.geojson")
        with open(geojson_out, "w", encoding="utf-8") as f:
            json.dump(results["geojson"], f, indent=2)

        # Save OSM XML
        osm_out = os.path.join(args.outdir, "roads.osm")
        engine.exporter.to_osm_xml(g, output_path=osm_out)

        # Save Visual Overlay
        vis_out = os.path.join(args.outdir, "roads_overlay.png")
        vis = engine.render_overlay_visualization(args.image, g)
        vis.save(vis_out)

        print(f"--> Extraction Complete!")
        print(f"    Nodes: {g.number_of_nodes()} | Edges: {g.number_of_edges()} | Total Length: {t['total_length_km']:.2f} km")
        print(f"    Saved: {geojson_out}")
        print(f"    Saved: {osm_out}")
        print(f"    Saved: {vis_out}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
