"""
Routability and Graph Pathfinding Engine.

Provides high-performance routing analytics for extracted topological networks:
- A* and Dijkstra shortest path pathfinding with Euclidean heuristic;
- Coordinate snapping (snapping arbitrary query points to nearest road nodes/edges);
- Turn-by-turn navigation maneuvers (bearing calculations, turn angles);
- Comprehensive GIS topological metrics (network diameter, circuity, reachability).
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import networkx as nx


class RoadRoutingEngine:
    """
    Routability verification and pathfinding engine for TopoGrapher road networks.
    """

    def __init__(self, graph: nx.Graph, pixel_to_meter_scale: float = 0.5):
        """
        Parameters
        ----------
        graph : nx.Graph
            Topological road graph with 'x', 'y' node attributes and 'weight' edge attributes.
        pixel_to_meter_scale : float
            Ground Sampling Distance (GSD) in meters per pixel.
        """
        self.graph = graph
        self.scale = pixel_to_meter_scale

    def find_nearest_node(self, x: float, y: float) -> Tuple[int, float]:
        """
        Finds the closest node to the specified coordinates.

        Returns
        -------
        nearest_node : int
        distance_px : float
        """
        best_node = None
        min_dist = float('inf')

        for n, data in self.graph.nodes(data=True):
            nx_coord = data["x"]
            ny_coord = data["y"]
            dist = np.hypot(nx_coord - x, ny_coord - y)
            if dist < min_dist:
                min_dist = dist
                best_node = n

        if best_node is None:
            raise ValueError("Graph has no vertices to query.")

        return best_node, float(min_dist)

    def shortest_path(
        self,
        start_xy: Tuple[float, float],
        end_xy: Tuple[float, float]
    ) -> Dict[str, Any]:
        """
        Computes the optimal A* path between arbitrary coordinates.

        Parameters
        ----------
        start_xy : Tuple[float, float]
            (x, y) start coordinates.
        end_xy : Tuple[float, float]
            (x, y) target coordinates.

        Returns
        -------
        result : Dict[str, Any]
            Path geometry, total length in meters, waypoints, and turn directions.
        """
        start_node, d_start = self.find_nearest_node(start_xy[0], start_xy[1])
        end_node, d_end = self.find_nearest_node(end_xy[0], end_xy[1])

        if not nx.has_path(self.graph, start_node, end_node):
            return {
                "success": False,
                "error": "No reachable path between coordinates in connected components.",
                "start_node": start_node,
                "end_node": end_node,
                "distance_m": 0.0,
                "path_coords": []
            }

        # A* Heuristic: Euclidean distance in pixels
        def heuristic(u: int, v: int) -> float:
            p1 = (self.graph.nodes[u]["x"], self.graph.nodes[u]["y"])
            p2 = (self.graph.nodes[v]["x"], self.graph.nodes[v]["y"])
            return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

        node_path = nx.astar_path(self.graph, start_node, end_node, heuristic=heuristic, weight='weight')
        path_length_px = nx.astar_path_length(self.graph, start_node, end_node, heuristic=heuristic, weight='weight')

        # Collect path coordinates
        path_coords = [
            (self.graph.nodes[n]["x"], self.graph.nodes[n]["y"])
            for n in node_path
        ]

        # Generate turn maneuvers
        maneuvers = self._generate_maneuvers(path_coords)

        return {
            "success": True,
            "start_node": start_node,
            "end_node": end_node,
            "node_count": len(node_path),
            "distance_px": float(path_length_px),
            "distance_m": float(path_length_px * self.scale),
            "path_coords": path_coords,
            "maneuvers": maneuvers
        }

    def _generate_maneuvers(self, coords: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
        """Calculates turn angles and bearings along waypoints."""
        if len(coords) < 2:
            return []

        maneuvers = []
        for i in range(len(coords) - 1):
            p1, p2 = coords[i], coords[i + 1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            bearing = (np.degrees(np.arctan2(dx, -dy)) + 360) % 360  # Compass bearing (0 = North)
            dist_seg_m = np.hypot(dx, dy) * self.scale

            instruction = "Proceed"
            if i > 0:
                prev_p = coords[i - 1]
                prev_dx = p1[0] - prev_p[0]
                prev_dy = p1[1] - prev_p[1]
                prev_bearing = (np.degrees(np.arctan2(prev_dx, -prev_dy)) + 360) % 360
                turn_diff = (bearing - prev_bearing + 180) % 360 - 180

                if turn_diff > 35.0:
                    instruction = "Turn Right"
                elif turn_diff < -35.0:
                    instruction = "Turn Left"
                else:
                    instruction = "Continue straight"

            maneuvers.append({
                "step": i + 1,
                "instruction": instruction,
                "bearing_deg": float(bearing),
                "distance_m": float(dist_seg_m),
                "coord": p1
            })

        return maneuvers

    def compute_network_telemetry(self) -> Dict[str, Any]:
        """
        Computes comprehensive GIS topological summary of the network.
        """
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()

        if num_nodes == 0:
            return {
                "nodes": 0,
                "edges": 0,
                "total_length_km": 0.0,
                "junctions": 0,
                "dead_ends": 0,
                "connected_components": 0,
                "largest_component_ratio": 0.0
            }

        total_length_px = sum(
            d.get("length_px", d.get("weight", 0.0))
            for _, _, d in self.graph.edges(data=True)
        )
        total_length_km = (total_length_px * self.scale) / 1000.0

        degrees = dict(self.graph.degree())
        junctions = sum(1 for d in degrees.values() if d >= 3)
        dead_ends = sum(1 for d in degrees.values() if d == 1)

        components = list(nx.connected_components(self.graph))
        largest_comp = max(len(c) for c in components) if components else 0
        reachability_ratio = largest_comp / float(num_nodes) if num_nodes > 0 else 0.0

        return {
            "nodes": num_nodes,
            "edges": num_edges,
            "total_length_km": float(total_length_km),
            "junctions": junctions,
            "dead_ends": dead_ends,
            "connected_components": len(components),
            "largest_component_ratio": float(reachability_ratio),
            "average_degree": float(2.0 * num_edges / num_nodes) if num_nodes > 0 else 0.0
        }
