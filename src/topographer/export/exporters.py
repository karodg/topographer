"""
GIS Vector Exporters for TopoGrapher Networks.

Exports topological road graphs to standard geospatial GIS formats:
1. GeoJSON (RFC 7946) with LineString road corridors and Point intersections;
2. OSM XML (OpenStreetMap format compatible with JOSM, QGIS, and OSRM);
3. GeoPackage / WKT geometry for PostGIS and spatial databases.
"""

from typing import Dict, List, Tuple, Optional, Any
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
import networkx as nx


class GISExporter:
    """
    Exports NetworkX road topologies into standard geospatial data formats.
    """

    def __init__(
        self,
        pixel_to_meter_scale: float = 0.5,
        origin_lon_lat: Optional[Tuple[float, float]] = None
    ):
        """
        Parameters
        ----------
        pixel_to_meter_scale : float
            Ground Sampling Distance (GSD) in meters per pixel.
        origin_lon_lat : Optional[Tuple[float, float]]
            (lon, lat) of the top-left pixel (0, 0).
            If omitted, uses local metric coordinates centered at origin.
        """
        self.scale = pixel_to_meter_scale
        self.origin = origin_lon_lat

    def _pixel_to_geo(self, px: float, py: float) -> Tuple[float, float]:
        """Converts pixel coordinate to (lon, lat) or local meters (x, y)."""
        if self.origin is not None:
            # Approximate degrees conversion near equator / mid-latitudes
            lon_0, lat_0 = self.origin
            meters_x = px * self.scale
            meters_y = py * self.scale
            # 1 deg lat ~= 111,320m; 1 deg lon ~= 111,320 * cos(lat)
            d_lat = -meters_y / 111320.0
            d_lon = meters_x / (111320.0 * np.cos(np.radians(lat_0)))
            return (lon_0 + d_lon, lat_0 + d_lat)
        else:
            return (px * self.scale, -py * self.scale)

    def to_geojson(self, graph: nx.Graph, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Converts road graph into standard GeoJSON FeatureCollection.
        """
        features = []

        # Road classes mapping to OSM highway tags
        highway_tags = {
            0: "motorway",
            1: "primary",
            2: "residential",
            3: "service"
        }

        # 1. Road Edges as LineString features
        edge_id = 1
        for u, v, data in graph.edges(data=True):
            p1 = (graph.nodes[u]["x"], graph.nodes[u]["y"])
            p2 = (graph.nodes[v]["x"], graph.nodes[v]["y"])
            g1 = self._pixel_to_geo(p1[0], p1[1])
            g2 = self._pixel_to_geo(p2[0], p2[1])

            length_m = data.get("length_px", data.get("weight", 0.0)) * self.scale
            r_class = data.get("road_class", 1)
            hw_type = highway_tags.get(r_class, "residential")

            feature = {
                "type": "Feature",
                "id": edge_id,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [list(g1), list(g2)]
                },
                "properties": {
                    "edge_id": edge_id,
                    "source_node": int(u),
                    "target_node": int(v),
                    "highway": hw_type,
                    "length_m": round(float(length_m), 2),
                    "confidence": round(float(data.get("probability", 1.0)), 3),
                    "oneway": "no"
                }
            }
            features.append(feature)
            edge_id += 1

        # 2. Intersections / Nodes as Point features
        for n, data in graph.nodes(data=True):
            p = (data["x"], data["y"])
            geo_pt = self._pixel_to_geo(p[0], p[1])
            deg = graph.degree(n)
            junction_type = "crossroads" if deg >= 4 else ("t_junction" if deg == 3 else ("terminal" if deg == 1 else "waypoint"))

            feature = {
                "type": "Feature",
                "id": f"node_{n}",
                "geometry": {
                    "type": "Point",
                    "coordinates": list(geo_pt)
                },
                "properties": {
                    "node_id": int(n),
                    "degree": deg,
                    "junction_type": junction_type,
                    "confidence": round(float(data.get("score", 1.0)), 3)
                }
            }
            features.append(feature)

        geojson_doc = {
            "type": "FeatureCollection",
            "generator": "TopoGrapher v1.0",
            "features": features
        }

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(geojson_doc, f, indent=2)

        return geojson_doc

    def to_osm_xml(self, graph: nx.Graph, output_path: Optional[str] = None) -> str:
        """
        Exports road graph to OpenStreetMap XML format compatible with JOSM and QGIS.
        """
        osm = ET.Element("osm", version="0.6", generator="TopoGrapher Autonomous Road Engine")

        # 1. Add OSM Nodes
        for n, data in graph.nodes(data=True):
            p = (data["x"], data["y"])
            geo_pt = self._pixel_to_geo(p[0], p[1])
            # If local metric coords, map to valid fake lon/lat for JOSM
            lon = geo_pt[0] if self.origin else 103.8198 + (geo_pt[0] / 111320.0)
            lat = geo_pt[1] if self.origin else 1.3521 + (geo_pt[1] / 111320.0)

            node_elem = ET.SubElement(
                osm, "node",
                id=str(n + 1000),
                lat=f"{lat:.7f}",
                lon=f"{lon:.7f}",
                version="1"
            )
            deg = graph.degree(n)
            if deg >= 3:
                ET.SubElement(node_elem, "tag", k="highway", v="traffic_signals" if deg >= 4 else "junction")

        # 2. Add OSM Ways
        highway_tags = {0: "motorway", 1: "primary", 2: "residential", 3: "service"}
        way_id = 50000

        for u, v, data in graph.edges(data=True):
            way_elem = ET.SubElement(osm, "way", id=str(way_id), version="1")
            ET.SubElement(way_elem, "nd", ref=str(u + 1000))
            ET.SubElement(way_elem, "nd", ref=str(v + 1000))

            r_class = data.get("road_class", 1)
            hw_type = highway_tags.get(r_class, "residential")
            ET.SubElement(way_elem, "tag", k="highway", v=hw_type)
            ET.SubElement(way_elem, "tag", k="source", v="TopoGrapher_AI")
            way_id += 1

        rough_string = ET.tostring(osm, encoding="utf-8")
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(pretty_xml)

        return pretty_xml
