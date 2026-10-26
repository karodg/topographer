"""
Test fetching real satellite orthophoto tile and OpenStreetMap vector ground truth.
"""

import urllib.request
import json
import numpy as np
from PIL import Image
import networkx as nx


def test_osm_fetch():
    # Las Vegas Strip coordinates bbox: min_lat, min_lon, max_lat, max_lon
    # Small test bbox around 36.1147, -115.1728
    lat = 36.1147
    lon = -115.1728
    dlat = 0.005
    dlon = 0.005
    min_lat, max_lat = lat - dlat, lat + dlat
    min_lon, max_lon = lon - dlon, lon + dlon

    print(f"Testing Overpass API query for bbox: [{min_lat}, {min_lon}, {max_lat}, {max_lon}]...")
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:25];
    (
      way["highway"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out body;
    >;
    out skel qt;
    """
    req = urllib.request.Request(
        overpass_url,
        data=query.encode("utf-8"),
        headers={"User-Agent": "TopoGrapher-Benchmark/1.0"}
    )
    import ssl
    ctx = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
            elements = data.get("elements", [])
            nodes = {el["id"]: (el["lon"], el["lat"]) for el in elements if el.get("type") == "node"}
            ways = [el for el in elements if el.get("type") == "way"]
            print(f"Overpass API SUCCESS: Retrieved {len(nodes)} nodes, {len(ways)} road ways!")
            return True
    except Exception as e:
        print(f"Overpass API error: {e}")
        return False


if __name__ == "__main__":
    test_osm_fetch()
