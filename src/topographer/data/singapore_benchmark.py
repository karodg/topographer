"""
Singapore Island & Marina Bay Aerial Scene Benchmark Generator.

Generates realistic coastal-urban satellite imagery mimicking Singapore:
- Deep blue coastline and harbour water bodies;
- High-capacity coastal expressways (e.g., East Coast Parkway / Marina Coastal Expressway);
- Dense rectilinear grid blocks of the Central Business District (CBD);
- Tropical rain-tree foliage corridors (Albizia / Samanea saman canopies) occluding road segments;
- Full topological ground truth graph for SpaceNet APLS and autonomous routing benchmarks.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter

from topographer.data.procedural_generator import ProceduralRoadScene


def generate_singapore_scene(
    width: int = 1024,
    height: int = 1024,
    seed: int = 2026
) -> ProceduralRoadScene:
    """
    Synthesizes a Singapore coastal/urban satellite scene with ground truth graph.
    """
    np.random.seed(seed)

    # 1. Base Terrain: Lush tropical soil and coastal sand
    r = np.random.normal(70, 6, (height, width))
    g = np.random.normal(115, 10, (height, width))
    b = np.random.normal(55, 6, (height, width))
    terrain = np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)
    terrain = gaussian_filter(terrain, sigma=1.2)

    img = Image.fromarray(terrain)
    draw = ImageDraw.Draw(img)

    # 2. Render Coastline & Marina Harbour Water Body (South / South-East)
    water_color = (25, 55, 85)
    # Sinusoidal coastline across the bottom 25% of the scene
    coast_pts = [(0, height)]
    for x in range(0, width + 50, 50):
        y_water = height - 160 + int(35 * np.sin(x * 0.006) + 15 * np.cos(x * 0.015))
        coast_pts.append((x, y_water))
    coast_pts.append((width, height))
    draw.polygon(coast_pts, fill=water_color)

    # Coastal sandy beach strip along waterline
    for k in range(len(coast_pts) - 2):
        p1 = coast_pts[k + 1]
        p2 = coast_pts[k + 2]
        sand_color = (195, 180, 140)
        draw.line([p1, p2], fill=sand_color, width=8)

    # 3. Design Road Network Topology (Nodes & Edges)
    vertices = []
    edges = []
    polylines = []
    v_idx = 0

    # A. Coastal Expressway (Dual-Lane Primary Highway running parallel to coastline)
    express_y = height - 230
    xs_express = list(range(40, width - 40, 110))
    express_nodes = []
    for x in xs_express:
        y_pos = express_y + int(20 * np.sin(x * 0.005))
        vertices.append((float(x), float(y_pos), 0))
        express_nodes.append(v_idx)
        v_idx += 1

    for i in range(len(express_nodes) - 1):
        u = express_nodes[i]
        v = express_nodes[i + 1]
        edges.append((u, v, 0))  # Class 0: Motorway
        polylines.append([(vertices[u][0], vertices[u][1]), (vertices[v][0], vertices[v][1])])

    # B. CBD Urban Street Grid (North-West sector: 6x6 grid blocks)
    grid_xs = list(range(80, 520, 85))
    grid_ys = list(range(80, 520, 85))
    grid_matrix = {}

    for r_i, gy in enumerate(grid_ys):
        for c_i, gx in enumerate(grid_xs):
            jx = float(gx + np.random.randint(-4, 5))
            jy = float(gy + np.random.randint(-4, 5))
            vertices.append((jx, jy, 0 if (0 < r_i < len(grid_ys)-1 and 0 < c_i < len(grid_xs)-1) else 1))
            grid_matrix[(r_i, c_i)] = v_idx
            v_idx += 1

    for r_i in range(len(grid_ys)):
        for c_i in range(len(grid_xs)):
            curr = grid_matrix[(r_i, c_i)]
            p_c = (vertices[curr][0], vertices[curr][1])

            # Horizontal edge
            if c_i + 1 < len(grid_xs):
                nxt = grid_matrix[(r_i, c_i + 1)]
                edges.append((curr, nxt, 1))  # Class 1: Arterial
                polylines.append([p_c, (vertices[nxt][0], vertices[nxt][1])])

            # Vertical edge
            if r_i + 1 < len(grid_ys):
                nxt = grid_matrix[(r_i + 1, c_i)]
                edges.append((curr, nxt, 1))
                polylines.append([p_c, (vertices[nxt][0], vertices[nxt][1])])

    # C. Connecting Arterial Avenues (CBD to Expressway & Marina Bay Island Bridge)
    # Connector 1
    u_cbd = grid_matrix[(len(grid_ys) - 1, 1)]
    u_exp = express_nodes[1]
    edges.append((u_cbd, u_exp, 1))
    polylines.append([(vertices[u_cbd][0], vertices[u_cbd][1]), (vertices[u_exp][0], vertices[u_exp][1])])

    # Connector 2
    u_cbd2 = grid_matrix[(len(grid_ys) - 1, len(grid_xs) - 2)]
    u_exp2 = express_nodes[4]
    edges.append((u_cbd2, u_exp2, 1))
    polylines.append([(vertices[u_cbd2][0], vertices[u_cbd2][1]), (vertices[u_exp2][0], vertices[u_exp2][1])])

    # D. East Residential Meandering Corridors (East sector)
    east_seeds = [
        (650.0, 150.0), (820.0, 130.0), (940.0, 220.0),
        (720.0, 310.0), (880.0, 380.0), (680.0, 480.0),
        (850.0, 560.0), (960.0, 490.0)
    ]
    east_nodes = []
    for sx, sy in east_seeds:
        vertices.append((sx, sy, 2))
        east_nodes.append(v_idx)
        v_idx += 1

    east_links = [
        (0, 1), (1, 2), (0, 3), (3, 4), (4, 2),
        (3, 5), (5, 6), (6, 7), (4, 7)
    ]
    for e_u, e_v in east_links:
        u_idx = east_nodes[e_u]
        v_idx_t = east_nodes[e_v]
        edges.append((u_idx, v_idx_t, 2))  # Class 2: Residential
        polylines.append([(vertices[u_idx][0], vertices[u_idx][1]), (vertices[v_idx_t][0], vertices[v_idx_t][1])])

    # Connect East sector to CBD and Expressway
    edges.append((grid_matrix[(1, len(grid_xs)-1)], east_nodes[0], 1))
    polylines.append([(vertices[grid_matrix[(1, len(grid_xs)-1)]][0], vertices[grid_matrix[(1, len(grid_xs)-1)]][1]),
                      (vertices[east_nodes[0]][0], vertices[east_nodes[0]][1])])

    edges.append((east_nodes[6], express_nodes[7], 1))
    polylines.append([(vertices[east_nodes[6]][0], vertices[east_nodes[6]][1]),
                      (vertices[express_nodes[7]][0], vertices[express_nodes[7]][1])])

    # 4. Render Asphalt Pavement & Lane Markings
    gt_mask = np.zeros((height, width), dtype=np.uint8)
    mask_img = Image.fromarray(gt_mask)
    mask_draw = ImageDraw.Draw(mask_img)

    cos_map = np.zeros((height, width), dtype=np.float32)
    sin_map = np.zeros((height, width), dtype=np.float32)

    for (u, v, r_class), pts in zip(edges, polylines):
        widths = {0: 16, 1: 11, 2: 7, 3: 5}
        w_px = widths.get(r_class, 7)

        # Draw road
        asphalt = (52, 53, 58)
        draw.line(pts, fill=asphalt, width=w_px, joint='round')
        mask_draw.line(pts, fill=1, width=w_px, joint='round')

        # Center line for wide roads
        if w_px >= 11:
            draw.line(pts, fill=(230, 230, 210), width=1, joint='round')

        # Compute orientation
        for k in range(len(pts) - 1):
            p1, p2 = pts[k], pts[k + 1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            theta = np.arctan2(dy, dx)
            cos_2th = np.cos(2.0 * theta)
            sin_2th = np.sin(2.0 * theta)

            seg_img = Image.new('L', (width, height), 0)
            seg_draw = ImageDraw.Draw(seg_img)
            seg_draw.line([p1, p2], fill=255, width=w_px)
            seg_mask = np.array(seg_img) > 0

            cos_map[seg_mask] = cos_2th
            sin_map[seg_mask] = sin_2th

    # 5. Add Dense Tropical Tree Canopies and Building Shadows
    # Dense Rain Trees along roads
    for _ in range(80):
        poly = polylines[np.random.randint(0, len(polylines))]
        t = np.random.uniform(0.1, 0.9)
        p1, p2 = poly[0], poly[-1]
        tx = p1[0] + t * (p2[0] - p1[0]) + np.random.uniform(-8, 8)
        ty = p1[1] + t * (p2[1] - p1[1]) + np.random.uniform(-8, 8)
        r_tree = np.random.randint(9, 22)
        leaf_color = (np.random.randint(25, 45), np.random.randint(80, 130), np.random.randint(20, 45))
        draw.ellipse([tx - r_tree, ty - r_tree, tx + r_tree, ty + r_tree], fill=leaf_color)

    # Building Shadows in CBD
    for _ in range(16):
        bx = np.random.uniform(70, 480)
        by = np.random.uniform(70, 480)
        bw = np.random.uniform(30, 70)
        bh = np.random.uniform(30, 70)
        shadow_poly = [(bx, by), (bx + bw, by), (bx + bw + 35, by + bh + 45), (bx + 35, by + bh + 45)]
        s_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        s_draw = ImageDraw.Draw(s_img)
        s_draw.polygon(shadow_poly, fill=(12, 16, 20, np.random.randint(120, 180)))
        img.paste(Image.alpha_composite(img.convert('RGBA'), s_img).convert('RGB'), (0, 0))

    # 6. Render Keypoint Heatmaps
    kp_heatmaps = np.zeros((3, height, width), dtype=np.float32)
    y_grid, x_grid = np.ogrid[:height, :width]
    for vx, vy, k_type in vertices:
        dist_sq = (x_grid - vx)**2 + (y_grid - vy)**2
        blob = np.exp(-dist_sq / (2.0 * 3.5**2))
        kp_heatmaps[k_type] = np.maximum(kp_heatmaps[k_type], blob)

    norm = np.sqrt(cos_map**2 + sin_map**2 + 1e-6)
    orient_tensor = np.stack([cos_map / norm, sin_map / norm], axis=0).astype(np.float32)

    return ProceduralRoadScene(
        image_rgb=np.array(img, dtype=np.uint8),
        vertices=vertices,
        edges=edges,
        road_mask=np.array(mask_img, dtype=np.uint8),
        orientation_field=orient_tensor,
        keypoint_heatmaps=kp_heatmaps
    )
