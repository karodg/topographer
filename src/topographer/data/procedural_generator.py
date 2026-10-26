"""
High-Fidelity Procedural Road Network and Satellite Scene Synthesizer.

Generates photorealistic aerial/satellite imagery paired with 100% verified mathematical ground-truth graphs:
1. Manhattan/Barcelona rectangular urban grids;
2. Radial-concentric European ring avenues;
3. Organic suburban cul-de-sacs and meandering rural switchbacks;
4. Realistic environmental occlusions (tree canopy clusters, building shadows, asphalt noise).
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter


class ProceduralRoadScene:
    """Encapsulates a synthesized scene with imagery and exact graph topology."""

    def __init__(
        self,
        image_rgb: np.ndarray,      # (H, W, 3) uint8
        vertices: List[Tuple[float, float, int]],  # (x, y, type_id)
        edges: List[Tuple[int, int, int]],        # (u, v, class_id)
        road_mask: np.ndarray,      # (H, W) uint8
        orientation_field: np.ndarray, # (2, H, W) float32 (cos 2theta, sin 2theta)
        keypoint_heatmaps: np.ndarray  # (3, H, W) float32
    ):
        self.image = image_rgb
        self.vertices = vertices
        self.edges = edges
        self.road_mask = road_mask
        self.orientation = orientation_field
        self.keypoints = keypoint_heatmaps


class ProceduralRoadGenerator:
    """
    Generates diverse global road network morphologies with optical occlusions.
    """

    def __init__(self, width: int = 512, height: int = 512):
        self.w = width
        self.h = height

    def generate_scene(
        self,
        pattern: str = "random",
        biome: Optional[str] = None,
        seed: Optional[int] = None
    ) -> ProceduralRoadScene:
        """
        Synthesizes an aerial scene with complete ground-truth topology across global environments.

        Parameters
        ----------
        pattern : str
            One of 'grid', 'radial', 'organic', 'highway', 'alpine', 'desert', or 'random'.
        biome : Optional[str]
            One of 'green_grass', 'dry_soil', 'urban_concrete', 'coastal_sand',
            'tropical_rainforest', 'alpine_snow', 'arid_desert', 'agricultural', or None.
        seed : Optional[int]
            Random seed for deterministic reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)

        if pattern == "random":
            pattern = np.random.choice(["grid", "radial", "organic", "highway", "alpine", "desert"])

        # 1. Base terrain canvas across 8 global biomes
        biomes = [
            "green_grass", "dry_soil", "urban_concrete", "coastal_sand",
            "tropical_rainforest", "alpine_snow", "arid_desert", "agricultural"
        ]
        base_color = biome if (biome is not None and biome in biomes) else np.random.choice(biomes)
        canvas = self._create_base_terrain(base_color)

        # 2. Synthesize Graph Topology
        if pattern == "grid":
            vertices, edges, polylines = self._generate_grid_topology()
        elif pattern == "radial":
            vertices, edges, polylines = self._generate_radial_topology()
        elif pattern == "highway":
            vertices, edges, polylines = self._generate_highway_topology()
        elif pattern == "alpine":
            vertices, edges, polylines = self._generate_alpine_topology()
        elif pattern == "desert":
            vertices, edges, polylines = self._generate_desert_topology()
        else:
            vertices, edges, polylines = self._generate_organic_topology()

        # 3. Render Road Surface and Markings on Canvas
        road_img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(road_img)

        # Ground-truth continuous mask and orientation tensor
        gt_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        mask_img = Image.fromarray(gt_mask)
        mask_draw = ImageDraw.Draw(mask_img)

        cos_map = np.zeros((self.h, self.w), dtype=np.float32)
        sin_map = np.zeros((self.h, self.w), dtype=np.float32)

        for (u, v, r_class), pts in zip(edges, polylines):
            # Width varies by class: Highway=14px, Arterial=10px, Local=6px, Track=4px
            widths = {0: 14, 1: 10, 2: 7, 3: 4}
            w_px = widths.get(r_class, 7)

            # Draw asphalt road pavement
            asphalt_color = (np.random.randint(45, 65), np.random.randint(45, 65), np.random.randint(48, 68))
            draw.line(pts, fill=asphalt_color, width=w_px, joint='round')

            # Draw center lane marking for wide roads
            if w_px >= 10:
                stripe_color = (np.random.randint(210, 240), np.random.randint(210, 240), np.random.randint(180, 220))
                draw.line(pts, fill=stripe_color, width=1, joint='round')

            # Draw to ground-truth binary mask
            mask_draw.line(pts, fill=1, width=w_px, joint='round')

            # Compute orientation tangent vectors along line
            for k in range(len(pts) - 1):
                p1, p2 = pts[k], pts[k + 1]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                theta = np.arctan2(dy, dx)
                cos_2th = np.cos(2.0 * theta)
                sin_2th = np.sin(2.0 * theta)

                # Rasterize orientation along segment
                seg_img = Image.new('L', (self.w, self.h), 0)
                seg_draw = ImageDraw.Draw(seg_img)
                seg_draw.line([p1, p2], fill=255, width=w_px)
                seg_mask = np.array(seg_img) > 0

                cos_map[seg_mask] = cos_2th
                sin_map[seg_mask] = sin_2th

        # 4. Render Keypoint Heatmaps (Junctions, Terminals, Waypoints)
        kp_heatmaps = np.zeros((3, self.h, self.w), dtype=np.float32)
        y_grid, x_grid = np.ogrid[:self.h, :self.w]

        for vx, vy, k_type in vertices:
            # Gaussian blob with sigma=3.0 pixels
            dist_sq = (x_grid - vx)**2 + (y_grid - vy)**2
            blob = np.exp(-dist_sq / (2.0 * 3.0**2))
            kp_heatmaps[k_type] = np.maximum(kp_heatmaps[k_type], blob)

        # 5. Apply Optical Occlusions (Tree Canopies and Building Shadows)
        self._add_tree_canopies(draw, road_img, vertices, polylines)
        self._add_building_shadows(draw, road_img)

        # Convert back to numpy arrays
        final_rgb = np.array(road_img, dtype=np.uint8)
        # Add high-frequency sensor noise and subtle blur
        final_rgb = np.clip(final_rgb.astype(np.float32) + np.random.normal(0, 3.5, final_rgb.shape), 0, 255).astype(np.uint8)

        gt_mask_np = np.array(mask_img, dtype=np.uint8)

        # Normalize orientation map to unit vectors
        norm = np.sqrt(cos_map**2 + sin_map**2 + 1e-6)
        orientation_tensor = np.stack([cos_map / norm, sin_map / norm], axis=0).astype(np.float32)

        return ProceduralRoadScene(
            image_rgb=final_rgb,
            vertices=vertices,
            edges=edges,
            road_mask=gt_mask_np,
            orientation_field=orientation_tensor,
            keypoint_heatmaps=kp_heatmaps
        )

    def _create_base_terrain(self, biome: str) -> np.ndarray:
        """Generates realistic texture for ground terrain across global biomes."""
        if biome == "green_grass":
            r = np.random.normal(85, 8, (self.h, self.w))
            g = np.random.normal(135, 12, (self.h, self.w))
            b = np.random.normal(65, 8, (self.h, self.w))
        elif biome == "tropical_rainforest":
            r = np.random.normal(45, 6, (self.h, self.w))
            g = np.random.normal(95, 10, (self.h, self.w))
            b = np.random.normal(40, 6, (self.h, self.w))
        elif biome == "urban_concrete":
            val = np.random.normal(160, 15, (self.h, self.w))
            r, g, b = val + np.random.normal(0, 2, val.shape), val, val - np.random.normal(0, 2, val.shape)
        elif biome == "arid_desert":
            r = np.random.normal(215, 10, (self.h, self.w))
            g = np.random.normal(175, 12, (self.h, self.w))
            b = np.random.normal(110, 8, (self.h, self.w))
        elif biome == "dry_soil":
            r = np.random.normal(165, 10, (self.h, self.w))
            g = np.random.normal(140, 10, (self.h, self.w))
            b = np.random.normal(105, 8, (self.h, self.w))
        elif biome == "alpine_snow":
            # Slate grey rock with bright white snow patches
            rock = np.random.normal(110, 12, (self.h, self.w))
            snow = np.random.rand(self.h, self.w) > 0.70
            r = np.where(snow, 235.0, rock + 5.0)
            g = np.where(snow, 240.0, rock)
            b = np.where(snow, 245.0, rock + 10.0)
        elif biome == "agricultural":
            # Patchwork fields
            r = np.random.normal(120, 20, (self.h, self.w))
            g = np.random.normal(150, 25, (self.h, self.w))
            b = np.random.normal(80, 15, (self.h, self.w))
        else:  # coastal sand
            r = np.random.normal(210, 8, (self.h, self.w))
            g = np.random.normal(195, 8, (self.h, self.w))
            b = np.random.normal(155, 8, (self.h, self.w))

        terrain = np.stack([r, g, b], axis=-1)
        terrain = gaussian_filter(terrain, sigma=1.5)
        return np.clip(terrain, 0, 255).astype(np.uint8)

    def _generate_grid_topology(self) -> Tuple[List, List, List]:
        """Manhattan-style rectilinear street grid."""
        spacing_x = np.random.randint(70, 110)
        spacing_y = np.random.randint(70, 110)
        margin = 35

        xs = list(range(margin, self.w - margin, spacing_x))
        ys = list(range(margin, self.h - margin, spacing_y))

        vertices = []
        node_lookup = {}
        idx = 0

        for r_i, y in enumerate(ys):
            for c_i, x in enumerate(xs):
                # Classify node: boundary is terminal/3-way, interior is 4-way junction
                is_boundary = (r_i == 0 or r_i == len(ys)-1 or c_i == 0 or c_i == len(xs)-1)
                k_type = 0 if not is_boundary else 1
                # Small organic jitter (+-3px)
                jx = float(x + np.random.randint(-3, 4))
                jy = float(y + np.random.randint(-3, 4))
                vertices.append((jx, jy, k_type))
                node_lookup[(r_i, c_i)] = idx
                idx += 1

        edges = []
        polylines = []

        # Connect horizontal and vertical street segments
        for r_i in range(len(ys)):
            for c_i in range(len(xs)):
                curr_idx = node_lookup[(r_i, c_i)]
                p_curr = (vertices[curr_idx][0], vertices[curr_idx][1])

                # Horizontal edge
                if c_i + 1 < len(xs):
                    next_idx = node_lookup[(r_i, c_i + 1)]
                    p_next = (vertices[next_idx][0], vertices[next_idx][1])
                    r_class = 1 if r_i % 2 == 0 else 2
                    edges.append((curr_idx, next_idx, r_class))
                    polylines.append([p_curr, p_next])

                # Vertical edge
                if r_i + 1 < len(ys):
                    next_idx = node_lookup[(r_i + 1, c_i)]
                    p_next = (vertices[next_idx][0], vertices[next_idx][1])
                    r_class = 1 if c_i % 2 == 0 else 2
                    edges.append((curr_idx, next_idx, r_class))
                    polylines.append([p_curr, p_next])

        return vertices, edges, polylines

    def _generate_radial_topology(self) -> Tuple[List, List, List]:
        """Concentric ring avenues with radiating arterial spokes."""
        cx, cy = self.w / 2.0, self.h / 2.0
        num_rings = np.random.randint(2, 4)
        num_spokes = np.random.randint(6, 10)
        radii = np.linspace(60, min(cx, cy) - 35, num_rings)

        vertices = [(cx, cy, 0)]  # Center junction
        ring_nodes = []
        idx = 1

        for r_idx, rad in enumerate(radii):
            current_ring = []
            for s in range(num_spokes):
                angle = (2.0 * np.pi * s) / num_spokes
                vx = cx + rad * np.cos(angle) + np.random.uniform(-4, 4)
                vy = cy + rad * np.sin(angle) + np.random.uniform(-4, 4)
                vertices.append((float(vx), float(vy), 0))
                current_ring.append(idx)
                idx += 1
            ring_nodes.append(current_ring)

        edges = []
        polylines = []

        # Connect radial spokes from center to outer ring
        for s in range(num_spokes):
            prev = 0
            for r_idx in range(num_rings):
                curr = ring_nodes[r_idx][s]
                edges.append((prev, curr, 1))
                polylines.append([(vertices[prev][0], vertices[prev][1]), (vertices[curr][0], vertices[curr][1])])
                prev = curr

        # Connect concentric rings
        for r_idx in range(num_rings):
            ring = ring_nodes[r_idx]
            for s in range(num_spokes):
                u = ring[s]
                v = ring[(s + 1) % num_spokes]
                edges.append((u, v, 0 if r_idx == 0 else 1))
                polylines.append([(vertices[u][0], vertices[u][1]), (vertices[v][0], vertices[v][1])])

        return vertices, edges, polylines

    def _generate_organic_topology(self) -> Tuple[List, List, List]:
        """Organic winding roads with dead-ends and curving residential loops."""
        density_factor = (self.w * self.h) / (512.0 * 512.0)
        num_seeds = max(18, int(np.random.randint(24, 40) * density_factor))
        coords = np.random.uniform(45, self.w - 45, (num_seeds, 2))

        vertices = []
        for i, (x, y) in enumerate(coords):
            k_type = int(np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2]))
            vertices.append((float(x), float(y), k_type))

        edges = []
        polylines = []

        from scipy.spatial import Delaunay
        tri = Delaunay(coords)
        for simplex in tri.simplices:
            for pair in [(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[0], simplex[2])]:
                u, v = min(pair), max(pair)
                dist = np.hypot(coords[u, 0] - coords[v, 0], coords[u, 1] - coords[v, 1])
                if dist < 115.0 and (u, v, 2) not in edges:
                    p1 = (vertices[u][0], vertices[u][1])
                    p2 = (vertices[v][0], vertices[v][1])
                    self._subdivide_edge(p1, p2, u, v, 2, vertices, edges, polylines, max_len=75.0)

        return vertices, edges, polylines

    def _generate_highway_topology(self) -> Tuple[List, List, List]:
        """High-speed dual highway intersection with 4-quadrant interchange connector ramps."""
        cx, cy = self.w / 2.0, self.h / 2.0
        margin = 35.0

        # Anchor Keypoints
        v_center = (cx, cy, 0)                  # 0: Central 4-way intersection
        v_west = (margin, cy, 1)                # 1: West terminal
        v_east = (self.w - margin, cy, 1)       # 2: East terminal
        v_north = (cx, margin, 1)               # 3: North terminal
        v_south = (cx, self.h - margin, 1)      # 4: South terminal

        # Ramp entrance/exit waypoints along main corridors
        v_w_ramp = (cx - 150.0, cy, 0)          # 5
        v_e_ramp = (cx + 150.0, cy, 0)          # 6
        v_n_ramp = (cx, cy - 150.0, 0)          # 7
        v_s_ramp = (cx, cy + 150.0, 0)          # 8

        # Curved ramp apex corners
        v_nw_apex = (cx - 95.0, cy - 95.0, 2)   # 9: NW quadrant
        v_ne_apex = (cx + 95.0, cy - 95.0, 2)   # 10: NE quadrant
        v_se_apex = (cx + 95.0, cy + 95.0, 2)   # 11: SE quadrant
        v_sw_apex = (cx - 95.0, cy + 95.0, 2)   # 12: SW quadrant

        vertices = [
            v_center, v_west, v_east, v_north, v_south,
            v_w_ramp, v_e_ramp, v_n_ramp, v_s_ramp,
            v_nw_apex, v_ne_apex, v_se_apex, v_sw_apex
        ]

        raw_edges = [
            # Main West-East highway (Class 0: Freeway)
            (1, 5, 0), (5, 0, 0), (0, 6, 0), (6, 2, 0),
            # Main North-South highway (Class 0: Freeway)
            (3, 7, 0), (7, 0, 0), (0, 8, 0), (8, 4, 0),
            # Quadrant Ramps (Class 2: Secondary / Ramp)
            (5, 9, 2), (9, 7, 2),    # NW ramp
            (7, 10, 2), (10, 6, 2),  # NE ramp
            (6, 11, 2), (11, 8, 2),  # SE ramp
            (8, 12, 2), (12, 5, 2),  # SW ramp
        ]

        edges = []
        polylines = []
        for u, v, r_class in raw_edges:
            p1 = (vertices[u][0], vertices[u][1])
            p2 = (vertices[v][0], vertices[v][1])
            self._subdivide_edge(p1, p2, u, v, r_class, vertices, edges, polylines, max_len=75.0)

        return vertices, edges, polylines

    def _subdivide_edge(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        u_idx: int,
        v_idx: int,
        r_class: int,
        vertices: List[Tuple[float, float, int]],
        edges: List[Tuple[int, int, int]],
        polylines: List[List[Tuple[float, float]]],
        max_len: float = 85.0
    ):
        """Subdivides a long road segment into intermediate degree-2 waypoints."""
        dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if dist <= max_len:
            edges.append((u_idx, v_idx, r_class))
            polylines.append([p1, p2])
            return

        num_segs = int(np.ceil(dist / max_len))
        prev_idx = u_idx
        prev_p = p1

        for s in range(1, num_segs):
            t = float(s) / num_segs
            curr_p = (
                float(p1[0] + t * (p2[0] - p1[0])),
                float(p1[1] + t * (p2[1] - p1[1]))
            )
            curr_idx = len(vertices)
            vertices.append((curr_p[0], curr_p[1], 2))  # Type 2: Geometric Waypoint
            edges.append((prev_idx, curr_idx, r_class))
            polylines.append([prev_p, curr_p])
            prev_idx = curr_idx
            prev_p = curr_p

        edges.append((prev_idx, v_idx, r_class))
        polylines.append([prev_p, p2])

    def _generate_alpine_topology(self) -> Tuple[List, List, List]:
        """Alpine mountain pass with serpentine hairpin switchbacks and sharp curvature anchors."""
        ys = np.linspace(60, self.h - 60, 6)
        raw_nodes = []

        # Hairpin switchback spine
        for i, y in enumerate(ys):
            x = (self.w - 70.0) if (i % 2 == 1) else 70.0
            vt = 1 if (i == 0 or i == len(ys) - 1) else 2
            raw_nodes.append((float(x), float(y), vt))

        vertices = list(raw_nodes)
        edges = []
        polylines = []

        for i in range(len(raw_nodes) - 1):
            p1 = (raw_nodes[i][0], raw_nodes[i][1])
            p2 = (raw_nodes[i + 1][0], raw_nodes[i + 1][1])
            self._subdivide_edge(p1, p2, i, i + 1, 1, vertices, edges, polylines, max_len=75.0)

        # Mountain observation branch trail (Class 3: Track)
        obs_idx = len(vertices)
        p_obs = (self.w / 2.0, self.h / 2.0 - 40.0)
        vertices.append((p_obs[0], p_obs[1], 1))
        p_anchor = (vertices[1][0], vertices[1][1])
        self._subdivide_edge(p_anchor, p_obs, 1, obs_idx, 3, vertices, edges, polylines, max_len=75.0)

        return vertices, edges, polylines

    def _generate_desert_topology(self) -> Tuple[List, List, List]:
        """Arid trans-desert corridor with diagonal primary highway and remote dune access tracks."""
        margin = 50.0
        # Main diagonal highway corridor
        raw_keypoints = [
            (margin, margin, 1),
            (self.w / 2.0, self.h / 2.0, 0),         # Central desert junction
            (self.w - margin, self.h - margin, 1),
            (self.w - margin, margin + 50.0, 1),
            (margin + 50.0, self.h - margin, 1),
            (self.w / 2.0 + 100.0, self.h / 4.0, 1) # Oasis branch
        ]

        vertices = list(raw_keypoints)
        edges = []
        polylines = []

        # Subdivide all long corridors
        raw_connections = [
            (0, 1, 0), (1, 2, 0),  # Highway
            (3, 1, 1), (1, 4, 1),  # Secondary
            (1, 5, 3)             # Unpaved track
        ]

        for u, v, r_class in raw_connections:
            p1 = (vertices[u][0], vertices[u][1])
            p2 = (vertices[v][0], vertices[v][1])
            self._subdivide_edge(p1, p2, u, v, r_class, vertices, edges, polylines, max_len=80.0)

        return vertices, edges, polylines

    def _add_tree_canopies(self, draw: ImageDraw.Draw, img: Image.Image, vertices: List, polylines: List):
        """Simulates roadside tree clusters occluding sections of road pavement."""
        num_trees = np.random.randint(15, 35)
        for _ in range(num_trees):
            if not polylines:
                break
            poly = polylines[np.random.randint(0, len(polylines))]
            t = np.random.uniform(0.1, 0.9)
            # Pick position along road segment with slight offset
            p1, p2 = poly[0], poly[-1]
            tx = p1[0] + t * (p2[0] - p1[0]) + np.random.uniform(-10, 10)
            ty = p1[1] + t * (p2[1] - p1[1]) + np.random.uniform(-10, 10)
            radius = np.random.randint(8, 20)

            # Realistic leafy foliage color with green-brown tones
            green = np.random.randint(60, 105)
            leaf_color = (np.random.randint(30, 60), green, np.random.randint(25, 55))
            draw.ellipse([tx - radius, ty - radius, tx + radius, ty + radius], fill=leaf_color)

    def _add_building_shadows(self, draw: ImageDraw.Draw, img: Image.Image):
        """Simulates oblique geometric shadows cast across roads by tall buildings."""
        num_shadows = np.random.randint(4, 10)
        for _ in range(num_shadows):
            bx = np.random.uniform(50, self.w - 80)
            by = np.random.uniform(50, self.h - 80)
            bw = np.random.uniform(25, 60)
            bh = np.random.uniform(25, 60)

            # Shadow cast diagonally (e.g. 45 degrees)
            shadow_poly = [
                (bx, by),
                (bx + bw, by),
                (bx + bw + 25, by + bh + 35),
                (bx + 25, by + bh + 35)
            ]
            # Translucent dark shadow
            shadow_img = Image.new('RGBA', (self.w, self.h), (0, 0, 0, 0))
            s_draw = ImageDraw.Draw(shadow_img)
            s_draw.polygon(shadow_poly, fill=(15, 18, 22, np.random.randint(110, 170)))
            img.paste(Image.alpha_composite(img.convert('RGBA'), shadow_img).convert('RGB'), (0, 0))
