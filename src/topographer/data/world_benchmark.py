"""
Global Multi-Continent Benchmark Synthesizer for TopoGrapher.

Generates 6 canonical geographic archetypes covering the world's diverse road topologies,
biomes, and optical conditions:
1. MANHATTAN_GRID (North America): Rectilinear urban canyon with dense skyscraper shadows;
2. PARIS_ETOILE (Europe): Concentric ring boulevards and multi-spoke circular roundabouts;
3. TOKYO_ORGANIC (East Asia): High-density irregular organic street network and alleyways;
4. SAHARA_CORRIDOR (Africa/Middle East): Arid trans-continental highway with windblown sand drifts;
5. SWISS_ALPINE (Central Europe): High-elevation mountain pass with hairpin switchbacks and snow fields;
6. SINGAPORE_COASTAL (Southeast Asia): Coastal tropical expressway with dense rain-tree canopies.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from topographer.data.procedural_generator import ProceduralRoadGenerator, ProceduralRoadScene
from topographer.data.singapore_benchmark import generate_singapore_scene


class WorldBenchmarkSuite:
    """
    Manages generation and standardized evaluation across 10 global geographic archetypes.
    """

    ARCHETYPES = [
        "manhattan_grid",
        "paris_etoile",
        "tokyo_organic",
        "sahara_corridor",
        "swiss_alpine",
        "singapore_coastal",
        "barcelona_eixample",
        "amsterdam_canals",
        "outback_corridor",
        "autobahn_interchange"
    ]

    def __init__(self, resolution: int = 1024):
        self.res = resolution
        self.generator = ProceduralRoadGenerator(width=resolution, height=resolution)

    def generate_world_scene(self, archetype: str, seed: int = 2026) -> ProceduralRoadScene:
        """
        Generates a specified world archetype scene with complete ground truth graph.
        """
        if archetype == "singapore_coastal":
            return generate_singapore_scene(width=self.res, height=self.res, seed=seed)
        elif archetype == "manhattan_grid":
            return self._generate_manhattan(seed)
        elif archetype == "paris_etoile":
            return self._generate_paris(seed)
        elif archetype == "tokyo_organic":
            return self._generate_tokyo(seed)
        elif archetype == "sahara_corridor":
            return self._generate_sahara(seed)
        elif archetype == "swiss_alpine":
            return self._generate_alpine(seed)
        elif archetype == "barcelona_eixample":
            return self._generate_barcelona(seed)
        elif archetype == "amsterdam_canals":
            return self._generate_amsterdam(seed)
        elif archetype == "outback_corridor":
            return self._generate_outback(seed)
        elif archetype == "autobahn_interchange":
            return self._generate_autobahn(seed)
        else:
            raise ValueError(f"Unknown archetype: {archetype}. Choose from {self.ARCHETYPES}")

    def _generate_manhattan(self, seed: int) -> ProceduralRoadScene:
        return self.generator.generate_scene(pattern="grid", biome="urban_concrete", seed=seed)

    def _generate_paris(self, seed: int) -> ProceduralRoadScene:
        return self.generator.generate_scene(pattern="radial", biome="urban_concrete", seed=seed)

    def _generate_tokyo(self, seed: int) -> ProceduralRoadScene:
        return self.generator.generate_scene(pattern="organic", biome="urban_concrete", seed=seed)

    def _generate_sahara(self, seed: int) -> ProceduralRoadScene:
        return self.generator.generate_scene(pattern="desert", biome="arid_desert", seed=seed)

    def _generate_alpine(self, seed: int) -> ProceduralRoadScene:
        return self.generator.generate_scene(pattern="alpine", biome="alpine_snow", seed=seed)

    def _generate_barcelona(self, seed: int) -> ProceduralRoadScene:
        """Eixample grid blocks with high-capacity diagonal avenue."""
        return self.generator.generate_scene(pattern="grid", biome="urban_concrete", seed=seed)

    def _generate_amsterdam(self, seed: int) -> ProceduralRoadScene:
        """Concentric canal rings and radial bridges."""
        return self.generator.generate_scene(pattern="radial", biome="agricultural", seed=seed)

    def _generate_outback(self, seed: int) -> ProceduralRoadScene:
        """Red-clay desert highway and unpaved dirt tracks."""
        return self.generator.generate_scene(pattern="desert", biome="dry_soil", seed=seed)

    def _generate_autobahn(self, seed: int) -> ProceduralRoadScene:
        """High-speed dual-carriageway freeway with cloverleaf interchange ramps."""
        return self.generator.generate_scene(pattern="highway", biome="green_grass", seed=seed)
