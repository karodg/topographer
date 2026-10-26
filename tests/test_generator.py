"""
Unit tests for Procedural Road Generator and Dataset.
"""

import pytest
import numpy as np
import torch
from topographer.data.procedural_generator import ProceduralRoadGenerator, ProceduralRoadScene
from topographer.data.dataset import ProceduralRoadDataset, collate_topographer_batch


def test_generator_patterns():
    gen = ProceduralRoadGenerator(width=256, height=256)

    for pattern in ["grid", "radial", "organic", "highway", "alpine", "desert"]:
        scene = gen.generate_scene(pattern=pattern, seed=42)
        assert isinstance(scene, ProceduralRoadScene)
        assert scene.image.shape == (256, 256, 3)
        assert scene.image.dtype == np.uint8
        assert scene.road_mask.shape == (256, 256)
        assert scene.orientation.shape == (2, 256, 256)
        assert scene.keypoints.shape == (3, 256, 256)
        assert len(scene.vertices) > 0
        assert len(scene.edges) > 0


def test_procedural_dataset_collation():
    dataset = ProceduralRoadDataset(num_samples=4, image_size=256, augment=True)
    assert len(dataset) == 4

    sample = dataset[0]
    assert "image" in sample
    assert "keypoints" in sample
    assert "road_mask" in sample
    assert "orientation" in sample
    assert "edge_pairs" in sample
    assert "edge_labels" in sample

    assert sample["image"].shape == (3, 256, 256)
    assert sample["keypoints"].shape == (3, 128, 128)
    assert sample["road_mask"].shape == (1, 128, 128)
    assert sample["orientation"].shape == (2, 128, 128)

    # Collation
    batch = collate_topographer_batch([dataset[0], dataset[1]])
    assert batch["images"].shape == (2, 3, 256, 256)
    assert batch["keypoints"].shape == (2, 3, 128, 128)
    assert len(batch["edge_pairs_list"]) == 2
