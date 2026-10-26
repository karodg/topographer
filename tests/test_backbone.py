"""
Unit tests for TopoGrapher ConvNeXt-FPN Backbone and Sub-Pixel Heads.
"""

import pytest
import torch
from topographer.models.backbone import FeaturePyramidEncoder, ConvNeXtBlock
from topographer.models.keypoint_head import KeypointHead, extract_peaks_from_heatmap
from topographer.models.orientation_head import OrientationHead
from topographer.models.graph_transformer import EdgeTransformerClassifier
from topographer.models.full_model import TopoGrapherNet


def test_convnext_block():
    block = ConvNeXtBlock(dim=32)
    x = torch.randn(2, 32, 64, 64)
    out = block(x)
    assert out.shape == (2, 32, 64, 64)
    assert not torch.isnan(out).any()


def test_fpn_encoder_multiscale():
    encoder = FeaturePyramidEncoder(in_channels=3, base_channels=16, out_channels=32)
    x = torch.randn(2, 3, 128, 128)
    fused, stages = encoder(x)

    # Fused feature map should be stride 2 (H/2, W/2)
    assert fused.shape == (2, 32, 64, 64)
    assert len(stages) == 4
    assert stages[0].shape == (2, 16, 64, 64)   # H/2
    assert stages[1].shape == (2, 32, 32, 32)   # H/4
    assert stages[2].shape == (2, 64, 16, 16)   # H/8
    assert stages[3].shape == (2, 128, 8, 8)    # H/16


def test_keypoint_head_and_subpixel_peaks():
    head = KeypointHead(in_channels=32, num_keypoint_types=3)
    feat = torch.randn(2, 32, 64, 64)
    kp_map = head(feat)
    assert kp_map.shape == (2, 3, 64, 64)
    assert (kp_map >= 0.0).all() and (kp_map <= 1.0).all()

    # Peak extraction
    np_map = kp_map[0].detach().cpu().numpy()
    peaks = extract_peaks_from_heatmap(np_map, threshold=0.1, min_distance=2)
    assert isinstance(peaks, list)
    for p in peaks:
        x, y, score, ktype = p
        assert 0 <= x < 64
        assert 0 <= y < 64
        assert 0.0 <= score <= 1.0
        assert ktype in [0, 1, 2]


def test_orientation_head_unit_vectors():
    head = OrientationHead(in_channels=32)
    feat = torch.randn(2, 32, 64, 64)
    road_prob, tangent, width = head(feat)

    assert road_prob.shape == (2, 1, 64, 64)
    assert tangent.shape == (2, 2, 64, 64)
    assert width.shape == (2, 1, 64, 64)

    # Verify unit-length normalization of tangent field
    norm = torch.sqrt(torch.sum(tangent**2, dim=1))
    assert torch.allclose(norm, torch.ones_like(norm), atol=1.5e-2)


def test_edge_transformer_attention():
    trans = EdgeTransformerClassifier(feature_dim=32, num_samples=16)
    feat = torch.randn(1, 32, 64, 64)
    pairs = torch.tensor([
        [-0.5, -0.5, 0.5, 0.5],
        [0.0, -0.8, 0.0, 0.8]
    ], dtype=torch.float32)

    probs, logits, embed = trans(feat, pairs)
    assert probs.shape == (2, 1)
    assert logits.shape == (2, 4)
    assert embed.shape == (2, 32)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_graph_refiner_gnn():
    from topographer.models.graph_refiner import GraphTopologyRefiner
    refiner = GraphTopologyRefiner(embed_dim=32, num_heads=4)
    embeds = torch.randn(4, 32)
    pairs = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=torch.long)
    refined = refiner(embeds, pairs)
    assert refined.shape == (4, 1)
    assert (refined >= 0.0).all() and (refined <= 1.0).all()
