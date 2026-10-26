"""
Example 04: Evaluating on Official SpaceNet 3 Satellite Imagery.

Demonstrates downloading a sample WorldView-3 tile from the AWS SpaceNet S3 bucket,
parsing the Ground Truth GeoJSON via GeoTIFF affine transform, and calculating
the official SpaceNet Average Path Length Similarity (APLS) metric.
"""

import os
from topographer.benchmark import run_spacenet_benchmark

print("Launching SpaceNet 3 Benchmark Evaluation across real satellite imagery...")
results = run_spacenet_benchmark(
    weights_path="d:/Workspace/checkpoints/topographer_weights.pt",
    output_dir="spacenet_results"
)

print("\nSpaceNet 3 Benchmark Complete! Inspect 'spacenet_results' for GeoJSONs and overlays.")
