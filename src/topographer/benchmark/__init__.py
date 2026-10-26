"""
Benchmark module for TopoGrapher.
"""

from topographer.benchmark.world import run_world_benchmark
from topographer.benchmark.spacenet import run_spacenet_benchmark

__all__ = ["run_world_benchmark", "run_spacenet_benchmark"]
