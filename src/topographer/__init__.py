"""
TopoGrapher: Autonomous Neural Road Network and Topology Graph Extraction Engine.

Converts aerial and satellite raster imagery directly into topologically closed,
routable vector graphs without broken links or spurious gaps.
"""

from topographer.pipeline import TopoGrapherEngine
from topographer.models.full_model import TopoGrapherNet
from topographer.graph.tracer import GraphTracer
from topographer.graph.regularizer import GraphRegularizer
from topographer.graph.simplifier import GraphTopologySimplifier
from topographer.graph.routing import RoadRoutingEngine
from topographer.metrics.apls import APLSMetric
from topographer.export.exporters import GISExporter

__version__ = "1.0.0"

__all__ = [
    "TopoGrapherEngine",
    "TopoGrapherNet",
    "GraphTracer",
    "GraphRegularizer",
    "GraphTopologySimplifier",
    "RoadRoutingEngine",
    "APLSMetric",
    "GISExporter"
]
