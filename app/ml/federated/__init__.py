"""Federated learning modules."""

from app.ml.federated.aggregation import FLAggregator, decode_backbone_blob

__all__ = ["FLAggregator", "decode_backbone_blob"]
