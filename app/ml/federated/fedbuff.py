"""Pure FedBuff aggregation math."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class FedBuffUpload(Protocol):
    backbone_version: int
    interaction_count: int
    weights: dict[str, np.ndarray]


def aggregate_fedbuff(
    current_weights: dict[str, np.ndarray],
    uploads: list[FedBuffUpload],
    base_weights_by_version: dict[int, dict[str, np.ndarray]],
    current_version: int,
    server_lr: float,
    staleness_alpha: float,
) -> dict[str, np.ndarray]:
    """
    Apply FedBuff with interaction-count and staleness weighting.

    For each upload:
        delta = base_weights - local_weights
        scale = interaction_count / (1 + staleness) ** staleness_alpha
    The server applies: current_weights - server_lr * weighted_average_delta.
    """
    param_keys = list(current_weights)
    weighted_deltas = {
        key: np.zeros_like(current_weights[key], dtype=np.float32) for key in param_keys
    }
    total_weight = 0.0

    for upload in uploads:
        base_weights = base_weights_by_version[upload.backbone_version]
        staleness = max(0, current_version - upload.backbone_version)
        scale = float(upload.interaction_count) / (
            (1.0 + staleness) ** staleness_alpha
        )

        total_weight += scale
        for key in param_keys:
            weighted_deltas[key] += scale * (base_weights[key] - upload.weights[key])

    if total_weight <= 0:
        raise ValueError("FedBuff weight normalizer is non-positive.")

    return {
        key: current_weights[key] - server_lr * (weighted_deltas[key] / total_weight)
        for key in param_keys
    }
