"""
model.py
========
PyTorch modules for pre-training the backbone.

The Backbone architecture matches the production weight key convention:
    backbone.0.weight  (HIDDEN_DIM, INPUT_DIM)   — Linear
    backbone.0.bias    (HIDDEN_DIM,)
    backbone.2.weight  (OUTPUT_DIM, HIDDEN_DIM)  — Linear
    backbone.2.bias    (OUTPUT_DIM,)
    (backbone.1 = ReLU, backbone.3 = Tanh — no parameters)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from app.db.seeding.seed_backbone import HIDDEN_DIM, INPUT_DIM, OUTPUT_DIM


class Backbone(nn.Module):
    """Two-layer feature extractor used by the TS algorithm."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class BackboneWithHead(nn.Module):
    """
    Backbone + temporary projection head for pre-training.

    The head maps the 32D embedding to a scalar in [0, 1] that predicts
    the synthetic recommendation quality score.  After training the head
    is discarded — only the backbone weights are saved.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = Backbone()
        self.head = nn.Sequential(
            nn.Linear(OUTPUT_DIM, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.backbone(x)
        return self.head(embedding).squeeze(-1)

    def get_backbone_weights_numpy(self) -> dict[str, np.ndarray]:
        """Extract backbone parameters as a numpy dict matching the DB format."""
        state = self.backbone.state_dict()
        return {key: value.cpu().detach().numpy() for key, value in state.items()}
