"""Centralized model definitions and optimizer setup."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

RETRAIN_LR_BACKBONE = 1e-4
RETRAIN_LR_PREDICTOR = 1e-3
RETRAIN_WEIGHT_DECAY = 0.0


class BackboneEncoder(nn.Module):
    def __init__(self, input_dim: int = 21, latent_dim: int = 32):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.backbone(x)

    def embed(self, context: list[float]) -> torch.Tensor:
        x = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return self.backbone(x).squeeze(0)


class RewardPredictor(nn.Module):
    def __init__(self, input_dim: int = 32):
        super().__init__()
        self.net = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.net(x)


def _split_decay_params(module: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    decay, no_decay = [], []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if name.endswith("bias") else decay).append(parameter)
    return decay, no_decay


def build_optimizer(
    backbone: BackboneEncoder,
    reward_predictor: RewardPredictor,
) -> optim.Optimizer:
    bb_decay, bb_no_decay = _split_decay_params(backbone)
    pred_decay, pred_no_decay = _split_decay_params(reward_predictor)
    return optim.Adam([
        {
            "params": bb_decay,
            "lr": RETRAIN_LR_BACKBONE,
            "weight_decay": RETRAIN_WEIGHT_DECAY,
        },
        {"params": bb_no_decay, "lr": RETRAIN_LR_BACKBONE, "weight_decay": 0.0},
        {
            "params": pred_decay,
            "lr": RETRAIN_LR_PREDICTOR,
            "weight_decay": RETRAIN_WEIGHT_DECAY,
        },
        {"params": pred_no_decay, "lr": RETRAIN_LR_PREDICTOR, "weight_decay": 0.0},
    ])
