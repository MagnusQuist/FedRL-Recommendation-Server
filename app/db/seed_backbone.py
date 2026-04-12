"""
seed_backbone.py
================
Writes the initial federated backbone (version 1) to the database.

Version 1 represents a randomly initialised backbone — it is the starting
point that Raspberry Pi clients download before they have accumulated enough
interactions to trigger a FedAvg round.

The same pretrained weights are used to initialise both:
  - the federated backbone  (stored in GlobalBackboneVersion, algorithm="ts")
  - the centralized backbone (initialised in-process by CentralizedService)

Environment variables:
    BACKBONE_INIT_SEED=42
"""

from __future__ import annotations

import base64
import gzip
import json
from app.logger import logger
import os

import numpy as np
from sqlalchemy import select

from app.db import AsyncSessionLocal
from app.db.models.backbone import GlobalBackboneVersion

INPUT_DIM = 16
HIDDEN_DIM = 64
OUTPUT_DIM = 32
INITIAL_VERSION = 1

# The single algorithm used by the federated backbone.
FEDERATED_ALGORITHM = "ts"

BASE_SEED = int(os.getenv("BACKBONE_INIT_SEED", "42"))


def _kaiming_uniform(
    rng: np.random.Generator,
    fan_in: int,
    fan_out: int,
) -> np.ndarray:
    bound = np.sqrt(1.0 / fan_in)
    return rng.uniform(-bound, bound, (fan_out, fan_in)).astype(np.float32)


def _bias_uniform(
    rng: np.random.Generator,
    fan_in: int,
    size: int,
) -> np.ndarray:
    bound = np.sqrt(1.0 / fan_in)
    return rng.uniform(-bound, bound, (size,)).astype(np.float32)


def init_backbone_weights(seed: int) -> dict[str, np.ndarray]:
    """
    Create a freshly initialised backbone state dict.

    Keys follow the pretrain/model.py convention (backbone.N.weight/bias) so
    that both the federated and centralized arms can load them with the same
    mapping logic.
    """
    rng = np.random.default_rng(seed)

    return {
        "backbone.0.weight": _kaiming_uniform(rng, INPUT_DIM, HIDDEN_DIM),
        "backbone.0.bias": _bias_uniform(rng, INPUT_DIM, HIDDEN_DIM),
        "backbone.2.weight": _kaiming_uniform(rng, HIDDEN_DIM, OUTPUT_DIM),
        "backbone.2.bias": _bias_uniform(rng, HIDDEN_DIM, OUTPUT_DIM),
    }


def serialise_weights(weights: dict[str, np.ndarray]) -> str:
    payload = {name: value.tolist() for name, value in weights.items()}
    compressed = gzip.compress(json.dumps(payload).encode("utf-8"))
    return base64.b64encode(compressed).decode("utf-8")


async def _federated_backbone_exists() -> bool:
    """Return True if the initial seeded federated backbone already exists."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(GlobalBackboneVersion)
            .where(GlobalBackboneVersion.algorithm == FEDERATED_ALGORITHM)
            .where(GlobalBackboneVersion.version == INITIAL_VERSION)
            .limit(1)
        )
        return result.scalar_one_or_none() is not None


async def seed_federated_backbone(seed: int = BASE_SEED) -> None:
    """
    Seed the initial federated backbone (version 1) in the database if it does
    not already exist.  Uses the same pretrained weights as the centralized arm.
    """
    if await _federated_backbone_exists():
        logger.info(
            "Initial federated backbone (algorithm='%s', version=%d) already exists — skipping.",
            FEDERATED_ALGORITHM,
            INITIAL_VERSION,
        )
        return

    weights = init_backbone_weights(seed)
    blob = serialise_weights(weights)

    async with AsyncSessionLocal() as db:
        backbone = GlobalBackboneVersion(
            version=INITIAL_VERSION,
            weights_blob=blob,
            algorithm=FEDERATED_ALGORITHM,
            client_count=0,
            total_interactions=0,
        )
        db.add(backbone)
        await db.commit()
        await db.refresh(backbone)

    for key, arr in weights.items():
        logger.info("  %s shape=%s dtype=%s", key, arr.shape, arr.dtype)

    logger.info(
        "Seeded initial federated backbone id=%d version=%d (algorithm='%s', seed=%d, blob_size=%d bytes).",
        backbone.id,
        backbone.version,
        FEDERATED_ALGORITHM,
        seed,
        len(blob),
    )
