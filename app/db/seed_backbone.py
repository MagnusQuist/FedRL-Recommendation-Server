"""
seed_backbone.py
================
Seeds both backbone models from pretrained weights.

Pretrained weights are loaded from the .npz file at PRETRAINED_WEIGHTS_PATH.
If the file does not exist, random Kaiming initialisation is used as a fallback
with a clear warning.

Both models (federated and centralized) are seeded from the same weights to
ensure experimental fairness at time-zero.

Environment variables:
    PRETRAINED_WEIGHTS_PATH   Path to the backbone_weights.npz produced by the
                              pretrain script.
                              Default: data/pretrained/pretrained_backbone_weights.npz
"""

from __future__ import annotations

import base64
import gzip
import json
import os
from pathlib import Path

import numpy as np
from sqlalchemy import select

from app.db import AsyncSessionLocal
from app.db.models.backbone import GlobalBackboneVersion
from app.db.models.centralized import CentralizedModelVersion
from app.logger import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_DIM = 16
HIDDEN_DIM = 64
OUTPUT_DIM = 32
INITIAL_VERSION = 1

FEDERATED_ALGORITHM = "ts"

PRETRAINED_WEIGHTS_PATH = Path(
    os.getenv("PRETRAINED_WEIGHTS_PATH", "data/pretrained/pretrained_backbone_weights.npz")
)

_NUDGE_TYPES = ["N1", "N2", "N3", "N4"]


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------
def _random_weights() -> dict[str, np.ndarray]:
    """Kaiming uniform fallback when no pretrained .npz is available."""
    rng = np.random.default_rng(42)

    def ku(fan_in: int, fan_out: int) -> np.ndarray:
        b = np.sqrt(1.0 / fan_in)
        return rng.uniform(-b, b, (fan_out, fan_in)).astype(np.float32)

    def bu(fan_in: int, size: int) -> np.ndarray:
        b = np.sqrt(1.0 / fan_in)
        return rng.uniform(-b, b, (size,)).astype(np.float32)

    return {
        "backbone.0.weight": ku(INPUT_DIM, HIDDEN_DIM),
        "backbone.0.bias":   bu(INPUT_DIM, HIDDEN_DIM),
        "backbone.2.weight": ku(HIDDEN_DIM, OUTPUT_DIM),
        "backbone.2.bias":   bu(HIDDEN_DIM, OUTPUT_DIM),
    }


def load_pretrained_weights(
    path: Path = PRETRAINED_WEIGHTS_PATH,
) -> dict[str, np.ndarray]:
    """
    Load backbone weights from a .npz file saved by pretrain/run.py.
    Falls back to random Kaiming init if the file is not found.
    """
    if path.exists():
        data = np.load(path)
        weights = {k: data[k].astype(np.float32) for k in data.files}
        logger.info("Loaded pretrained backbone weights from '%s'.", path)
        for k, v in weights.items():
            logger.info("  %s  shape=%s  dtype=%s", k, v.shape, v.dtype)
        return weights

    logger.warning(
        "Pretrained weights not found at '%s' — falling back to random Kaiming init. "
        "Run the pretrain script and set PRETRAINED_WEIGHTS_PATH to use real weights.",
        path,
    )
    return _random_weights()


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------
def _encode(obj: object) -> str:
    """Serialize obj to a gzip-compressed, base64-encoded JSON string."""
    return base64.b64encode(gzip.compress(json.dumps(obj).encode())).decode()


def serialise_weights(weights: dict[str, np.ndarray]) -> str:
    return _encode({k: v.tolist() for k, v in weights.items()})


# ---------------------------------------------------------------------------
# Default head states for the centralized model
# ---------------------------------------------------------------------------
def _default_item_head() -> dict:
    return {"params": {}, "lam": 1.0, "v": 0.5, "max_items": 200}


def _default_price_head() -> dict:
    return {"A": 1.0, "b": 0.0, "lam": 1.0, "v": 0.5}


def _default_nudge_head() -> dict:
    return {
        "params": {
            n: {"mu": 0.5, "tau": 1.0, "sum": 0.0, "count": 0}
            for n in _NUDGE_TYPES
        },
        "interaction_count": 0,
        "rr_index": 0,
        "last_reward": 0.0,
    }


def _default_reward_predictor() -> dict:
    """Zero-initialised reward predictor weights (net.0.weight, net.0.bias)."""
    return {
        "net.0.weight": np.zeros((1, OUTPUT_DIM), dtype=np.float32).tolist(),
        "net.0.bias":   np.zeros(1, dtype=np.float32).tolist(),
    }


# ---------------------------------------------------------------------------
# Existence checks
# ---------------------------------------------------------------------------
async def _federated_backbone_exists() -> bool:
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(GlobalBackboneVersion)
            .where(GlobalBackboneVersion.algorithm == FEDERATED_ALGORITHM)
            .where(GlobalBackboneVersion.version == INITIAL_VERSION)
            .limit(1)
        )
        return result.scalar_one_or_none() is not None


async def _centralized_backbone_exists() -> bool:
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(CentralizedModelVersion)
            .where(CentralizedModelVersion.version == INITIAL_VERSION)
            .limit(1)
        )
        return result.scalar_one_or_none() is not None


# ---------------------------------------------------------------------------
# Public seeding functions
# ---------------------------------------------------------------------------
async def seed_federated_backbone() -> None:
    """
    Insert the initial federated backbone (version 1) into GlobalBackboneVersion
    if it does not already exist.  Weights are loaded from PRETRAINED_WEIGHTS_PATH.
    """
    if await _federated_backbone_exists():
        logger.info(
            "Federated backbone (algorithm='%s', version=%d) already exists — skipping.",
            FEDERATED_ALGORITHM, INITIAL_VERSION,
        )
        return

    weights = load_pretrained_weights()
    blob = serialise_weights(weights)

    async with AsyncSessionLocal() as db:
        row = GlobalBackboneVersion(
            version=INITIAL_VERSION,
            weights_blob=blob,
            algorithm=FEDERATED_ALGORITHM,
            client_count=0,
            total_interactions=0,
        )
        db.add(row)
        await db.commit()
        await db.refresh(row)

    logger.info(
        "Seeded federated backbone id=%d version=%d (blob=%d bytes).",
        row.id, row.version, len(blob),
    )


async def seed_centralized_backbone() -> None:
    """
    Insert the initial centralized backbone (version 1) into CentralizedModelVersion
    if it does not already exist.  Backbone weights are loaded from the same
    PRETRAINED_WEIGHTS_PATH used by the federated arm; all heads start at their
    default (uninformed) priors and the tuple pool starts empty.
    """
    if await _centralized_backbone_exists():
        logger.info(
            "Centralized backbone version=%d already exists — skipping.", INITIAL_VERSION,
        )
        return

    weights = load_pretrained_weights()

    async with AsyncSessionLocal() as db:
        row = CentralizedModelVersion(
            version=INITIAL_VERSION,
            backbone_blob=serialise_weights(weights),
            reward_predictor_blob=_encode(_default_reward_predictor()),
            item_head_blob=_encode(_default_item_head()),
            price_head_blob=_encode(_default_price_head()),
            nudge_head_blob=_encode(_default_nudge_head()),
            tuple_pool_blob=_encode([]),
        )
        db.add(row)
        await db.commit()
        await db.refresh(row)

    logger.info(
        "Seeded centralized backbone id=%d version=%d (blob=%d bytes).",
        row.id, row.version, len(row.backbone_blob),
    )
