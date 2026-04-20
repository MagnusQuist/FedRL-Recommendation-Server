"""Seed federated and centralized backbone rows from one pretrained ``.npz``.

Weights: ``PRETRAINED_WEIGHTS_PATH`` (default: ``data/pretrained/pretrained_backbone_weights.npz``),
or Kaiming init if missing. Federated and centralized share the same backbone tensors at v1.
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
from app.db.models.centralized import CentralizedModelVersion
from app.db.models.federated import FederatedBackboneVersion
from app.logger import logger

INPUT_DIM = 18
HIDDEN_DIM = 64
OUTPUT_DIM = 32
INITIAL_VERSION = 1

_DEFAULT_PRETRAINED_WEIGHTS_PATH = (
    Path(__file__).resolve().parent / "data" / "pretrained" / "pretrained_backbone_weights.npz"
)

PRETRAINED_WEIGHTS_PATH = Path(
    os.getenv("PRETRAINED_WEIGHTS_PATH", str(_DEFAULT_PRETRAINED_WEIGHTS_PATH))
)

_NUDGE_TYPES = ["N1", "N2", "N3", "N4", "N5", "N6"]


def _random_weights() -> dict[str, np.ndarray]:
    """Kaiming-uniform backbone weights when no ``.npz`` is found."""
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
    """Load backbone arrays from ``pretrain`` output, else ``_random_weights()``."""
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


def _encode(obj: object) -> str:
    """gzip + base64 JSON for DB blobs."""
    return base64.b64encode(gzip.compress(json.dumps(obj).encode())).decode()


def serialise_weights(weights: dict[str, np.ndarray]) -> str:
    """Encode numpy backbone dict for ``weights_blob`` / ``backbone_blob``."""
    return _encode({k: v.tolist() for k, v in weights.items()})


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
    """Zero init for ``RewardPredictor`` linear layer."""
    return {
        "net.0.weight": np.zeros((1, OUTPUT_DIM), dtype=np.float32).tolist(),
        "net.0.bias":   np.zeros(1, dtype=np.float32).tolist(),
    }


async def _federated_backbone_exists() -> bool:
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(FederatedBackboneVersion)
            .where(FederatedBackboneVersion.version == INITIAL_VERSION)
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


async def seed_federated_backbone() -> None:
    """Insert v1 federated backbone row if missing (from ``load_pretrained_weights``)."""
    if await _federated_backbone_exists():
        logger.info(
            "Federated backbone version=%d already exists — skipping.",
            INITIAL_VERSION,
        )
        return

    weights = load_pretrained_weights()
    blob = serialise_weights(weights)

    async with AsyncSessionLocal() as db:
        row = FederatedBackboneVersion(
            version=INITIAL_VERSION,
            weights_blob=blob,
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
    """Insert v1 centralized row if missing: same backbone as federated; default heads; empty pool."""
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
            client_count=0,
        )
        db.add(row)
        await db.commit()
        await db.refresh(row)

    logger.info(
        "Seeded centralized backbone id=%d version=%d (blob=%d bytes).",
        row.id, row.version, len(row.backbone_blob),
    )

