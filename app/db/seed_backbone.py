"""
seed_backbone.py
================
Writes an initial global backbone (version 1) to the database.

Version 1 represents a randomly initialised backbone — it is the starting
point that Raspberry Pi clients download before they have accumulated enough
interactions to trigger a FedAvg round.

Environment variables (.env supported):
    SUPPORTED_BACKBONE_ALGORITHMS=ts,dqn
    DEFAULT_BACKBONE_ALGORITHMS=ts,dqn
    BACKBONE_INIT_SEED=42
"""

from __future__ import annotations

import base64
import gzip
import json
from app.logger import logger
import os
from typing import Iterable

import numpy as np
from dotenv import load_dotenv
from sqlalchemy import select

from app.db import AsyncSessionLocal
from app.api.models.backbone import GlobalBackboneVersion

load_dotenv()

INPUT_DIM = 28
HIDDEN_DIM = 64
OUTPUT_DIM = 32
INITIAL_VERSION = 1

FALLBACK_SUPPORTED_ALGORITHMS = ("ts", "dqn")
FALLBACK_DEFAULT_ALGORITHMS = ("ts", "dqn")
FALLBACK_BASE_SEED = 42


def _parse_csv_env(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _load_algorithm_config() -> tuple[tuple[str, ...], tuple[str, ...]]:
    supported = _parse_csv_env(os.getenv("SUPPORTED_BACKBONE_ALGORITHMS"))
    default = _parse_csv_env(os.getenv("DEFAULT_BACKBONE_ALGORITHMS"))

    if not supported:
        supported = list(FALLBACK_SUPPORTED_ALGORITHMS)

    if not default:
        default = list(FALLBACK_DEFAULT_ALGORITHMS)

    supported_set = set(supported)
    invalid_defaults = [algo for algo in default if algo not in supported_set]
    if invalid_defaults:
        raise ValueError(
            "DEFAULT_BACKBONE_ALGORITHMS contains values not present in "
            f"SUPPORTED_BACKBONE_ALGORITHMS: {invalid_defaults}"
        )

    return tuple(supported), tuple(default)


SUPPORTED_ALGORITHMS, DEFAULT_ALGORITHMS = _load_algorithm_config()
BASE_SEED = int(os.getenv("BACKBONE_INIT_SEED", str(FALLBACK_BASE_SEED)))


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

    Using the same seed for TS and DQN ensures both algorithm families begin
    from the same initial parameter state for fair comparison.
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


async def initial_version_exists(algorithm: str) -> bool:
    """Return True if the initial seeded version already exists for algorithm."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(GlobalBackboneVersion)
            .where(GlobalBackboneVersion.algorithm == algorithm)
            .where(GlobalBackboneVersion.version == INITIAL_VERSION)
            .limit(1)
        )
        return result.scalar_one_or_none() is not None


async def seed_algorithm(algorithm: str, seed: int = BASE_SEED) -> None:
    """
    Seed the initial backbone (version 1) for one algorithm if it does not
    already exist.
    """
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. "
            f"Supported algorithms: {SUPPORTED_ALGORITHMS}"
        )

    if await initial_version_exists(algorithm):
        logger.info(
            "Initial backbone for algorithm='%s' (version=%d) already exists — skipping.",
            algorithm,
            INITIAL_VERSION,
        )
        return

    weights = init_backbone_weights(seed)
    blob = serialise_weights(weights)

    async with AsyncSessionLocal() as db:
        backbone = GlobalBackboneVersion(
            version=INITIAL_VERSION,
            weights_blob=blob,
            algorithm=algorithm,
            client_count=0,
            total_interactions=0,
        )
        db.add(backbone)
        await db.commit()
        await db.refresh(backbone)

    for key, arr in weights.items():
        logger.info("  %s shape=%s dtype=%s", key, arr.shape, arr.dtype)

    logger.info(
        "Seeded initial backbone id=%d version=%d for algorithm='%s' "
        "(seed=%d, blob_size=%d bytes).",
        backbone.id,
        backbone.version,
        algorithm,
        seed,
        len(blob),
    )


async def seed_algorithms(
    algorithms: Iterable[str],
    base_seed: int = BASE_SEED,
) -> None:
    for algorithm in algorithms:
        logger.info("Seeding backbone for algorithm='%s' ...", algorithm)
        await seed_algorithm(algorithm, seed=base_seed)