"""
seed_backbone.py
================
Writes a version 0 global backbone to the database.

Version 0 represents a randomly initialised backbone — it is the starting
point that all Raspberry Pi clients download before they have accumulated
enough interactions to trigger a FedAvg round.

Usage (from the repo root, with PostgreSQL running):

    python -m app.db.seed_backbone                    # default: ts algorithm
    python -m app.db.seed_backbone --algorithm dqn    # seed a DQN backbone
    python -m app.db.seed_backbone --algorithm ts --algorithm dqn  # seed both

The script is idempotent for version 0: if a version 0 backbone already
exists for the given algorithm it will not create a duplicate.
"""

import argparse
import asyncio
import base64
import gzip
import json
import logging

import numpy as np
from sqlalchemy import select

from app.db import AsyncSessionLocal
from app.api.schemas.backbone import GlobalBackboneVersion

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backbone architecture constants — must match the client implementation
# MLP: input(28) -> Linear -> ReLU -> Linear -> output(32)
# ---------------------------------------------------------------------------
INPUT_DIM = 28
HIDDEN_DIM = 64
OUTPUT_DIM = 32


def _init_backbone_weights(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """
    Kaiming uniform initialisation for both Linear layers.
    This matches PyTorch's default weight initialisation so the seed backbone
    is compatible with a freshly constructed nn.Sequential on the client.
    """
    def kaiming_uniform(fan_in: int, fan_out: int) -> np.ndarray:
        bound = np.sqrt(1.0 / fan_in)
        return rng.uniform(-bound, bound, (fan_out, fan_in)).astype(np.float32)

    def bias_uniform(fan_in: int, size: int) -> np.ndarray:
        bound = np.sqrt(1.0 / fan_in)
        return rng.uniform(-bound, bound, (size,)).astype(np.float32)

    return {
        "backbone.0.weight": kaiming_uniform(INPUT_DIM, HIDDEN_DIM),   # (64, 28)
        "backbone.0.bias":   bias_uniform(INPUT_DIM, HIDDEN_DIM),      # (64,)
        "backbone.2.weight": kaiming_uniform(HIDDEN_DIM, OUTPUT_DIM),  # (32, 64)
        "backbone.2.bias":   bias_uniform(HIDDEN_DIM, OUTPUT_DIM),     # (32,)
    }


def _serialise(weights: dict[str, np.ndarray]) -> str:
    """Convert weight arrays to gzip-compressed, base64-encoded JSON blob."""
    weights_json = {k: v.tolist() for k, v in weights.items()}
    compressed = gzip.compress(json.dumps(weights_json).encode())
    return base64.b64encode(compressed).decode()


async def seed_algorithm(algorithm: str, seed: int = 42) -> None:
    async with AsyncSessionLocal() as db:
        # Check if version 0 already exists for this algorithm
        result = await db.execute(
            select(GlobalBackboneVersion)
            .where(GlobalBackboneVersion.algorithm == algorithm)
            .order_by(GlobalBackboneVersion.version.asc())
            .limit(1)
        )
        existing = result.scalar_one_or_none()

        if existing is not None:
            logger.info(
                "Version 0 backbone for algorithm '%s' already exists "
                "(earliest version in db: %d) — skipping.",
                algorithm, existing.version,
            )
            return

        rng = np.random.default_rng(seed)
        weights = _init_backbone_weights(rng)
        blob = _serialise(weights)

        backbone = GlobalBackboneVersion(
            weights_blob=blob,
            algorithm=algorithm,
            client_count=0,
            total_interactions=0,
        )
        db.add(backbone)
        await db.commit()
        await db.refresh(backbone)

        # Log weight shapes for verification
        for key, arr in weights.items():
            logger.info("  %s  shape=%s  dtype=%s", key, arr.shape, arr.dtype)

        logger.info(
            "Seeded version %d backbone for algorithm='%s' "
            "(random seed=%d, blob_size=%d bytes).",
            backbone.version, algorithm, seed, len(blob),
        )


async def main(algorithms: list[str]) -> None:
    for algo in algorithms:
        logger.info("Seeding backbone for algorithm='%s' ...", algo)
        await seed_algorithm(algo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed version 0 backbone weights.")
    parser.add_argument(
        "--algorithm",
        choices=["ts", "dqn"],
        action="append",
        dest="algorithms",
        default=None,
        help="Algorithm to seed. Pass multiple times for multiple algorithms. Default: ts",
    )
    args = parser.parse_args()
    algorithms = args.algorithms or ["ts"]
    asyncio.run(main(algorithms))