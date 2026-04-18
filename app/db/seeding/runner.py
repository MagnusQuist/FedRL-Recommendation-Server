"""Bootstrap helpers used by server startup and ``python -m app.db.seeding``."""

from __future__ import annotations

from sqlalchemy.exc import IntegrityError

from app.db import Base, engine
from app.db.seeding.seed_backbone import (
    seed_centralized_backbone,
    seed_federated_backbone,
)
from app.db.seeding.seed_catalogue import seed_catalogue
from app.db.seeding.seed_status import has_tables
from app.logger import logger


async def ensure_models() -> None:
    """Create missing tables from ``Base.metadata`` (safe across Uvicorn workers)."""
    import app.db.models  # noqa: F401 — register models on Base.metadata

    logger.info("Creating missing database tables...")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except IntegrityError:
        logger.info("Tables already created by another worker — skipping.")
        return
    logger.info("Table creation complete.")


async def seed_all() -> None:
    """Run backbone seeders then catalogue (each skips if already present where applicable)."""
    logger.info("Seeding backbone weights...")
    await seed_federated_backbone()
    await seed_centralized_backbone()

    logger.info("Seeding food catalogue...")
    await seed_catalogue()


async def bootstrap_if_empty() -> bool:
    """If no tables yet: create schema and seed. Otherwise no-op. Returns whether bootstrap ran."""
    if await has_tables():
        logger.info("Database already initialised — skipping bootstrap.")
        return False

    logger.info("Fresh database detected — creating tables and seeding.")
    await ensure_models()
    await seed_all()
    logger.info("Bootstrap complete.")
    return True
