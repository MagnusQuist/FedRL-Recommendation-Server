"""Database bootstrap orchestration.

High-level helpers shared by the server's startup lifecycle and the
``python -m app.db.seeding`` CLI.
"""

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
    """Create any missing tables from the SQLAlchemy models.

    Safe to call concurrently from multiple Uvicorn workers.
    """
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
    """Seed backbones + catalogue. Individual seeders are idempotent."""
    logger.info("Seeding backbone weights...")
    await seed_federated_backbone()
    await seed_centralized_backbone()

    logger.info("Seeding food catalogue...")
    await seed_catalogue()


async def bootstrap_if_empty() -> bool:
    """Create tables and seed data when the database is fresh.

    Detects a fresh database via ``has_tables()`` (see ``seed_status``).
    No-op when the schema already exists, making it safe to call on every
    server startup.

    Returns True when bootstrap ran, False when the database was already
    initialised.
    """
    if await has_tables():
        logger.info("Database already initialised — skipping bootstrap.")
        return False

    logger.info("Fresh database detected — creating tables and seeding.")
    await ensure_models()
    await seed_all()
    logger.info("Bootstrap complete.")
    return True
