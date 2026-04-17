import os

from sqlalchemy.exc import IntegrityError

from app.logger import logger

from app.db import Base, engine
from app.db.seed_backbone import seed_federated_backbone, seed_centralized_backbone
from app.db.seed_catalogue import seed_catalogue


async def ensure_models(*, check_env: bool = False) -> None:
    """Create any missing tables from database models.

    When *check_env* is True the function is a no-op unless AUTO_CREATE_MODELS=true.
    Safe to call from multiple Uvicorn workers concurrently.
    """
    if check_env and os.getenv("AUTO_CREATE_MODELS", "false").strip().lower() != "true":
        logger.info("Skipping create_all (AUTO_CREATE_MODELS=false).")
        return

    import app.db.models  # noqa: F401 — register models on Base.metadata

    logger.info("Ensuring database models (create_all for missing tables)...")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except IntegrityError:
        logger.info("Tables already created by another worker — skipping.")
        return
    logger.info("Model creation complete.")


async def ensure_seed_backbones() -> None:
    """Seed the initial backbone (version 1) for both federated and centralized arms."""
    await seed_federated_backbone()
    await seed_centralized_backbone()


async def ensure_seed_catalogue() -> None:
    """Seed the food catalogue once if it has not already been loaded."""
    logger.info("Seeding food catalogue if necessary...")
    await seed_catalogue()
