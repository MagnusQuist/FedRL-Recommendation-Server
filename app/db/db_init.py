from app.logger import logger

from app.db import Base, engine
from app.db.seed_backbone import seed_federated_backbone
from app.db.seed_catalogue import seed_catalogue


async def ensure_models() -> None:
    """Create any missing tables from database models.

    Does not alter or drop existing tables/columns. If you change the database model
    shape on a non-empty database, use manual SQL or reset the database.
    """
    import app.db.models  # Register database models on Base.metadata

    logger.info("Ensuring database models (create_all for missing tables)...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Model creation complete.")


async def ensure_seed_backbones() -> None:
    """Ensure the initial federated backbone (version 1) exists in the database."""
    await seed_federated_backbone()


async def ensure_seed_catalogue() -> None:
    """Seed the food catalogue once if it has not already been loaded."""
    logger.info("Seeding food catalogue if necessary...")
    await seed_catalogue()
