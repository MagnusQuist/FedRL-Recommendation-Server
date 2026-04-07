from app.logger import logger

from app.db import Base, engine
from app.db.seed_backbone import DEFAULT_ALGORITHMS, seed_algorithms
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


async def ensure_seed_backbones(algorithms: list[str] | None = None) -> None:
    """Ensure the initial backbone (version 1) exists for the requested algorithms."""
    algos = algorithms or list(DEFAULT_ALGORITHMS)
    await seed_algorithms(algos)


async def ensure_seed_catalogue() -> None:
    """Seed the food catalogue once if it has not already been loaded."""
    logger.info("Seeding food catalogue if necessary...")
    await seed_catalogue()
