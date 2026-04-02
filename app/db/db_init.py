from app.logger import logger

from app.db import Base, engine
from app.db.seed_backbone import DEFAULT_ALGORITHMS, seed_algorithms
from app.db.seed_catalogue import seed_catalogue


async def ensure_schema() -> None:
    """Create any missing tables from ORM models.

    Does not alter or drop existing tables/columns. If you change the model
    shape on a non-empty database, use manual SQL or reset the DB.
    """
    import app.api.schemas  # noqa: F401 — register all Table objects on Base.metadata

    logger.info("Ensuring database schema (create_all for missing tables)...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Schema check complete.")


async def ensure_seed_backbones(algorithms: list[str] | None = None) -> None:
    """Ensure the initial backbone (version 1) exists for the requested algorithms."""
    algos = algorithms or list(DEFAULT_ALGORITHMS)
    await seed_algorithms(algos)


async def ensure_seed_catalogue() -> None:
    """Seed the food catalogue once if it has not already been loaded."""
    logger.info("Seeding food catalogue if necessary...")
    await seed_catalogue()
