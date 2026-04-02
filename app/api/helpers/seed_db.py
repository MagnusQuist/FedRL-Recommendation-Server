import os

from app.db.db_init import ensure_schema, ensure_seed_backbones, ensure_seed_catalogue
from app.logger import logger

from app.db.seed_status import is_database_seeded


def _auto_create_schema_enabled() -> bool:
    return os.getenv("AUTO_CREATE_SCHEMA", "false").strip().lower() == "true"


async def ensure_schema_if_enabled() -> None:
    """Create missing tables before serving when AUTO_CREATE_SCHEMA=true."""
    if _auto_create_schema_enabled():
        await ensure_schema()
        logger.info("Schema creation phase done.")
    else:
        logger.info("Skipping create_all (AUTO_CREATE_SCHEMA=false).")


async def seed_data_if_needed() -> None:
    """Backbone + catalogue seed when the DB has no categories yet (can be slow)."""
    if await is_database_seeded():
        logger.info("Existing data detected; skipping seed data population.")
        return

    await ensure_seed_backbones()
    logger.info("Backbone seeding done")

    await ensure_seed_catalogue()
    logger.info("Catalogue seeding done")


async def seed_db() -> None:
    """Ensure schema (if enabled) then seed data — for manual/CLI use; blocks until complete."""
    try:
        await ensure_schema_if_enabled()
        await seed_data_if_needed()
    except Exception:
        logger.exception("seed_db failed.")
        raise
